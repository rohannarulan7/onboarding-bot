from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import json
import pysolr
from tqdm import tqdm
import requests
import os
from dotenv import load_dotenv
import os
load_dotenv()

class LLMConfig:
    
    model_name = 'google/gemma-2-9b-it' # for gpt - "gpt-4o"
    source = 'vllm' # use 'gpt' / 'vllm'
    temperature = 0
    vllm_port = '5010'
    vllm_host = '34.145.11.180'

from openai import OpenAI

class LLMExtractor:
    def __init__(self):
        self.model_name = LLMConfig.model_name
        self.source = LLMConfig.source
        self.temperature = LLMConfig.temperature

        if self.source == "vllm":
            base_url = f"http://{LLMConfig.vllm_host}:{LLMConfig.vllm_port}/v1"
            self.llm = OpenAI(
                base_url=base_url,
                api_key="EMPTY"  # vLLM doesn’t check keys by default
            )

class OnboardingChatbot:
    def __init__(self):
        self.embedding_host = os.getenv("EMBEDDING_HOST")
        self.embedding_url = f"http://{self.embedding_host}"+os.getenv("EMBEDDING_URL")
        self.pred_url = os.getenv('PRED_URL')
        self.pred_handle = pysolr.Solr(self.pred_url)
    def add_vectors(self):
        embedding_host = self.embedding_host
        url = self.embedding_url
        headers = {
            'Content-Type': 'application/json'
        }
        #query = f"type:OBS AND {SolrConfig.CREATED_DATE}:[{self.extraction_start_time} TO *] AND {SolrConfig.CREATED_BY}:system AND -merged:true AND -active:false" if self.extraction_start_time else "type:OBS AND -merged:true AND -active:false"
        query = '*:*'
        pred_handle = self.pred_handle
        all_obs = pred_handle.search(
            q=query, 
            rows=10000
        ).docs

        to_add = []
        progress_desc = f"Adding vectors to {query} observations"

        for i in tqdm(range(0, len(all_obs), 5), total=(len(all_obs) // 5) + 1, desc=progress_desc):
            batch = all_obs[i:i+5]
            payload = json.dumps({
                "sentences": [doc.get('case_description', '') for doc in batch]

            })
            resp = requests.post(url, headers=headers, data=payload).json()
            if "embeddings" not in resp:
                print("Error from embedding API:", resp)
                continue
            response = resp["embeddings"]
                        

            to_add.extend({"id": doc["id"], "vector": vector} for doc, vector in zip(batch, response))

        for i in range(0,len(to_add),5):
            pred_handle.add(to_add[i:i+5],fieldUpdates={"vector":"set"}, commit=True) 



    def fetch_relevant_docs(self, user_query: str, top_k: int = 5):
        """
        Takes a user query, generates its embedding, and retrieves top-K relevant documents from Solr.
        """
        # Step 1: Generate embedding for query
        embedding_host = self.embedding_host
        embedding_url = self.embedding_url
        url = embedding_url
        headers = {"Content-Type": "application/json"}
        payload = json.dumps({"sentences": [user_query]})
        response = requests.post(url, headers=headers, data=payload).json()
        query_embedding = response["embeddings"][0]

        # Step 2: Format Solr KNN query
        # Solr expects the embedding vector as a JSON array string
        vector_str = "[" + ",".join(map(str, query_embedding)) + "]"
        solr_query = "{!knn f=vector topK=" + str(top_k) + "}" + vector_str

        pred_handle = self.pred_handle

        # Step 3: Execute search in Solr
        results = pred_handle.search(
            q=solr_query,
            fl="id,case_description,score"  # return id, text, and similarity score
        ).docs

        return results
    def generate_solution(self, user_query: str, relevant_docs: list, memory: list = None, model_name: str = "gpt-4o"):
        """
        
        Takes user query + relevant documents, and generates a solution using the model.
        """
        # Extract just the text content from docs
        docs_text = "\n\n".join(
            [doc.get("case_description", "") for doc in relevant_docs]
        )

        # Format memory into a conversation string
        memory_text = ""
        if memory:
            memory_text = "\n".join(
                [f"User: {m['query']}\nAssistant: {m['answer']}" for m in memory]
            )


        # Prompt template
        template = """You are an onboarding assistant.
        Use the relevant documents below, along with the chat history, to answer the user's query.

        Chat History:
        {memory}

        User Query:
        {query}

        Relevant Documents:
        {docs}

        Answer clearly and step by step, using the documents and history when possible.
        """

        prompt = PromptTemplate.from_template(template)
        final_prompt = prompt.format(
            query=user_query,
            docs=docs_text,
            memory=memory_text
        )

        # Call model
        # llm = ChatOpenAI(model=model_name, temperature=0)  # deterministic
        # answer = llm.predict(final_prompt)
        model = LLMExtractor()
        resp = model.llm.chat.completions.create(
        model=model.model_name,
        messages=[{"role": "user", "content": final_prompt}],
        temperature=model.temperature
        )
        answer = resp.choices[0].message.content

        return answer
