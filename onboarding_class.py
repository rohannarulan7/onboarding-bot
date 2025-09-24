from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import json
import pysolr
from tqdm import tqdm
import requests
from dotenv import load_dotenv
load_dotenv()

class OnboardingChatbot:
    def __init__(self):
        
        self.embedding_host = ""
        self.embedding_url = ""
        self.pred_handle = ""
    def add_vectors(self):
        embedding_host = self.embedding_host
        embedding_url = self.embedding_url
        url = embedding_url
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
    def generate_solution(self, user_query: str, relevant_docs: list, model_name: str = "gpt-4o"):
        """
        
        Takes user query + relevant documents, and generates a solution using the model.
        """
        # Extract just the text content from docs
        docs_text = "\n\n".join(
            [doc.get("case_description", "") for doc in relevant_docs]
        )

        # Prompt template
        template = """You are an onboarding assistant.
            Use the relevant documents below to answer the user's query.

            User Query:
            {query}

            Relevant Documents:
            {docs}

            Answer clearly and step by step, using the documents when possible.
            """

        prompt = PromptTemplate.from_template(template)
        final_prompt = prompt.format(query=user_query, docs=docs_text)

        # Call model
        llm = ChatOpenAI(model=model_name, temperature=0)  # deterministic
        answer = llm.predict(final_prompt)

        return answer
