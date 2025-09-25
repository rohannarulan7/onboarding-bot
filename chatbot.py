from collections import deque
from onboarding_class import OnboardingChatbot
from sentence_transformers import SentenceTransformer, util
import torch

class ChatManager:
    def __init__(self, max_memory=5):
        self.chatbot = OnboardingChatbot()
        self.memory = deque(maxlen=max_memory)  # store last N turns

    def ask(self, user_query: str):
        # Step 1: Retrieve docs
        docs = self.chatbot.fetch_relevant_docs(user_query, top_k=5)
        # Step 2: Generate answer
        answer = self.chatbot.generate_solution(user_query, docs)

        # Update memory
        self.memory.append({"query": user_query, "answer": answer})

        return answer, list(self.memory)
    


class ChatManager2:
    def __init__(self, max_memory=5):
        self.chatbot = OnboardingChatbot()
        self.memory = deque(maxlen=max_memory)  # store last N turns
        # Load re-ranking model once
        self.rerank_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def ask(self, user_query: str):
        # Step 1: Retrieve docs
        docs = self.chatbot.fetch_relevant_docs(user_query, top_k=10)

        # Step 2: Re-rank docs using all-MiniLM-L6-v2
        doc_texts = [d.get("case_description", "") for d in docs]
        query_emb = self.rerank_model.encode(user_query, convert_to_tensor=True)
        doc_embs = self.rerank_model.encode(doc_texts, convert_to_tensor=True)

        scores = util.cos_sim(query_emb, doc_embs)[0]
        ranked_indices = torch.argsort(scores, descending=True)

        reranked_docs = [docs[i] for i in ranked_indices]

        # Step 3: Use top 5 after re-ranking for answer generation
        top_docs = reranked_docs[:5]
        answer = self.chatbot.generate_solution(user_query, top_docs, memory=self.memory)

        # Update memory
        self.memory.append({"query": user_query, "answer": answer})

        return answer, list(self.memory)
 
