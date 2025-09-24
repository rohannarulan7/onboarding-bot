from collections import deque
from onboarding_class import OnboardingChatbot

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
