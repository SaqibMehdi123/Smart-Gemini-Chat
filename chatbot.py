import os
import requests
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

class GeminiChat:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.file_path = os.path.join(os.path.dirname(__file__), "chat-history.txt")
        self.url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        self.headers = {'Content-Type': 'application/json', 'X-goog-api-key': self.api_key}
        self.conversation_history = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model
        self.embeddings = []
        self.messages = []
        
        # load existing chat
        self.load_chat()
    
    def load_chat(self):
        """Load existing chat history"""
        if not os.path.exists(self.file_path):
            return
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("You: "):
                        msg = line[5:]
                        self.conversation_history.append({"role": "user", "parts": [{"text": msg}]})
                        self.messages.append(msg)
                    elif line.startswith("Bot: "):
                        msg = line[5:]
                        self.conversation_history.append({"role": "model", "parts": [{"text": msg}]})
                        self.messages.append(msg)
            
            # generate embeddings for loaded messages
            if self.messages:
                self.embeddings = self.model.encode(self.messages)
                
            print(f"Loaded {len(self.conversation_history)} messages with embeddings.")
        except Exception as e:
            print(f"Error loading chat: {e}")
    
    def save_message(self, prefix, message):
        """Save message to file"""
        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write(f"{prefix}: {message}\n")
    
    def send_message(self, message):
        """Send message to Gemini API"""
        self.conversation_history.append({"role": "user", "parts": [{"text": message}]})
        
        payload = {
            "contents": self.conversation_history,
            "generationConfig": {"temperature": 0.7, "topP": 0.9, "topK": 40, "maxOutputTokens": 2048}
        }
        
        response = requests.post(self.url, headers=self.headers, json=payload)
        if response.status_code != 200:
            return f"Error: {response.status_code}"
        
        bot_response = response.json()['candidates'][0]['content']['parts'][0]['text']
        self.conversation_history.append({"role": "model", "parts": [{"text": bot_response}]})
        
        return bot_response
    
    def check_similarity(self, new_message):
        """Check cosine similarity with previous messages"""
        if not self.messages:
            return "No previous messages to compare."
        
        # get embedding for new message
        new_embedding = self.model.encode([new_message])
        
        # calculate cosine similarities
        similarities = cosine_similarity(new_embedding, self.embeddings)[0]
        
        # find most similar message
        max_idx = np.argmax(similarities)
        max_similarity = similarities[max_idx]
        
        report = f"\n--- Similarity Report ---"
        report += f"\nMost similar to: '{self.messages[max_idx][:50]}...'"
        report += f"\nSimilarity score: {max_similarity:.4f}"
        
        # show top 3 similarities if more than 3 messages
        if len(similarities) >= 3:
            top_3_idx = np.argsort(similarities)[-3:][::-1]
            report += f"\nTop 3 similarities:"
            for i, idx in enumerate(top_3_idx, 1):
                report += f"\n{i}. {similarities[idx]:.4f} - '{self.messages[idx][:30]}...'"
        
        return report
    
    def clear_history(self):
        """Clear all history"""
        self.conversation_history = []
        self.messages = []
        self.embeddings = []
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
        print("Chat history cleared!")
    
    def run(self):
        """Main chat loop"""
        print("Gemini Chat with Sentence Transformers - Type 'exit' to quit, 'clear' to clear history")
        print("-" * 60)
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'clear':
                self.clear_history()
                continue
            
            # check similarity before processing
            similarity_report = self.check_similarity(user_input)
            
            # save and process message
            self.save_message("You", user_input)
            self.messages.append(user_input)
            
            # update embeddings
            if len(self.embeddings) == 0:
                self.embeddings = self.model.encode(self.messages)
            else:
                new_embedding = self.model.encode([user_input])
                self.embeddings = np.vstack([self.embeddings, new_embedding])
            
            # get bot response
            bot_response = self.send_message(user_input)
            print(f"Gemini: {bot_response}")
            
            # save bot response and update embeddings
            self.save_message("Bot", bot_response)
            self.messages.append(bot_response)
            new_bot_embedding = self.model.encode([bot_response])
            self.embeddings = np.vstack([self.embeddings, new_bot_embedding])
            
            # show similarity report
            print(similarity_report)
        
        print("Chat session ended.")

if __name__ == "__main__":
    chat = GeminiChat()
    chat.run()