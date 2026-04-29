import os
import requests
from dotenv import load_dotenv


class LLMGenerator:
    def __init__(self):
        load_dotenv()

        self.api_key = os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("LLM_BASE_URL")
        self.model = os.getenv("LLM_MODEL")

        if not self.api_key or not self.base_url or not self.model:
            raise ValueError("Missing LLM_API_KEY, LLM_BASE_URL, or LLM_MODEL in .env")

        self.url = f"{self.base_url}/chat/completions"

    def build_prompt(self, query, retrieved_results):
        context = "\n\n".join(
            [f"Q: {r['question']}\nA: {r['answer']}" for r in retrieved_results[:5]]
        )

        prompt = f"""
You are a professional Amazon customer support assistant.

A user asked:
"{query}"

Here are similar past support cases:

{context}

Based on these, Generate a helpful, clear, and concise support response.

Do NOT ask for sensitive or personal information such as:
- order numbers
- addresses
- phone numbers
- payment details

If needed, ask the user to contact support through official channels.
Do not mention that this is retrieved data.
Answer naturally like a real support agent.
"""

        return prompt

    def generate(self, query, retrieved_results):
        prompt = self.build_prompt(query, retrieved_results)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(self.url, headers=headers, json=data, timeout=60)

        if response.status_code != 200:
            return (
                "The LLM service is currently unavailable. "
                "Here is a suggested response based on similar support cases:\n\n"
                f"{retrieved_results[0]['answer']}"
            )

        result = response.json()

        try:
            return result["choices"][0]["message"]["content"]
        except:
            return str(result)