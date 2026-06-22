import os
import requests
import requests.exceptions
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
        if not retrieved_results:
            context = "No similar support cases were found."
        else:
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

    def fallback_answer(self, retrieved_results):
        if retrieved_results and len(retrieved_results) >= 1 and retrieved_results[0].get("answer"):
            answer = retrieved_results[0]["answer"]
            return (
                "Based on similar support cases, here is the most relevant available guidance:\n\n" + answer
            )

        return (
            "I'm sorry, I couldn't find enough relevant information in the support knowledge base. "
            "Please contact official customer support for more accurate help."
        )

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

        try:
            response = requests.post(self.url, headers=headers, json=data, timeout=60)
        except requests.exceptions.RequestException:
            return self.fallback_answer(retrieved_results)

        if response.status_code != 200:
            return self.fallback_answer(retrieved_results)

        try:
            result = response.json()
        except Exception:
            return self.fallback_answer(retrieved_results)

        try:
            content = result["choices"][0]["message"]["content"]
        except Exception:
            return self.fallback_answer(retrieved_results)

        if not content or str(content).strip() == "":
            return self.fallback_answer(retrieved_results)

        return content