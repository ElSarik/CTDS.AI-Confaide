import os
import time
import requests
from types import SimpleNamespace

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

class DeepseekAPIAgent():
    def __init__(self, kwargs: dict):
        self.api_key = "DEEPSEEK_API_KEY"
        if not self.api_key:
            raise ValueError("Please set DEEPSEEK_API_KEY environment variable")
            
        self.args = SimpleNamespace(**kwargs)
        self._set_default_args()

    def _set_default_args(self):
        if not hasattr(self.args, 'model'):
            self.args.model = "deepseek-chat"
        if not hasattr(self.args, 'temperature'):
            self.args.temperature = 1.0
        if not hasattr(self.args, 'max_tokens'):
            self.args.max_tokens = 256
        if not hasattr(self.args, 'top_p'):
            self.args.top_p = 1.0

        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def generate(self, prompt):
        payload = {
            "model": self.args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.args.temperature,
            "max_tokens": self.args.max_tokens,
            "top_p": self.args.top_p
        }

        while True:
            try:
                response = requests.post(DEEPSEEK_API_URL, json=payload, headers=self.headers)
                if response.status_code == 200:
                    break
                print(f">>> Error: {response.text}\nRetrying...")
                time.sleep(1)
            except Exception as e:
                print(f">>> Error: {str(e)}\nRetrying...")
                time.sleep(1)

        return response

    def parse_basic_text(self, response):
        output = response.json()['choices'][0]['message']['content'].strip()
        return output

    def interact(self, prompt):
        response = self.generate(prompt)
        output = self.parse_basic_text(response)
        return output

    def batch_interact(self, batch_texts):
        responses = []
        for text in batch_texts:
            response = self.interact(text)
            responses.append(response)
        return responses 