import os
import time
from anthropic import Anthropic
from types import SimpleNamespace

class ClaudeAgent():
    def __init__(self, kwargs: dict):
        self.api_key = 'ANTHROPIC_API_KEY'
        if not self.api_key:
            raise ValueError("Please set ANTHROPIC_API_KEY environment variable")
            
        self.client = Anthropic(api_key=self.api_key)
        self.args = SimpleNamespace(**kwargs)
        self._set_default_args()

    def _set_default_args(self):
        if not hasattr(self.args, 'model'):
            self.args.model = "claude-3-sonnet-20240229"
        if not hasattr(self.args, 'temperature'):
            self.args.temperature = 1.0
        if not hasattr(self.args, 'max_tokens'):
            self.args.max_tokens = 256
        if not hasattr(self.args, 'top_p'):
            self.args.top_p = 1.0

    def generate(self, prompt):
        while True:
            try:
                response = self.client.messages.create(
                    model=self.args.model,
                    max_tokens=self.args.max_tokens,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    messages=[{"role": "user", "content": prompt}]
                )
                break
            except Exception as e:
                print(f">>> Error: {str(e)}\nRetrying...")
                time.sleep(1)

        return response

    def parse_basic_text(self, response):
        output = response.content[0].text.strip()
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