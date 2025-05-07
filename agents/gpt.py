import os
import time
from openai import OpenAI
from types import SimpleNamespace

class GPT3BaseAgent():
    def __init__(self, kwargs: dict):
        self.client = OpenAI(api_key="OPENAI_API_KEY")
        self.args = SimpleNamespace(**kwargs)
        self._set_default_args()

    def _set_default_args(self):
        if not hasattr(self.args, 'engine'):
            self.args.engine = "text-davinci-003"
        if not hasattr(self.args, 'temperature'):
            self.args.temperature = 0.9
        if not hasattr(self.args, 'max_tokens'):
            self.args.max_tokens = 256
        if not hasattr(self.args, 'top_p'):
            self.args.top_p = 0.9
        if not hasattr(self.args, 'frequency_penalty'):
            self.args.frequency_penalty = 0.7
        if not hasattr(self.args, 'presence_penalty'):
            self.args.presence_penalty = 0

    def generate(self, prompt):
        while True:
            try:
                completion = self.client.Completion.create(
                    engine=self.args.engine,
                    prompt=prompt,
                    temperature=self.args.temperature,
                    max_tokens=self.args.max_tokens,
                    top_p=self.args.top_p,
                    frequency_penalty=self.args.frequency_penalty,
                    presence_penalty=self.args.presence_penalty,
                    stop=self.args.stop_tokens if hasattr(self.args, 'stop_tokens') else None,
                    logprobs=self.args.logprobs if hasattr(self.args, 'logprobs') else 0,
                    echo=self.args.echo if hasattr(self.args, 'echo') else False
                )
                break
            except Exception as e:
                print("Error: {}".format(e))
                time.sleep(2)
                continue

        return completion

    def parse_basic_text(self, response):
        output = response.choices[0].text.strip()

        return output

    def parse_ordered_list(self, numbered_items):
        ordered_list = numbered_items.split("\n")
        output = [item.split(".")[-1].strip() for item in ordered_list if item.strip() != ""]

        return output

    def interact(self, prompt):
        response = self.generate(prompt)
        output = self.parse_basic_text(response)

        return output

class ConversationalGPTBaseAgent(GPT3BaseAgent):
    def __init__(self, kwargs: dict):
        super().__init__(kwargs)

    def _set_default_args(self):
        if not hasattr(self.args, 'model'):
            self.args.model = "gpt-4-0613"
        if not hasattr(self.args, 'temperature'):
            self.args.temperature = 0.9
        if not hasattr(self.args, 'max_tokens'):
            self.args.max_tokens = 256
        if not hasattr(self.args, 'top_p'):
            self.args.top_p = 0.9
        if not hasattr(self.args, 'frequency_penalty'):
            self.args.frequency_penalty = 0.7
        if not hasattr(self.args, 'presence_penalty'):
            self.args.presence_penalty = 0

    def generate(self, prompt):
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.args.model,
                    messages=[{"role": "user", "content": "{}".format(prompt)}]
                )
                break
            except Exception as e: 
                print("Error: {}".format(e))
                time.sleep(2)
                continue

        return completion

    def parse_basic_text(self, response):
        output = response.choices[0].message.content.strip()
        return output
    
    def batch_interact(self, batch_texts):
        responses = []
        for text in batch_texts:
            response = self.interact(text)
            responses.append(response)
        return responses