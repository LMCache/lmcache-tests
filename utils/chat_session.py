from openai import OpenAI
import threading
import sys
from io import StringIO
import time


class ChatSession:
    def __init__(self, context_file, port):
        self.context_file = context_file
        with open(context_file, "r") as fin:
            self.context = fin.read()
        
        self.openai_api_key = "EMPTY"
        self.openai_api_base = f"http://localhost:{port}/v1"
        
        self.client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=self.openai_api_key,
            base_url=self.openai_api_base,
        )

        models = self.client.models.list()
        self.model = models.data[0].id

        self.messages = [
            {
                "role": "user",
                "content": f"I've got a document, here's the content:```\n{self.context}\n```."
            }, 
            {
                "role": "assistant",
                "content": "I've got your document"
            }, 
        ]

        print(f"\033[33mLoaded context file: {self.context_file}\033[0m")

        #self.printer = Printer()

    def on_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def on_server_message(self, message):
        self.messages.append({"role": "assistant", "content": message})

    def chat(self, text, max_tokens=1, temperature=0):
        self.on_user_message(text)


        chat_completion = self.client.chat.completions.create(
            messages=self.messages,
            model=self.model,
            temperature=temperature,
            stream=False,
            max_tokens=1,
        )
        
        server_message = chat_completion.choices[0].message.content
        return server_message





