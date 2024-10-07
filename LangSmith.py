import os 
import json 

env = {}
with open("config.json", "r") as config:
    env = json.load(config)

for k, v in env.items():
    os.environ[k] = v 

from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
llm.invoke("Hello, world!")