'''    
    https://python.langchain.com/docs/integrations/platforms/huggingface/
'''
import os 
import getpass 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub import login
from langchain import HuggingFaceHub

if __name__ == '__main__':
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    # login(token = "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")
    model_id = "openai-community/gpt2"
    llm = HuggingFaceHub(repo_id = model_id, model_kwargs={"temperature":1})
    print(llm.invoke("Hello how are you?"))

