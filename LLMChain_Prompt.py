from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain import HuggingFaceHub, LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
import logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(filename)s - Line: %(lineno)d - %(message)s',  
)
import json
from huggingface_hub import login
import os

examples = [
    {
        "context": "Muhammad Ali was 74 years old when he died.",
        "question": "How long did Muhammad live?"
        
    },
    {
        "context": "Craig Newmark was born on December 6, 1952.",
        "question": "When was the founder of craigslist born?"
    },
    {
        "context": "The mother of George Washington was Mary Ball Washington. " + \
                   "The father of Mary Ball Washington was Joseph Ball.",
        "question": "Who was the maternal grandfather of George Washington?"

    },
    {
        "context":  "Follow up: Who is the director of Jaws? " + \
                    "Intermediate Answer: The director of Jaws is Steven Spielberg. " + \
                    "Follow up: Where is Steven Spielberg from? " + \
                    "Intermediate Answer: The United States. " + \
                    "Follow up: Who is the director of Casino Royale? " + \
                    "Intermediate Answer: The director of Casino Royale is Martin Campbell. " + \
                    "Follow up: Where is Martin Campbell from? " + \
                    "Intermediate Answer: New Zealand. " + \
                    "So the final answer is: No. ",
        "question": "Are both the directors of Jaws and Casino Royale from the same country?",
    },
]
    
def load_env_vars():
    with open("config.json", "r") as file:
        env = json.load(file)
    for k, v in env.items():
        os.environ[k] = v

if __name__ == '__main__':
    # login(token = "XXX")
    load_env_vars()
    template0 = "Context: {context}\n\nQuestion: {question}\n\nAnswer: "
    example_prompt = PromptTemplate(intput = ["context", "question"], template = template0)
    
    model_id = "openai-community/gpt2"
    llm = HuggingFaceHub(repo_id = model_id, model_kwargs={"temperature": 0.1})
    print(example_prompt.format(**examples[0]))
    # Building LLM chain.
    rag_model = (
        example_prompt
        | llm
        | StrOutputParser()
    )
    print(rag_model.invoke(examples[0]))
    # print(example_prompt.input_schema.schema())
    # print(llm.input_schema.schema())