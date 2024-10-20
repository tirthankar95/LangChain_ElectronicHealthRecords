from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

import logging 
logger = logging.getLogger(__name__)
logging.basicConfig(format='[TM] %(pathname)s:%(lineno)d  %(message)s', level = logging.WARNING)

# GET env variables.
import os 
import json 
with open("../config.json", "rb") as file:
    env_dict = json.load(file)
for k, v in env_dict.items():
    os.environ[f"{k}"] = v

# -> Not all models can use a tool.
model_id = "google/flan-t5-xxl" # -> Bad
model_id = "baffo32/decapoda-research-llama-7B-hf" # -> Bad
model_id = "microsoft/Phi-3.5-mini-instruct"
# model_id = "meta-llama/Llama-3.1-8B-Instruct"

llm = HuggingFaceEndpoint(
                            repo_id = model_id,
                            task = "text-generation",
                            max_new_tokens = 512, 
                            do_sample=False,
                            repetition_penalty=1.03
                        )
llm = ChatHuggingFace(llm = llm)
messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(
        content="What happens when an unstoppable force meets an immovable object?"
    ),
]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2
logging.warning(f'{magic_function.name}')
logging.warning(magic_function.description)
logging.warning(magic_function.args)

tools = [magic_function]
llm_tools = llm.bind_tools(tools)

agent = create_tool_calling_agent(llm_tools, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, \
                               handle_parsing_errors=True, max_iterations=5)

query = "Call function magic_function(3) from tools and tell me the result."
messages = [
    SystemMessage(content="You're an ai assistant, who uses tools."),
    HumanMessage(content=f"{query}"),
]
'''
logging.info(
    agent_executor.invoke(
        {"input": f"{query}"}
        # prompt.invoke( -> doesn't work with agent_executor.
        #     {
        #         "input": query,
        #         "agent_scratchpad": [
        #             ("ai", "Use the search tool."),
        #             ("ai", "Does the results make sense."),
        #         ]
        #     }
        # )
    )
)
'''
logging.info(
    llm.invoke(
    # Put in prompt.
        prompt.invoke(
            {
                "input": "Can you summarize recent advancements in AI?", 
                "agent_scratchpad": [
                    ("ai", "I don't have access to current events, but I can summarize known advancements in AI."),
                    ("ai", "Deep learning has seen continued growth, particularly with transformers."),
                ]
            }
        )
    )
)


