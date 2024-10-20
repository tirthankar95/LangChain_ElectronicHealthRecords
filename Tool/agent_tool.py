
# https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/
# https://python.langchain.com/v0.1/docs/modules/agents/agent_types/tool_calling/
# https://python.langchain.com/v0.1/docs/modules/agents/how_to/custom_agent/
# https://github.com/langchain-ai/langchain/discussions/23845
# https://github.com/langchain-ai/langchain/discussions/23845
# https://python.langchain.com/docs/integrations/chat/huggingface/

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain import HuggingFaceHub
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
import logging 
from langchain_core.output_parsers import StrOutputParser
logger = logging.getLogger(__name__)
logging.basicConfig(format='[TM] %(pathname)s:%(lineno)d  %(message)s', level = logging.WARNING)

# GET env variables.
import os 
import json 
with open("../config.json", "rb") as file:
    env_dict = json.load(file)
for k, v in env_dict.items():
    os.environ[f"{k}"] = v

# LOAD model
'''
This function initializes an instance of HuggingFaceEndpoint, which connects to a Hugging Face model endpoint for generating text. Let's break down each argument:

    repo_id="HuggingFaceH4/zephyr-7b-beta":
        This specifies the repository ID of the model you are using from Hugging Face. In this case, the model is named "zephyr-7b-beta", and it belongs to the organization or user "HuggingFaceH4". This tells the endpoint which model to use for text generation.

    task="text-generation":
        The task argument defines what kind of task the model will perform. Here, "text-generation" indicates that the model will be used to generate text sequences. Other possible tasks could include text classification, question answering, etc., depending on the model's capabilities.

    max_new_tokens=512:
        This parameter controls the maximum number of new tokens (words, subwords, or characters) the model is allowed to generate. In this case, the model can generate up to 512 tokens in its response. If the generation is supposed to be shorter, the process will stop earlier, but this sets an upper limit.

    do_sample=False:
        This determines whether sampling is used during generation. If do_sample=False, the model will use greedy decoding or beam search (depending on other settings), which chooses the most likely next token at each step. If set to True, the model will sample from the probability distribution of the next token, allowing for more diverse outputs.

    repetition_penalty=1.03:
        The repetition penalty discourages the model from generating repetitive sequences. A value greater than 1 reduces the likelihood of repeating the same token multiple times. In this case, 1.03 slightly penalizes repeated tokens, helping to produce more varied and natural output. The higher this value, the stronger the penalty on repeating words.
'''
model_id = "openai-community/gpt2"
llm = HuggingFaceEndpoint(
                            repo_id = model_id,
                            task = "text-generation",
                            max_new_tokens = 512, 
                            do_sample=False,
                            repetition_penalty=1.03
                        )
chat_model = ChatHuggingFace(llm = llm)
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature = 0.1)

@tool
def bad_search(query: str) -> str:
    """
    This is a bad search engine which returns non-sense string ~ Langchain
    """
    return "LangChain is an open-source framework that helps developers build applications using large language models (LLMs)." 
logging.debug(f'{bad_search.name}')
logging.debug(bad_search.description)
logging.debug(bad_search.args)
logging.debug(bad_search.invoke("$$"))

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
prompt_value = prompt.invoke(
    {
        "input": "Can you summarize recent advancements in AI?",
        "agent_scratchpad": [
            ("ai", "I don't have access to current events, but I can summarize known advancements in AI."),
            ("ai", "Deep learning has seen continued growth, particularly with transformers."),
        ]
    }
)
logging.debug(prompt_value)

tools = [bad_search]
# Construct the Tools agent
# chat_model_with_tools = chat_model.bind_tools(tools)
# agent = create_tool_calling_agent(chat_model_with_tools, tools, prompt)
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: x["intermediate_steps"]
    }
    | prompt
    | llm
    | StrOutputParser()
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
logging.warning(agent.invoke({"input": "what is LangChain?"}))

# Logistic Regression.
# def predict_lr() -> int: 
#     '''
#     - pass the name of the document.
#     - select model according to document name.
#     - show prediction for the patient.
#     '''
#     pass 

