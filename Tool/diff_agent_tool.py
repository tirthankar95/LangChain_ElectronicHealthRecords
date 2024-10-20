from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import logging 
logger = logging.getLogger(__name__)
logging.basicConfig(format='[TM] %(pathname)s:%(lineno)d  %(message)s', \
                    level = logging.WARN)

model_id = "black-forest-labs/FLUX.1-dev" # -> doesn't work
model_id = "google/flan-t5-large" # -> doesn't work
model_id = "meta-llama/Llama-2-7b-chat-hf" # -> access required
model_id = "google/gemma-2-2b" # -> too big to be loaded. Use spaces or inference endpoints.
model_id = "HuggingFaceH4/zephyr-7b-beta" # -> works
model_id = "microsoft/Phi-3.5-mini-instruct" # -> works

llm = HuggingFaceEndpoint(
                            repo_id = model_id,
                            task = "text-generation",
                            do_sample=False,
                            repetition_penalty=1.03
                        )
llm = ChatHuggingFace(llm = llm)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
logging.info(wiki.run("langchain"))

# list of tools.
@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2

tools = ['wiki','magic_function']
tool_fn = [wiki, magic_function]
llm_tools = llm.bind_tools(tool_fn)

def StopHallucinations(response):
    return response.split("Question:")[0]
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an helpful ai assistant, if you can't answer a question "+\
                    "name the appropriate tool name followed by the input that goes "+\
                    f"to the tool. Here are the list of tools [{", ".join(tools)}]"),
        ("human", "{query}")
    ]
)
agent = (
    {"query": lambda x: x["query"]}
    | prompt
    | llm 
    | StrOutputParser()
    # | StopHallucinations
)

# query1 = "Call function magic_function(3) from tools and tell me the result."
# logging.warning(agent.invoke({"query": f"{query1}"}))

# query2 = "Can you list the name of tools provided to you in the context."
# logging.warning(agent.invoke({"query": f"{query2}"}))

# query3 = "Call function magic_function(3), what [Tool] from the list provided and [Input] should I use."
# logging.warning(agent.invoke({"query": f"{query3}"}))

def ask_me_anything(query, history):
    return agent.invoke({"query": f"{query}"})