# https://python.langchain.com/docs/tutorials/rag/
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json 
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='[TM] %(pathname)s:%(lineno)d  %(message)s', level = logging.WARNING)
import os
def load_env_vars():
    with open("config.json", "r") as file:
        env = json.load(file)
    for k, v in env.items():
        os.environ[k] = v

# STEP 1 ~ Document Loader.
loader = CSVLoader("Disease.csv")
docs = loader.load()
logging.info(len(docs))
logging.info(docs[0].page_content)

# STEP 2 ~ Split Documents.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200, 
    add_start_index = True
)
all_splits = text_splitter.split_documents(docs)
logging.info(len(all_splits))
logging.info(all_splits[1])


# STEP 3 ~ Create Vector Store & Retriever.
'''
Now we need to index our 66 text chunks so that we can search over them at runtime. The most common way to do this is to embed 
the contents of each document split and insert these embeddings into a vector database (or vector store). When we want to 
search over our splits, we take a text search query, embed it, and perform some sort of “similarity” search to identify the 
stored splits with the most similar embeddings to our query embedding. The simplest similarity measure is cosine similarity
 — we measure the cosine of the angle between each pair of embeddings (which are high dimensional vectors).
'''
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings 
load_env_vars()
# https://huggingface.co/sentence-transformers/all-mpnet-base-v2
model_embed_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(model_name = model_embed_name, model_kwargs = model_kwargs, encode_kwargs = encode_kwargs)
vectorstore, vector_db_dir = None, "./db"
try:
    vectorstore = Chroma(embedding_function = hf, persist_directory = vector_db_dir)
except Exception as e:
    logging.error(e)
    logging.warning(f'Creating vector store from scratch.')
    vectorstore = Chroma.from_documents(documents = all_splits, embedding = hf, persist_directory = vector_db_dir)
logging.info(vectorstore)
# Display all the records of P4
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
retrieved_docs = retriever.invoke("Display all the records of P4")
logging.info(retrieved_docs[0].page_content)


# Step 4 ~ Create LLM chain.
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate



template1 = """You are an AI assistant in the healthcare industry who is supposed to answer questions based 
on electronic health records of patients provided in the context, 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(intput = ["context", "question"], template = template1)

'''
template = \"\"\"You are superman.
{context}
Question: {question}
Answer:\"\"\"

def get_context(): return \"This is the context from a function.\"
def get_question(): return \"What is the capital of France?\"

filled_template = template.format(context=get_context(), question=get_question())
logging.info(filled_template)
'''

# model_id = "meta-llama/Llama-3.2-1B"
model_id = "microsoft/Phi-3.5-mini-instruct"
llm = HuggingFaceEndpoint(repo_id = model_id, temperature = 1.0)

def QandA(question):
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    rdocs = retriever.invoke(question)
    fmt_docs = format_docs(rdocs)
    return {"context": fmt_docs, "question": question}

def StopHallucinations(response):
    return response.split("Question:")[0]

rag_chain = (
    QandA
    | prompt
    | llm
    | StrOutputParser()
    | StopHallucinations
)

logging.info(rag_chain.invoke("For patient name: P4 what is the average result on test f0?"))

from langchain.agents import AgentExecutor, create_tool_calling_agent


# Logistic Regression.
@tool 
def predict_lr() -> int: 
    pass 