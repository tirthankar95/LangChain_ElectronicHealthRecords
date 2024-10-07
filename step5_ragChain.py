# https://python.langchain.com/docs/tutorials/rag/
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = CSVLoader("Disease.csv")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200, 
    add_start_index = True
)
all_splits = text_splitter.split_documents(docs)

'''
Now we need to index our 66 text chunks so that we can search over them at runtime. The most common way to do this is to embed 
the contents of each document split and insert these embeddings into a vector database (or vector store). When we want to 
search over our splits, we take a text search query, embed it, and perform some sort of “similarity” search to identify the 
stored splits with the most similar embeddings to our query embedding. The simplest similarity measure is cosine similarity
 — we measure the cosine of the angle between each pair of embeddings (which are high dimensional vectors).
'''

from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings 
import os 

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# https://huggingface.co/sentence-transformers/all-mpnet-base-v2
model_embed_name = "sentence-transformers/all-mpnet-base-v2"

model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(model_name = model_embed_name, model_kwargs = model_kwargs, encode_kwargs = encode_kwargs)
vectorstore = Chroma.from_documents(documents = all_splits, embedding = hf)

# Display all the records of P4
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})
'''
retrieved_docs = retriever.invoke("Display all the records of P4")
print(retrieved_docs[0].page_content)
'''

# Create rag chain.
from langchain import HuggingFaceHub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

template = """You are an AI assistant in the healthcare industry who is supposed to answer questions based 
on electronic health records of patients provided in the context, 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Answer:"""
prompt = PromptTemplate.from_template(template)

'''
template = \"\"\"You are 

{context}

Question: {question}

Answer:\"\"\"

def get_context(): return \"This is the context from a function.\"
def get_question(): return \"What is the capital of France?\"

filled_template = template.format(context=get_context(), question=get_question())
print(filled_template)
'''

model_id = "openai-community/gpt2"
llm = HuggingFaceHub(repo_id = model_id, model_kwargs={"temperature": 1.0})

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm 
    | StrOutputParser()
)

print(rag_chain.invoke("For patient P4 what is the average result on test f0?"))

# Integrate prompt & load_model.
# Use LangSmith.
# Try to save vector dB to make things faster.