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
model_name = "sentence-transformers/all-mpnet-base-v2"

model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(model_name = model_name, model_kwargs = model_kwargs, encode_kwargs = encode_kwargs)
vectorstore = Chroma.from_documents(documents = all_splits, embedding = hf)

# Display all the records of P4
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
retrieved_docs = retriever.invoke("Display all the records of P4")
print(retrieved_docs[0].page_content)