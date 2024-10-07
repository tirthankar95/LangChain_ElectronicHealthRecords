# https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/
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
print(len(all_splits))
print(all_splits[1])


