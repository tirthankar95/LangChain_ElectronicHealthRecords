# https://python.langchain.com/docs/integrations/document_loaders/microsoft_excel/
from langchain_community.document_loaders.csv_loader import CSVLoader
loader = CSVLoader("Disease.csv")
docs = loader.load()

print(len(docs))
print(docs[0].page_content)