from langchain.document_loaders import PyPDFium2Loader

loader = PyPDFium2Loader("운수좋은날.pdf")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
)

texts = text_splitter.split_documents(data)

from langchain.embeddings import HuggingFaceEmbeddings

model_name = "jhgan/ko-sroberta-multitask"
ko_embedding= HuggingFaceEmbeddings(
    model_name=model_name
)

from langchain.vectorstores import Chroma

db = Chroma.from_documents(data, ko_embedding, persist_directory="./chroma_db")