from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

data_list = ['전문', '예상질문', '인물특성', '인물평가']

def load_and_split_txt(txt_path, chunk_size=100, chunk_overlap=0):
    # TXT 파일 로드
    loader = TextLoader(txt_path, encoding="utf-8")
    data = loader.load()

    # 텍스트 분할기 초기화
    text_splitter = RecursiveCharacterTextSplitter(
        separators = '\n',
        chunk_size = chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    # 텍스트 분할 후 반환
    return text_splitter.split_documents(data)


# OpenAI 임베딩 초기화
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


for data in data_list:
    file_path = f'data/{data}.txt'
    
    if os.path.exists(file_path):
        texts = load_and_split_txt(file_path)
        db = Chroma.from_documents(
            documents = texts, 
            embedding = embeddings, 
            persist_directory = f"./data/embedding/{data}_chroma_db"
        )
        print(f"✅ {data} 처리 완료")
