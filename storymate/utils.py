import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def initialize_chroma_db(persist_directory: str) -> Chroma:
    """
    Chroma DB를 초기화하여 반환합니다.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )


def fetch_data(retriever, query: str, max_docs: int = 3) -> list:
    """
    retriever.invoke(query) 결과에서 최대 max_docs개의 문서만 추출,
    각 문서의 page_content를 리스트로 반환
    """
    docs = retriever.invoke(query)
    results = []
    for i, doc in enumerate(docs):
        if i >= max_docs:
            break
        results.append(doc.page_content)
    return results

def initialize_retriever(db, k: int = 3):
    """
    Chroma DB로부터 Retriever를 생성하여 반환합니다.
    - search_type="similarity_score_threshold"
    - score_threshold=0.8
    """
    return db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.8}
    )

def initialize_llm(model_name: str = "gpt-3.5-turbo", temperature: float = 0):
    """
    ChatOpenAI 모델을 초기화하고 반환합니다.
    """
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=OPENAI_API_KEY
    )