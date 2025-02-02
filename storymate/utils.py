import os
import json
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ JSON 파일 경로
STORE_PATH = "chat_history.json"

class ChatHistoryManager:
    """
    session_id 기반으로 chat_history를 JSON 파일에 저장 및 불러오는 클래스
    """
    def __init__(self, store_path=STORE_PATH):
        self.store_path = store_path
        self.store = self._load_store()

    def _load_store(self):
        """ JSON 파일에서 대화 기록(store)을 불러옴 """
        if os.path.exists(self.store_path):
            with open(self.store_path, "r", encoding="utf-8") as f:
                return json.load(f)  # JSON 파일 읽기
        return {}

    def _save_store(self):
        """ 현재 store을 JSON 파일에 저장 """
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self.store, f, ensure_ascii=False, indent=4)  # JSON 저장

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        """
        session_id 기반으로 chat history를 가져옴.
        - session_id가 store에 있으면 해당 기록을 불러옴.
        - 없으면 새로 생성하여 store에 저장.
        """
        if session_id not in self.store:
            self.store[session_id] = {"messages": []}  # 새로운 세션 생성
        return ChatMessageHistory(messages=self.store[session_id]["messages"])

    def update_session_history(self, session_id: str, user_message: str, ai_message: str):
        """
        session_id에 대한 대화 기록을 업데이트하고 JSON 파일에 저장
        """
        history = self.get_session_history(session_id)
        history.add_user_message(user_message)
        history.add_ai_message(ai_message)

        # ✅ store에 최신 대화 내용 저장 후 JSON 저장
        self.store[session_id] = {"messages": history.messages}
        self._save_store()

def initialize_chroma_db(persist_directory: str) -> Chroma:
    """
    Chroma DB를 초기화하여 반환합니다.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )


# ✅ ChatHistoryManager 인스턴스 생성 (전역적으로 사용)
chat_manager = ChatHistoryManager()



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

def initialize_llm(model_name: str = "gpt-4o", temperature: float = 0):
    """
    ChatOpenAI 모델을 초기화하고 반환합니다.
    """
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=OPENAI_API_KEY
    )