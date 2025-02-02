import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from utils import (
    initialize_chroma_db, 
    fetch_data, 
    initialize_retriever, 
    initialize_llm, 
    chat_manager
)
from template import get_template

load_dotenv()

class ChatBot:
    def __init__(self, book_title: str, character_name: str, session_id: str):
        """
        챗봇을 초기화
        - session_id를 기반으로 대화 기록 저장
        - 해당 session_id가 존재하면 기존 대화 기록 불러오기
        """
        self.session_id = session_id  # ✅ 세션 ID 저장

        # 1) DB 경로 설정
        base_path = f"C:/storymate/{book_title}/data/embedding"

        # 2) DB & 리트리버 초기화
        self.q_db = initialize_chroma_db(f"{base_path}/예상질문_chroma_db")
        self.e_db = initialize_chroma_db(f"{base_path}/인물평가_chroma_db")
        self.n_db = initialize_chroma_db(f"{base_path}/전문_chroma_db")
        self.c_db = initialize_chroma_db(f"{base_path}/인물특성_chroma_db")

        self.q_retriever = initialize_retriever(self.q_db)
        self.e_retriever = initialize_retriever(self.e_db)
        self.n_retriever = initialize_retriever(self.n_db)
        self.c_retriever = initialize_retriever(self.c_db)

        # 3) 템플릿 & LLM 초기화
        self.prompt_template = get_template(character_name)
        self.llm = initialize_llm()

        # 4) 체인 결합
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def get_answer(self, query: str) -> tuple:
        """
        사용자의 질문을 받고 DB에서 context 검색 후, LLM으로 답변 생성.
        """
        # ✅ session_id 기반 chat_history 가져오기
        chat_history = chat_manager.get_session_history(self.session_id).messages

        # ✅ DB에서 관련 context 검색
        retrieved_contexts = {
            "context_doc1": fetch_data(self.n_retriever, query),
            "context_doc2": fetch_data(self.e_retriever, query),
            "context_doc3": fetch_data(self.c_retriever, query),
            "context_doc4": fetch_data(self.q_retriever, query),
        }

        # ✅ 체인 입력 데이터
        input_data = {
            "query": query,
            "chat_history": chat_history,  # ✅ 이전 대화 기록 추가
            **retrieved_contexts
        }

        # ✅ 체인 실행
        response = self.chain.invoke(input_data)

        # ✅ session_id별로 대화 기록 저장
        chat_manager.update_session_history(self.session_id, query, response)

        return self.session_id, response