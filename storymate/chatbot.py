import os
import json
from dotenv import load_dotenv

# langchain_core나 utils 등은 기존 프로젝트 구조를 따른다고 가정
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 모듈에서 필요한 함수들을 임포트
from utils import (
    initialize_chroma_db, fetch_data, initialize_retriever, initialize_llm
)
from template import get_character_template

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class ChatBot:
    # ✅ ChatBot 클래스 초기화
    def __init__(self, character_name, book_title):
        # 1) DB(Chroma) 경로 설정
        base_path = f"{book_title}/data/embedding"
        self.character_name = character_name
        self.book_title = book_title

        # 2) DB & 리트리버 초기화
        self.q_db = initialize_chroma_db(f"{base_path}/예상질문_chroma_db")
        self.e_db = initialize_chroma_db(f"{base_path}/인물평가_chroma_db")
        self.n_db = initialize_chroma_db(f"{base_path}/전문_chroma_db")
        self.c_db = initialize_chroma_db(f"{base_path}/인물특성_chroma_db")

        self.q_retriever = initialize_retriever(self.q_db)
        self.e_retriever = initialize_retriever(self.e_db)
        self.n_retriever = initialize_retriever(self.n_db)
        self.c_retriever = initialize_retriever(self.c_db)

        # 3) 템플릿 & LLM
        self.prompt_template = get_character_template(book_title, character_name)
        self.llm = initialize_llm(model_name="gpt-4o")

        # 4) 체인 결합 (PromptTemplate → LLM → StrOutputParser)
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        # 5) JSON 저장/로드를 위한 준비
        self.chat_history_file = "chat_history.json"
        self.chat_data = self._load_chat_data()


    # --------------------------------------
    # JSON 기반 대화 기록 관리 함수들
    # --------------------------------------
    def _load_chat_data(self):
        """JSON 파일에서 전체 대화 기록을 불러오는 내부 함수"""
        if os.path.exists(self.chat_history_file):
            with open(self.chat_history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return {}

    def _save_chat_data(self):
        """JSON 파일에 전체 대화 기록을 저장하는 내부 함수"""
        with open(self.chat_history_file, "w", encoding="utf-8") as f:
            json.dump(self.chat_data, f, indent=4, ensure_ascii=False)

    def load_chat_history(self, session_id):
        """
        특정 session_id의 대화 기록을 JSON에서 불러옴
        """
        return self.chat_data.get(session_id, [])

    def save_chat_history(self, session_id, role, content):
        """
        session_id에 해당하는 대화 기록을 JSON 파일에 저장
        """
        if session_id not in self.chat_data:
            self.chat_data[session_id] = []
        self.chat_data[session_id].append({"role": role, "content": content})
        self._save_chat_data()

    def add_conversation(self, session_id, user_query, response):
        """
        새로운 대화를 self.chat_data에 저장
        """
        self.save_chat_history(session_id, "human", user_query)
        self.save_chat_history(session_id, "ai", response)


    # ✅ 대화 요약
    def summarize_history(self, session_id):
        """
        특정 세션의 대화 기록을 불러와 요약
        """
        history = self.load_chat_history(session_id)

        if not history:
            return "이전 대화 기록이 없습니다."

        # 요약할 텍스트 변환
        conversation_text = "\n".join(
            [f"{'사용자' if msg['role'] == 'human' else self.character_name}: {msg['content']}" 
             for msg in history]
        )

        # 요약 프롬프트
        summary_prompt = """
        [이전 대화 요약]
        - 마크다운 형식으로 작성합니다.
        - 사용자의 정보와 대화 주제를 강조하여 chat_history를 요약하세요.

        chat_history : {conversation_text}
        """

        summary_chain = ChatPromptTemplate.from_template(summary_prompt) | self.llm | StrOutputParser()
        summary = summary_chain.invoke({"conversation_text": conversation_text})
        return summary


    # ✅ 최종 답변 생성 함수
    def get_answer(self, session_id: str, user_query: str) -> str:
        """
        세션 ID 기반으로 검색 후 캐릭터 템플릿과 LLM으로 답변 생성
        """
        if not isinstance(user_query, str) or not user_query.strip():
            return "오류: 질문이 없습니다."

        # 각 DB에서 context를 검색
        question_context  = fetch_data(self.q_retriever, user_query)
        evaluate_context  = fetch_data(self.e_retriever, user_query)
        novel_context     = fetch_data(self.n_retriever, user_query)
        character_context = fetch_data(self.c_retriever, user_query)

        # 이전 대화 내용 요약
        summarized_history = self.summarize_history(session_id)

        # 체인에 넣을 input_data
        input_data = {
            "context_doc1": novel_context,
            "context_doc2": evaluate_context,
            "context_doc3": character_context,
            "context_doc4": question_context,
            "query": user_query,
            "chat_history": summarized_history,
        }

        response = self.chain.invoke(input_data)

        # 대화 저장
        self.add_conversation(session_id, user_query, response)

        return response
