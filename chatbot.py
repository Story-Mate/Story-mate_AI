import os
import json
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 모듈에서 필요한 함수들을 임포트
from utils import (
    initialize_chroma_db, fetch_data, initialize_retriever, initialize_llm, get_session_history
)
from template import get_template

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# JSON 저장 파일 경로
JSON_FILE_PATH = "chat_history.json"

class ChatBot:
    def __init__(self, character_name="김첨지", book_title="운수좋은날"):
        # 2) DB 경로 설정
        base_path = f"C:/storymate/{book_title}/data/embedding"
        self.character_name = character_name
        # 3) DB & 리트리버 초기화
        self.q_db = initialize_chroma_db(f"{base_path}/예상질문_chroma_db")
        self.e_db = initialize_chroma_db(f"{base_path}/인물평가_chroma_db")
        self.n_db = initialize_chroma_db(f"{base_path}/전문_chroma_db")
        self.c_db = initialize_chroma_db(f"{base_path}/인물특성_chroma_db")

        self.q_retriever = initialize_retriever(self.q_db)
        self.e_retriever = initialize_retriever(self.e_db)
        self.n_retriever = initialize_retriever(self.n_db)
        self.c_retriever = initialize_retriever(self.c_db)

        # 4) 템플릿 & LLM
        self.prompt_template = get_template(character_name)
        self.llm = initialize_llm(model_name="gpt-4o")

        # 5) 체인 결합 (PromptTemplate → LLM → StrOutputParser)
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        # 6) 대화 기록 저장소
        self.store = {}

        # 7) JSON 파일에서 기존 대화 기록 불러오기
        self.load_chat_history()

    # ✅ JSON에서 대화 기록 불러오기
    def load_chat_history(self):
        """
        JSON 파일에서 기존 대화 기록을 불러와 store에 저장
        """
        if not os.path.exists(JSON_FILE_PATH):
            return  # 파일이 없으면 아무것도 안 함

        with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
            chat_data = json.load(f)

        # JSON 데이터를 store에 복원
        for session_id, messages in chat_data.items():
            self.store[session_id] = messages

        print("✅ 대화 기록이 JSON에서 store로 복원되었습니다.")

    # ✅ JSON에 대화 기록 저장
    def save_chat_history(self):
        """
        store의 대화 기록을 JSON 파일에 저장
        """
        with open(JSON_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(self.store, f, ensure_ascii=False, indent=4)

        print("✅ 대화 기록이 JSON 파일에 저장되었습니다.")

    # ✅ 대화 내용을 요약하는 함수
    def summarize_history(self, session_id):
        """
        특정 세션의 대화 기록을 JSON 파일에서 불러와 요약하는 함수
        """
        # JSON에서 해당 세션의 대화 기록 불러오기
        if not os.path.exists(JSON_FILE_PATH):
            return "이전 대화 기록이 없습니다."

        with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
            chat_data = json.load(f)

        history = chat_data.get(session_id, [])

        if not history:
            return "이전 대화 기록이 없습니다."

        # 요약할 텍스트 변환 (role을 붙여서 정리)
        conversation_text = "\n".join(
            [f"{'사용자' if msg['role'] == 'human' else self.character_name}: {msg['content']}" for msg in history]
        )

        # 요약 프롬프트 설정
        summary_prompt = (
        """
        [이전 대화 요약]
        - 마크다운 형식으로 작성합니다.
        - 사용자의 정보와 대화 주제를 강조하여 chat_history를 요약합니다.

        chat_history : {conversation_text}"""
        )

        # 요약 실행
        summary_chain = ChatPromptTemplate.from_template(summary_prompt) | self.llm | StrOutputParser()
        summary = summary_chain.invoke({"conversation_text": conversation_text})

        print(f"[대화 요약]: {summary}")
        return summary 

    # ✅ 세션 ID 기반으로 대화 기록 불러오기
    def get_session_history(self, session_id):
        """
        특정 session_id에 대한 대화 기록을 반환 (없으면 새로 생성)
        """
        print(f"[대화 세션ID]: {session_id}")

        if session_id not in self.store:  # 세션 ID가 store에 없는 경우
            self.store[session_id] = []

        return self.store[session_id]

    # ✅ 새로운 대화 추가
    def add_conversation(self, session_id, user_query, response):
        """
        새로운 대화를 store에 추가하고 JSON에 저장
        """
        if session_id not in self.store:
            self.store[session_id] = []

        self.store[session_id].append({"role": "human", "content": user_query})
        self.store[session_id].append({"role": "ai", "content": response})

        # 변경된 대화 기록을 JSON에 저장
        self.save_chat_history()

    # ✅ 최종 답변 생성 함수
    def get_answer(self, session_id: str, user_query: str) -> str:
        """
        세션 ID 기반으로 DB 검색 후, 캐릭터 템플릿과 LLM으로 답변 생성
        """
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

        # 체인 실행
        response = self.chain.invoke(input_data)

        # 대화 저장
        self.add_conversation(session_id, user_query, response)

        return response