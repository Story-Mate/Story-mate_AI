import pymysql
# MariaDB 연결 함수
def get_db_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="1234",  # HeidiSQL에서 설정한 MariaDB 비밀번호 입력
        database="chatdb",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )





import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 모듈에서 필요한 함수들을 임포트
from utils import (
    initialize_chroma_db, fetch_data, initialize_retriever, initialize_llm
)
from template import get_template

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")




class ChatBot:

    # ✅ ChatBot 클래스 초기화
    def __init__(self, character_name="김첨지", book_title="운수좋은날"):
        # 2) DB 경로 설정
        base_path = f"{book_title}/data/embedding"
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




    # ✅ DB에서 대화 기록 불러오기
    def load_chat_history(self, session_id):
        """
        특정 session_id의 대화 기록을 MariaDB에서 불러옴
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT role, content FROM conversations WHERE session_id = %s ORDER BY created_at", (session_id,))
        messages = cursor.fetchall()
        conn.close()
        return messages



    # ✅ DB에 대화 기록 저장
    def save_chat_history(self, session_id, role, content):
        """
        MariaDB에 새로운 대화 내용을 저장
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (session_id, role, content) VALUES (%s, %s, %s)",
            (session_id, role, content)
        )
        conn.commit()
        conn.close()



    # ✅ 새로운 대화 추가 (사용자 → 챗봇)
    def add_conversation(self, session_id, user_query, response):
        """
        새로운 대화를 MariaDB에 저장
        """
        self.save_chat_history(session_id, "human", user_query)
        self.save_chat_history(session_id, "ai", response)



    # ✅ 대화 내용을 요약하는 함수
    def summarize_history(self, session_id):
        """
        특정 세션의 대화 기록을 MariaDB에서 불러와 요약하는 함수
        """

        history = self.load_chat_history(session_id)


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



    # ✅ 최종 답변 생성 함수
    def get_answer(self, session_id: str, user_query: str) -> str:
        """
        세션 ID 기반으로 DB 검색 후, 캐릭터 템플릿과 LLM으로 답변 생성
            """
        
        if not isinstance(user_query, str) or not user_query.strip():
            return "오류: 질문이 없습니다."

        print(f"📌 get_answer()에서 user_query 확인: {user_query} (type: {type(user_query)})")
        
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