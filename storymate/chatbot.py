import os
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser

# 모듈에서 필요한 함수들을 임포트
from utils import initialize_chroma_db, fetch_data, initialize_retriever, initialize_llm
from template import get_template

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class ChatBot:
    def __init__(self, character_name="김첨지", book_title="운수좋은날"):
        # 2) DB 경로 설정
        base_path = f"C:/storymate/{book_title}/data/embedding"

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
        self.llm = initialize_llm(model_name="gpt-3.5-turbo")

        # 5) 체인 결합 (PromptTemplate → LLM → StrOutputParser)
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def get_answer(self, user_query: str) -> str:
        """
        DB 검색 후, 캐릭터 템플릿과 LLM으로 답변 생성
        """
        # 각 DB에서 context를 검색
        question_context  = fetch_data(self.q_retriever, user_query)
        evaluate_context  = fetch_data(self.e_retriever, user_query)
        novel_context     = fetch_data(self.n_retriever, user_query)
        character_context = fetch_data(self.c_retriever, user_query)

        # 체인에 넣을 input_data
        input_data = {
            "context_doc1": novel_context,
            "context_doc2": evaluate_context,
            "context_doc3": character_context,
            "context_doc4": question_context,
            "query": user_query,
        }

        # 체인 실행
        response = self.chain.invoke(input_data)
        return response