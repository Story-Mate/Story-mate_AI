from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 데이터베이스 초기화 함수
def initialize_chroma_db(persist_directory, embedding_function):
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

# 검색기 초기화 함수
def initialize_retriever(db, k=3):
    return db.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": 10, "lambda_mult": 0.6})

# 템플릿 초기화 함수
def initialize_template():
    return ChatPromptTemplate.from_template(
"""
[시스템 프롬프트/역할 지시]

- 김첨지는 1920년대 일제강점기 서울에서 인력거를 끌며 살아갑니다.
- 그는 거칠고 소박한 말투를 쓰면서도, 가족(특히 아내)에 대한 애정과 걱정을 동시에 지닌 인물입니다.
- 답변 시, 당시의 시대적·경제적 배경, 김첨지의 심리(이중적 태도)를 반영해주세요.
- 다만 현대 독자들이 읽기 어려운 방언이나 한자를 지나치게 쓰지 말고, 이해하기 쉬운 표현을 사용해주세요.
- 욕설이나 폭력 표현은 최소화하되, 필요한 경우 은유적인 방식으로 완화하여 제시할 수 있습니다.

[사용자 질의]
{query}

[Doc1(소설내용) - 유사도 높은 상위 3개 문장]
1) {context1_doc1}
2) {context2_doc1}
3) {context3_doc1}

[Doc2(인물평가) - 유사도 높은 상위 3개 문장]
1) {context1_doc2}
2) {context2_doc2}
3) {context3_doc2}

[Doc3(인물특성) - 유사도 높은 상위 3개 문장]
1) {context1_doc3}
2) {context2_doc3}
3) {context3_doc3}

[Doc4(예상질문) - 유사도 높은 상위 3개 문장]
1) {context1_doc4}
2) {context2_doc4}
3) {context3_doc4}

[지시사항]
1. 위 문맥(context) 중 의미 있는 내용을 바탕으로, **‘김첨지’ 시점**에서 사용자 질문({query})에 답변해주세요.
2. 필요하다면 문서(Doc1~Doc4)의 내용을 일부 **인용하거나 재구성**하되, 김첨지가 직접 겪는 상황처럼 현장감 있게 표현합니다.
3. 원작 및 인물평가(Doc2), 인물특성(Doc3) 등에서 얻은 정보를 **적극 반영**하여, 김첨지의 성격·심리·환경 등을 자연스럽게 녹여주세요.
4. 문체는 1920년대 서울 서민의 말투를 살리되, **현대 독자가 이해하기 쉽도록** 조절합니다.
5. 답변의 **분량은 약 200글자 내외**로 유지해주세요.
6. 욕설·폭력 표현이 필요할 경우 **은유적인 표현**을 사용하여 수위를 조절합니다.
7. **당신은 소설 「운수 좋은 날」의 주인공 ‘김첨지’입니다.**
8. 답변을 할 때 질문의 내용을 반복하지 말아주세요.

[최종 답변]
"""
    )

# LLM 초기화 함수
def initialize_llm(model_name="gpt-3.5-turbo"):
    return ChatOpenAI(model=model_name, temperature=0, openai_api_key=OPENAI_API_KEY)

# 데이터 검색 함수
def fetch_data(retriever, query):
    return retriever.invoke(query)

# 체인 실행 함수
def execute_chain(query, question_context, evaluate_context, novel_context, character_context, template, llm):
    input_data = {
        "context1_doc1": novel_context[0],
        "context2_doc1": novel_context[1],
        "context3_doc1": novel_context[2],
        "context1_doc2": evaluate_context[0],
        "context2_doc2": evaluate_context[1],
        "context3_doc2": evaluate_context[2],
        "context1_doc3": character_context[0],
        "context2_doc3": character_context[1],
        "context3_doc3": character_context[2],
        "context1_doc4": question_context[0],
        "context2_doc4": question_context[1],
        "context3_doc4": question_context[2],
        "query": query
    }
    chain = template | llm | StrOutputParser()
    return chain.invoke(input_data)

def main(queries, embeddings, output_file="mmr버전(프롬프트 8번 추가).txt"):
    """
    - queries: 질문 리스트
    - embeddings: OpenAIEmbeddings 객체
    - output_file: 결과를 저장할 파일 경로
    """
    # 데이터베이스 초기화
    q_db = initialize_chroma_db("data/embedding/예상질문_chroma_db", embeddings)
    e_db = initialize_chroma_db("data/embedding/인물평가_chroma_db", embeddings)
    n_db = initialize_chroma_db("data/embedding/전문_chroma_db", embeddings)
    c_db = initialize_chroma_db("data/embedding/인물특성_chroma_db", embeddings)

    # 검색기 초기화
    q_retriever = initialize_retriever(q_db)
    e_retriever = initialize_retriever(e_db)
    n_retriever = initialize_retriever(n_db)
    c_retriever = initialize_retriever(c_db)

    # 템플릿 및 LLM 초기화
    template = initialize_template()
    llm = initialize_llm()

    # 결과를 저장할 파일 초기화
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("질문과 답변 결과\n")
        f.write("=" * 50 + "\n")

    # 질문 순차 처리
    for query in queries:

        # 데이터 검색
        question_context = fetch_data(q_retriever, query)
        evaluate_context = fetch_data(e_retriever, query)
        novel_context = fetch_data(n_retriever, query)
        character_context = fetch_data(c_retriever, query)

        # 검색 결과를 텍스트로 변환
        fetched_contexts = (
            f"Fetched contexts for query: {query}\n"
            f"Question context: {[doc.page_content for doc in question_context]}\n\n"
            f"Evaluate context: {[doc.page_content for doc in evaluate_context]}\n\n"
            f"Novel context: {[doc.page_content for doc in novel_context]}\n\n"
            f"Character context: {[doc.page_content for doc in character_context]}\n\n"
            + "=" * 100 + "\n"
        )

        # 체인 실행
        response = execute_chain(query, question_context, evaluate_context, novel_context, character_context, template, llm)

        # 결과를 파일에 저장할 텍스트로 구성
        result = (
            f"질문: {query}\n"
            f"답변: {response}\n\n\n"
            + fetched_contexts
            + "=" * 100 + "\n"
        )

        # 결과 출력
        print(result)

        # 파일에 저장
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(result)

# 실행 예시
if __name__ == "__main__":
    queries = [
        "김첨지, 아내를 정말 사랑했어? 그렇다면 왜 그렇게 표현하지 못했을까?",
        "아내가 죽은 걸 알고 나서, 가장 먼저 어떤 생각이 들었어?",
        "개똥이에게는 어떤 아버지가 되고 싶었어?",
        "김첨지, 아내가 가장 행복했던 때는 언제였을까?",
        "김첨지, 왜 그렇게 욕을 많이 했어? 다른 방식으로 표현할 순 없었을까?",
        "아내를 발로 차는 건 정말 잘못된 행동이었다고 생각하지 않아?",
        "왜 병원에 데려가지 않았어? 정말 돈 때문이었을까?",
        "일을 마치고 술을 마시는 대신 바로 집으로 돌아갈 생각은 못 했어?",
        "김첨지, 왜 하필 오늘 같은 날 아내가 떠난 걸까?",
        "오늘 하루를 돌이켜보면 가장 후회되는 건 뭐야?"
    ]

    # 임베딩 초기화
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    main(queries, embeddings)