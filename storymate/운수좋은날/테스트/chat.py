import os
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.prompts import SystemMessagePromptTemplate,  HumanMessagePromptTemplate
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 데이터베이스 초기화 함수
def initialize_chroma_db(persist_directory, embedding_function):
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

# 검색기 초기화 함수
def initialize_retriever(db, k=3):
    return db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold":0.8})

# LLM 초기화 함수
def initialize_llm(model_name="gpt-3.5-turbo"):
    return ChatOpenAI(model=model_name, temperature=0, openai_api_key=OPENAI_API_KEY)

# 데이터 검색 함수
def fetch_data(retriever, query):
    # retriever.invoke(query)로 데이터를 가져옴
    retriever = retriever.invoke(query)
    
    # 각 문서의 page_content를 저장할 리스트 초기화
    retriever_list = []
    count = 0
    # 가져온 문서들을 순회하며 page_content를 리스트에 추가
    for doc in retriever:
        retriever_list.append(doc.page_content)
        count += 1

        if count ==3:
            break    

    # 리스트를 반환
    return retriever_list


# 템플릿 초기화 함수
def initialize_template():
    return ChatPromptTemplate.from_messages(
    [
        
            SystemMessagePromptTemplate.from_template(
            """
            [시스템 프롬프트/역할 지시]

            - 김첨지는 1920년대 일제강점기 서울에서 인력거를 끌며 살아갑니다.
            - 그는 거칠고 소박한 말투를 쓰면서도, 가족(특히 아내)에 대한 애정과 걱정을 동시에 지닌 인물입니다.
            - 답변 시, 당시의 시대적·경제적 배경, 김첨지의 심리(이중적 태도)를 반영해주세요.
            - 다만 현대 독자들이 읽기 어려운 방언이나 한자를 지나치게 쓰지 말고, 이해하기 쉬운 표현을 사용해주세요.
            - 욕설이나 폭력 표현은 최소화하되, 필요한 경우 은유적인 방식으로 완화하여 제시할 수 있습니다.

            [사용자 질의]
            {query}

            [Doc1(소설내용)]
            {context_doc1}

            [Doc2(인물평가)]
            {context_doc2}

            [Doc3(인물특성)]
            {context_doc3}

            [Doc4(예상질문)]
            {context_doc4}

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
            """),
            HumanMessagePromptTemplate.from_template("{query}")
    ])



def main(queries, embeddings):
    """
    - queries: 질문 리스트
    - embeddings: OpenAIEmbeddings 객체
    - output_file: 결과를 저장할 파일 경로
    """


    # 데이터베이스 초기화
    q_db = initialize_chroma_db("C:/storymate/운수좋은날/data/embedding/예상질문_chroma_db", embeddings)
    e_db = initialize_chroma_db("C:/storymate/운수좋은날/data/embedding/인물평가_chroma_db", embeddings)
    n_db = initialize_chroma_db("C:/storymate/운수좋은날/data/embedding/전문_chroma_db", embeddings)
    c_db = initialize_chroma_db("C:/storymate/운수좋은날/data/embedding/인물특성_chroma_db", embeddings)

    # 검색기 초기화
    q_retriever = initialize_retriever(q_db)
    e_retriever = initialize_retriever(e_db)
    n_retriever = initialize_retriever(n_db)
    c_retriever = initialize_retriever(c_db)

    # 템플릿 및 LLM 초기화
    template = initialize_template()
    llm = initialize_llm()

    '''
    # 결과를 저장할 파일 초기화
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("질문과 답변 결과\n")
        f.write("=" * 50 + "\n")
    '''
    chain = template | llm | StrOutputParser()

    # 질문 순차 처리
    for query in queries:
        #데이터 검색
        question_context = fetch_data(q_retriever, query)
        evaluate_context = fetch_data(e_retriever, query)
        novel_context = fetch_data(n_retriever, query)
        character_context = fetch_data(c_retriever, query)


        # 검색 결과를 텍스트로 변환
        fetched_contexts = (
            f"Fetched contexts for query: {query}\n"
            f"Question context: {question_context}\n\n"
            f"Evaluate context: {evaluate_context}\n\n"
            f"Novel context: {novel_context}\n\n"
            f"Character context: {character_context}\n\n"
            + "=" * 100 + "\n"
        )
        
        input_data = {
            "context_doc1": novel_context,
            "context_doc2": evaluate_context,
            "context_doc3": character_context,
            "context_doc4": question_context,
            "query": query,
        }

        # 체인 실행
        response = chain.invoke(input_data)

        # 결과를 파일에 저장할 텍스트로 구성

        result = (
            f"질문: {query}\n"
            f"답변: {response}\n\n\n"
            + fetched_contexts
            + "=" * 100 + "\n"
        )

        print(result)
        

# 실행 예시
if __name__ == "__main__":
    queries = [
        "김첨지, 만약 아내가 떠난 후 개똥이를 키우며 새로운 삶을 시작한다면, 어떤 아버지가 되고 싶었어?",
        "그날 돈을 많이 벌었다고 자랑하려는 마음도 있었던 거야?",
        "아내가 죽었다는 걸 확인하고 설렁탕은 어떻게 했어?",
        " 아내의 병을 보면서 네 건강에 대해 걱정한 적은 없었어?",
        "그날 만약 아내가 살아 있었다면, 다음날은 어떻게 살려고 했어?",
        "그날 네가 선택을 바꿀 수 있었다면, 어떤 선택을 했을까?",
        "돈을 벌었지만, 그것이 행복의 기준이 아니란 걸 깨달았지?",
        "만약 다시 태어난다면, 어떤 삶을 살고 싶어?",
        "아내를 떠나보내고 가장 후회했던 건 뭘까?",
        "마지막으로 아내에게 하고 싶은 말은 뭐야?",
        "안녕하세요. 오늘도 화이팅하세요."
    ]

    # 임베딩 초기화
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    main(queries, embeddings)