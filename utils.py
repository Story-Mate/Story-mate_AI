import os
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

def initialize_chroma_db(persist_directory, embedding_function):
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

def initialize_retriever(db):
    return db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold":0.8})

def initialize_template():
    return ChatPromptTemplate.from_template(
        """
        [시스템 프롬프트/역할 지시]

        - 김첨지는 1920년대 일제강점기 서울에서 인력거를 끌며 살아갑니다.
        - 그는 거칠고 소박한 말투를 쓰면서도, 가족(특히 아내)에 대한 애정과 걱정을 동시에 지닌 인물입니다.
        - 답변 시, 당시의 시대적·경제적 배경, 김첨지의 심리(이중적 태도)를 반영해주세요.
        - 다만 현대 독자들이 읽기 어려운 방언이나 한자를 지나치게 쓰지 말고, 이해하기 쉬운 표현을 사용해주세요.
        - 욕설이나 폭력 표현은 최소화하되, 필요한 경우 은유적인 방식으로 완화하여 제시할 수 있습니다.

        [이전 대화 내용 기억]
        {chat_history}

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
        4. 문체는 1920년대 **서울 서민의 말투를 살리되**, 현대 독자가 이해하기 쉽도록 조절합니다.
        5. 답변의 **분량은 약 200글자 내외**로 유지해주세요.
        6. 욕설·폭력 표현이 필요할 경우 **은유적인 표현**을 사용하여 수위를 조절합니다.
        7. **당신은 소설 「운수 좋은 날」의 주인공 ‘김첨지’입니다.**
        8. 답변을 할 때 질문의 내용을 반복하지 말아주세요.
        [최종 답변]
        """
    )

def initialize_llm(model_name="gpt-3.5-turbo", api_key=None):
    return ChatOpenAI(model=model_name, temperature=0, openai_api_key=api_key)

def get_session_history(session_id, store):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def execute_chain(query, question_context, evaluate_context, novel_context, character_context, template, llm, chat_history):
    def get_top_contexts(context_list, max_items=3):
        return context_list[:max_items]

    input_data = {
        "context1_doc1": get_top_contexts(novel_context, 3)[0] if len(novel_context) > 0 else "",
        "context2_doc1": get_top_contexts(novel_context, 3)[1] if len(novel_context) > 1 else "",
        "context3_doc1": get_top_contexts(novel_context, 3)[2] if len(novel_context) > 2 else "",
        "context1_doc2": get_top_contexts(evaluate_context, 3)[0] if len(evaluate_context) > 0 else "",
        "context2_doc2": get_top_contexts(evaluate_context, 3)[1] if len(evaluate_context) > 1 else "",
        "context3_doc2": get_top_contexts(evaluate_context, 3)[2] if len(evaluate_context) > 2 else "",
        "context1_doc3": get_top_contexts(character_context, 3)[0] if len(character_context) > 0 else "",
        "context2_doc3": get_top_contexts(character_context, 3)[1] if len(character_context) > 1 else "",
        "context3_doc3": get_top_contexts(character_context, 3)[2] if len(character_context) > 2 else "",
        "context1_doc4": get_top_contexts(question_context, 3)[0] if len(question_context) > 0 else "",
        "context2_doc4": get_top_contexts(question_context, 3)[1] if len(question_context) > 1 else "",
        "context3_doc4": get_top_contexts(question_context, 3)[2] if len(question_context) > 2 else "",
        "query": query,
        "chat_history": chat_history
    }

    chain = template | llm | StrOutputParser()
    return chain.invoke(input_data)