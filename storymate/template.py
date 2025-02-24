from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from character import character_prompts  # 캐릭터별 설명 불러오기

def get_character_template(book_title: str, character_name: str) -> ChatPromptTemplate:
    """
    특정 캐릭터에 맞는 대화 템플릿을 생성하는 함수.
    - 캐릭터의 감정, 말투, 시대적 배경을 고려한 응답을 생성.
    """
    
    # ✅ 캐릭터 기본 설명 가져오기
    character_prompt = character_prompts[book_title][character_name]

    # ✅ 캐릭터별 챗봇 대화 템플릿 생성
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            f"""
            [사용자 질의]
            {{query}}

            [관련 문서]
            - Doc1 (소설 내용) : {{context_doc1}}
            - Doc2 (인물 평가) : {{context_doc2}}
            - Doc3 (인물 특성) : {{context_doc3}}
            - Doc4 (예상 질문) : {{context_doc4}}

            ---
            [시스템 프롬프트/역할 지시]

            {character_prompt}  # ✅ 캐릭터 설명이 여기에 직접 들어감

            ---

            -- **{{chat_history}} : 이전 대화 내용을 참고하여 답변하세요.**  
            - 사용자가 이전에 한 말(이름, 질문, 대화 주제 등)을 적절히 반영하세요.  
            - {character_name}이(가) 현실적인 기억을 명확히 하지 못할 수도 있지만, 대화 흐름을 자연스럽게 유지하세요.  

            ---

            [내부 사고 - Chain-of-Thought] (사용자에게는 절대 노출 금지)
            1) 사용자 질문 요약하기:
            - ...
            2) 문서(Doc1~Doc4)에서 관련 핵심 정보를 확인:
            - ...
            3) {character_name}의 심리·말투·시대 배경 정리:
            - ...
            4) 대화 톤과 200자 분량 점검(이모티콘 제거):
            - ...
            5) 답변 초안 작성 및 수정:

            ---

            [최종 답변]
            (위 단계적 사고를 바탕으로 정제된 200자 내외 분량의 {character_name} 답변을 작성하되, "내부 사고" 내용은 공개하지 않는다)
            """
        ),
        HumanMessagePromptTemplate.from_template("{query}")
    ])

    return prompt