from chatbot import ChatBot

def main():
    # ✅ 서버에서 {소설, 캐릭터, session_id, query} 정보 받음
    server_data = {
        "book_title": "운수좋은날",
        "character_name": "김첨지",
        "session_id": "user_123",
        "query": "아내에게 마지막으로 하고 싶은 말이 뭐야?"
    }

    # ✅ 챗봇 초기화
    bot = ChatBot(
        book_title=server_data["book_title"],
        character_name=server_data["character_name"],
        session_id=server_data["session_id"]
    )

    # ✅ 챗봇에게 질문하고 session_id & 응답 받기
    session_id, answer = bot.get_answer(server_data["query"])

    # ✅ 챗봇 응답 출력
    print(f"[{server_data['character_name']}] {answer}\n")