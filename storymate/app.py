from flask import Flask, request, jsonify
from chatbot import ChatBot

app = Flask(__name__)


@app.route('/', methods=['POST'])
def chat():
    # 여기부터는 POST 요청일 때 동작
    data = request.get_json()
    session_id = data.get("session_id")
    character_name = data.get("character_name")
    query = data.get("query")

    print(f"📥 요청 받음 - 세션: {session_id}, 캐릭터: {character_name}, 질문: {query}")

    bot = ChatBot(character_name=character_name)
    response_text = bot.get_answer(user_query=query, session_id=session_id)

    print(f"🤖 챗봇 응답 - {character_name}: {response_text}")

    response_data = {
        "session_id": session_id,
        "character_name": character_name,
        "response": response_text
    }

    print(f"반환 데이터 {response_data}")

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
