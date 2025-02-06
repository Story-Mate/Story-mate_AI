from flask import Flask, request, jsonify
from chatbot import ChatBot  # 챗봇 모델 불러오기

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    """ 백엔드에서 세션ID, 캐릭터이름, 질문을 받아 챗봇을 실행하고 답변을 반환 """
    data = request.json  # JSON 데이터 받기

    # 필수 데이터 확인
    if "session_id" not in data or "character_name" not in data or "question" not in data:
        return jsonify({"error": "Missing required fields"}), 400

    session_id = data["session_id"]
    character_name = data["character_name"]
    question = data["question"]

    print(f"📥 요청 받음 - 세션: {session_id}, 캐릭터: {character_name}, 질문: {question}")

    # 챗봇 실행 (character_name에 맞는 챗봇 생성 후 응답 받기)
    bot = ChatBot(character_name=character_name)
    response_text = bot.get_answer(question)  # 챗봇 응답

    print(f"🤖 챗봇 응답 - {character_name}: {response_text}")

    # 응답 데이터 구성
    response_data = {
        "session_id": session_id,
        "character_name": character_name,
        "response": response_text
    }

    return jsonify(response_data), 200  # JSON 형태로 반환

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Flask 서버 실행
