# main.py
import sys
from chatbot import ChatBot

def main():
    # 1) 캐릭터 이름과 책 제목(프로젝트마다 다를 수 있음)을 지정합니다.
    character_name = "김첨지"
    book_title = "운수좋은날"

    # 2) ChatBot 인스턴스 생성
    chatbot = ChatBot(character_name, book_title)

    # 3) 세션 ID를 설정합니다. 여기서는 간단히 'session_1'로 고정
    session_id = "session_1"

    print("대화를 시작합니다. 질문을 입력해 주세요. (종료하려면 'exit' 또는 'quit' 입력)")
    
    while True:
        try:
            user_input = input("사용자: ")
            if user_input.lower() in ["exit", "quit"]:
                print("대화를 종료합니다.")
                break

            # 4) ChatBot에 질문을 전달하여 답변 생성
            answer = chatbot.get_answer(session_id, user_input)

            # 5) AI의 답변 출력
            print(f"{character_name}: {answer}\n")

        except (KeyboardInterrupt, EOFError):
            # 사용자가 Ctrl+C / Ctrl+D 등을 입력한 경우 종료
            print("\n대화를 종료합니다.")
            sys.exit(0)

if __name__ == "__main__":
    main()
