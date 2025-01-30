from chatbot import ChatBot
from character import WORKS_TO_CHARACTERS  # 작품-캐릭터 목록 불러오기

def main():
    # 1) 작품 목록 출력
    print("=== 작품 목록 ===")
    work_titles = list(WORKS_TO_CHARACTERS.keys())
    for idx, title in enumerate(work_titles, start=1):
        print(f"{idx}) {title}")
    
    choice = input("작품 번호를 선택하세요: ").strip()
    # 사용자가 잘못된 번호를 입력했을 경우 대비
    try:
        work_index = int(choice) - 1
        if work_index < 0 or work_index >= len(work_titles):
            raise ValueError
    except ValueError:
        print("잘못된 입력이므로 기본값(1번)으로 진행합니다.")
        work_index = 0

    book_title = work_titles[work_index]

    # 2) 캐릭터 선택
    possible_chars = WORKS_TO_CHARACTERS[book_title]
    print(f"\n=== '{book_title}' 작품의 캐릭터 목록 ===")
    for c_idx, char_name in enumerate(possible_chars, start=1):
        print(f"{c_idx}) {char_name}")

    char_choice = input("캐릭터 번호를 선택하세요: ").strip()
    try:
        char_index = int(char_choice) - 1
        if char_index < 0 or char_index >= len(possible_chars):
            raise ValueError
    except ValueError:
        print("잘못된 입력이므로 첫 번째 캐릭터로 진행합니다.")
        char_index = 0

    character_name = possible_chars[char_index]

    # 3) 챗봇 초기화
    bot = ChatBot(character_name=character_name, book_title=book_title)

    print(f"\n--- '{book_title}' 작품의 [{character_name}] 챗봇과 대화를 시작합니다. ---")
    print("대화를 종료하려면 'exit' 또는 'quit'를 입력하세요.\n")

    # 4) 대화 루프
    while True:
        user_query = input("[사용자] ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print("\n대화를 종료합니다.")
            break

        # 챗봇에게 질문
        answer = bot.get_answer(user_query)

        # 답변 출력
        print(f"[{character_name}] {answer}\n")

if __name__ == "__main__":
    main()