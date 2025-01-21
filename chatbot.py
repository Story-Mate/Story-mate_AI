import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from utils import (
    initialize_chroma_db,
    initialize_retriever,
    initialize_template,
    initialize_llm,
    get_session_history,
    execute_chain
)

def main():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    q_db = initialize_chroma_db("C:/storymate/운수좋은날/data/embedding/예상질문_chroma_db", embeddings)
    e_db = initialize_chroma_db("C:/storymate/운수좋은날/data/embedding/인물평가_chroma_db", embeddings)
    n_db = initialize_chroma_db("C:/storymate/운수좋은날/data/embedding/전문_chroma_db", embeddings)
    c_db = initialize_chroma_db("C:/storymate/운수좋은날/data/embedding/인물특성_chroma_db", embeddings)

    q_retriever = initialize_retriever(q_db)
    e_retriever = initialize_retriever(e_db)
    n_retriever = initialize_retriever(n_db)
    c_retriever = initialize_retriever(c_db)

    template = initialize_template()
    llm = initialize_llm(api_key=OPENAI_API_KEY)

    store = {}

    print("김첨지: 반갑네. 나는 김첨지라는 사람이네. 나에 대해 궁금한게 있나?")
    session_id = "default_session"

    while True:
        query = input("사용자: ")
        if query.lower() == "exit":
            print("조심히 가시게!")
            break

        chat_history = get_session_history(session_id, store).messages

        question_context = q_retriever.invoke(query)
        evaluate_context = e_retriever.invoke(query)
        novel_context = n_retriever.invoke(query)
        character_context = c_retriever.invoke(query)

        response = execute_chain(query, question_context, evaluate_context, novel_context, character_context, template, llm, chat_history)

        print(f"김첨지: {response}")

        store[session_id].add_user_message(query)
        store[session_id].add_ai_message(response)

if __name__ == "__main__":
    main()