# main.py
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

    queries = [
        "김첨지, 만약 아내가 떠난 후 개똥이를 키우며 새로운 삶을 시작한다면, 어떤 아버지가 되고 싶었어?",
        "그날 돈을 많이 벌었다고 자랑하려는 마음도 있었던 거야?",
        "아내가 죽었다는 걸 확인하고 설렁탕은 어떻게 했어?",
    ]

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

    for query in queries:
        session_id = "default_session"
        chat_history = get_session_history(session_id, store).messages

        question_context = q_retriever.invoke(query)
        evaluate_context = e_retriever.invoke(query)
        novel_context = n_retriever.invoke(query)
        character_context = c_retriever.invoke(query)

        response = execute_chain(query, question_context, evaluate_context, novel_context, character_context, template, llm, chat_history)

        print(f"질문: {query}\n답변: {response}\n")

if __name__ == "__main__":
    main()
