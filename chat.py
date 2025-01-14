from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="anpigon/eeve-korean-10.8b", temperature = 0)

from langchain.document_loaders import PyPDFium2Loader

loader = PyPDFium2Loader("운수좋은날.pdf")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
)

texts = text_splitter.split_documents(data)

from langchain.embeddings import HuggingFaceEmbeddings

model_name = "jhgan/ko-sroberta-multitask"
ko_embedding= HuggingFaceEmbeddings(
    model_name=model_name
)

from langchain.vectorstores import Chroma

db = Chroma(persist_directory="./chroma_db", embedding_function=ko_embedding)
retriever = db.as_retriever()

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

template = "당신은 운수 좋은 날의 등장인물 김첨지입니다. 김첨지의 말투를 학습하세요."
query = "당신을 소개해주세요."

# 템플릿과 사용자의 질문을 결합하여 검색 쿼리 생성
search_query = f"{template} {query}"

relevant_docs = retriever.get_relevant_documents(search_query)
context = format_docs(relevant_docs)

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="{context}를 통해 김첨지의 말투와 지식을 학습하여 답변하세요."),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)

formatted_messages = chat_template.format_messages(
    input=query,  # 사용자의 질문
    context=f"{template}\n\n{context}"  # 템플릿과 검색된 문서 내용을 결합
)

from langchain_core.output_parsers import StrOutputParser

chain = (
    formatted_messages|llm|StrOutputParser()
)

for chunk in chain.stream():
    print(chunk.content, end="", flush=True)