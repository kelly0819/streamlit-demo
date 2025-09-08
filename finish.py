## streamlit 관련 모듈 불러오기
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
import os
import fitz  # PyMuPDF
import re

## 환경변수 불러오기
from dotenv import load_dotenv
load_dotenv()


############################### 1단계 : RAG 기능 구현과 관련된 함수들 ##########################

@st.cache_data
def process_question(user_question: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 벡터 DB 호출 (이미 생성되어 있어야 함)
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    retrieve_docs: List[Document] = retriever.invoke(user_question)
    
    chain = get_rag_chain()
    response = chain.invoke({"question": user_question, "context": retrieve_docs})
    
    return response, retrieve_docs


def get_rag_chain() -> Runnable:
    template = """
    다음의 컨텍스트를 활용해서 질문에 답변해줘
    - 질문에 대한 응답을 해줘
    - 간결하게 5줄 이내로 해줘
    - 곧바로 응답결과를 말해줘

    컨텍스트 : {context}

    질문: {question}

    응답:"""
    
    custom_rag_prompt = PromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4o-mini")
    
    return custom_rag_prompt | model | StrOutputParser()


############################### 2단계 : PDF 이미지 관련 함수 ##########################
@st.cache_data(show_spinner=False)
def convert_pdf_to_images(pdf_path: str, dpi: int = 250) -> List[str]:
    doc = fitz.open(pdf_path)
    image_paths = []
    output_folder = "PDF_이미지"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)
        
    return image_paths


def display_pdf_page(image_path: str, page_number: int) -> None:
    image_bytes = open(image_path, "rb").read()
    st.image(image_bytes, caption=f"Page {page_number}", output_format="PNG", width=600)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]


############################### 3단계 : Streamlit 앱 ##########################
def main():
    st.set_page_config("청약 FAQ 챗봇", layout="wide")
    st.header("청약 FAQ 챗봇")

    # 사용자 질문 입력
    user_question = st.text_input(
        placeholder="무순위 청약 시에도 부부 중복신청이 가능한가요?"
    )

    if user_question:
        response, context = process_question(user_question)
        st.write(response)

        # 관련 문서 및 이미지 표시
        for i, document in enumerate(context):
            with st.expander("관련 문서"):
                st.write(document.page_content)
                page_number = document.metadata.get('page', 0) + 1
                image_folder = "PDF_이미지"
                images = sorted(os.listdir(image_folder), key=natural_sort_key)
                image_paths = [os.path.join(image_folder, image) for image in images]
                if page_number <= len(image_paths):
                    display_pdf_page(image_paths[page_number - 1], page_number)


if __name__ == "__main__":
    main()
