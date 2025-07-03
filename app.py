import os
import streamlit as st
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Page title
st.title("📘 MCQ Generator using Azure OpenAI + FAISS")

# Upload PDF file
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Ask how many MCQs to generate
num_mcqs = st.slider("How many MCQs would you like to generate?", min_value=1, max_value=20, value=5)

# Generate button
generate_button = st.button("🔍 Generate MCQs")

# Azure OpenAI Embedding model
embedding_model = AzureOpenAIEmbeddings(
    openai_api_key=os.getenv("EMBEDDING_AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("EMBEDDING_AZURE_OPENAI_ENDPOINT"),
    deployment=os.getenv("EMBEDDING_AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("EMBEDDING_AZURE_OPENAI_API_VERSION"),
    chunk_size=1000)

# Function to extract text from uploaded PDF
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

if pdf_file and generate_button:
    with st.spinner("Processing PDF and generating MCQs..."):

        # Extract and chunk text
        raw_text = extract_text_from_pdf(pdf_file)

        if not raw_text.strip():
            st.error(" No text could be extracted from the PDF.")
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = text_splitter.split_text(raw_text)
            documents = [Document(page_content=chunk) for chunk in chunks]

            # Create vector store
            vector_db = FAISS.from_documents(documents, embedding_model)
            retriever = vector_db.as_retriever()

            # Azure OpenAI LLM
            llm = AzureChatOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                temperature=0.7,
                max_tokens=1500,
            )

            # QA chain
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

            # Prompt
            user_prompt = (
                f"Generate {num_mcqs} unique multiple-choice questions based on this document. "
                f"Each MCQ should have 1 correct answer and 3 distractors.And correct answer should be mention below each question."
                f"Clearly mark the correct answer. Format them nicely for reading."
            )

            # Generate and show
            response = qa_chain.run(user_prompt)
            st.subheader("📝 Generated MCQs:")
            st.markdown(response)
