import streamlit as st
import re
from openai import AuthenticationError, RateLimitError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader

st.set_page_config(page_title="Track A RAG Bot")

st.title("📄 Document Q&A Chatbot (Track A)")
st.write("Upload a document and ask questions")

st.success("✅ App loaded successfully")


def local_fallback_answer(chunks, question, top_k=3):
    question_tokens = set(re.findall(r"[a-zA-Z0-9_]+", question.lower()))
    if not question_tokens:
        return "Please enter a question with some keywords."

    scored_chunks = []
    for chunk in chunks:
        chunk_tokens = set(re.findall(r"[a-zA-Z0-9_]+", chunk.lower()))
        score = len(question_tokens.intersection(chunk_tokens))
        if score > 0:
            scored_chunks.append((score, chunk))

    if not scored_chunks:
        return "I couldn't find a relevant answer in the uploaded document. Try a more specific question."

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    top_chunks = [chunk for _, chunk in scored_chunks[:top_k]]
    return "\n\n".join(top_chunks)

secret_api_key = st.secrets.get("OPENAI_API_KEY", "").strip()
OPENAI_API_KEY = secret_api_key

use_openai = True
if not OPENAI_API_KEY or "paste_your_openai_api_key_here" in OPENAI_API_KEY.lower():
    use_openai = False

uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file:
    text = ""

    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    else:
        text = uploaded_file.read().decode("utf-8")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    if not chunks:
        st.warning("No readable text found in the uploaded file.")
        st.stop()

    vectorstore = None
    llm = None
    if use_openai:
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vectorstore = FAISS.from_texts(chunks, embeddings)
            llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
        except AuthenticationError:
            use_openai = False
        except RateLimitError:
            use_openai = False

    question = st.text_input("Ask a question")

    if question:
        if use_openai and vectorstore is not None and llm is not None:
            try:
                docs = vectorstore.similarity_search(question, k=4)
                context = "\n\n".join(doc.page_content for doc in docs)
                prompt = (
                    "Use the provided context to answer the user's question. "
                    "If the answer is not in the context, say you don't know.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                )
                answer = llm.invoke(prompt).content
                st.success(answer)
            except RateLimitError:
                st.success(local_fallback_answer(chunks, question))
        else:
            st.success(local_fallback_answer(chunks, question))