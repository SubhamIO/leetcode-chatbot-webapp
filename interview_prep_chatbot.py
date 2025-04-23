from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import streamlit as st
import time
import re

loader = TextLoader("DSA.md")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)


embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


vectorstore = FAISS.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

## Load environment variables

load_dotenv()
# groq_api_key=os.getenv("GROQ_API_KEY")
groq_api_key = st.secrets['GROQ_API_KEY']

model=ChatGroq(groq_api_key=groq_api_key,model="Llama3-8b-8192") ## llama 3 model



qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=retriever
)

def query_rag(question):
    response = qa_chain.run(question)
    return response


def query_llm_sync(question):
    try:
        response = model.invoke(question)

        # Extract content safely if it's inside a dict-like object
        if hasattr(response, 'content'):
            response_str = response.content
        else:
            response_str = response if isinstance(response, str) else str(response)

        # Look for lines that appear to be code (starting with def/class/indentation)
        code_pattern = re.compile(r"(?P<code>(class\s+\w+|def\s+\w+|\s{4}|\t).+)", re.MULTILINE)
        matches = list(code_pattern.finditer(response_str))

        if matches:
            # Wrap the entire response in code block if code is detected
            response_str = f"```\n{response_str.strip()}\n```"

        return response_str

    except Exception as e:
        return f"Error: {e}"
    


st.set_page_config(page_title="LeetCode Chatbot", page_icon="🤖", layout="centered")
st.title("LeetCode Chatbot 🤖")
user_input = st.text_input("Ask me anything related to LeetCode:")
if user_input:
    st.text_input("Ask me anything related to LeetCode:", value=user_input, disabled=True)
    with st.spinner("Processing your query in the Vector DB..."):
        vector_db_answer = query_rag(user_input)
    st.markdown("### Answer from Vector DB:")
    st.write(vector_db_answer)
    with st.spinner("Processing your query with the LLM..."):
        llm_answer = query_llm_sync(user_input)
    st.markdown("### Answer from LLM:")
    st.write(llm_answer)
else:
    st.info("Please enter a question related to LeetCode.")