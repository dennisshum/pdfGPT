import os
from langchain.llms import OpenAI
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature=0.9)
#print(llm("Hello World!"))

# App framework
st.title("pdfGPT")

# upload file
pdfs = st.file_uploader("Upload your PDF", type="pdf", accept_multiple_files=True)

# extract the text
if pdfs:
  text = ""
  for pdf in pdfs:
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
      text += page.extract_text()

  # split into chunks
  text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
  )
  chunks = text_splitter.split_text(text)

  # create embeddings
  embeddings = OpenAIEmbeddings()
  knowledge_base = FAISS.from_texts(chunks, embeddings)
  
  # show user input
  user_question = st.text_input("Question:")
  if user_question:
    docs = knowledge_base.similarity_search(user_question)
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=user_question)
    st.write(response)
  