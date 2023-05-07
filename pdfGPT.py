import os
from langchain.llms import OpenAI
from PyPDF2 import PdfReader
import streamlit as st

os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature=0.9)
#print(llm("Hello World!"))

# App framework
st.title("pdfGPT")

# upload file
pdf = st.file_uploader("Upload your PDF", type="pdf")
# extract the text
if pdf is not None:
  pdf_reader = PdfReader(pdf)
  text = ""
  for page in pdf_reader.pages:
    text += page.extract_text()

  st.write(text)