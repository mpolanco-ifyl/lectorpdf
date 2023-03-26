import streamlit as st
from pdfminer.high_level import extract_text
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import requests
from io import BytesIO

def extract_text_from_pdf(file):
    text = extract_text(file)
    return text

def generate_citation(text):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    input_text = f"Resumen y cita del siguiente texto: {text}"
    input_tokens = tokenizer.encode(input_text, return_tensors='pt')
    output_tokens = model.generate(input_tokens, max_length=200, num_return_sequences=1)
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return output_text

st.title("Generador de citas a partir de un PDF")
uploaded_file = st.file_uploader("Carga un archivo PDF", type=['pdf'])

if uploaded_file is not None:
    with st.spinner('Extrayendo texto del PDF...'):
        text = extract_text_from_pdf(BytesIO(uploaded_file.read()))
    st.subheader("Texto extraído del PDF:")
    st.write(text)
    with st.spinner('Generando cita...'):
        citation = generate_citation(text)
    st.subheader("Cita generada:")
    st.write(citation)
