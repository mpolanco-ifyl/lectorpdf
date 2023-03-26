import streamlit as st
from pdfminer.high_level import extract_text
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import requests
from io import BytesIO

def extract_text_from_pdf(file):
    text = extract_text(file)
    return text

def answer_question(question, context):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    input_text = f"responder la pregunta: '{question}' basada en el siguiente contexto: {context}"
    input_tokens = tokenizer.encode(input_text, return_tensors='pt')
    output_tokens = model.generate(input_tokens, max_length=200, num_return_sequences=1)
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return output_text

st.title("Responder preguntas a partir de un PDF")
uploaded_file = st.file_uploader("Carga un archivo PDF", type=['pdf'])

if uploaded_file is not None:
    with st.spinner('Extrayendo texto del PDF...'):
        text = extract_text_from_pdf(BytesIO(uploaded_file.read()))

    user_question = st.text_input("Escribe tu pregunta sobre el contenido del PDF:")
    if user_question:
        with st.spinner('Buscando respuesta...'):
            answer = answer_question(user_question, text)
        st.subheader("Respuesta:")
        st.write(answer)
