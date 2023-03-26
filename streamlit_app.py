import streamlit as st
from pdfminer.high_level import extract_text
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import requests
from io import BytesIO

def extract_text_from_pdf(file):
    text = extract_text(file)
    return text

def answer_question(question, context):
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    inputs = tokenizer(question, context, return_tensors="pt")
    outputs = model(**inputs)
    answer_start = outputs.start_logits.argmax().item()
    answer_end = outputs.end_logits.argmax().item() + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    return answer

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
