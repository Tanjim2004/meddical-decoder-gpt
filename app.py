import streamlit as st
from PIL import Image
import easyocr
import openai
import os

st.set_page_config(page_title="🧬 Medical Decoder GPT", layout="centered")

st.title("🧬 Medical Decoder GPT")
st.markdown("Upload a medical image (e.g. prescription, report, or scan) to decode it and chat with GPT about symptoms.")

# --- OpenAI API Key ---
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
if openai_api_key:
    openai.api_key = openai_api_key

# --- OCR Section ---
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])
extracted_text = ""

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    with st.spinner("Extracting text..."):
        reader = easyocr.Reader(['en'])
        result = reader.readtext(image)
        extracted_text = "\n".join([text for _, text, _ in result])
    if extracted_text.strip():
        st.subheader("📝 Extracted Text:")
        st.code(extracted_text)
    else:
        st.warning("Couldn't extract text from image. Try a clearer image or a text-based scan.")
else:
    st.info("Please upload a medical image to get started.")
st.header("💬 Chat with Local AI (DialoGPT)")

from transformers import pipeline

@st.cache_resource
def load_chatbot():
    return pipeline("conversational", model="microsoft/DialoGPT-medium")

chatbot = load_chatbot()

default_prompt = extracted_text if extracted_text else "Describe your symptoms or ask a medical question."
user_input = st.text_area("Enter your symptoms or question:", value=default_prompt, height=100)
if st.button("Ask Local AI"):
    if user_input.strip():
        with st.spinner("AI is thinking..."):
            response = chatbot(user_input)
            st.success(response[0]['generated_text'])
    else:
        st.warning("Please enter your symptoms or question.")
st.markdown("---")

# --- GPT Chat Section ---


st.markdown("---")
st.caption("Built by Tanjim Tanur")
