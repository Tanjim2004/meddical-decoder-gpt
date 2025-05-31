import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="🩺 Clinical Camel Medical Chatbot", layout="centered")

st.title("🩺  Medical Chatbot")
st.markdown("Ask any medical question or describe your symptoms to chat with a free, open-source AI assistant. (For informational purposes only.)")

@st.cache_resource
def load_camel():
    return pipeline("text2text-generation", model="google/flan-t5-base")

camel = load_camel()

user_input = st.text_area(
    "Enter your medical question or symptoms:",
    value="What are the symptoms of migraine?",
    height=100,
    key="chat_input"
)

if st.button("Ask your AI"):
    if user_input.strip():
        with st.spinner("AI is thinking..."):
           prompt = (
                " Act as if You are a helpful medical assistant. "
                "You are a medical expert with years of experience. "
                "You are able to provide detailed and accurate medical information. "
                "You can answer questions about symptoms, diseases, treatments, and medical conditions. "
                "dont repeat the question, just answer it. "
                "if the question is about symptoms, the list all possible symptoms, "
                "if the question is about a disease, list all possible causes, "
                "You can also provide advice on how to manage symptoms and when to seek medical attention. "
                "Give a detailed, step-by-step answer to the following question:\n"
                f"{user_input.strip()}"
            )
            response = camel(prompt, max_length=1024)
            answer = response[0]['generated_text'].strip()
            st.success(answer)
    else:
        st.warning("Please enter your symptoms or question.")

st.markdown("---")
st.caption("Built by Tanjim Tanur")
