import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="🧬 BioGPT Medical Chatbot", layout="centered")

st.title("🧬 BioGPT Medical Chatbot")
st.markdown("Ask any biomedical or medical question. (For informational purposes only.)")

@st.cache_resource
def load_biogpt():
    return pipeline("text-generation", model="microsoft/BioGPT-Large")

biogpt = load_biogpt()

user_input = st.text_area(
    "Enter your medical question:",
    value="What are the symptoms of migraine?",
    height=100,
    key="chat_input"
)

if st.button("Ask BioGPT"):
    if user_input.strip():
        with st.spinner("AI is thinking..."):
            prompt = f"{user_input.strip()}"
            response = biogpt(prompt, max_length=128, do_sample=True, temperature=0.8)
            answer = response[0]['generated_text'].replace(prompt, "").strip()
            st.success(answer)
    else:
        st.warning("Please enter your symptoms or question.")

st.markdown("---")
st.caption("Built by Tanjim Tanur")
