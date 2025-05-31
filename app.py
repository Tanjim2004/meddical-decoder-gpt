import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="🩺 Clinical Camel Medical Chatbot", layout="centered")

st.title("🩺 Clinical Camel Medical Chatbot")
st.markdown("Ask any medical question or describe your symptoms to chat with a free, open-source AI assistant. (For informational purposes only.)")

@st.cache_resource
def load_camel():
    return pipeline(
        "text-generation",
        model="Writer/camel-5b-hf",
        device_map="auto",  # Use "cuda" for GPU, "cpu" for CPU
        torch_dtype="auto"
    )

camel = load_camel()

user_input = st.text_area(
    "Enter your medical question or symptoms:",
    value="What are the symptoms of migraine?",
    height=100,
    key="chat_input"
)

if st.button("Ask Clinical Camel"):
    if user_input.strip():
        with st.spinner("AI is thinking..."):
            prompt = f"### Instruction:\n{user_input.strip()}\n### Response:"
            response = camel(prompt, max_length=256, do_sample=True, temperature=0.7)
            answer = response[0]['generated_text'].replace(prompt, "").strip()
            st.success(answer)
    else:
        st.warning("Please enter your symptoms or question.")

st.markdown("---")
st.caption("Built by Tanjim Tanur")
