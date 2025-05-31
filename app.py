import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="🧬 Medical Chatbot", layout="centered")

st.title("🧬 Medical Chatbot")
st.markdown("Ask any medical question or describe your symptoms to chat with a free AI assistant.")

@st.cache_resource
def load_flan_t5():
    return pipeline("text2text-generation", model="google/flan-t5-base")

flan_t5 = load_flan_t5()

user_input = st.text_area(
    "Enter your symptoms or question:",
    value="Describe your symptoms or ask a medical question.",
    height=100,
    key="chat_input"
)

if st.button("Ask AI"):
    if user_input.strip():
        with st.spinner("AI is thinking..."):
            # Prompt engineering for better answers
            prompt = (
                "You are a helpful and knowledgeable medical assistant. "
                "Answer the following in clear, simple language:\n"
                f"{user_input}"
            )
            response = flan_t5(prompt, max_length=256, do_sample=True, temperature=0.7)
            answer = response[0]['generated_text'].strip()
            # Remove repeated instruction if present
            for prefix in [
                "You are a helpful and knowledgeable medical assistant.",
                "Answer the following in clear, simple language:",
                "Medical question:"
            ]:
                if answer.lower().startswith(prefix.lower()):
                    answer = answer[len(prefix):].strip()
            st.success(answer)
    else:
        st.warning("Please enter your symptoms or question.")

st.markdown("---")
st.caption("Built by Tanjim Tanur")
