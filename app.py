import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="🩺 Clinical Camel Medical Chatbot", layout="centered")

st.title("🩺 Medical Chatbot")
st.markdown(
    "Ask any medical question or describe your symptoms to chat with a free, open-source AI assistant. "
    "For best results, ask clear, specific questions. (For informational purposes only.)"
)

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
            # Strong, clear prompt for detailed, structured answers
            prompt = (
                "You are a helpful and knowledgeable medical assistant. "
                "When asked about symptoms, list all symptoms in a bullet-point list. "
                "When asked about treatments, list all treatments in a bullet-point list. "
                "Give a detailed, step-by-step answer in clear, simple language. "
                "If the question is about a disease, include causes, symptoms, and possible treatments. "
                "If the question is about symptoms, include possible conditions and advice. "
                "Do not repeat the question in your answer. "
                "Here is the question:\n"
                f"{user_input.strip()}"
            )
            # Generate multiple answers (if supported)
            response = camel(
                prompt,
                max_length=512,
                num_return_sequences=3,
                do_sample=True,
                temperature=0.9
            )
            # Display each answer, removing repeated prompt if present
            for idx, result in enumerate(response, 1):
                answer = result['generated_text'].strip()
                # Remove prompt echo if present
                if answer.lower().startswith(prompt.lower()):
                    answer = answer[len(prompt):].strip()
                st.markdown(f"**Answer {idx}:**\n{answer}")
    else:
        st.warning("Please enter your symptoms or question.")

st.markdown("---")
st.caption("Built by Tanjim Tanur")
