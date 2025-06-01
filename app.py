import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

st.set_page_config(page_title="🧬 AI Medical Chatbot", layout="centered")
st.title("🧬 AI Medical Chatbot ")
st.markdown("Ask any biomedical or medical question. (For informational purposes only.)")

@st.cache_resource
def load_llama3():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # lightweight and fast
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

llama_pipe = load_llama3()

user_input = st.text_area(
    "Enter your medical question:",
    value="What are the symptoms of ______?",
    height=100,
    key="chat_input"
)

if st.button("Ask your AI"):
    if user_input.strip():
        with st.spinner("AI is thinking..."):
            prompt = f"[INST] You are a helpful medical assistant. {user_input.strip()} [/INST]"
            result = llama_pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
            response = result[0]['generated_text'].replace(prompt, "").strip()
            st.success(response)
    else:
        st.warning("Please enter your symptoms or question.")

st.markdown("---")
st.caption("Built by Tanjim Tanur")
