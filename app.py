import streamlit as st
from PIL import Image
import easyocr
from transformers import pipeline

st.set_page_config(page_title="🧬 Medical Decoder GPT", layout="centered")

st.title("🧬 Medical Decoder GPT")
st.markdown("Upload a medical image (e.g. prescription, report, or scan) to decode it and chat with a free AI about symptoms.")

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

st.markdown("---")

# --- Free AI Chat Section (Flan-T5) ---
st.header("💬 Chat with your AI companion")

@st.cache_resource
def load_flan_t5():
    return pipeline("text2text-generation", model="google/flan-t5-base")

flan_t5 = load_flan_t5()

default_prompt = extracted_text if extracted_text else "Describe your symptoms or ask a medical question."
user_input = st.text_area("Enter your symptoms or question:", value=default_prompt, height=100)
default_prompt = extracted_text if extracted_text else "Describe your symptoms or ask a medical question."
user_input = st.text_area("Enter your symptoms or question:", value=default_prompt, height=100, key="chat_input")
if st.button("Ask AI"):   
    if user_input.strip():
        with st.spinner("AI is thinking..."):
            prompt = f"Medical question: {user_input}"
            response = flan_t5(prompt, max_length=256, do_sample=True, temperature=0.7)
            answer = response[0]['generated_text']
            # Remove repeated instruction if present
            if answer.lower().startswith("medical question:"):
                answer = answer[len("medical question:"):].strip()
            st.success(answer)
    else:
        st.warning("Please enter your symptoms or question.")
st.markdown("---")
st.caption("Built by Tanjim Tanur")
