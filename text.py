import streamlit as st
import requests
import os
from PIL import Image
from io import BytesIO
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OCR_API_KEY = os.getenv("OCR_API_KEY")  # Optional for OCR.space

st.set_page_config(page_title="Handwriting Recognition", layout="centered")
st.title("📝 Handwriting Recognition using Groq")

uploaded_file = st.file_uploader("Upload a handwritten image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Extracting text using OCR..."):
        try:
            # Prepare image buffer
            image_bytes = BytesIO()
            image.save(image_bytes, format='PNG')
            image_bytes.seek(0)

            files = {
                "file": ("image.png", image_bytes, "image/png")
            }

            # OCR API call
            response = requests.post(
                "https://api.ocr.space/parse/image",
                files=files,
                data={"OCREngine": "2"},
                headers={"apikey": OCR_API_KEY} if OCR_API_KEY else {}
            )

            result = response.json()

            # Error Handling
            if not result.get("IsErroredOnProcessing") and "ParsedResults" in result:
                extracted_text = result["ParsedResults"][0]["ParsedText"]
            else:
                raise ValueError("OCR API Error: " + result.get("ErrorMessage", "Unknown error"))

            st.text_area("🧾 Extracted Text", extracted_text, height=150)

            if extracted_text.strip():
                with st.spinner("💡 Interpreting with Groq..."):
                    llm = ChatGroq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are an AI that explains handwritten notes."),
                        ("human", "{handwriting}")
                    ])
                    chain = prompt | llm | StrOutputParser()
                    answer = chain.invoke({"handwriting": extracted_text})
                    st.success("📘 Interpretation:")
                    st.markdown(answer)
            else:
                st.warning("⚠️ OCR succeeded but no text was found.")

        except Exception as e:
            st.error(f"❌ Error: {e}")
