import streamlit as st
import requests
import os
from PIL import Image
from io import BytesIO
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OCR_API_KEY = os.getenv("OCR_API_KEY")

st.set_page_config(page_title="Handwriting Recognition", layout="centered")
st.title("📝 Handwriting Recognition using Groq + OCR")

uploaded_file = st.file_uploader("Upload a handwritten image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("🔍 Compressing image and extracting text..."):
            # Convert to RGB and compress image
            image = image.convert("RGB")
            max_size = (1024, 1024)
            image.thumbnail(max_size, Image.Resampling.LANCZOS) 

            image_bytes = BytesIO()
            image.save(image_bytes, format='JPEG', quality=70)
            image_bytes.seek(0)

            files = {
                "file": ("image.jpg", image_bytes, "image/jpeg")
            }

            response = requests.post(
                "https://api.ocr.space/parse/image",
                files=files,
                data={"OCREngine": "2"},
                headers={"apikey": OCR_API_KEY} if OCR_API_KEY else {}
            )

            result = response.json()

            # Error Handling
            if result.get("IsErroredOnProcessing"):
                error_message = result.get("ErrorMessage", "Unknown error")
                if isinstance(error_message, list):
                    error_message = ", ".join(error_message)
                raise ValueError(f"OCR API Error: {error_message}")

            parsed_results = result.get("ParsedResults")
            if not parsed_results:
                raise KeyError("OCR response missing 'ParsedResults' key.")

            extracted_text = parsed_results[0].get("ParsedText", "").strip()

            if not extracted_text:
                st.warning("OCR did not extract any text.")
            else:
                st.text_area("🧾 Extracted Text", extracted_text, height=150)

                with st.spinner("🤖 Interpreting text using Groq..."):
                    llm = ChatGroq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are an AI that explains handwritten notes."),
                        ("human", "{handwriting}")
                    ])
                    chain = prompt | llm | StrOutputParser()
                    answer = chain.invoke({"handwriting": extracted_text})

                    st.success("💡 Interpretation by LLaMA3:")
                    st.markdown(answer)

    except Exception as e:
        st.error(f"❌ Error: {e}")
