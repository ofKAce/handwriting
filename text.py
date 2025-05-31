import streamlit as st
import requests
import os
from PIL import Image
from io import BytesIO
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OCR_API_KEY = os.getenv("OCR_API_KEY")

# Streamlit configuration
st.set_page_config(page_title="Handwriting Recognition", layout="centered")
st.title("📝 Handwriting Recognition")

# Upload image
uploaded_file = st.file_uploader("Upload a handwritten image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Extracting text using OCR..."):
        image_bytes = BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)

        # Send image to OCR.space
        response = requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": image_bytes},
            data={"OCREngine": "2"},
            headers={"apikey": OCR_API_KEY} if OCR_API_KEY else {}
        )

        try:
            result = response.json()

            # Debug: show full API response (optional)
            # st.code(result, language='json')

            # Handle API errors
            if result.get("IsErroredOnProcessing"):
                error_message = result.get("ErrorMessage", ["Unknown error"])[0]
                raise ValueError(f"OCR API Error: {error_message}")

            if "ParsedResults" not in result:
                raise KeyError("The response from OCR API does not contain 'ParsedResults'. Full response: " + str(result))

            extracted_text = result["ParsedResults"][0]["ParsedText"]
            st.text_area("🧾 Extracted Text", extracted_text, height=150)

            if extracted_text.strip():
                with st.spinner("🤖 Interpreting the handwriting using Groq..."):
                    llm = ChatGroq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are an AI that explains handwritten notes."),
                        ("human", "{handwriting}")
                    ])
                    chain = prompt | llm | StrOutputParser()
                    answer = chain.invoke({"handwriting": extracted_text})
                    st.success("💡 Interpretation:")
                    st.markdown(answer)
            else:
                st.warning("⚠️ OCR failed to extract any text.")

        except Exception as e:
            error_type = type(e).__name__
            st.error(f"❌ {error_type}: {str(e)}")
