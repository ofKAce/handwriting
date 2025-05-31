import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import pytesseract

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Configure Streamlit UI
st.set_page_config(page_title="Handwriting Reader", page_icon="📝")
st.title("📝 Handwriting Recognition")

# Upload image
uploaded_file = st.file_uploader("Upload an image of handwriting", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    try:
        # OCR with pytesseract
        extracted_text = pytesseract.image_to_string(image)
        st.subheader("✍️ Extracted Text from Image:")
        st.code(extracted_text)

        if extracted_text.strip():
            # Setup Groq LLM
            llm = ChatGroq(
                model_name="llama3-70b-8192",
                groq_api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.2,
            )

            # LangChain prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an AI assistant that helps interpret handwriting."),
                ("human", "Here is some handwritten text:\n\n{text}\n\nPlease explain or summarize what it says.")
            ])

            chain = prompt | llm | StrOutputParser()

            # Ask Groq
            response = chain.invoke({"text": extracted_text})
            st.subheader("🤖 Interpretation:")
            st.markdown(response)

        else:
            st.warning("No text could be detected from the image.")

    except Exception as e:
        st.exception(e)
