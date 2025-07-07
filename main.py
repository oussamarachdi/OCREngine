import streamlit as st
import tempfile
from ocr_utils import ocr_with_gpt4_vision, ocr_with_gemini, process_easyocr_pipeline

st.set_page_config(page_title="Intelligent OCR App", layout="centered")
st.title("Intelligent OCR Pipeline")
st.markdown("Upload an image and choose the OCR engine (Gemini API Or OpenAI GPT-4 API)")

uploaded_file = st.file_uploader("Upload an image file ", type=["jpg", "jpeg", "png"])  

ocr_model = st.selectbox("Select OCR Engine", ["gpt-4o", "gemini", "easyocr"])
if ocr_model == "easyocr":
    selected_langs = st.multiselect(
        "Select Languages for EasyOCR",
        options = ["en", "ar"],
        default="ar"
    )
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_file.read())
        image_path = tmp_file.name

    st.image(image_path, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Run OCR"):
        if ocr_model == "easyocr":
            md_output, json_output = process_easyocr_pipeline(image_path, langs=selected_langs)
        elif ocr_model == "gpt-4o":
            md_output = ocr_with_gpt4_vision(image_path)
            json_output = "{}"
        elif ocr_model == "gemini":
            md_output = ocr_with_gemini(image_path)
            json_output = "{}"
        else:
            st.error("Unsupported engine selected.")
            md_output, json_output = "", ""
    


        st.subheader("OCR Result")
        st.markdown(json_output)

        st.code(md_output, language="markdown")