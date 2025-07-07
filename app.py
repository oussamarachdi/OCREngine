import streamlit as st
import tempfile
from ocr_utils import ocr_with_gpt4_vision, ocr_with_gemini, process_easyocr_pipeline
from pdf2image import convert_from_path, exceptions
import os

def convert_pdf_to_images(pdf_path, max_pages=1, first_page=1):
    try:
        images = convert_from_path(pdf_path, first_page=first_page, last_page=first_page + max_pages)
        temp_paths = []
        for i, image in enumerate(images):
            temp_path = f"{pdf_path}_page_{i+1}.png"
            image.save(temp_path, "PNG")
            temp_paths.append(temp_path)
        return temp_paths
    except exceptions.PDFInfoNotInstalledError:
        st.error("Poppler is not installed. Please install it to process PDF files.")
        return []
    except Exception as e:
        st.error(f"Failed to convert PDF to images: {e}")
        return []

# --- Streamlit UI ---
st.set_page_config(page_title="Intelligent OCR App", layout="centered")
st.title("🧠 Intelligent OCR Pipeline")
st.markdown("Upload an image or a PDF file and choose your OCR engine:")

uploaded_file = st.file_uploader("📄 Upload an image or PDF", type=["jpg", "jpeg", "png", "pdf"])
ocr_model = st.selectbox("🤖 Select OCR Engine", ["gpt-4o", "gemini", "easyocr"])

# EasyOCR language support
if ocr_model == "easyocr":
    selected_langs = st.multiselect("🌐 Select Languages", ["en", "ar"], default="ar")

if uploaded_file:
    file_suffix = uploaded_file.name.split('.')[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_suffix}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        input_path = tmp_file.name

    # Convert to images (for PDF) or use single image
    image_paths = []
    process_all = st.checkbox("🔁 Run OCR on all loaded pages", value=False)

    if file_suffix == "pdf":
        max_pages = st.number_input("Max Number of Pages to Convert", min_value=1, max_value=1000, value=1, help="Limit the number of pages from the PDF to process (starts from page 1)")
        first_page = st.number_input("First Page to Convert", min_value=1, max_value=1000, value=1, help="The first page to start converting from the PDF")

        if first_page > max_pages:
            st.error("First page cannot be greater than max pages.")
            first_page = 1

        max_pages = max(max_pages, first_page)

        with st.spinner("📚 Converting PDF pages to images..."):
            image_paths = convert_pdf_to_images(input_path, max_pages=max_pages, first_page=first_page)
    else:
        image_paths = [input_path]

    if image_paths:
        st.success(f"{len(image_paths)} page(s) loaded.")

        if not process_all:
            # Single-page logic
            page_options = [f"Page {i+1}" for i in range(len(image_paths))]
            selected_page_index = st.selectbox("📃 Select a page to process", range(len(image_paths)), format_func=lambda x: page_options[x])
            selected_image_path = image_paths[selected_page_index]

            st.image(selected_image_path, caption=f"Preview: {page_options[selected_page_index]}", use_container_width=True)

            if st.button("🔍 Run OCR on selected page"):
                with st.spinner(f"Running {ocr_model} on Page {selected_page_index + 1}..."):
                    try:
                        if ocr_model == "easyocr":
                            md_output, json_output = process_easyocr_pipeline(selected_image_path, langs=selected_langs)
                        elif ocr_model == "gpt-4o":
                            md_output = ocr_with_gpt4_vision(selected_image_path)
                            json_output = "{}"
                        elif ocr_model == "gemini":
                            md_output = ocr_with_gemini(selected_image_path)
                            json_output = "{}"
                        else:
                            st.error("Unsupported OCR engine.")
                            md_output, json_output = "", ""
                    except Exception as e:
                        st.error(f"OCR failed: {str(e)}")
                        md_output, json_output = "", ""

                if md_output:
                    st.success("✅ OCR Complete!")
                    st.subheader("📝 OCR Result")
                    st.markdown(json_output)
                    st.code(md_output, language="markdown")

        else:
            # Multi-page OCR logic
            if st.button("🔁 Run OCR on all pages"):
                all_md = []
                all_json = []

                with st.spinner(f"Running {ocr_model} on {len(image_paths)} page(s)..."):
                    for idx, image_path in enumerate(image_paths):
                        st.info(f"Processing Page {idx + 1}")
                        try:
                            if ocr_model == "easyocr":
                                md_output, json_output = process_easyocr_pipeline(image_path, langs=selected_langs)
                            elif ocr_model == "gpt-4o":
                                md_output = ocr_with_gpt4_vision(image_path)
                                json_output = "{}"
                            elif ocr_model == "gemini":
                                md_output = ocr_with_gemini(image_path)
                                json_output = "{}"
                            else:
                                st.error("Unsupported OCR engine.")
                                continue
                        except Exception as e:
                            st.error(f"OCR failed on page {idx+1}: {str(e)}")
                            continue

                        all_md.append(f"### Page {idx+1}\n{md_output}")
                        all_json.append(f'"page_{idx+1}": {json_output}')

                # Display results
                st.success("✅ OCR Complete for all pages!")
                st.subheader("📝 Combined Markdown")
                st.code("\n\n".join(all_md), language="markdown")

                st.subheader("📦 Combined JSON")
                st.code("{\n" + ",\n".join(all_json) + "\n}", language="json")

    else:
        st.warning("⚠️ No pages found in the uploaded file or conversion failed.")
