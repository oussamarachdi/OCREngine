import os
import cv2
import json
import base64
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
import openai
from openai import OpenAI
import easyocr
from langdetect import detect
import language_tool_python as language_tool

# Load environment variables
load_dotenv()

# Initialize clients
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ===== GPT-4o OCR =====
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def ocr_with_gpt4_vision(image_path):
    image_base64 = image_to_base64(image_path)

    response =client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the text and format it in Markdown. Identify titles, lists, and tables."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ],
            }
        ],
        max_tokens=4096,
        temperature=0.5,
    )

    return response.choices[0].message.content

# ===== Gemini Vision OCR =====
def ocr_with_gemini(image_path):
    model = genai.GenerativeModel("gemini-1.5-flash")

    with Image.open(image_path) as image:
        response = model.generate_content([
            "Extract all text in Markdown. Format sections, titles, lists, tables.",
            image
        ])

    return response.text

# ===== EasyOCR Pipeline =====

def preprocess_image_for_easyocr(image_path):
    image = cv2.imread(image_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def extract_text_easyocr(image_path, langs=['ar', 'en']):
    reader = easyocr.Reader(langs)
    result = reader.readtext(image_path)
    return '\n'.join([r[1] for r in result])

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def correct_text(text, lang_code='en'):
    tool = language_tool.LanguageTool(lang_code)
    matches = tool.check(text)
    corrected = language_tool.utils.correct(text, matches)
    return corrected

def detect_structure(text):
    lines = text.split('\n')
    structured = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.endswith(':'):
            structured.append({"type": "heading", "text": line})
        elif line.startswith(("-", "*", "•", "1.", "2.", "3.")):
            structured.append({"type": "list", "text": line})
        else:
            structured.append({"type": "paragraph", "text": line})

    return structured

def to_markdown(structure):
    md = []
    for block in structure:
        if block["type"] == "heading":
            md.append(f"## {block['text']}")
        elif block["type"] == "list":
            md.append(f"- {block['text']}")
        elif block["type"] == "paragraph":
            md.append(block["text"])
    return "\n\n".join(md)

def to_json(structure):
    return json.dumps(structure, indent=2, ensure_ascii=False)

def process_easyocr_pipeline(image_path, langs=['ar', 'en']):
    # Preprocess image
    preprocessed = preprocess_image_for_easyocr(image_path)

    # Save preprocessed image to temp
    tmp_image_path = image_path.replace(".png", "_preprocessed.png")
    cv2.imwrite(tmp_image_path, preprocessed)

    # OCR with EasyOCR
    raw_text = extract_text_easyocr(tmp_image_path, langs=langs)
    lang = detect_language(raw_text)
    corrected = correct_text(raw_text, lang_code=lang if lang != "unknown" else 'en')
    structure = detect_structure(corrected)
    return to_markdown(structure), to_json(structure)
