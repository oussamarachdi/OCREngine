from fastapi import FastAPI, UploadFIle, File, Form
from fastapi.responses import JSONResponse
from ocr_utils import ocr_with_gpt4_vision, ocr_with_gemini, process_easyocr_pipeline