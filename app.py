import os
import io
import cv2
import numpy as np
from PIL import Image
import pytesseract
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file

# Google Gemini Official New SDK
from google import genai
from google.genai import types

# Document Generators
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

# ==========================================
# GOOGLE GEMINI API CONFIGURATION
# ==========================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Direct Client Initialize
ai_client = genai.Client(
    api_key=GEMINI_API_KEY
)

MODEL_NAME = "gemini-flash-latest"


def preprocess_image_for_ocr(image_bytes):
    """OpenCV Preprocessing Engine with RAM Optimization"""
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image format or corrupted file.")

    # Convert to PIL Image for Resizing (RAM ပြည့်ပြီး Server မလဲစေရန်)
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil_img.thumbnail((1200, 1200))  # Max dimensions: 1200x1200px
    
    # Convert back to OpenCV format
    image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return Image.fromarray(binary)


@app.route('/')
def index():
    return render_template('index.html')


# ==========================================
# OCR SCAN ENDPOINT
# ==========================================
@app.route('/ocr', methods=['POST'])
@app.route('/scan', methods=['POST'])
def scan_ocr():
    file = request.files.get('file') or request.files.get('image')
    
    if not file or file.filename == '':
        return jsonify({'success': False, 'error': 'No image file uploaded'}), 400

    lang = request.form.get('lang', 'mya+eng')

    try:
        image_bytes = file.read()
        processed_img = preprocess_image_for_ocr(image_bytes)

        custom_config = r'--oem 3 --psm 6'
        extracted_text = pytesseract.image_to_string(
            processed_img,
            lang=lang,
            config=custom_config
        )

        return jsonify({
            'success': True,
            'text': extracted_text.strip(),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    except Exception as e:
        print(f"[OCR Error]: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==========================================
# GOOGLE GEMINI AI ASSISTANT ENDPOINT
# ==========================================
@app.route('/ai/process', methods=['POST'])
def ai_process():
    data = request.get_json() or {}
    text = data.get('text', '')
    action = data.get('action', '')
    target_lang = data.get('target_lang', '')
    user_prompt = data.get('prompt') or data.get('query', '')

    if not text and action != 'ask':
        return jsonify({'success': False, 'error': 'No text provided for AI processing'}), 400

    try:
        if action == 'translate':
            if target_lang == 'en':
                lang_instruction = "Translate the input text into clear and standard English."
            elif target_lang == 'my':
                lang_instruction = "Translate the input text into natural Myanmar (Burmese) language."
            else:
                lang_instruction = (
                    "If the input text is primarily in Myanmar, translate it to English. "
                    "If it is in English, translate it to Myanmar."
                )

            prompt = (
                f"You are a professional translator. {lang_instruction}\n"
                f"Preserve document structure, formatting, and line breaks. "
                f"Provide ONLY the final translation result without explanations or greetings.\n\n"
                f"Input Text:\n{text}"
            )

        elif action == 'summarize':
            prompt = (
                f"Please provide a concise, well-structured bullet-point summary of the following text. "
                f"If the source text is in Myanmar, reply in Myanmar language. If English, reply in English:\n\n{text}"
            )

        elif action == 'explain':
            if user_prompt:
                prompt = f"Task: {user_prompt}\n\nTarget Text:\n{text}"
            else:
                prompt = (
                    f"Explain the key concepts, main points, and contextual details of the following text in simple, clear terms. "
                    f"Match the primary language of the source text:\n\n{text}"
                )

        elif action == 'ask':
            prompt = (
                f"Context Document:\n{text}\n\n"
                f"User Question: {user_prompt}\n\n"
                f"Answer the user's question accurately based on the provided context document. Match the language of the user's question."
            )

        else:
            return jsonify({'success': False, 'error': 'Invalid AI action'}), 400

        # API Call
        response = ai_client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are a helpful and intelligent AI assistant specialized in OCR document processing and translation.",
                temperature=0.3
            )
        )

        result_text = response.text.strip()
        return jsonify({'success': True, 'result': result_text})

    except Exception as e:
        print(f"\n[AI Process Error Log]: {str(e)}\n")
        return jsonify({'success': False, 'error': f"Gemini AI Error: {str(e)}"}), 500


# ==========================================
# EXPORT ROUTES (DOCX & PDF)
# ==========================================
@app.route('/export/docx', methods=['POST'])
def export_docx():
    data = request.get_json() or {}
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    doc = Document()
    doc.add_heading('OCR Extracted Document', level=1)
    
    for line in text.split('\n'):
        if line.strip():
            doc.add_paragraph(line)

    doc_io = io.BytesIO()
    doc.save(doc_io)
    doc_io.seek(0)

    return send_file(
        doc_io,
        mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        as_attachment=True,
        download_name='ocr_result.docx'
    )


@app.route('/export/pdf', methods=['POST'])
def export_pdf():
    data = request.get_json() or {}
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    pdf_io = io.BytesIO()
    doc = SimpleDocTemplate(pdf_io, pagesize=letter)
    styles = getSampleStyleSheet()
    
    story = []
    title_style = styles['Heading1']
    body_style = styles['Normal']
    body_style.fontSize = 11
    body_style.leading = 16

    story.append(Paragraph("OCR Extracted Document", title_style))
    story.append(Spacer(1, 18))

    for line in text.split('\n'):
        if line.strip():
            clean_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(clean_line, body_style))
            story.append(Spacer(1, 8))

    doc.build(story)
    pdf_io.seek(0)

    return send_file(
        pdf_io,
        mimetype='application/pdf',
        as_attachment=True,
        download_name='ocr_result.pdf'
    )


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
