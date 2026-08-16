import os
import io
import socket
import cv2
import numpy as np
from PIL import Image
import pytesseract
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import time
from flask import send_file, request
# .env ဖိုင်မှ API Key များကို ဖတ်ရန်
from dotenv import load_dotenv
load_dotenv()

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
# GOOGLE GEMINI API CONFIGURATION & MODELS
# ==========================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Direct Client Initialize (New SDK)
ai_client = genai.Client(
    api_key=GEMINI_API_KEY
)

# ယခုလက်ရှိ အသုံးဝင်သော Active Model စာရင်းအသစ်များ (Model Fallback Chain)
AVAILABLE_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash"
]


# ==========================================
# INTERNET / VPN CONNECTIVITY CHECKER
# ==========================================
def is_connected(host="8.8.8.8", port=53, timeout=3):
    """အင်တာနက် သို့မဟုတ် VPN ချိတ်ဆက်မှု ရှိမရှိ စစ်ဆေးခြင်း"""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False


# ==========================================
# MULTI-MODEL FALLBACK & RETRY MECHANISM
# ==========================================
def call_gemini_with_backoff(contents, system_instruction="You are a helpful and intelligent AI assistant."):
    """Model တစ်ခုချင်းစီ၏ Quota နှင့် Server Busy များကို စစ်ဆေးပြီး အခြား Model သို့ အလိုအလျောက် ကူးပြောင်းပေးသည်"""
    
    for model_name in AVAILABLE_MODELS:
        delay = 2  # စတင်စောင့်ရမည့် အချိန် (စက္ကန့်)
        
        for attempt in range(2):  # မော်ဒယ် တစ်ခုချင်းစီအတွက် ၂ ကြိမ်စီ ကြိုးစားမည်
            try:
                response = ai_client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.1
                    )
                )
                print(f"✅ Successfully used Gemini Model: {model_name}")
                return response
            
            except Exception as e:
                err_str = str(e)
                
                # Quota ပြည့်သွားပါက (သို့မဟုတ်) Model ရှာမတွေ့ပါက ဤ Model ကို ကျော်ပြီး နောက် Model သို့ ချက်ချင်း ကူးမည်
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str or "404" in err_str:
                    print(f"⚠️ Model '{model_name}' အလုပ်မလုပ်ပါ (သို့) Quota ကုန်သွားပါပြီ။ နောက် Model တစ်ခုသို့ ပြောင်းနေပါသည်...")
                    break  # Inner loop မှ ထွက်ပြီး next model သို့ သွားမည်
                
                # Server Busy ဖြစ်ပါက ခဏစောင့်ပြီး ထပ်ကြိုးစားမည်
                if any(x in err_str for x in ["503", "UNAVAILABLE", "disconnected", "high demand", "timed out"]):
                    print(f"⚠️ [{model_name}] Server Busy နေလို့ {delay} စက္ကန့် စောင့်ပြီး ပြန်ကြိုးစားနေပါသည်... (ကြိုးစားမှု - {attempt+1})")
                    time.sleep(delay)
                    delay *= 2
                else:
                    print(f"⚠️ [{model_name}] Error: {err_str}")
                    break
                    
    raise Exception("Google AI Free Tier Quota အားလုံး (သို့မဟုတ်) ဆာဗာချိတ်ဆက်မှု အပြည့်အဝ ကုန်ဆုံးသွားပါပြီ။ ကျေးဇူးပြု၍ ခဏစောင့်ပြီးမှ ထပ်ကြိုးစားပါ။")


def preprocess_image_for_gemini(image_bytes):
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image format or corrupted file.")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)


# ==========================================
# HIGH ACCURACY LOCAL TESSERACT PREPROCESSING
# ==========================================
def preprocess_image_for_tesseract(image_bytes):
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image format or corrupted file.")

    height, width = image.shape[:2]
    if width < 1500:
        scale = 1500 / width
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)

    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return Image.fromarray(sharpened)


@app.route('/')
def index():
    return render_template('index.html')


# ==========================================
# OCR SCAN ENDPOINT (Smart Internet Check + Fallback)
# ==========================================
@app.route('/ocr', methods=['POST'])
@app.route('/scan', methods=['POST'])
def scan_ocr():
    file = request.files.get('file') or request.files.get('image')
    
    if not file or file.filename == '':
        return jsonify({'success': False, 'error': 'No image file uploaded'}), 400

    image_bytes = file.read()
    extracted_text = ""

    if not is_connected():
        print("🌐 No internet/VPN connection detected. Using High-Accuracy Local Tesseract OCR directly.")
        try:
            processed_img = preprocess_image_for_tesseract(image_bytes)
            custom_config = r'--oem 3 --psm 3'
            extracted_text = pytesseract.image_to_string(
                processed_img,
                lang='mya+eng',
                config=custom_config
            ).strip()
            print("✅ OCR successfully extracted using Local Tesseract (Offline Mode).")
        except Exception as t_err:
            return jsonify({'success': False, 'error': f"Local OCR Extraction Failed: {str(t_err)}"}), 500

    else:
        try:
            pil_img = preprocess_image_for_gemini(image_bytes)
            ocr_prompt = (
                "Extract all text from this image with absolute accuracy. "
                "Pay close attention to Myanmar script (including diacritics and vowels), "
                "tables, math equations, and special symbols. "
                "Preserve the original layout and line breaks. "
                "Provide ONLY the extracted text without any introductory remarks."
            )
            sys_inst = "You are an expert OCR and document analysis engine specialized in extracting text, tables, and handwriting with high accuracy."
            
            response = call_gemini_with_backoff([pil_img, ocr_prompt], system_instruction=sys_inst)
            extracted_text = response.text.strip()
            print("✅ OCR successfully extracted using Gemini Vision AI.")

        except Exception as e:
            print(f"🔄 Falling back to Optimized Local Tesseract OCR due to: {str(e)}")
            try:
                processed_img = preprocess_image_for_tesseract(image_bytes)
                custom_config = r'--oem 3 --psm 3'
                extracted_text = pytesseract.image_to_string(
                    processed_img,
                    lang='mya+eng',
                    config=custom_config
                ).strip()
                print("✅ OCR successfully extracted using Optimized Local Tesseract (Fallback).")
            except Exception as t_err:
                return jsonify({'success': False, 'error': f"OCR Extraction Failed: {str(t_err)}"}), 500

    return jsonify({
        'success': True,
        'text': extracted_text,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


# ==========================================
# GOOGLE GEMINI AI ASSISTANT ENDPOINT (Multi-Model Fallback)
# ==========================================
@app.route('/ai/process', methods=['POST'])
def ai_process():
    data = request.get_json() or {}
    text = data.get('text', '')
    action = data.get('action', '')
    target_lang = data.get('target_lang', '')
    user_prompt = data.get('prompt') or data.get('query', '')

    if not is_connected():
        return jsonify({'success': False, 'error': 'AI features require an active internet connection or VPN.'}), 400

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

        elif action == 'fix' or action == 'spellcheck':
            prompt = (
                f"Please review and fix all spelling errors, grammar mistakes, and typos in the following text. "
                f"Maintain the original meaning and formatting. Provide ONLY the corrected text:\n\n{text}"
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

        sys_inst = "You are a helpful and intelligent AI assistant specialized in document processing, translation, and text correction."
        response = call_gemini_with_backoff(prompt, system_instruction=sys_inst)

        result_text = response.text.strip()
        return jsonify({'success': True, 'result': result_text})

    except Exception as e:
        err_msg = str(e)
        print(f"\n[AI Process Error Log]: {err_msg}\n")
        return jsonify({'success': False, 'error': f"Gemini AI Error: {err_msg}"}), 500


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

@app.route('/export/txt', methods=['POST'])
def export_txt():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        # Text ကို Memory ထဲမှာ Bytes အဖြစ်ပြောင်းပြီး ဖိုင်အဖြစ် ပို့ပေးခြင်း
        byte_io = io.BytesIO(text.encode('utf-8'))
        
        return send_file(
            byte_io,
            mimetype='text/plain',
            as_attachment=True,
            download_name='Extracted_Text.txt'
        )
    except Exception as e:
        return {'error': str(e)}, 500

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
        download_name='result.pdf'
    )


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)