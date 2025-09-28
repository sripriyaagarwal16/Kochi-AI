import os
import fitz  # PyMuPDF
import fasttext
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from IndicTransToolkit.processor import IndicProcessor
import torch
from PIL import Image
import requests
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()  # will read from .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

TRANSLATION_MODEL_REPO_ID = "ai4bharat/indictrans2-indic-en-1B"
OCR_MODEL_ID = "microsoft/trocr-base-printed"
LANGUAGE_TO_TRANSLATE = "mal"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {DEVICE}")

# --- GLOBAL MODEL LOADING ---
print("Loading fastText language detector...")
ft_model_path = hf_hub_download(
    repo_id="facebook/fasttext-language-identification",
    filename="model.bin"
)
lang_detect_model = fasttext.load_model(ft_model_path)
print("✅ fastText loaded.")

print("Loading OCR model...")
ocr_pipeline = pipeline("image-to-text", model=OCR_MODEL_ID, device=-1)
print("✅ OCR loaded.")

print(f"Loading tokenizer & model: {TRANSLATION_MODEL_REPO_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_REPO_ID, trust_remote_code=True)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(
    TRANSLATION_MODEL_REPO_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)
print("✅ Translation model loaded.")

ip = IndicProcessor(inference=True)
print("✅ IndicProcessor initialized.")


# --- UTILITY FUNCTIONS ---
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        txt = ""
        for p in doc:
            txt += p.get_text("text") + "\n"
        doc.close()
        return txt
    except Exception as e:
        print(f"PDF extract error: {e}")
        return None

def read_text_from_txt(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"TXT read error: {e}")
        return None

def extract_text_from_image(path):
    try:
        with Image.open(path) as img:
            out = ocr_pipeline(img)
        return out[0]["generated_text"] if out else ""
    except Exception as e:
        print(f"Image OCR error: {e}")
        return None

def detect_language(text_snippet):
    s = text_snippet.replace("\n", " ").strip()
    if not s:
        return None
    preds = lang_detect_model.predict(s, k=1)
    if preds and preds[0]:
        label = preds[0][0]
        code = label.split("__")[-1]
        return code
    return None

def translate_chunk(chunk, src_lang="mal_Mlym", tgt_lang="eng_Latn"):
    if not chunk.strip():
        return ""
    batch = ip.preprocess_batch([chunk], src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        generated_tokens = translation_model.generate(
            **inputs, use_cache=False, min_length=0, max_length=256,
            num_beams=5, num_return_sequences=1,
        )
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    translations = ip.postprocess_batch(decoded, lang=tgt_lang)
    return translations[0]

def summarize_with_gemini(text_to_summarize, api_key):
    if not api_key:
        return {"error": "Gemini API key is not configured."}

    print("\nAnalyzing the document with Gemini...")
    model = 'gemini-1.5-flash-latest'
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    prompt = f"""
    You are an expert AI assistant for KMRL (Kochi Metro Rail Limited) document management.
    You have been given the following document. Your task is to analyze it and extract key information.

    **Document Content:**
    ---
    {text_to_summarize}
    ---

    Based on the document, perform the following actions and provide the output in a valid JSON format:
    1. Summarize the document in 2-3 concise sentences highlighting key points.
    2. Identify specific actions required. For each action, detect any timeline/deadline, assign a priority ("High", "Medium", or "Low"), and add brief notes for traceability.
    3. Suggest a list of departments that should be notified.
    4. Detect if this document references or relates to previous incidents, maintenance logs, or similar documents and flag any recurring issues.
    """

    json_schema = {
        "type": "OBJECT",
        "properties": {
            "summary": {"type": "STRING"},
            "actions_required": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "action": {"type": "STRING"},
                        "priority": {"type": "STRING", "enum": ["High", "Medium", "Low"]},
                        "deadline": {"type": "STRING"},
                        "notes": {"type": "STRING"}
                    },
                    "required": ["action", "priority", "deadline", "notes"]
                }
            },
            "departments_to_notify": {"type": "ARRAY", "items": {"type": "STRING"}},
            "cross_document_flags": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "related_document_type": {"type": "STRING"},
                        "related_issue": {"type": "STRING"}
                    },
                    "required": ["related_document_type", "related_issue"]
                }
            }
        },
        "required": ["summary", "actions_required", "departments_to_notify", "cross_document_flags"]
    }

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": json_schema
        }
    }

    try:
        response = requests.post(api_url, headers={"Content-Type": "application/json"}, json=payload)
        response.raise_for_status()
        result = response.json()
        if 'candidates' in result and result['candidates']:
            json_text = result['candidates'][0]['content']['parts'][0]['text']
            print("✅ Gemini analysis successful.")
            return json.loads(json_text)
        else:
            return {"error": "Invalid response from Gemini", "raw": result}
    except Exception as e:
        return {"error": str(e)}


# --- MAIN PROCESSING FUNCTION ---
def process_and_analyze_document(input_file_path):
    ext = os.path.splitext(input_file_path)[1].lower()
    if ext == ".pdf":
        original_text = extract_text_from_pdf(input_file_path)
    elif ext == ".txt":
        original_text = read_text_from_txt(input_file_path)
    elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        original_text = extract_text_from_image(input_file_path)
    else:
        return {"error": "Unsupported file type. Please upload a .pdf, .txt, or image file."}

    if not original_text or not original_text.strip():
        return {"error": "Could not extract any text from the document."}

    lines = original_text.split("\n")
    translated_lines = []

    for i, ln in enumerate(lines):
        if not ln.strip():
            translated_lines.append("")
            continue

        lang = detect_language(ln)
        if lang == LANGUAGE_TO_TRANSLATE:
            print(f"  -> Translating chunk {i+1} (Malayalam)...")
            translated = translate_chunk(ln, src_lang="mal_Mlym", tgt_lang="eng_Latn")
            translated_lines.append(translated)
        else:
            translated_lines.append(ln)

    translated_text = "\n".join(translated_lines)

    if not translated_text.strip():
        return {"error": "The document was empty after translation."}

    return summarize_with_gemini(translated_text, GEMINI_API_KEY)


# --- FLASK APP ---
app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    temp_path = os.path.join("/tmp", file.filename)
    file.save(temp_path)

    result = process_and_analyze_document(temp_path)
    return jsonify(result)


@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "KMRL Document Analysis API is running."})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
