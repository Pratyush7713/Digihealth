import base64
import hashlib
import io
import json
import os
import re
import time
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import fitz
import pandas as pd
import pytesseract
import streamlit as st
from deep_translator import GoogleTranslator
from groq import Groq
from gtts import gTTS
from PIL import Image

DATA_PATH = os.path.join("data", "users.csv")
STORAGE_ROOT = os.path.join("storage")
HISTORY_FILE = "disease_history.json"

# Configure Tesseract OCR path for Windows
if os.name == 'nt':  # Windows
    tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_cmd):
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

st.set_page_config(page_title="Swasthi Health Locker (MVP)", page_icon="🩺", layout="wide")

def get_groq_client():
    """Initialize Groq client with API key from environment or Streamlit secrets (Groq is free!)"""
    try:
        # First try environment variable
        api_key = os.getenv("GROQ_API_KEY")
        
        # If not in environment, try Streamlit secrets
        if not api_key:
            try:
                if hasattr(st, 'secrets') and st.secrets:
                    api_key = st.secrets.get("GROQ_API_KEY", "")
            except Exception as e:
                # If secrets not available, return None
                api_key = ""
        
        # For free tier, users can get API key from https://console.groq.com
        if not api_key:
            return None
        
        return Groq(api_key=api_key)
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def fetch_disease_info(disease_name: str) -> Optional[Dict]:
    """Fetch detailed information about a disease using Groq (free AI)"""
    client = get_groq_client()
    if not client:
        return None
    
    prompt = f"""Provide detailed medical information about the disease/condition: {disease_name}

Format the response as a JSON object with the following structure:
{{
    "name": "disease name",
    "information": "brief overview (2-3 sentences)",
    "symptoms": ["symptom1", "symptom2", "symptom3", ...],
    "precautions": ["precaution1", "precaution2", "precaution3", ...],
    "treatment": "treatment/cure information",
    "medications": ["medication1", "medication2", ...],
    "doctor_type": "type of doctor to consult (e.g., Cardiologist, General Physician, etc.)"
}}

Only return valid JSON, no additional text."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Free and fast model from Groq
            messages=[
                {"role": "system", "content": "You are a medical information assistant. Always return valid JSON only. Do not include any markdown formatting or code blocks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        content = response.choices[0].message.content.strip()
        
        # Clean the content - remove markdown code blocks if present
        if content.startswith("```"):
            parts = content.split("```")
            if len(parts) > 1:
                content = parts[1]
                if content.startswith("json"):
                    content = content[4:].strip()
        # Remove any remaining backticks or whitespace
        content = content.strip().strip('`').strip()
        
        # Try to parse JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError as je:
            # Try to find JSON object in the content
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except:
                    pass
            # If all parsing fails, return None
            return None
            
    except Exception as e:
        # Return None on error - don't show error to user
        return None

def validate_lab_value(value_str: str, normal_range: str) -> str:
    """Validate if a lab value is actually within normal range and return correct status"""
    try:
        # Parse normal range (e.g., "4000-11000" or "12-16")
        range_match = re.search(r'([\d.]+)\s*-\s*([\d.]+)', normal_range)
        if not range_match:
            return "unknown"
        
        min_val = float(range_match.group(1))
        max_val = float(range_match.group(2))
        
        # Parse actual value
        value_match = re.search(r'([\d.]+)', value_str.replace(',', ''))
        if not value_match:
            return "unknown"
        
        actual_val = float(value_match.group(1))
        
        # Determine status
        if actual_val < min_val:
            return "low"
        elif actual_val > max_val:
            return "high"
        else:
            return "normal"
    except:
        return "unknown"

@st.cache_data(ttl=3600)
def analyze_lab_values(text: str) -> Dict:
    """Analyze lab report text to find abnormal values and suggest possible diseases"""
    client = get_groq_client()
    if not client:
        return None
    
    prompt = f"""Analyze the following medical report/lab report text and identify:
1. ALL lab values with their test names, values, and normal ranges
2. ONLY abnormal lab values (values that are ACTUALLY outside the normal range)
3. Possible diseases or conditions these abnormalities could indicate

IMPORTANT: Only mark values as "high" or "low" if they are ACTUALLY outside the normal range provided.
If a value is within the normal range, do NOT include it in abnormal_values.

Text:
{text[:2000]}

Return a JSON object with this structure:
{{
    "abnormal_values": [
        {{"test_name": "WBC", "value": "15000", "normal_range": "4000-11000", "status": "high", "unit": "cells/μL"}},
        {{"test_name": "Hemoglobin", "value": "10.5", "normal_range": "12-16", "status": "low", "unit": "g/dL"}}
    ],
    "possible_diseases": [
        {{"disease": "Infection", "reason": "High WBC count indicates possible bacterial infection"}},
        {{"disease": "Anemia", "reason": "Low hemoglobin suggests iron deficiency or anemia"}}
    ],
    "summary": "Brief 2-3 sentence summary of ONLY actual abnormalities found"
}}

Return only valid JSON, no additional text."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a medical lab analyzer. Only include values that are ACTUALLY outside normal ranges. Always return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        content = response.choices[0].message.content.strip()
        
        # Clean and parse JSON
        if content.startswith("```"):
            parts = content.split("```")
            if len(parts) > 1:
                content = parts[1]
                if content.startswith("json"):
                    content = content[4:].strip()
        content = content.strip().strip('`').strip()
        
        try:
            result = json.loads(content)
            # Validate each abnormal value
            if result.get("abnormal_values"):
                validated_values = []
                for abv in result["abnormal_values"]:
                    value_str = str(abv.get("value", ""))
                    normal_range = abv.get("normal_range", "")
                    actual_status = validate_lab_value(value_str, normal_range)
                    
                    # Only include if actually abnormal
                    if actual_status in ["high", "low"]:
                        abv["status"] = actual_status
                        validated_values.append(abv)
                    # If AI said it was abnormal but validation shows it's normal, exclude it
                
                result["abnormal_values"] = validated_values
            
            return result
        except json.JSONDecodeError:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return None
    except:
        return None

@st.cache_data(ttl=3600)
def generate_short_summary(disease_info: Dict, symptoms: str, doctor_type: str, language: str = "en") -> str:
    """Generate a short 2-5 line summary of the disease, symptoms, and doctor type"""
    client = get_groq_client()
    if not client:
        # Fallback summary
        summary = f"The patient is suffering from {disease_info.get('name', 'a medical condition')}."
        if symptoms:
            summary += f" Symptoms include: {symptoms[:100]}."
        if doctor_type:
            summary += f" Consultation with a {doctor_type} is recommended."
        return summary
    
    lang_codes = {"english": "en", "hindi": "hi", "marathi": "mr", "telugu": "te", "tamil": "ta", "kannada": "kn"}
    lang_code = lang_codes.get(language.lower(), "en")
    
    disease_name = disease_info.get('name', 'medical condition')
    disease_info_text = disease_info.get('information', '')
    symptoms_text = symptoms or 'Not specified'
    
    prompt = f"""Create a very short summary (2-5 lines maximum, about 50-100 words) in {language} language about:
- Disease/condition: {disease_name}
- Information: {disease_info_text}
- Symptoms: {symptoms_text}
- Doctor type to consult: {doctor_type}

Write in {language} language. Make it concise, clear, and informative. Do not include markdown formatting."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": f"You are a medical translator and summarizer. Respond in {language} language only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except:
        # Fallback
        return f"Patient has {disease_name}. Symptoms: {symptoms_text}. Consult {doctor_type}."

def generate_overall_medical_summary(history: List[Dict]) -> str:
    """Generate an AI-powered overall medical summary from complete history"""
    client = get_groq_client()
    if not client:
        # Fallback summary
        if not history:
            return "No medical history available."
        
        summary_parts = []
        summary_parts.append(f"### Medical History Overview\n")
        summary_parts.append(f"**Total Records:** {len(history)}\n")
        
        # Get unique diseases
        diseases = []
        for entry in history:
            disease = entry.get('disease', 'Not specified')
            if disease != 'Not specified' and disease not in diseases:
                diseases.append(disease)
        
        if diseases:
            summary_parts.append(f"**Conditions:** {', '.join(diseases[:5])}\n")
        
        return "\n".join(summary_parts)
    
    # Sort history by date (latest first) - already sorted but ensure it
    sorted_history = sorted(history, key=lambda x: x.get('date', ''), reverse=True)
    
    # Build comprehensive prompt with recent history emphasized
    recent_entries = sorted_history[:5]  # Last 5 most recent reports
    older_entries = sorted_history[5:]   # Remaining reports
    
    prompt = f"""Analyze the following complete medical history and provide a comprehensive overall summary with recommendations.

RECENT MEDICAL REPORTS (Most Important - Last 5 visits):
{json.dumps(recent_entries, indent=2)[:3000]}

PREVIOUS MEDICAL REPORTS (for context):
{json.dumps(older_entries, indent=2)[:2000] if older_entries else "None"}

Provide a structured medical summary with:
1. **Current Health Status** - Primary focus on latest reports
2. **Medical Conditions Overview** - All conditions mentioned across history
3. **Treatment History** - Key medications and therapies
4. **Progress Analysis** - Trends in health over time
5. **Recommendations** - Specific actionable advice including:
   - Doctor consultation suggestions
   - Lifestyle recommendations
   - Preventive measures
   - Follow-up care suggestions

Write professionally in English. Use bullet points for recommendations. Be specific and practical.
Return as formatted markdown text (no JSON)."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a medical advisor analyzing patient history. Provide clear, actionable recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except:
        # Fallback
        return "Unable to generate summary. Please try again."

def translate_text(text: str, target_lang: str = "en") -> str:
    """Translate text to target language"""
    try:
        lang_codes = {"english": "en", "hindi": "hi", "marathi": "mr", "telugu": "te", "tamil": "ta", "kannada": "kn"}
        lang_code = lang_codes.get(target_lang.lower(), "en")
        
        if lang_code == "en" or not text.strip():
            return text
        
        translator = GoogleTranslator(source='en', target=lang_code)
        translated = translator.translate(text)
        return translated if translated else text
    except Exception as e:
        return text  # Return original on error

def text_to_speech(text: str, language: str = "en", speed: float = 1.0) -> bytes:
    """Convert text to speech audio"""
    try:
        lang_codes = {"english": "en", "hindi": "hi", "marathi": "mr", "telugu": "te", "tamil": "ta", "kannada": "kn"}
        lang_code = lang_codes.get(language.lower(), "en")
        
        # Generate TTS audio
        tts = gTTS(text=text, lang=lang_code, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()
    except Exception as e:
        return None

def create_audio_player_html(audio_bytes: bytes, playback_rate: float = 1.0, player_id: str = "audioPlayer") -> str:
    """Create HTML5 audio player with speed control"""
    import base64
    audio_b64 = base64.b64encode(audio_bytes).decode()
    audio_data_url = f"data:audio/mpeg;base64,{audio_b64}"
    
    return f"""
    <audio id="{player_id}" controls style="width: 100%;">
        <source src="{audio_data_url}" type="audio/mpeg">
    </audio>
    <script>
        (function() {{
            const audioId = '{player_id}';
            const audio = document.getElementById(audioId);
            if (audio) {{
                audio.addEventListener('loadedmetadata', function() {{
                    this.playbackRate = {playback_rate};
                }});
                // Also set immediately if already loaded
                if (audio.readyState >= 1) {{
                    audio.playbackRate = {playback_rate};
                }}
            }}
        }})();
    </script>
    """

def extract_disease_names(text: str, parsed: dict) -> List[str]:
    """Extract disease names from text"""
    diseases = []
    
    # Try to get from parsed diagnosis
    diagnosis = parsed.get("diagnosis") or parsed.get("final diagnosis") or parsed.get("impression") or parsed.get("assessment")
    if diagnosis:
        # Split by common separators
        for d in re.split(r'[,;]|\sand\s', diagnosis):
            d = d.strip()
            if d and len(d) > 2:
                diseases.append(d)
    
    # Also search in text for common disease patterns
    common_diseases = [
        "asthma", "diabetes", "hypertension", "migraine", "bronchitis", "pneumonia",
        "flu", "cold", "fever", "covid", "dengue", "malaria", "typhoid", "hepatitis",
        "tuberculosis", "arthritis", "anemia", "obesity", "anxiety", "depression"
    ]
    
    text_lower = text.lower()
    for disease in common_diseases:
        if disease in text_lower and disease not in [d.lower() for d in diseases]:
            diseases.append(disease.capitalize())
    
    return diseases[:5] if diseases else ["Unknown Condition"]

def get_history_file_path(aadhaar: str) -> str:
    """Get path to user's disease history file"""
    user_dir = ensure_user_dir(aadhaar)
    return os.path.join(user_dir, HISTORY_FILE)

def load_disease_history(aadhaar: str) -> List[Dict]:
    """Load user's disease history"""
    history_path = get_history_file_path(aadhaar)
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_disease_history(aadhaar: str, history: List[Dict]):
    """Save user's disease history"""
    history_path = get_history_file_path(aadhaar)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

def delete_from_history(aadhaar: str, file_name: str):
    """Delete an entry from disease history by file name"""
    history = load_disease_history(aadhaar)
    history = [entry for entry in history if entry.get("file_name") != file_name]
    save_disease_history(aadhaar, history)
    return len(history)

def delete_user_file(user_dir: str, file_name: str) -> bool:
    """Delete a file from user's storage"""
    try:
        file_path = os.path.join(user_dir, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    except Exception as e:
        return False

def add_to_history(aadhaar: str, file_name: str, disease_name: str, parsed: dict, meds: List[str], text: str = ""):
    """Add a new entry to disease history with comprehensive information"""
    history = load_disease_history(aadhaar)
    
    # Extract date from filename or use current date
    date_str = datetime.now().strftime("%Y-%m-%d")
    if re.search(r'\d{8}', file_name):
        match = re.search(r'(\d{8})', file_name)
        if match:
            try:
                date_str = datetime.strptime(match.group(1), "%Y%m%d").strftime("%Y-%m-%d")
            except:
                pass
    
    # Extract doctor name from parsed data or text
    doctor_name = "Not specified"
    for key in ["doctor", "physician", "consulted doctor", "dr.", "dr ", "doctor name", "physician name", "attending physician"]:
        if key in parsed and parsed[key].strip():
            doctor_name = parsed[key].strip()
            break
    
    # Also try to find doctor pattern in text
    if doctor_name == "Not specified" and text:
        doctor_match = re.search(r'(?:dr|doctor|physician)[\s\.:]+([a-z\s]+(?:\s+[a-z]+)?)', text.lower())
        if doctor_match:
            doctor_name = doctor_match.group(1).strip().title()
    
    # Extract diagnosis/disease information
    diagnosis = parsed.get("diagnosis") or parsed.get("final diagnosis") or parsed.get("impression") or parsed.get("assessment") or disease_name
    
    # Extract symptoms - try multiple sources
    symptoms = "Not specified"
    symptoms_list = []
    for key in ["symptoms", "chief complaint", "complaints", "presenting complaint", "history of present illness", "hpi"]:
        if key in parsed and parsed[key].strip():
            symptoms = parsed[key].strip()
            # Try to split into list if it's a semicolon or comma separated string
            if ";" in symptoms or "," in symptoms:
                symptoms_list = [s.strip() for s in re.split(r'[;,]', symptoms) if s.strip()]
            else:
                symptoms_list = [symptoms]
            break
    
    # Extract allergies
    allergies = []
    for key in ["allergies", "list any allergies", "known allergies"]:
        if key in parsed and parsed[key].strip():
            allergy_text = parsed[key].strip()
            if ";" in allergy_text or "," in allergy_text:
                allergies = [a.strip() for a in re.split(r'[;,]', allergy_text) if a.strip()]
            else:
                allergies = [allergy_text]
            break
    
    # Extract medication duration
    duration = "Not specified"
    for key in ["duration", "treatment duration", "medication duration", "course duration", "treatment period", "medication interval"]:
        if key in parsed and parsed[key].strip():
            duration = parsed[key].strip()
            break
    
    # Extract medication frequency/dosage if available
    medication_details = []
    for i, med in enumerate(meds[:10] if meds else []):
        med_detail = {"name": med, "frequency": "Not specified", "duration": duration}
        medication_details.append(med_detail)
    
    # Extract visit date/consultation date
    visit_date = date_str
    for key in ["date", "visit date", "consultation date", "report date"]:
        if key in parsed and parsed[key].strip():
            visit_date = parsed[key].strip()
            break
    
    # Extract patient name if available
    patient_name = parsed.get("patient_name") or parsed.get("patient name") or parsed.get("name") or "Not specified"
    
    # Check if this file already exists in history (update instead of duplicate)
    existing_idx = None
    for idx, entry in enumerate(history):
        if entry.get("file_name") == file_name:
            existing_idx = idx
            break
    
    entry = {
        "date": date_str,
        "visit_date": visit_date,
        "file_name": file_name,
        "patient_name": patient_name,
        "disease": disease_name if disease_name != "Unknown Condition" else diagnosis,
        "diagnosis": diagnosis,
        "doctor": doctor_name,
        "medications": meds[:15] if meds else [],
        "medication_details": medication_details,
        "duration": duration,
        "symptoms": symptoms,
        "symptoms_list": symptoms_list if symptoms_list else ([symptoms] if symptoms != "Not specified" else []),
        "allergies": allergies,
        "notes": parsed.get("notes") or parsed.get("remarks") or ""
    }
    
    if existing_idx is not None:
        # Update existing entry
        history[existing_idx] = entry
    else:
        # Insert at the beginning (latest first)
        history.insert(0, entry)
    
    # Keep only last 100 entries
    history = history[:100]
    
    save_disease_history(aadhaar, history)

@st.cache_data
def load_users():
    return pd.read_csv(DATA_PATH, dtype={"aadhaar_id": str, "name": str, "dob": str})


def ensure_user_dir(aadhaar: str) -> str:
    user_dir = os.path.join(STORAGE_ROOT, aadhaar)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def save_uploaded_file(uploaded_file, dest_dir: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{ts}__{uploaded_file.name}"
    filepath = os.path.join(dest_dir, filename)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return filepath


def list_user_files(user_dir: str):
    if not os.path.isdir(user_dir):
        return []
    files = []
    for name in sorted(os.listdir(user_dir)):
        path = os.path.join(user_dir, name)
        if os.path.isfile(path):
            size_kb = os.path.getsize(path) / 1024.0
            files.append({"name": name, "path": path, "size_kb": size_kb})
    return files


def file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@st.cache_data
def extract_text(path: str, key: str) -> str:
    lower = path.lower()
    if lower.endswith(".pdf"):
        text_parts = []
        doc = fitz.open(path)
        for page in doc:
            text_parts.append(page.get_text())
        return "\n".join(text_parts)
    if lower.endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(path)
        return pytesseract.image_to_string(img)
    return ""


def parse_report_text(text: str) -> dict:
    result = {}
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in lines:
        if ":" in l:
            k, _, v = l.partition(":")
            k = k.strip().lower()
            v = v.strip()
            if k and v and k not in result:
                result[k] = v
    patterns = [
        ("date", r"\b(\d{4}-\d{2}-\d{2}|\d{2}[/-]\d{2}[/-]\d{4})\b"),
        ("dob", r"\b(\d{4}-\d{2}-\d{2}|\d{2}[/-]\d{2}[/-]\d{4})\b"),
    ]
    for key, pat in patterns:
        if key not in result:
            m = re.search(pat, text)
            if m:
                result[key] = m.group(1)
    name_keys = ["patient name", "name", "patient"]
    for nk in name_keys:
        if nk in result:
            result["patient_name"] = result[nk]
            break
    return result


def generate_structured_summary(text: str, parsed: dict, file_name: str = "", aadhaar: str = "", language: str = "english") -> tuple:
    """Generate enhanced structured summary with AI-powered disease information.
    Returns (summary_markdown, disease_name, meds_list, short_summary, lab_analysis)"""
    lines = [l.strip() for l in text.splitlines()]
    lower_lines = [l.lower().strip() for l in lines]

    def extract_block(start_keys):
        idx = -1
        for i, ll in enumerate(lower_lines):
            if any(k in ll for k in start_keys):
                idx = i
                break
        if idx == -1:
            return []
        items = []
        header_line = lines[idx].strip()
        if ":" in header_line:
            tail = header_line.split(":", 1)[1].strip()
            if tail:
                items.append(tail)
        for j in range(idx + 1, len(lines)):
            if not lines[j].strip():
                break
            items.append(lines[j].strip())
        cleaned = []
        for it in items:
            t = re.sub(r"\s+", " ", it).strip("-• ")
            if t and t.lower() not in {"yes", "no"}:
                cleaned.append(t)
        return list(dict.fromkeys(cleaned))

    def first_value(keys):
        for k in keys:
            if k in parsed and parsed[k].strip():
                return parsed[k].strip()
        return ""

    diagnosis = first_value(["diagnosis", "final diagnosis", "impression", "assessment", "conclusion", "result", "results"])
    if not diagnosis:
        probs = extract_block(["list any medical problems", "medical problems", "problem list"])
        if probs:
            diagnosis = ", ".join(probs[:5])

    symptom_text = first_value(["symptoms", "chief complaint", "complaints", "presenting complaint", "history of present illness", "hpi"])
    if not symptom_text:
        sym_list = extract_block(["symptom", "symptoms", "chief complaint", "complaints", "presenting complaint", "history of present illness", "hpi"])
        if sym_list:
            symptom_text = "; ".join(sym_list[:6])

    allergies = extract_block(["allergies", "list any allergies"])
    meds = extract_block(["medication taken regularly", "current medications", "medications"]) or extract_block(["list any medication taken regularly"])

    # Extract disease names
    diseases = extract_disease_names(text, parsed)
    primary_disease = diseases[0] if diseases else "Unknown Condition"
    
    # Analyze lab values if no disease is found
    lab_analysis = None
    short_summary = None
    if not diseases or diseases[0] == "Unknown Condition":
        lab_analysis = analyze_lab_values(text)

    # Build enhanced markdown with AI-powered information
    sections = []
    sections.append("## 📋 Summarised Report\n")
    
    if not diseases or diseases[0] == "Unknown Condition":
        if lab_analysis:
            sections.append("### 🔬 Lab Analysis & Abnormal Values")
            abnormal_values = lab_analysis.get("abnormal_values", [])
            if abnormal_values:
                sections.append("**Abnormal Test Results:**")
                for abv in abnormal_values[:10]:
                    status = abv.get("status", "").lower()
                    # Double-check validation
                    value_str = str(abv.get("value", ""))
                    normal_range = abv.get("normal_range", "")
                    validated_status = validate_lab_value(value_str, normal_range)
                    
                    # Only show if actually abnormal
                    if validated_status in ["high", "low"]:
                        status_emoji = "⬆️" if validated_status == "high" else "⬇️"
                        sections.append(f"- {status_emoji} **{abv.get('test_name', 'Test')}**: {abv.get('value', 'N/A')} {abv.get('unit', '')} (Normal: {abv.get('normal_range', 'N/A')}) - **{validated_status.upper()}**")
                sections.append("")
            else:
                sections.append("**All test values are within normal ranges.** ✅\n")
            
            possible_diseases = lab_analysis.get("possible_diseases", [])
            if possible_diseases:
                sections.append("### 🦠 Possible Conditions:")
                for pd in possible_diseases[:8]:
                    sections.append(f"- **{pd.get('disease', 'Unknown')}**: {pd.get('reason', 'Based on abnormal lab values')}")
                sections.append("")
            
            summary_text = lab_analysis.get("summary", "")
            if summary_text:
                sections.append(f"**Summary:** {summary_text}\n")
        else:
            sections.append("### ℹ️ General Health Information")
            if diagnosis:
                sections.append(f"**Diagnosis/Findings:** {diagnosis}\n")
            if symptom_text:
                sections.append(f"**Symptoms:** {symptom_text}\n")
            if meds:
                sections.append(f"**Medications:**")
                for med in meds[:8]:
                    sections.append(f"- {med}")
                sections.append("")
    else:
        # Fetch detailed disease information for each disease
        for disease in diseases[:3]:  # Limit to 3 diseases
            disease_info = fetch_disease_info(disease)
            
            sections.append(f"### 🦠 {disease}")
            
            if disease_info:
                if disease_info.get("information"):
                    sections.append(f"**Information:** {disease_info.get('information')}\n")
                
                if disease_info.get("symptoms"):
                    symptoms_list = disease_info.get("symptoms", [])
                    sections.append(f"**Symptoms:**")
                    for sym in symptoms_list[:8]:
                        sections.append(f"- {sym}")
                    sections.append("")
                
                if disease_info.get("precautions"):
                    precautions_list = disease_info.get("precautions", [])
                    sections.append(f"**Precautions:**")
                    for prec in precautions_list[:8]:
                        sections.append(f"- {prec}")
                    sections.append("")
                
                if disease_info.get("treatment"):
                    sections.append(f"**Treatment/Cure:** {disease_info.get('treatment')}\n")
                
                if disease_info.get("medications"):
                    meds_list = disease_info.get("medications", [])
                    if meds_list:
                        sections.append(f"**Common Medications:**")
                        for med in meds_list[:6]:
                            sections.append(f"- {med}")
                        sections.append("")
                
                if disease_info.get("doctor_type"):
                    sections.append(f"**Consult:** {disease_info.get('doctor_type')}\n")
            else:
                # Fallback if AI fetch fails - show extracted info
                if diagnosis:
                    sections.append(f"**Diagnosis:** {diagnosis}\n")
                if symptom_text:
                    sections.append(f"**Symptoms:** {symptom_text}\n")
                sections.append("**Note:** *To get detailed disease information, get a free Groq API key from https://console.groq.com and set GROQ_API_KEY environment variable.*\n")
            
            sections.append("---\n")

    # Add medications from report (if different from AI-provided ones or if AI info not available)
    if meds:
        sections.append(f"**Medications from Report:**")
        for med in meds[:8]:
            sections.append(f"- {med}")
        sections.append("")
    
    if allergies:
        sections.append(f"**Allergies:**")
        for allergy in allergies[:5]:
            sections.append(f"- {allergy}")
        sections.append("")

    # Generate short summary for all cases
    short_summary = None
    if diseases and diseases[0] != "Unknown Condition":
        disease_info = fetch_disease_info(primary_disease)
        if disease_info:
            symptoms_str = symptom_text or (", ".join(disease_info.get("symptoms", [])[:3]) if disease_info.get("symptoms") else "Not specified")
            doctor_type = disease_info.get("doctor_type", "General Physician")
            short_summary = generate_short_summary(disease_info, symptoms_str, doctor_type, language)
    elif lab_analysis and lab_analysis.get("possible_diseases"):
        # Create short summary from lab analysis
        possible = lab_analysis.get("possible_diseases", [])
        diseases_list = ", ".join([p.get("disease", "") for p in possible[:3]])
        summary_text = lab_analysis.get("summary", "")
        if summary_text:
            short_summary = f"{summary_text} Possible conditions include: {diseases_list}. Please consult a healthcare provider for proper diagnosis."
        else:
            short_summary = f"Lab analysis shows abnormal values indicating possible {diseases_list}. Please consult a healthcare provider for proper diagnosis and treatment."
    elif symptom_text or diagnosis:
        # Generate basic summary from available info
        summary_parts = []
        if diagnosis:
            summary_parts.append(f"Diagnosis: {diagnosis}")
        if symptom_text:
            summary_parts.append(f"Symptoms: {symptom_text[:100]}")
        if summary_parts:
            short_summary = ". ".join(summary_parts) + ". Please consult a healthcare provider."
    
    content = "\n".join(sections)
    return (content if content.strip() else "No key findings detected.", primary_disease, meds, short_summary, lab_analysis)


def login_view():
    st.title("Swasthi Health Locker – MVP")
    st.caption("Login with Aadhaar ID and Date of Birth")

    users = load_users()

    with st.form("login_form", clear_on_submit=False):
        aadhaar = st.text_input("Aadhaar ID (12 digits)", max_chars=12)
        dob_input = st.date_input("Date of Birth", value=date(1990, 1, 1))
        submitted = st.form_submit_button("Login")

    if submitted:
        aadhaar = (aadhaar or "").strip()
        dob_str = dob_input.strftime("%Y-%m-%d")
        if len(aadhaar) != 12 or not aadhaar.isdigit():
            st.error("Enter a valid 12-digit Aadhaar ID.")
            return
        row = users.loc[(users["aadhaar_id"] == aadhaar) & (users["dob"] == dob_str)]
        if row.empty:
            st.error("Invalid credentials. Please check Aadhaar and DOB.")
            return
        user = row.iloc[0].to_dict()
        st.session_state["logged_in"] = True
        st.session_state["user"] = user
        st.success(f"Welcome, {user['name']}!")
        st.rerun()


def header_bar(user):
    cols = st.columns([1, 2, 3, 1])
    with cols[0]:
        st.subheader("🩺 Swasthi")
    with cols[1]:
        st.caption("Digital Health Locker • MVP")
    with cols[2]:
        st.caption(f"Logged in as: {user['name']} ({user['aadhaar_id']})")
    with cols[3]:
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()


def dashboard_view(user):
    header_bar(user)

    user_dir = ensure_user_dir(user["aadhaar_id"])

    tab_upload, tab_camera, tab_records, tab_summary = st.tabs(["Upload Files", "Camera Capture", "My Records", "Summarize Report"])

    with tab_upload:
        st.markdown("#### Upload Lab Reports, Prescriptions, Images or PDFs")
        uploads = st.file_uploader(
            "Select files",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )
        if uploads:
            if "saved_hashes" not in st.session_state:
                st.session_state["saved_hashes"] = set()
            for uf in uploads:
                try:
                    h = hashlib.sha256(uf.getbuffer()).hexdigest()
                    if h in st.session_state["saved_hashes"]:
                        st.info(f"Already uploaded: {uf.name}")
                    else:
                        saved = save_uploaded_file(uf, user_dir)
                        st.session_state["saved_hashes"].add(h)
                        st.success(f"Saved: {os.path.basename(saved)}")
                except Exception:
                    saved = save_uploaded_file(uf, user_dir)
                    st.success(f"Saved: {os.path.basename(saved)}")

    with tab_camera:
        st.markdown("#### Capture a Document using your Camera")
        enable_cam = st.checkbox("Enable camera", value=False, key="enable_camera")
        if enable_cam:
            img_file = st.camera_input("Camera", key="camera_input")
            if img_file is not None:
                if "saved_hashes" not in st.session_state:
                    st.session_state["saved_hashes"] = set()
                try:
                    h = hashlib.sha256(img_file.getbuffer()).hexdigest()
                    if h in st.session_state["saved_hashes"]:
                        st.info("This capture is already saved.")
                    else:
                        saved = save_uploaded_file(img_file, user_dir)
                        st.session_state["saved_hashes"].add(h)
                        st.success(f"Captured and saved: {os.path.basename(saved)}")
                except Exception:
                    saved = save_uploaded_file(img_file, user_dir)
                    st.success(f"Captured and saved: {os.path.basename(saved)}")
                try:
                    image = Image.open(io.BytesIO(img_file.getbuffer()))
                    st.image(image, caption="Captured Preview", use_column_width=True)
                except Exception:
                    pass

    with tab_records:
        st.markdown("#### Your Stored Records")
        files = list_user_files(user_dir)
        if not files:
            st.info("No files yet. Upload or capture to get started.")
        else:
            for f in files:
                with st.container(border=True):
                    # First row: File info and actions
                    col_info, col_actions = st.columns([3, 1])
                    with col_info:
                        st.write(f"**{f['name']}** • {f['size_kb']:.1f} KB")
                    with col_actions:
                        # Delete button with confirmation
                        delete_key = f"delete_file_{f['name']}"
                        if st.button("🗑️ Delete", key=delete_key, type="secondary", use_container_width=True):
                            try:
                                # Delete file from storage
                                if delete_user_file(user_dir, f['name']):
                                    # Also remove from history if it exists
                                    delete_from_history(user["aadhaar_id"], f['name'])
                                    st.success(f"✅ Successfully deleted: {f['name']}")
                                    time.sleep(0.5)  # Brief delay for user to see success message
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed to delete file: {f['name']}")
                            except Exception as e:
                                st.error(f"❌ Error deleting file: {str(e)}")
                    
                    # Second row: Preview and action buttons
                    lower = f["name"].lower()
                    if lower.endswith((".png", ".jpg", ".jpeg")):
                        # Image preview - small size
                        col_preview, col_buttons = st.columns([2, 1])
                        with col_preview:
                            try:
                                st.image(f["path"], width=300)
                            except Exception:
                                pass
                        with col_buttons:
                            st.write("")  # Spacing
                            st.write("")  # Spacing
                            # Open button - show in expandable modal
                            with st.expander("🔓 Open Full View", expanded=False):
                                try:
                                    st.image(f["path"], use_column_width=True)
                                except Exception as e:
                                    st.error(f"Error loading image: {e}")
                            with open(f["path"], "rb") as filedata:
                                st.download_button(
                                    label="📥 Download",
                                    data=filedata,
                                    file_name=f["name"],
                                    mime="application/octet-stream",
                                    use_container_width=True,
                                    key=f"download_{f['name']}"
                                )
                    
                    elif lower.endswith(".pdf"):
                        # PDF preview - show first page as thumbnail
                        col_preview, col_buttons = st.columns([2, 1])
                        with col_preview:
                            try:
                                doc = fitz.open(f["path"])
                                page = doc[0]
                                # Convert to image with a reasonable zoom for thumbnail
                                mat = fitz.Matrix(2, 2)  # Zoom factor
                                pix = page.get_pixmap(matrix=mat)
                                img_bytes = pix.tobytes("png")
                                st.image(img_bytes, width=300, caption="First page preview")
                                doc.close()
                            except Exception:
                                st.info("📄 PDF file")
                        with col_buttons:
                            st.write("")  # Spacing
                            st.write("")  # Spacing
                            # Open button - show all PDF pages in expander
                            with st.expander("🔓 Open Full PDF", expanded=False):
                                try:
                                    doc = fitz.open(f["path"])
                                    for page_num in range(len(doc)):
                                        page = doc[page_num]
                                        mat = fitz.Matrix(3, 3)  # Higher resolution for full view
                                        pix = page.get_pixmap(matrix=mat)
                                        img_bytes = pix.tobytes("png")
                                        st.image(img_bytes, caption=f"Page {page_num + 1} of {len(doc)}", use_column_width=True)
                                    doc.close()
                                except Exception as e:
                                    st.error(f"Error loading PDF: {e}")
                            with open(f["path"], "rb") as filedata:
                                st.download_button(
                                    label="📥 Download",
                                    data=filedata,
                                    file_name=f["name"],
                                    mime="application/pdf",
                                    use_container_width=True,
                                    key=f"download_{f['name']}"
                                )

    with tab_summary:
        summary_subtab1, summary_subtab2 = st.tabs(["📄 Report Summaries", "📊 Overall Summary"])
        
        with summary_subtab1:
            st.markdown("#### Summarize Reports from PDFs and Images")
            # Check if Groq API key is configured
            api_key_available = get_groq_client() is not None
            if api_key_available:
                st.success("✅ Groq API key configured - AI disease information enabled!")
                
                # Add a button to test API and clear cache
                col_test, col_cache = st.columns(2)
                with col_test:
                    if st.button("🧪 Test API Connection"):
                        with st.spinner("Testing Groq API..."):
                            test_result = fetch_disease_info("Asthma")
                            if test_result:
                                st.success(f"✅ API working! Got info for: {test_result.get('name', 'Unknown')}")
                            else:
                                st.error("❌ API call failed. Check your API key and network connection.")
                with col_cache:
                    if st.button("🔄 Clear Cache"):
                        # Clear the cache for fetch_disease_info
                        fetch_disease_info.clear()
                        st.success("Cache cleared! Please reprocess your files.")
                        st.rerun()
            else:
                # Try to help debug
                env_key = os.getenv("GROQ_API_KEY")
                try:
                    secrets_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, 'secrets') else ""
                except:
                    secrets_key = ""
                
                if not env_key and not secrets_key:
                    st.warning("⚠️ Groq API key not found. Please restart Streamlit after adding the key to `.streamlit/secrets.toml` or set GROQ_API_KEY environment variable.")
                elif secrets_key:
                    st.warning("⚠️ Secrets file found but API key not loaded. Please **restart Streamlit** for changes to take effect.")
            files = [f for f in list_user_files(user_dir) if f["name"].lower().endswith((".pdf", ".png", ".jpg", ".jpeg"))]
            if not files:
                st.info("No PDF or image records found.")
            else:
                display_names = [f["name"] for f in files]
                selection = st.multiselect("Select files to summarize", display_names, default=[])
                to_process = [f for f in files if f["name"] in selection]
                # Language selection
                languages = ["English", "Hindi", "Marathi", "Telugu", "Tamil", "Kannada"]
                selected_lang = st.selectbox("🌐 Select Language for Summary & Audio", languages, index=0, key="summary_language")
                
                # Initialize processed files in session state
                processed_files_key = "processed_files_summary"
                if processed_files_key not in st.session_state:
                    st.session_state[processed_files_key] = {}
                
                # Process and Clear buttons
                col_process, col_clear = st.columns([1, 1])
                with col_process:
                    if st.button("Process Selected", use_container_width=True):
                        # Clear previous summaries before processing new ones
                        st.session_state[processed_files_key] = {}
                        
                        for f in to_process:
                            try:
                                h = file_sha256(f["path"])
                                text = extract_text(f["path"], h)
                                summary = parse_report_text(text)
                                summary["text"] = text
                                summary_md, disease_name, meds, short_summary, lab_analysis = generate_structured_summary(text, summary, f["name"], user["aadhaar_id"], selected_lang)
                                add_to_history(user["aadhaar_id"], f["name"], disease_name, summary, meds, text)
                                
                                # Store processed results in session state
                                st.session_state[processed_files_key][f["name"]] = {
                                    "file_name": f["name"],
                                    "summary_md": summary_md,
                                    "disease_name": disease_name,
                                    "short_summary": short_summary,
                                    "text": text,
                                    "selected_lang": selected_lang
                                }
                            except Exception as e:
                                st.error(f"Failed to process {f['name']}: {e}")
                
                with col_clear:
                    if st.button("Clear Summary", use_container_width=True):
                        st.session_state[processed_files_key] = {}
                        st.rerun()
                
                # Display processed files (persists across reruns)
                for file_name, file_data in st.session_state[processed_files_key].items():
                    if file_name in [f["name"] for f in files]:  # Only show if file still exists
                        f = next(f for f in files if f["name"] == file_name)
                        summary_md = file_data["summary_md"]
                        short_summary = file_data["short_summary"]
                        text = file_data["text"]
                        current_lang = file_data.get("selected_lang", selected_lang)
                        
                        with st.container(border=True):
                            st.subheader(file_name)
                            
                            cols = st.columns([1, 1])
                            
                            with cols[0]:
                                if short_summary:
                                    st.markdown("#### 📝 Quick Summary (2-5 lines)")
                                    st.info(short_summary)
                                    
                                    # TTS section
                                    col_tts, col_lang = st.columns([2, 1])
                                    with col_tts:
                                        tts_key = f"tts_{file_name}"
                                        audio_key = f"audio_{file_name}"
                                        
                                        # Generate audio on demand
                                        if st.button("🔊 Speak Summary", key=tts_key, use_container_width=True):
                                            with st.spinner("Generating audio..."):
                                                audio_data = text_to_speech(short_summary, current_lang, speed=1.0)
                                                if audio_data:
                                                    st.session_state[audio_key] = audio_data
                                                    st.rerun()
                                                else:
                                                    st.error("Failed to generate audio.")
                                        
                                        # Display audio if available
                                        if audio_key in st.session_state:
                                            audio_bytes = st.session_state[audio_key]
                                            # Use HTML audio player with playbackRate for speed control
                                            player_id = f"audio_player_{file_name.replace('.', '_').replace(' ', '_')}"
                                            audio_html = create_audio_player_html(audio_bytes, playback_rate=1.0, player_id=player_id)
                                            st.markdown(audio_html, unsafe_allow_html=True)
                                    
                                    with col_lang:
                                        st.caption(f"🌐 {current_lang}")
                                    
                                    st.markdown("---")
                                
                                # Full Summary - translated to selected language
                                if summary_md and summary_md.strip():
                                    st.markdown("#### 📋 Detailed Summary")
                                    # Translate the detailed summary to selected language
                                    if current_lang.lower() != "english":
                                        detailed_summary_display = translate_text(summary_md, current_lang)
                                        st.markdown(detailed_summary_display)
                                    else:
                                        st.markdown(summary_md)
                                else:
                                    st.info("No key findings detected.")
                            
                            with cols[1]:
                                st.markdown("##### Extracted Text")
                                st.text_area("", text if text else "(empty)", height=400, key=f"text_{file_name}", disabled=True)
                        st.markdown("")  # Spacing
        
        with summary_subtab2:
            st.markdown("## 📊 Overall Summary")
            st.markdown("#### Complete Medical History - Document by Document")
            
            history = load_disease_history(user["aadhaar_id"])
            
            if not history:
                st.info("No medical history found. Upload and process reports to build your medical history.")
            else:
                st.markdown(f"### Total Documents Processed: **{len(history)}**")
                st.markdown("---")
                
                # Generate AI-powered overall summary
                overall_summary_key = "overall_summary_cache"
                if overall_summary_key not in st.session_state:
                    st.session_state[overall_summary_key] = None
                
                with st.expander("📊 AI-Powered Overall Medical Summary", expanded=True):
                    if st.button("🔄 Generate Summary", key="generate_overall_summary"):
                        with st.spinner("Analyzing medical history..."):
                            overall_summary = generate_overall_medical_summary(history)
                            st.session_state[overall_summary_key] = overall_summary
                    
                    if st.session_state[overall_summary_key]:
                        st.markdown(st.session_state[overall_summary_key])
                    else:
                        st.info("Click 'Generate Summary' to get an AI-powered analysis of your medical history")
                
                st.markdown("---")
                
                # Display each document as a separate section
                for idx, entry in enumerate(history, 1):
                    with st.container(border=True):
                        # Header section with delete button
                        col_title, col_delete = st.columns([4, 1])
                        with col_title:
                            st.markdown(f"### 📄 Document #{idx}: {entry.get('file_name', 'Unknown File')}")
                        with col_delete:
                            st.markdown("<br>", unsafe_allow_html=True)
                            file_name_for_delete = entry.get('file_name', 'unknown')
                            # Create a safe key by replacing special characters
                            safe_key = re.sub(r'[^a-zA-Z0-9_]', '_', file_name_for_delete)
                            delete_summary_key = f"delete_summary_{safe_key}_{idx}"
                            if st.button("🗑️ Delete Summary", key=delete_summary_key, type="secondary", use_container_width=True):
                                file_name = entry.get('file_name', '')
                                if file_name:
                                    try:
                                        delete_from_history(user["aadhaar_id"], file_name)
                                        st.success(f"✅ Successfully deleted summary for: {file_name}")
                                        time.sleep(0.5)  # Brief delay for user to see success message
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ Error deleting summary: {str(e)}")
                                else:
                                    st.error("❌ Cannot delete: File name not found")
                        
                        # Date and basic info row
                        col_header1, col_header2 = st.columns([2, 1])
                        with col_header1:
                            st.markdown(f"**📅 Date:** {entry.get('visit_date', entry.get('date', 'Not specified'))}")
                            patient_name = entry.get('patient_name', 'Not specified')
                            if patient_name != "Not specified":
                                st.markdown(f"**👤 Patient:** {patient_name}")
                        with col_header2:
                            st.caption(f"Record #{idx}")
                        
                        st.markdown("---")
                        
                        # Disease/Diagnosis section
                        disease = entry.get('disease', entry.get('diagnosis', 'Not specified'))
                        diagnosis = entry.get('diagnosis', disease)
                        if disease != "Not specified" or diagnosis != "Not specified":
                            st.markdown("#### 🦠 Condition / Diagnosis")
                            if disease == diagnosis:
                                st.markdown(f"**{disease}**")
                            else:
                                st.markdown(f"**Primary Disease:** {disease}")
                                if diagnosis != disease:
                                    st.markdown(f"**Full Diagnosis:** {diagnosis}")
                        
                        # Symptoms section
                        symptoms_list = entry.get('symptoms_list', [])
                        symptoms_text = entry.get('symptoms', 'Not specified')
                        if symptoms_list or (symptoms_text and symptoms_text != "Not specified"):
                            st.markdown("#### 🔍 Symptoms")
                            if symptoms_list:
                                for symptom in symptoms_list[:10]:
                                    st.markdown(f"- {symptom}")
                            elif symptoms_text and symptoms_text != "Not specified":
                                st.markdown(f"{symptoms_text}")
                        
                        # Doctor and Duration row
                        col_doc, col_dur = st.columns(2)
                        with col_doc:
                            doctor = entry.get('doctor', 'Not specified')
                            st.markdown(f"#### 👨‍⚕️ Consulted Doctor")
                            st.markdown(f"**{doctor}**")
                        with col_dur:
                            duration = entry.get('duration', 'Not specified')
                            st.markdown(f"#### ⏱️ Treatment Duration")
                            st.markdown(f"**{duration}**")
                        
                        # Medications section
                        medications = entry.get('medications', [])
                        medication_details = entry.get('medication_details', [])
                        if medications:
                            st.markdown("#### 💊 Medications Taken")
                            if medication_details and len(medication_details) > 0:
                                # Display with details if available
                                for med_detail in medication_details[:10]:
                                    med_name = med_detail.get('name', med_detail) if isinstance(med_detail, dict) else med_detail
                                    med_freq = med_detail.get('frequency', '') if isinstance(med_detail, dict) else ''
                                    med_dur = med_detail.get('duration', '') if isinstance(med_detail, dict) else ''
                                    if med_freq and med_freq != "Not specified":
                                        st.markdown(f"- **{med_name}** - Frequency: {med_freq}" + (f", Duration: {med_dur}" if med_dur and med_dur != "Not specified" else ""))
                                    else:
                                        st.markdown(f"- **{med_name}**")
                            else:
                                # Simple list
                                med_cols = st.columns(min(len(medications), 3))
                                for i, med in enumerate(medications[:12]):
                                    with med_cols[i % len(med_cols)]:
                                        st.markdown(f"- {med}")
                        
                        # Allergies section
                        allergies = entry.get('allergies', [])
                        if allergies:
                            st.markdown("#### ⚠️ Allergies")
                            for allergy in allergies[:5]:
                                st.markdown(f"- {allergy}")
                        
                        # Notes section
                        notes = entry.get('notes', '')
                        if notes and notes.strip():
                            st.markdown("#### 📝 Additional Notes")
                            st.markdown(f"{notes}")
                        
                        # Footer with file info
                        st.markdown("---")
                        st.caption(f"📎 Source File: {entry.get('file_name', 'N/A')} | Processed on: {entry.get('date', 'N/A')}")
                        
                        if idx < len(history):
                            st.markdown("")  # Add spacing between documents


if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "saved_hashes" not in st.session_state:
    st.session_state["saved_hashes"] = set()

if not st.session_state["logged_in"]:
    login_view()
else:
    dashboard_view(st.session_state["user"]) 
