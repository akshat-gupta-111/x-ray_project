"""
Serimport os
import re
import json
import io
import base64
import cv2
import numpy as np
from typing import Any, Dict, List, Optional
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import google.generativeai as genai
from PIL import Image
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()AI App for Vercel deployment
- In-memory image processing (no file persistence)
- Optimized for serverless cold starts
- Real-time inference without file uploads
"""

import os
import re
import json
import io
import base64
import cv2
import numpy as np
from typing import Any, Dict, List, Optional
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import google.generativeai as genai
from PIL import Image
import tempfile

# ---------------- Configuration ---------------- #
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
FRACTURE_MODEL_PATH = os.environ.get("FRACTURE_MODEL_PATH", "best_fracture_yolov8.pt")
PNEUMONIA_CLASSIFIER_PATH = os.environ.get("PNEUMONIA_CLASSIFIER_PATH", "best_classifier.pt")
PNEUMONIA_DET_MODEL_PATH = os.environ.get("PNEUMONIA_DET_MODEL_PATH", "best_detection.pt")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ENABLE_VISION_IN_PROMPT = os.environ.get("ENABLE_VISION_IN_PROMPT", "true").lower() == "true"
LOG_LEVEL = os.environ.get("APP_LOG_LEVEL", "INFO").upper()

app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')
app.secret_key = os.environ.get('SECRET_KEY', 'supersecret-change-in-production')

# Global model cache
_models_cache = {}

def log(msg, level="INFO"):
    levels = ["DEBUG", "INFO", "WARN", "ERROR"]
    if levels.index(level) >= levels.index(LOG_LEVEL):
        print(f"[{level}] {msg}")

# ---------------- Model Loading with Caching ---------------- #
def get_model(model_type: str) -> Optional[YOLO]:
    """Load models with caching for serverless efficiency"""
    if model_type in _models_cache:
        return _models_cache[model_type]
    
    model_paths = {
        "fracture": FRACTURE_MODEL_PATH,
        "pneumonia_cls": PNEUMONIA_CLASSIFIER_PATH,
        "pneumonia_det": PNEUMONIA_DET_MODEL_PATH
    }
    
    path = model_paths.get(model_type)
    if not path or not os.path.exists(path):
        log(f"Model {model_type} not found at {path}", "WARN")
        _models_cache[model_type] = None
        return None
    
    try:
        model = YOLO(path)
        _models_cache[model_type] = model
        log(f"Loaded {model_type} model: {path}", "INFO")
        return model
    except Exception as e:
        log(f"Failed loading {model_type} model: {e}", "ERROR")
        _models_cache[model_type] = None
        return None

# ---------------- Gemini Setup ---------------- #
def get_gemini_client():
    """Initialize Gemini client"""
    if not GEMINI_API_KEY:
        log("GEMINI_API_KEY not set", "WARN")
        return None
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        client = genai.GenerativeModel("gemini-2.0-flash-exp")
        log("Gemini initialized.", "INFO")
        return client
    except Exception as e:
        log(f"Gemini init failed: {e}", "ERROR")
        return None

# ---------------- Helper Functions ---------------- #
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_bytes(image_array: np.ndarray) -> bytes:
    """Convert numpy array to bytes"""
    _, buffer = cv2.imencode('.jpg', image_array)
    return buffer.tobytes()

def bytes_to_image(image_bytes: bytes) -> np.ndarray:
    """Convert bytes to numpy array"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# ---------------- Detection Functions ---------------- #
def detect_fractures_memory(image_array: np.ndarray):
    """Detect fractures in memory without file I/O"""
    model = get_model("fracture")
    if model is None:
        raise RuntimeError("Fracture model not loaded.")
    
    results = model(image_array)
    annotated_img = image_array.copy()
    detections = []
    
    if len(results[0].boxes) == 0:
        return annotated_img, detections
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names.get(cls, f"class_{cls}")
        
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (220, 20, 60), 2)
        cv2.putText(annotated_img, f"{label} {conf:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 20, 60), 1)
        
        detections.append({
            "label": label,
            "confidence": conf,
            "box": [x1, y1, x2, y2],
            "area": (x2 - x1) * (y2 - y1)
        })
    
    return annotated_img, detections

def classify_pneumonia_memory(image_array: np.ndarray):
    """Classify pneumonia in memory"""
    model = get_model("pneumonia_cls")
    if model is None:
        raise RuntimeError("Pneumonia classification model not loaded.")
    
    results = model.predict(image_array, verbose=False)
    res = results[0]
    
    if not hasattr(res, "probs"):
        raise RuntimeError("No .probs on classification result.")
    
    pred_idx = int(res.probs.top1)
    conf = float(res.probs.top1conf)
    names = getattr(res, "names", getattr(model, "names", {}))
    label = names.get(pred_idx, f"class_{pred_idx}")
    probs_list = res.probs.data.tolist() if hasattr(res.probs, "data") else []
    class_names = [names[i] for i in range(len(probs_list))] if probs_list else []
    
    return label, conf, probs_list, class_names

def detect_pneumonia_regions_memory(image_array: np.ndarray, classification_label: str):
    """Detect pneumonia regions in memory"""
    if classification_label.lower() != "pneumonia":
        return image_array, []
    
    model = get_model("pneumonia_det")
    if model is None:
        return image_array, []
    
    results = model(image_array)
    annotated_img = image_array.copy()
    regions = []
    
    if len(results[0].boxes) == 0:
        return annotated_img, regions
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        raw_label = model.names.get(cls, f"class_{cls}")
        
        if raw_label.lower() == "normal":
            continue
        
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 140, 0), 2)
        cv2.putText(annotated_img, f"{raw_label} {conf:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 1)
        
        regions.append({
            "label": raw_label,
            "confidence": conf,
            "box": [x1, y1, x2, y2],
            "area": (x2 - x1) * (y2 - y1)
        })
    
    return annotated_img if regions else image_array, regions

# ---------------- Visual Inspection ---------------- #
def gemini_visual_inspect(image_bytes: bytes) -> Dict[str, Any]:
    """Visual inspection using Gemini"""
    client = get_gemini_client()
    if client is None:
        return {
            "raw_description": "Gemini unavailable.",
            "detected_type": "uncertain",
            "confidence_hint": 0.0,
            "rationale": "Skipped due to missing Gemini client."
        }
    
    try:
        prompt = (
            "Describe this image briefly (<=20 words). Then state if it is a medical radiographic X-ray (Yes/No) "
            "and your confidence 0-1. Format strictly as JSON with keys: description, is_xray (Yes|No|Uncertain), confidence."
        )
        
        response = client.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": image_bytes}
        ])
        
        text = (response.text or "").strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("No JSON in visual inspection response.")
        
        vi = json.loads(match.group(0))
        is_xray_field = vi.get("is_xray", "").lower()
        
        if is_xray_field.startswith("y"):
            dtype = "xray"
        elif is_xray_field.startswith("n"):
            dtype = "non_xray"
        else:
            dtype = "uncertain"
        
        return {
            "raw_description": vi.get("description", "").strip(),
            "detected_type": dtype,
            "confidence_hint": float(vi.get("confidence", 0.0)),
            "rationale": f"Model said is_xray={vi.get('is_xray')} conf={vi.get('confidence')}"
        }
    except Exception as e:
        log(f"Visual inspection failed: {e}", "WARN")
        return {
            "raw_description": "Inspection error.",
            "detected_type": "uncertain",
            "confidence_hint": 0.0,
            "rationale": f"Error: {e}"
        }

# ---------------- Gemini Structured Explanation ---------------- #
GEMINI_SCHEMA_KEYS = [
    "summary", "risk_level", "reasoning", "detection_analysis",
    "recommendations", "follow_up", "disclaimer"
]

def default_explanation(task: str) -> Dict[str, Any]:
    return {
        "summary": f"{task.replace('_', ' ').title()} analysis completed.",
        "risk_level": "unknown",
        "reasoning": "Detailed reasoning unavailable.",
        "detection_analysis": "No detailed region analysis.",
        "recommendations": "Consult a qualified medical professional.",
        "follow_up": "Monitor symptoms; seek care if they worsen.",
        "disclaimer": "AI-generated assistance. Not a diagnosis."
    }

def extract_json_from_text(text: str) -> str:
    text = text.strip()
    if text.startswith('{') and text.endswith('}'):
        return text
    fenced = re.search(r"\{[\s\S]*\}", text)
    if fenced:
        return fenced.group(0)
    raise ValueError("No JSON object found in Gemini response.")

def normalize_explanation(raw: Dict[str, Any]) -> Dict[str, Any]:
    norm = {}
    for k in GEMINI_SCHEMA_KEYS:
        val = raw.get(k)
        norm[k] = val if isinstance(val, str) and val.strip() else default_explanation("ai").get(k)
    return norm

def gemini_structured_advice(task: str, context: Dict[str, Any], image_bytes: bytes, visual_desc: str) -> Dict[str, Any]:
    """Get structured advice from Gemini"""
    client = get_gemini_client()
    if client is None:
        return default_explanation(task)
    
    context_lines = []
    if task == "fracture_detection":
        context_lines.append(f"num_detections={context.get('num_detections')}")
        for d in context.get("detections", []):
            context_lines.append(f"{d['label']} conf={d['confidence']:.2f} box={d['box']}")
    elif task == "pneumonia_analysis":
        cls = context.get("classification", {})
        context_lines.append(f"class_label={cls.get('label')} conf={cls.get('confidence')}")
        context_lines.append(f"regions_found={context.get('regions_found')}")
        for r in context.get("regional_detections", []):
            context_lines.append(f"region {r['label']} conf={r['confidence']:.2f} box={r['box']}")
    
    context_lines.append(f"visual_description={visual_desc[:120]}")
    
    schema_example = {
        "summary": "One or two sentence plain-language overview.",
        "risk_level": "none|low|moderate|high",
        "reasoning": "Brief rationale referencing key model signals.",
        "detection_analysis": "Interpretation of boxes (or 'None').",
        "recommendations": "2-4 short actions separated by ';'.",
        "follow_up": "One sentence follow-up.",
        "disclaimer": "Single sentence disclaimer (not diagnostic)."
    }
    
    prompt = (
        "You are a medical imaging assistant. Produce ONLY valid minified JSON with keys:\n"
        f"{list(schema_example.keys())}\n"
        "No extra keys, no markdown. \n"
        f"Task: {task}\n"
        "Context:\n" + "\n".join(context_lines) + "\n"
        "Rules:\n"
        "- summary: <= 26 words.\n"
        "- risk_level: choose one of none, low, moderate, high.\n"
        "- reasoning: <= 45 words referencing signals.\n"
        "- detection_analysis: <= 50 words or 'None'.\n"
        "- recommendations: 2-4 clauses separated by ';'.\n"
        "- follow_up: single sentence.\n"
        "- disclaimer: single sentence (not diagnostic).\n"
        "Return JSON only."
    )
    
    parts = [prompt]
    if ENABLE_VISION_IN_PROMPT:
        parts.append({"mime_type": "image/jpeg", "data": image_bytes})
    
    try:
        response = client.generate_content(parts)
        raw_text = (response.text or "").strip()
        json_str = extract_json_from_text(raw_text)
        parsed = json.loads(json_str)
        return normalize_explanation(parsed)
    except Exception as e:
        log(f"Gemini structured parse failed: {e}", "WARN")
        return default_explanation(task)

# ---------------- Severity Functions ---------------- #
def fracture_severity(detections):
    max_conf = max([d['confidence'] for d in detections], default=0.0)
    if max_conf >= 0.85: return "high"
    if max_conf >= 0.7: return "moderate"
    if max_conf > 0: return "low"
    return "none"

def pneumonia_severity(label: str, conf: float):
    if label.lower() != "pneumonia": return "none"
    if conf >= 0.9: return "high"
    if conf >= 0.75: return "moderate"
    return "low"

# ---------------- Result Builders ---------------- #
def build_fracture_result(detections, explanation, visual, orig_img_b64, annotated_img_b64):
    return {
        "task": "fracture_detection",
        "num_detections": len(detections),
        "detections": detections,
        "max_confidence": max([d['confidence'] for d in detections], default=0.0),
        "severity": fracture_severity(detections),
        "ai_explanation": explanation,
        "visual_inspection": visual,
        "original_image": orig_img_b64,
        "annotated_image": annotated_img_b64 if annotated_img_b64 != orig_img_b64 else None
    }

def build_pneumonia_result(label, conf, probs, class_names, regions, explanation, visual, orig_img_b64, annotated_img_b64):
    return {
        "task": "pneumonia_analysis",
        "classification": {
            "label": label,
            "confidence": conf,
            "probabilities": probs,
            "class_names": class_names
        },
        "regional_detections": regions,
        "regions_found": len(regions),
        "region_image": len(regions) > 0,
        "severity": pneumonia_severity(label, conf),
        "ai_explanation": explanation,
        "visual_inspection": visual,
        "original_image": orig_img_b64,
        "annotated_image": annotated_img_b64 if (len(regions) > 0 and annotated_img_b64 != orig_img_b64) else None
    }

def build_rejected_result(visual, task_requested):
    return {
        "task": "rejected_non_xray",
        "requested_task": task_requested,
        "severity": "none",
        "ai_explanation": {
            "summary": "Uploaded image does not appear to be a medical X-ray.",
            "risk_level": "none",
            "reasoning": visual.get("rationale", "Non-X-ray content"),
            "detection_analysis": "Medical analysis skipped.",
            "recommendations": "Please upload a chest radiograph (DICOM or standard X-ray image).",
            "follow_up": "Re-capture using proper imaging equipment if needed.",
            "disclaimer": "Safety filter: not a diagnosis."
        },
        "visual_inspection": visual,
        "original_image": None,
        "annotated_image": None
    }

# ---------------- Routes ---------------- #
@app.route("/", methods=["GET"])
def index():
    """Landing page"""
    return render_template("create.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    """Main analysis endpoint"""
    try:
        # Get task mode
        task_mode = request.form.get("task_mode", "fracture")
        
        # Get uploaded file
        file = request.files.get("file")
        if not file or not allowed_file(file.filename):
            return jsonify({"error": "Please upload a valid PNG/JPG image."}), 400
        
        # Read image into memory
        image_bytes = file.read()
        image_array = bytes_to_image(image_bytes)
        
        if image_array is None:
            return jsonify({"error": "Invalid image format."}), 400
        
        # Convert to base64 for frontend
        orig_img_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Visual inspection
        visual = gemini_visual_inspect(image_bytes)
        
        # Check if it's a valid X-ray
        if visual["detected_type"] == "non_xray":
            result = build_rejected_result(visual, task_mode)
            return jsonify(result)
        
        # Process based on task mode
        if task_mode == "fracture":
            if get_model("fracture") is None:
                return jsonify({"error": "Fracture model unavailable."}), 500
            
            annotated_img, detections = detect_fractures_memory(image_array)
            annotated_img_bytes = image_to_bytes(annotated_img)
            annotated_img_b64 = base64.b64encode(annotated_img_bytes).decode('utf-8')
            
            provisional = {
                "num_detections": len(detections),
                "detections": detections
            }
            
            explanation = gemini_structured_advice("fracture_detection", provisional, image_bytes, visual.get("raw_description", ""))
            result = build_fracture_result(detections, explanation, visual, orig_img_b64, annotated_img_b64)
            
        elif task_mode == "pneumonia":
            if get_model("pneumonia_cls") is None:
                return jsonify({"error": "Pneumonia classification model unavailable."}), 500
            
            label, conf, probs, class_names = classify_pneumonia_memory(image_array)
            region_img, regions = detect_pneumonia_regions_memory(image_array, label)
            
            if len(regions) > 0:
                region_img_bytes = image_to_bytes(region_img)
                annotated_img_b64 = base64.b64encode(region_img_bytes).decode('utf-8')
            else:
                annotated_img_b64 = orig_img_b64
            
            provisional = {
                "classification": {"label": label, "confidence": conf},
                "regional_detections": regions,
                "regions_found": len(regions)
            }
            
            explanation = gemini_structured_advice("pneumonia_analysis", provisional, image_bytes, visual.get("raw_description", ""))
            result = build_pneumonia_result(label, conf, probs, class_names, regions, explanation, visual, orig_img_b64, annotated_img_b64)
            
        else:
            return jsonify({"error": "Unknown task mode."}), 400
        
        return jsonify(result)
        
    except Exception as e:
        log(f"Analysis error: {e}", "ERROR")
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models": {
            "fracture": get_model("fracture") is not None,
            "pneumonia_cls": get_model("pneumonia_cls") is not None,
            "pneumonia_det": get_model("pneumonia_det") is not None
        },
        "gemini": get_gemini_client() is not None
    })

# For Vercel serverless
def app_handler(environ, start_response):
    return app(environ, start_response)

# Vercel expects a handler function
handler = app

if __name__ == "__main__":
    app.run(debug=True)
