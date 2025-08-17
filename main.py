"""
main.py - X-ray AI App with Structured Gemini Explanations + Visual Inspection

Enhancements in this revision:
- Reintroduce multimodal (image + text) Gemini usage so the model can recognize non-X-ray images (e.g., code screenshots).
- New gemini_visual_inspect() obtains a concise description + X-ray likelihood classification.
- Visual inspection result included in structured JSON under 'visual_inspection'.
- Optional: feed the raw image again into the structured explanation prompt (controlled by ENABLE_VISION_IN_PROMPT env).
- If visual inspection says 'non_xray', we short-circuit with a rejection response (unless you want to override).
"""

import os
import re
import json
import uuid
import cv2
from typing import Any, Dict, List
from flask import (
    Flask, request, render_template, redirect,
    url_for, send_from_directory, flash, jsonify
)
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import google.generativeai as genai

# ---------------- Configuration ---------------- #
UPLOAD_FOLDER = 'user_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

FRACTURE_MODEL_PATH = os.environ.get("FRACTURE_MODEL_PATH", "best_fracture_yolov8.pt")
PNEUMONIA_CLASSIFIER_PATH = os.environ.get("PNEUMONIA_CLASSIFIER_PATH", "best_classifier.pt")
PNEUMONIA_DET_MODEL_PATH = os.environ.get("PNEUMONIA_DET_MODEL_PATH", "best_detection.pt")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "key")
ENABLE_VISION_IN_PROMPT = os.environ.get("ENABLE_VISION_IN_PROMPT", "true").lower() == "true"

LOG_LEVEL = os.environ.get("APP_LOG_LEVEL", "INFO").upper()

app = Flask(__name__)
app.secret_key = 'supersecret'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def log(msg, level="INFO"):
    levels = ["DEBUG", "INFO", "WARN", "ERROR"]
    if levels.index(level) >= levels.index(LOG_LEVEL):
        print(f"[{level}] {msg}")

# ---------------- Model Loading ---------------- #
def load_yolo_model(path: str, label: str):
    if os.path.exists(path):
        try:
            m = YOLO(path)
            log(f"Loaded {label} model: {path}", "INFO")
            return m
        except Exception as e:
            log(f"Failed loading {label} model: {e}", "ERROR")
    else:
        log(f"Missing {label} model: {path}", "WARN")
    return None

fracture_model = load_yolo_model(FRACTURE_MODEL_PATH, "fracture")
pneumonia_cls_model = load_yolo_model(PNEUMONIA_CLASSIFIER_PATH, "pneumonia_classification")
pneumonia_det_model = load_yolo_model(PNEUMONIA_DET_MODEL_PATH, "pneumonia_detection")

# ---------------- Gemini Setup ---------------- #
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_client = genai.GenerativeModel("gemini-2.5-flash")
    log("Gemini initialized.", "INFO")
except Exception as e:
    gemini_client = None
    log(f"Gemini init failed: {e}", "ERROR")

# ---------------- Helpers ---------------- #
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- Detection / Classification ---------------- #
def detect_fractures(image_path: str):
    if fracture_model is None:
        raise RuntimeError("Fracture model not loaded.")
    results = fracture_model(image_path)
    img = cv2.imread(image_path)
    dets = []
    if len(results[0].boxes) == 0:
        out = image_path.replace('.', '_fracture.', 1)
        cv2.imwrite(out, img)
        return out, dets
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0]); cls = int(box.cls[0])
        label = fracture_model.names.get(cls, f"class_{cls}")
        cv2.rectangle(img, (x1, y1), (x2, y2), (220, 20, 60), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 20, 60), 1)
        dets.append({
            "label": label,
            "confidence": conf,
            "box": [x1, y1, x2, y2],
            "area": (x2 - x1) * (y2 - y1)
        })
    out = image_path.replace('.', '_fracture.', 1)
    cv2.imwrite(out, img)
    return out, dets

def classify_pneumonia(image_path: str):
    if pneumonia_cls_model is None:
        raise RuntimeError("Pneumonia classification model not loaded.")
    results = pneumonia_cls_model.predict(image_path, verbose=False)
    res = results[0]
    if not hasattr(res, "probs"):
        raise RuntimeError("No .probs on classification result.")
    pred_idx = int(res.probs.top1)
    conf = float(res.probs.top1conf)
    names = getattr(res, "names", getattr(pneumonia_cls_model, "names", {}))
    label = names.get(pred_idx, f"class_{pred_idx}")
    probs_list = res.probs.data.tolist() if hasattr(res.probs, "data") else []
    class_names = [names[i] for i in range(len(probs_list))] if probs_list else []
    return label, conf, probs_list, class_names

def detect_pneumonia_regions(image_path: str, classification_label: str):
    # Only run detection for pneumonia classification
    if classification_label.lower() != "pneumonia":
        return image_path, []
    if pneumonia_det_model is None:
        return image_path, []
    results = pneumonia_det_model(image_path)
    img = cv2.imread(image_path)
    regions = []
    if len(results[0].boxes) == 0:
        return image_path, regions
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0]); cls = int(box.cls[0])
        raw_label = pneumonia_det_model.names.get(cls, f"class_{cls}")
        if raw_label.lower() == "normal":  # skip any 'normal' boxes
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 140, 0), 2)
        cv2.putText(img, f"{raw_label} {conf:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 1)
        regions.append({
            "label": raw_label,
            "confidence": conf,
            "box": [x1, y1, x2, y2],
            "area": (x2 - x1) * (y2 - y1)
        })
    if not regions:
        return image_path, []
    out = image_path.replace('.', '_pneureg.', 1)
    cv2.imwrite(out, img)
    return out, regions

# ---------------- Visual Inspection (Multimodal) ---------------- #
def gemini_visual_inspect(image_path: str) -> Dict[str, Any]:
    """
    Sends the raw image with a neutral prompt to Gemini to identify if it's a medical X-ray.
    Returns dict: { 'raw_description', 'detected_type', 'confidence_hint', 'rationale' }
    detected_type in {'xray','non_xray','uncertain'}
    """
    if gemini_client is None:
        return {
            "raw_description": "Gemini unavailable.",
            "detected_type": "uncertain",
            "confidence_hint": 0.0,
            "rationale": "Skipped due to missing Gemini client."
        }
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        prompt = (
            "Describe this image briefly (<=20 words). Then state if it is a medical radiographic X-ray (Yes/No) "
            "and your confidence 0-1. Format strictly as JSON with keys: description, is_xray (Yes|No|Uncertain), confidence."
        )
        resp = gemini_client.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": img_bytes}
        ])
        text = (resp.text or "").strip()
        # Extract JSON
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

# ---------------- Structured Gemini Explanation ---------------- #
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

def gemini_structured_advice(task: str, context: Dict[str, Any], image_path: str, visual_desc: str) -> Dict[str, Any]:
    """
    Request structured JSON explanation from Gemini.
    Optionally passes the actual image again if ENABLE_VISION_IN_PROMPT is True
    so Gemini can leverage pixels for reasoning.
    """
    if gemini_client is None:
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

    # Include visual description line to let Gemini leverage its previous analysis.
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
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            parts.append({"mime_type": "image/jpeg", "data": img_bytes})
        except Exception as e:
            log(f"Failed attaching image to structured prompt: {e}", "WARN")

    try:
        response = gemini_client.generate_content(parts)
        raw_text = (response.text or "").strip()
        json_str = extract_json_from_text(raw_text)
        parsed = json.loads(json_str)
        return normalize_explanation(parsed)
    except Exception as e:
        log(f"Gemini structured parse failed: {e}", "WARN")
        return default_explanation(task)

# ---------------- Severity Heuristics ---------------- #
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

# ---------------- Structured Builders ---------------- #
def build_fracture_structured(dets, explanation, visual):
    return {
        "task": "fracture_detection",
        "num_detections": len(dets),
        "detections": dets,
        "max_confidence": max([d['confidence'] for d in dets], default=0.0),
        "severity": fracture_severity(dets),
        "ai_explanation": explanation,
        "visual_inspection": visual
    }

def build_pneumonia_structured(label, conf, probs, class_names, regions, explanation, annotated_available, visual):
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
        "region_image": annotated_available,
        "severity": pneumonia_severity(label, conf),
        "ai_explanation": explanation,
        "visual_inspection": visual
    }

def build_rejected_structured(visual, task_requested):
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
        "visual_inspection": visual
    }

# ---------------- Persistence ---------------- #
def save_outputs(user_dir: str, structured: dict):
    os.makedirs(user_dir, exist_ok=True)
    with open(os.path.join(user_dir, "summary.json"), "w") as f:
        json.dump(structured, f, indent=2)

# ---------------- Routes ---------------- #
@app.route("/", methods=["GET"])
def index():
    return redirect(url_for("create"))

@app.route("/debug_models")
def debug_models():
    return jsonify({
        "fracture_model_loaded": fracture_model is not None,
        "pneumonia_cls_model_loaded": pneumonia_cls_model is not None,
        "pneumonia_det_model_loaded": pneumonia_det_model is not None
    })

@app.route("/create", methods=["GET", "POST"])
def create():
    myid = str(uuid.uuid4())
    if request.method == "POST":
        task_mode = request.form.get("task_mode", "fracture")
        received_id = request.form.get("uuid", myid)
        file = request.files.get("file")

        if not file or not allowed_file(file.filename):
            flash("Upload a valid PNG/JPG image.")
            return render_template("create.html", myid=myid)

        filename = secure_filename(file.filename)
        user_dir = os.path.join(app.config['UPLOAD_FOLDER'], received_id)
        os.makedirs(user_dir, exist_ok=True)
        img_path = os.path.join(user_dir, filename)
        file.save(img_path)

        # NEW: Visual multimodal inspection BEFORE medical inference
        visual = gemini_visual_inspect(img_path)
        if visual["detected_type"] == "non_xray":
            structured = build_rejected_structured(visual, task_mode)
            save_outputs(user_dir, structured)
            return render_template(
                "result.html",
                task_mode="rejected_non_xray",
                orig_img=url_for("uploaded_file", uuid=received_id, filename=filename),
                annotated_img=None,
                structured=structured,
                explanation=structured["ai_explanation"]
            )

        try:
            if task_mode == "fracture":
                if fracture_model is None:
                    flash("Fracture model unavailable.")
                    return render_template("create.html", myid=myid)
                ann_path, dets = detect_fractures(img_path)
                provisional = {
                    "num_detections": len(dets),
                    "detections": dets
                }
                explanation = gemini_structured_advice("fracture_detection", provisional, img_path, visual.get("raw_description", ""))
                structured = build_fracture_structured(dets, explanation, visual)
                annotated_img = ann_path

            elif task_mode == "pneumonia":
                if pneumonia_cls_model is None:
                    flash("Pneumonia classification model unavailable.")
                    return render_template("create.html", myid=myid)
                label, conf, probs, class_names = classify_pneumonia(img_path)
                region_img, regions = detect_pneumonia_regions(img_path, label)
                provisional = {
                    "classification": {"label": label, "confidence": conf},
                    "regional_detections": regions,
                    "regions_found": len(regions)
                }
                explanation = gemini_structured_advice("pneumonia_analysis", provisional, img_path, visual.get("raw_description", ""))
                annotated_available = (label.lower() == "pneumonia" and regions and region_img != img_path)
                structured = build_pneumonia_structured(
                    label, conf, probs, class_names, regions, explanation, annotated_available, visual
                )
                annotated_img = region_img if annotated_available else None
            else:
                flash("Unknown task mode.")
                return render_template("create.html", myid=myid)

            save_outputs(user_dir, structured)
            return render_template(
                "result.html",
                task_mode=task_mode,
                orig_img=url_for("uploaded_file", uuid=received_id, filename=filename),
                annotated_img=(url_for("uploaded_file", uuid=received_id, filename=os.path.basename(annotated_img))
                               if annotated_img else None),
                structured=structured,
                explanation=structured["ai_explanation"]
            )
        except Exception as e:
            log(f"Processing error: {e}", "ERROR")
            flash(f"Processing error: {e}")
            return render_template("create.html", myid=myid)

    return render_template("create.html", myid=myid)

@app.route("/user_uploads/<uuid>/<filename>")
def uploaded_file(uuid, filename):
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], uuid), filename)

@app.route("/clear_uploads")
def clear_uploads():
    import shutil
    shutil.rmtree(app.config['UPLOAD_FOLDER'], ignore_errors=True)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    return "Uploads cleared."

if __name__ == "__main__":
    log("App starting...", "INFO")
    app.run(debug=True)
