"""Macro Tracker backend for Render.

Serves the static frontend (index.html) and provides a server-side
photo -> nutrition estimate endpoint. The Gemini API key lives only here
as the GEMINI_API_KEY environment variable -- it is never sent to the browser.

Uses the Gemini API free tier (no billing, no card; get a key at
https://aistudio.google.com) via Google's OpenAI-compatible chat-completions
endpoint. Free-tier Flash models accept image input at no charge.

Endpoints:
  GET  /                  -> the app
  GET  /healthz           -> health check
  POST /api/estimate      -> {image: dataURL} -> {name, serving, calories,
                             protein_g, carbs_g, fat_g, confidence}
"""
import base64
import io
import json
import os
import re

import requests
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.route("/healthz")
def healthz():
    return jsonify(ok=True)


def _extract_json(text):
    """Pull a JSON object out of model output, tolerating code fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])


@app.route("/api/estimate", methods=["POST"])
def estimate():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return jsonify(error="AI key not configured on server"), 503

    data = request.get_json(force=True, silent=True) or {}
    data_url = data.get("image", "")
    if "," not in data_url or not data_url.startswith("data:image/"):
        return jsonify(error="no image provided"), 400

    try:
        raw = base64.b64decode(data_url.split(",", 1)[1])
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img.thumbnail((1024, 1024))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        small_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return jsonify(error="could not read image"), 400

    prompt = (
        "Estimate the nutrition for the food in this photo, for one serving as eaten. "
        'Respond with ONLY a JSON object: {"name": string, "serving": string like "1 plate (350g)", '
        '"calories": number, "protein_g": number, "carbs_g": number, "fat_g": number, '
        '"confidence": "low", "medium" or "high"}. Use US Nutrition Facts conventions (Calories). '
        'If no food is visible, respond with ONLY {"error": "no food"}.'
    )
    try:
        resp = requests.post(
            GEMINI_URL,
            headers={"Authorization": "Bearer " + key},
            json={
                "model": GEMINI_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": small_url}},
                    ],
                }],
                "max_tokens": 300,
            },
            timeout=90,
        )
    except Exception:
        return jsonify(error="AI service unreachable"), 502

    if resp.status_code == 429:
        return jsonify(error="AI quota exhausted, try again in a minute"), 502
    if resp.status_code != 200:
        return jsonify(error="AI service error"), 502

    try:
        content = resp.json()["choices"][0]["message"]["content"]
        result = _extract_json(content)
    except Exception:
        return jsonify(error="AI response unreadable"), 502

    if result.get("error"):
        return jsonify(error="no food"), 422

    def num(v):
        try:
            v = float(v)
            return v if v == v and abs(v) != float("inf") else 0
        except (TypeError, ValueError):
            return 0

    return jsonify(
        name=str(result.get("name", ""))[:160],
        serving=str(result.get("serving", ""))[:80],
        calories=num(result.get("calories")),
        protein_g=num(result.get("protein_g")),
        carbs_g=num(result.get("carbs_g")),
        fat_g=num(result.get("fat_g")),
        confidence=str(result.get("confidence", "medium"))[:10],
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
