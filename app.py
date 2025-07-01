import os
import json
import tempfile
import subprocess
from flask import Flask, request, jsonify
from pdf2image import convert_from_path
import pytesseract
import requests

app = Flask(__name__)

# Let PATH find the tesseract binary instead of hardcoding full path.
pytesseract.pytesseract.tesseract_cmd = "tesseract"
POPPLER_PATH = "/usr/bin"  # This remains as a helper for pdf2image

def call_huggingface_inference(prompt):
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        raise Exception("No Hugging Face API token found in environment variables.")

    API_URL = "https://router.huggingface.co/novita/v3/openai/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "deepseek/deepseek-v3-0324"
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code}, {response.text}")
    response_data = response.json()
    try:
        return response_data["choices"][0]["message"]
    except (KeyError, IndexError):
        raise Exception("Unexpected response format from Hugging Face API")

@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    try:
        # Run 'tesseract --version' via the shell so that flags are correctly parsed.
        tesseract_output = subprocess.run(
            ["tesseract", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        # Run 'pdfinfo --version' similarly via the shell.
        # new
        poppler_output = subprocess.run(["pdfinfo", "-v"],      # <- use -v instead of --version
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        return jsonify({
            "tesseract": tesseract_output.stdout.decode().strip(),
            "poppler": poppler_output.stdout.decode().strip(),
            "status": "OK"
        })
    except subprocess.CalledProcessError as e:
        return jsonify({
            "error": "One or more dependencies are not installed correctly.",
            "details": e.stderr.decode().strip()
        }), 500

@app.route("/analyze", methods=["POST"])
def analyze_pdf():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded."}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            file.save(temp_pdf)
            temp_pdf_path = temp_pdf.name
    except Exception as e:
        return jsonify({"error": f"Could not save file: {str(e)}"}), 500

    try:
        images = convert_from_path(temp_pdf_path, poppler_path=POPPLER_PATH)
        extracted_text = ""
        for page in images:
            extracted_text += pytesseract.image_to_string(page) + "\n"
    except Exception as e:
        os.remove(temp_pdf_path)
        return jsonify({"error": f"Error processing the PDF: {str(e)}"}), 500

    os.remove(temp_pdf_path)

    prompt = (
        "You are an assistant that analyzes grade reports. "
        "Given the following grade report text, extract the overall GPA (on a 0.0 to 4.0 scale) "
        "and count the number of subjects with grade F. "
        "Return your answer as valid JSON in the following format:\n"
        '{"GPA": <number>, "F": <number>}\n'
        "Grade Report Text:\n"
        f"'''{extracted_text}'''"
    )

    try:
        generated_response = call_huggingface_inference(prompt)
        result = json.loads(generated_response) if isinstance(generated_response, str) else generated_response
    except Exception as e:
        return jsonify({"error": f"Error from Hugging Face inference: {str(e)}"}), 500

    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)