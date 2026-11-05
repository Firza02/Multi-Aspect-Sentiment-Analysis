import os
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from flask import Flask, render_template, request, jsonify, Response
from huggingface_hub import hf_hub_download
import json

app = Flask(__name__)

# --- KONFIGURASI MODEL ---
MODEL_NAME = "indobenchmark/indobert-base-p2"
DEVICE = torch.device("cpu")
MODEL_PATH = "focal_model_quantized2.pt"
BATCH_SIZE = 16  # sesuaikan: naikkan jika RAM cukup, turunkan jika crash

# --- KONFIGURASI HUGGING FACE HUB (model di-download otomatis saat start) ---
HF_REPO_ID = "firza02/focal-model"
HF_FILENAME = "focal_model_quantized2.pt"

ASPECT_NAMES = ["Effectiveness", "Efficiency", "Learnability", "Errors", "Satisfaction", "Cognitive Load"]
LABEL_MAP = {0: "Negatif", 1: "Positif", 2: "Aspek Tidak Disebut"}

class IndoBERTMultiAspek(nn.Module):
    def __init__(self, model_name, num_aspects, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifiers = nn.ModuleList([
            nn.Linear(self.bert.config.hidden_size, num_classes) for _ in range(num_aspects)
        ])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        return [clf(pooled) for clf in self.classifiers]

if not os.path.exists(MODEL_PATH):
    print(f"Model '{MODEL_PATH}' tidak ditemukan secara lokal, mengunduh dari Hugging Face Hub...")
    downloaded_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    MODEL_PATH = downloaded_path
    print(f"Model berhasil diunduh ke: {MODEL_PATH}")

print("Memuat model... Harap tunggu.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = IndoBERTMultiAspek(MODEL_NAME, num_aspects=6, num_classes=3)

qconfig_spec = {
    nn.Linear: torch.quantization.default_dynamic_qconfig,
    nn.Embedding: torch.quantization.float_qparams_weight_only_qconfig,
}
model = torch.quantization.quantize_dynamic(model, qconfig_spec, dtype=torch.qint8)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Sistem Siap!")


def get_prediction(text):
    """Prediksi untuk satu teks (dipakai di /predict)."""
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=128, padding='max_length'
    )
    with torch.no_grad():
        logits = model(inputs['input_ids'].to(DEVICE), inputs['attention_mask'].to(DEVICE))
    return {asp: LABEL_MAP[torch.argmax(logits[i], dim=1).item()] for i, asp in enumerate(ASPECT_NAMES)}


def get_predictions_batch(texts):
    """Prediksi untuk list teks sekaligus dalam satu forward pass."""
    inputs = tokenizer(
        texts, return_tensors="pt",
        truncation=True, max_length=128,
        padding=True  # padding dinamis, lebih efisien dari max_length
    )
    with torch.no_grad():
        logits = model(inputs['input_ids'].to(DEVICE), inputs['attention_mask'].to(DEVICE))

    # logits adalah list (per aspek) dengan shape [batch_size, num_classes]
    batch_preds = []
    batch_size = len(texts)
    for i in range(batch_size):
        row = {}
        for asp_idx, asp in enumerate(ASPECT_NAMES):
            pred_idx = torch.argmax(logits[asp_idx][i]).item()
            row[asp] = LABEL_MAP[pred_idx]
        batch_preds.append(row)
    return batch_preds


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "Teks kosong"}), 400
    res = get_prediction(text)
    return jsonify([{"aspek": k, "sentimen": v} for k, v in res.items()])


@app.route('/predict_file', methods=['POST'])
def predict_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    try:
        df = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)
    except Exception as e:
        return jsonify({"error": f"Gagal membaca file: {str(e)}"}), 400

    col = 'ulasan' if 'ulasan' in df.columns else df.columns[0]
    texts = df[col].astype(str).tolist()

    final_results = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start: start + BATCH_SIZE]
        batch_preds = get_predictions_batch(batch_texts)
        for text, pred in zip(batch_texts, batch_preds):
            pred['teks_asli'] = text
            final_results.append(pred)

    return jsonify(final_results)


if __name__ == '__main__':
    app.run(debug=True)