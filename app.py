import os
import sys
import math
import numpy as np
import pickle
import json  # <--- (ใหม่) ใช้จัดการไฟล์ประวัติ
import datetime # <--- (ใหม่) ใช้วันที่และเวลา
from collections import Counter
from flask import Flask, render_template, request, jsonify
from urllib.parse import urlparse

# --- Config Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
URL_MODEL_PATH = os.path.join(BASE_DIR, 'final_hybrid_model.keras')
URL_TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer.pkl')
URL_SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
SMS_MODEL_PATH = os.path.join(BASE_DIR, 'sms_model')
HISTORY_PATH = os.path.join(BASE_DIR, 'scan_history.json') # <--- (ใหม่) ไฟล์เก็บประวัติ

MAX_LEN_URL = 150 

app = Flask(__name__)

# --- Setup AI ---
try:
    import keras
    from keras.models import Model
    from keras.layers import Input, Embedding, SpatialDropout1D, Bidirectional, LSTM, Dropout, Dense, BatchNormalization, Concatenate
    from keras.preprocessing.sequence import pad_sequences
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import torch.nn.functional as F
    print(f"✅ AI Libraries Ready")
except ImportError as e:
    print(f"❌ Critical Error: {e}")
    sys.exit(1)

# Global Vars
url_model = None; url_tokenizer = None; url_scaler = None
sms_model = None; sms_tokenizer = None

# --- Rebuild Model Function ---
def build_url_model(vocab_size):
    url_input = Input(shape=(MAX_LEN_URL,), dtype="int32", name="url_input")
    x = Embedding(input_dim=vocab_size, output_dim=50)(url_input)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.5)(x)
    feat_input = Input(shape=(13,), dtype="float32", name="features_input")
    y = Dense(64, activation="relu")(feat_input)
    y = BatchNormalization()(y)
    y = Dropout(0.4)(y)
    y = Dense(32, activation="relu")(y)
    merged = Concatenate()([x, y])
    z = Dense(64, activation="relu")(merged)
    z = Dropout(0.5)(z)
    output = Dense(1, activation="sigmoid")(z)
    return Model(inputs=[url_input, feat_input], outputs=output)

# --- Load Resources ---
def load_resources():
    global url_model, url_tokenizer, url_scaler, sms_model, sms_tokenizer
    # URL
    if os.path.exists(URL_TOKENIZER_PATH):
        with open(URL_TOKENIZER_PATH, 'rb') as f: url_tokenizer = pickle.load(f)
    if os.path.exists(URL_SCALER_PATH):
        with open(URL_SCALER_PATH, 'rb') as f: url_scaler = pickle.load(f)
    if os.path.exists(URL_MODEL_PATH) and url_tokenizer:
        vocab_size = len(url_tokenizer.word_index) + 1
        url_model = build_url_model(vocab_size)
        try: url_model.load_weights(URL_MODEL_PATH)
        except: pass
    # SMS
    if os.path.exists(SMS_MODEL_PATH):
        try:
            sms_tokenizer = AutoTokenizer.from_pretrained(SMS_MODEL_PATH)
            sms_model = AutoModelForSequenceClassification.from_pretrained(SMS_MODEL_PATH)
        except: pass

load_resources()

# --- Helpers ---
def calculate_entropy(text):
    if not text: return 0
    counts = Counter(text)
    length = len(text)
    return -sum((c/length) * math.log2(c/length) for c in counts.values())

def extract_url_features(url):
    url = str(url)
    features = [
        len(url), url.count('.'), url.count('-'), url.count('@'), url.count('?'),
        url.count('&'), url.count('='), url.count('_'), sum(c.isdigit() for c in url),
        calculate_entropy(url), 1 if 'https' in url else 0, 1 if 'http' in url else 0,
        1 if 'www' in url else 0
    ]
    return np.array([features])

# --- Routes ---
@app.route('/')
def home(): return render_template('detec.html')

# ==========================================
# 🆕 HISTORY API (ระบบบันทึกประวัติลงไฟล์จริง)
# ==========================================
@app.route('/api/history', methods=['GET', 'POST', 'DELETE'])
def manage_history():
    # 1. อ่านประวัติ (GET)
    if request.method == 'GET':
        if os.path.exists(HISTORY_PATH):
            try:
                with open(HISTORY_PATH, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                return jsonify(history)
            except: return jsonify([])
        return jsonify([])

    # 2. บันทึกประวัติใหม่ (POST)
    if request.method == 'POST':
        data = request.json
        new_item = {
            'type': data.get('type'),
            'text': data.get('text'),
            'isDanger': data.get('isDanger'),
            'timestamp': datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        }
        
        history = []
        if os.path.exists(HISTORY_PATH):
            try:
                with open(HISTORY_PATH, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except: pass
        
        history.insert(0, new_item) # แทรกรายการใหม่ไว้บนสุด
        history = history[:20] # เก็บแค่ 20 รายการล่าสุด (กันไฟล์บวม)
        
        with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=4)
            
        return jsonify({'status': 'saved', 'history': history})

    # 3. ล้างประวัติ (DELETE)
    if request.method == 'DELETE':
        if os.path.exists(HISTORY_PATH):
            os.remove(HISTORY_PATH)
        return jsonify({'status': 'cleared'})

# ==========================================

@app.route('/predict_url', methods=['POST'])
def predict_url():
    url = request.form.get('text', '').strip()
    if not url_model: return jsonify({'status': 'error', 'msg': 'System Not Ready'})
    
    try:
        seq = url_tokenizer.texts_to_sequences([url])
        padded = pad_sequences(seq, maxlen=MAX_LEN_URL)
        feats = extract_url_features(url)
        if url_scaler: feats = url_scaler.transform(feats)
        
        pred = url_model.predict({'url_input': padded, 'features_input': feats}, verbose=0)
        score = float(pred[0][0]) * 100
        
        analysis_type = "Unknown"
        if score > 80: analysis_type = "High Risk Phishing"
        elif score > 50: analysis_type = "Suspicious Link"
        else: analysis_type = "Legitimate Website"

        advice = "ปลอดภัย สามารถเข้าใช้งานได้"
        if score > 50: advice = "อันตราย! ห้ามกดลิงก์ หรือกรอกข้อมูลส่วนตัวเด็ดขาด"
        
        return jsonify({
            'status': 'success',
            'score': score,
            'is_danger': score > 50,
            'type': analysis_type,
            'advice': advice
        })
    except Exception as e:
        print(e)
        return jsonify({'status': 'error', 'msg': 'Analysis Failed'})

@app.route('/predict_sms', methods=['POST'])
def predict_sms():
    text = request.form.get('text', '').strip()
    if not sms_model: return jsonify({'status': 'error', 'msg': 'SMS Model Not Ready'})

    # 1. Rule-Based
    risk_words = ["รับฟรี", "เครดิตฟรี", "เงินกู้", "ดอกเบี้ยต่ำ", "คลิกเลย", "เว็บตรง", "สล็อต", 
                  "คุณได้รับ", "ยืนยันสิทธิ์", "กรมที่ดิน", "การไฟฟ้า", "ใบขับขี่", "ไม่ต้องสอบ", "กรุงไทย", "ออมสิน"]
    has_link = any(x in text for x in ["http", ".com", ".net", "bit.ly"])
    
    found_keyword = None
    for word in risk_words:
        if word in text:
            found_keyword = word
            break
            
    if found_keyword:
        if has_link or found_keyword in ["เงินกู้", "ไม่ต้องสอบ", "เครดิตฟรี", "กรมที่ดิน"]:
            return jsonify({
                'status': 'success',
                'score': 99.99,
                'is_danger': True,
                'type': f"Scam Pattern Detected ({found_keyword})",
                'advice': "ตรวจพบคำโฆษณาชวนเชื่อหรือการแอบอ้างหน่วยงานรัฐ ห้ามโอนเงิน!"
            })

    # 2. AI Check
    try:
        inputs = sms_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            probs = F.softmax(sms_model(**inputs).logits, dim=1)
        
        prob_spam = probs[0][1].item() * 100
        
        analysis_type = "Normal Message"
        advice = "ข้อความดูปกติ ไม่พบความเสี่ยง"
        if prob_spam > 50:
            analysis_type = "Spam / Smishing"
            advice = "ข้อความนี้มีแนวโน้มเป็น Spam หลอกลวง"

        return jsonify({
            'status': 'success',
            'score': prob_spam,
            'is_danger': prob_spam > 50,
            'type': analysis_type,
            'advice': advice
        })
    except: return jsonify({'status': 'error', 'msg': 'Analysis Failed'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)