# Cyber Scanner AI
Application สำหรับตรวจจับ Phishing URL และ SMS Spam ภาษาไทยด้วย Deep Learning (Hybrid LSTM + WangchanBERTa)

# Prerequisites
- Python 3.10 หรือใหม่กว่า
- Windows 10/11 (สำหรับไฟล์ .exe)

# Terminal (CMD) ในโฟลเดอร์ URL 
# Environment Setup
 1. python -m venv venv
 2.venv\Scripts\activate

# ติดตั้ง Dependencies
# ติดตั้ง Library ตามเวอร์ชันที่ระบุเพื่อความเสถียร (TensorFlow 2.16.1 + Keras 3.3.3)
pip install flask pywebview numpy pandas scikit-learn
pip install tensorflow==2.16.1 keras==3.3.3
pip install torch transformers
pip install pyinstaller

# Development
python run_app.py
