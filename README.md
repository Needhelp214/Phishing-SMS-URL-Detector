#Terminal (CMD) ในโฟลเดอร์ URL 

# 1. สร้าง Environment
python -m venv venv

# 2. เปิดใช้งาน (Windows)
venv\Scripts\activate

# 3. ติดตั้ง Library ที่จำเป็น (สำคัญมากต้องตรงรุ่น)
pip install flask pywebview numpy pandas scikit-learn
pip install tensorflow==2.16.1 keras==3.3.3
pip install torch transformers

# 4. เมื่อติดตั้งทุกอย่างครบแล้ว สามารถเปิดใช้งานได้ทันทีด้วยคำสั่ง
python run_app.py
