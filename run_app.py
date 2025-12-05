import webview
import sys
import threading
import time
import os

# เพิ่ม path ปัจจุบันเพื่อให้หาไฟล์ app.py เจอแน่นอน
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app

def start_server():
    """ฟังก์ชันสำหรับรัน Flask Server ใน Thread แยก"""
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

if __name__ == '__main__':
    t = threading.Thread(target=start_server)
    t.daemon = True
    t.start()

    time.sleep(2)

    window = webview.create_window(
        title='Cyber Scanner (AI Powered)',
        url='http://127.0.0.1:5000',
        width=1000,
        height=720,
        resizable=True,
        min_size=(800, 600),
        background_color='#0f172a'
    )


    webview.start(debug=True)
    
    sys.exit()