let currentMode = 'url';

// ✅ โหลดประวัติจากเซิร์ฟเวอร์ทันทีที่เปิดแอป
document.addEventListener('DOMContentLoaded', loadHistory);

function switchMode(mode) {
    currentMode = mode;
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`tab-${mode}`).classList.add('active');

    const input = document.getElementById('inputField');
    input.value = '';
    document.getElementById('result-area').classList.remove('show');

    if (mode === 'url') {
        input.placeholder = "วางลิงก์เว็บไซต์ที่ต้องการตรวจสอบ...";
        input.rows = 1;
        input.style.fontFamily = "'JetBrains Mono', monospace";
    } else {
        input.placeholder = "วางข้อความ SMS ที่ได้รับเพื่อตรวจสอบ...";
        input.rows = 3;
        input.style.fontFamily = "'Sarabun', sans-serif";
    }
}

async function handleScan(e) {
    e.preventDefault();
    const text = document.getElementById('inputField').value.trim();
    if (!text) return;

    setLoading(true);
    const endpoint = currentMode === 'url' ? '/predict_url' : '/predict_sms';

    try {
        const formData = new FormData();
        formData.append('text', text);

        const response = await fetch(endpoint, { method: 'POST', body: formData });
        const data = await response.json();

        renderResult(data);
        
        // ✅ ถ้าสแกนสำเร็จ ให้ส่งข้อมูลไปบันทึกที่ Python Backend
        if(data.status === 'success') {
            await saveHistoryToServer(currentMode, text, data.is_danger);
        }

    } catch (error) {
        console.error(error);
        alert("การเชื่อมต่อขัดข้อง");
    } finally {
        setLoading(false);
    }
}

function setLoading(isLoading) {
    const btn = document.getElementById('scanBtn');
    const btnText = document.getElementById('btnText');
    const loader = document.getElementById('loader');
    
    btn.disabled = isLoading;
    btnText.style.display = isLoading ? 'none' : 'inline';
    loader.style.display = isLoading ? 'inline-block' : 'none';
}

function renderResult(data) {
    const resultArea = document.getElementById('result-area');
    resultArea.classList.add('show');
    
    const statusIcon = document.getElementById('statusIcon');
    const statusText = document.getElementById('statusText');
    const statusBadge = document.getElementById('statusBadge');
    const riskScore = document.getElementById('riskScore');
    const threatType = document.getElementById('threatType');
    const recommendation = document.getElementById('recommendation');
    const meterFill = document.getElementById('meterFill');

    statusIcon.className = ''; 

    if (data.status === 'error') {
        statusIcon.className = 'fas fa-triangle-exclamation';
        statusText.innerText = "Error";
        statusBadge.style.color = "#f59e0b";
        threatType.innerText = "System Error";
        recommendation.innerText = data.msg;
        return;
    }

    riskScore.innerText = `${data.score.toFixed(1)}%`;
    threatType.innerText = data.type;
    recommendation.innerText = data.advice;
    
    if (data.is_danger) {
        statusIcon.className = 'fas fa-shield-virus';
        statusText.innerText = "DANGEROUS";
        statusBadge.style.color = "#FF4D5E";
        meterFill.style.backgroundColor = "#FF4D5E";
        recommendation.style.color = "#FF4D5E";
    } else {
        statusIcon.className = 'fas fa-shield-check';
        statusText.innerText = "SAFE";
        statusBadge.style.color = "#4ADE80";
        meterFill.style.backgroundColor = "#4ADE80";
        recommendation.style.color = "#4ADE80";
    }
    
    meterFill.style.width = '0%';
    setTimeout(() => meterFill.style.width = `${data.score}%`, 100);
}

// --- History Logic (API Based) ---

async function saveHistoryToServer(type, text, isDanger) {
    try {
        const response = await fetch('/api/history', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type, text, isDanger })
        });
        const result = await response.json();
        // อัปเดตตารางประวัติทันทีหลังจากบันทึกเสร็จ
        if(result.history) {
            renderHistoryList(result.history); 
        }
    } catch (e) { console.error("Save History Failed", e); }
}

async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const history = await response.json();
        renderHistoryList(history);
    } catch (e) { console.error("Load History Failed", e); }
}

async function clearHistory() {
    if(!confirm("ต้องการล้างประวัติทั้งหมดหรือไม่?")) return;
    try {
        await fetch('/api/history', { method: 'DELETE' });
        renderHistoryList([]); // เคลียร์หน้าจอ
    } catch (e) { console.error("Clear History Failed", e); }
}

function renderHistoryList(history) {
    const list = document.getElementById('history-list');
    if (!list) return; // ป้องกัน error ถ้าหา element ไม่เจอ
    
    list.innerHTML = '';

    if (!history || history.length === 0) {
        list.innerHTML = '<div class="empty-state">ยังไม่มีประวัติการตรวจสอบ</div>';
        return;
    }

    history.forEach(item => {
        const div = document.createElement('div');
        div.className = 'history-item';
        
        const iconClass = item.type === 'url' ? 'fa-globe' : 'fa-comment-dots';
        const badgeClass = item.isDanger ? 'status-danger' : 'status-safe';
        const badgeText = item.isDanger ? 'DANGER' : 'SAFE';
        
        div.innerHTML = `
            <div class="h-left">
                <div class="h-icon ${item.type === 'url' ? 'blue' : 'green'}" style="background:rgba(255,255,255,0.1)">
                    <i class="fas ${iconClass}"></i>
                </div>
                <div class="h-content">
                    <span class="h-text">${escapeHtml(item.text)}</span>
                    <span class="h-date">${item.timestamp}</span>
                </div>
            </div>
            <div class="h-badge ${badgeClass}">${badgeText}</div>
        `;
        list.appendChild(div);
    });
}

function escapeHtml(text) {
    if (!text) return "";
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}