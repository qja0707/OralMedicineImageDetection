// 약품 DB
let pillDB = {};
fetch('/static/pill_info.json')
    .then(r => r.json())
    .then(data => { pillDB = data; })
    .catch(() => {});

// 약상자 (localStorage)
function getPillBox() {
    try { return JSON.parse(localStorage.getItem('pillbox') || '[]'); }
    catch { return []; }
}
function savePillBox(box) {
    localStorage.setItem('pillbox', JSON.stringify(box));
}

// 분류 → 증상 매핑
function classToSymptom(classNo) {
    if (!classNo) return { emoji: '💊', label: '일반' };
    const c = classNo.toLowerCase();
    if (c.includes('해열') || c.includes('진통') || c.includes('소염')) return { emoji: '🤒', label: '통증/감기' };
    if (c.includes('소화') || c.includes('궤양') || c.includes('위장') || c.includes('제산')) return { emoji: '🤢', label: '소화/위장' };
    if (c.includes('정신') || c.includes('신경')) return { emoji: '😰', label: '신경/수면' };
    if (c.includes('혈압') || c.includes('심장') || c.includes('혈관') || c.includes('순환')) return { emoji: '❤️', label: '심혈관' };
    if (c.includes('항생') || c.includes('감염') || c.includes('항균')) return { emoji: '🦠', label: '감염/항생' };
    if (c.includes('당뇨') || c.includes('혈당')) return { emoji: '🩸', label: '당뇨' };
    if (c.includes('알레르기') || c.includes('항히스타민')) return { emoji: '🤧', label: '알레르기' };
    if (c.includes('비타민') || c.includes('영양')) return { emoji: '💪', label: '영양/비타민' };
    if (c.includes('간장') || c.includes('간') || c.includes('담낭')) return { emoji: '🫁', label: '간/담' };
    if (c.includes('호흡') || c.includes('기관지') || c.includes('천식')) return { emoji: '🫁', label: '호흡기' };
    if (c.includes('골격') || c.includes('근육') || c.includes('관절')) return { emoji: '🦴', label: '근골격' };
    return { emoji: '💊', label: '일반' };
}

// 검출된 약들의 공통 증상 배너
function makeCategoryBanner(detections) {
    const symptoms = new Map();
    detections.forEach(det => {
        if (det.info && det.info.class_no) {
            const s = classToSymptom(det.info.class_no);
            symptoms.set(s.label, s);
        }
    });
    if (symptoms.size === 0) return '';
    const parts = [...symptoms.values()].map(s => `${s.emoji} ${s.label}`).join(', ');
    const msg = symptoms.size === 1
        ? `이 약은 <b>${[...symptoms.values()][0].label}</b> 일때 먹는 약입니다`
        : `<b>${parts}</b> 관련 약입니다`;
    return `<div class="category-banner">${msg}</div>`;
}

// 앱 시작 시 프로필 로드
document.addEventListener('DOMContentLoaded', () => {
    loadProfile();
    renderHomeSchedule();

    // 저장된 약사 복원
    const savedPh = localStorage.getItem('selectedPharmacist');
    if (savedPh && pharmacists[savedPh]) {
        currentPharmacist = pharmacists[savedPh];
        document.getElementById('tab-ai-icon').src = currentPharmacist.img;
        document.getElementById('mypage-pharmacist-img').src = currentPharmacist.img;
        document.getElementById('mypage-pharmacist-name').textContent = currentPharmacist.name;
    }
    const fs = localStorage.getItem('fontSize');
    if (fs) {
        const frame = document.getElementById('phone-frame');
        frame.style.fontSize = fs === 'small' ? '13px' : fs === 'large' ? '17px' : '15px';
    }
});

// 검색
const searchInput = document.getElementById('search-input');
const searchResults = document.getElementById('search-results');

searchInput.addEventListener('input', function() {
    const query = this.value.trim().toLowerCase();
    searchResults.innerHTML = '';
    if (query.length < 1) return;

    const matches = [];
    for (const [id, info] of Object.entries(pillDB)) {
        const searchable = [info.name, info.material, info.company, info.name_en].join(' ').toLowerCase();
        if (searchable.includes(query)) matches.push(info);
    }

    if (matches.length === 0) {
        searchResults.innerHTML = '<div class="search-empty">검색 결과가 없습니다.</div>';
        return;
    }

    matches.slice(0, 10).forEach(info => {
        const catId = Object.keys(pillDB).find(k => pillDB[k].name === info.name) || '';
        const card = document.createElement('div');
        card.className = 'search-card';
        card.innerHTML = `
            <div style="display:flex;align-items:center;gap:12px;">
                <img src="/static/pills/${catId}.png" style="width:44px;height:44px;border-radius:10px;object-fit:cover;" onerror="this.style.display='none'">
                <div>
                    <div class="search-card-name">${info.name}</div>
                    <div class="search-card-sub">${info.material || ''} · ${info.company || ''}</div>
                </div>
            </div>
        `;
        card.onclick = () => showPillDetail(info, catId);
        searchResults.appendChild(card);
    });
});

function showPillDetail(info, catId) {
    searchInput.value = '';
    searchResults.innerHTML = '';

    const det = { name: info.name, confidence: 1.0, info: info, catId: catId };
    document.getElementById('detect-image-section').style.display = 'none';
    const resultImgSection = document.getElementById('result-image-section');
    const resultImg = document.getElementById('result-image');
    if (catId) {
        resultImg.src = `/static/pills/${catId}.png`;
        resultImgSection.style.display = 'block';
        resultImg.style.width = '140px';
        resultImg.style.margin = '0 auto 12px';
        resultImg.style.borderRadius = '16px';
        resultImg.style.display = 'block';
    } else {
        resultImgSection.style.display = 'none';
    }
    lastDetections = [det];
    renderCards([det], 1);
    showPage('result');
}

// 토스트 알림
function showToast(msg, duration = 2000) {
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), duration);
}

// 이미지 팝업
function openPopup(src) {
    document.getElementById('popup-img').src = src;
    document.getElementById('img-popup').classList.add('active');
}
function closePopup() {
    document.getElementById('img-popup').classList.remove('active');
}

// 페이지 전환
let pageHistory = ['home'];

function showPage(pageId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById('page-' + pageId).classList.add('active');

    document.querySelectorAll('.tab-item').forEach(t => t.classList.remove('active'));
    const tab = document.querySelector(`.tab-item[data-page="${pageId}"]`);
    if (tab) tab.classList.add('active');

    document.getElementById('back-btn-fixed').style.display = (pageId === 'home') ? 'none' : 'flex';

    // 로딩 페이지는 히스토리에 넣지 않음
    if (pageId !== 'loading' && pageHistory[pageHistory.length - 1] !== pageId) {
        pageHistory.push(pageId);
    }

    if (pageId === 'home') renderHomeSchedule();
    if (pageId === 'pillbox') renderPillBox();
    if (pageId === 'mypage') { loadProfile(); renderHistory(); }
}

function goBack() {
    if (pageHistory.length > 1) {
        pageHistory.pop();
        showPage(pageHistory[pageHistory.length - 1]);
    } else {
        showPage('home');
    }
}

// 카메라/업로드
let webcamStream = null;

function openCamera() {
    // 모바일: 카메라 앱, PC: 웹캠
    if (/Mobi|Android|iPhone/i.test(navigator.userAgent)) {
        document.getElementById('camera-input').click();
    } else {
        startWebcam();
    }
}
function openUpload() { document.getElementById('upload-input').click(); }

async function startWebcam() {
    try {
        const video = document.getElementById('webcam-video');
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: 1280, height: 960 }
        });
        video.srcObject = webcamStream;
        showPage('webcam');
    } catch (err) {
        showToast('카메라를 사용할 수 없습니다');
        document.getElementById('camera-input').click();
    }
}

function captureWebcam() {
    const video = document.getElementById('webcam-video');
    const canvas = document.getElementById('webcam-canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    stopWebcamStream();
    showPage('loading');

    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'capture.jpg');

        try {
            const res = await fetch('/api/detect', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.error) { showToast(data.error); showPage('home'); return; }

            document.getElementById('result-image-section').style.display = 'none';
            document.getElementById('detect-image').src = 'data:image/jpeg;base64,' + data.image;
            document.getElementById('detect-image-section').style.display = 'block';

            lastDetections = data.detections;
            addHistory(data.detections);
            renderCards(data.detections, data.count);
            showPage('result');
        } catch (err) {
            showToast('검출 실패');
            showPage('home');
        }
    }, 'image/jpeg', 0.9);
}

function stopWebcam() {
    stopWebcamStream();
    goBack();
}

function stopWebcamStream() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(t => t.stop());
        webcamStream = null;
    }
}

document.getElementById('camera-input').addEventListener('change', handleFile);
document.getElementById('upload-input').addEventListener('change', handleFile);

let lastDetections = [];

async function handleFile(e) {
    const file = e.target.files[0];
    if (!file) return;
    showPage('loading');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/api/detect', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.error) { alert(data.error); showPage('home'); return; }

        document.getElementById('result-image-section').style.display = 'none';
        document.getElementById('detect-image').src = 'data:image/jpeg;base64,' + data.image;
        document.getElementById('detect-image-section').style.display = 'block';

        lastDetections = data.detections;
        addHistory(data.detections);
        renderCards(data.detections, data.count);
        showPage('result');
    } catch (err) {
        alert('검출 실패: ' + err.message);
        showPage('home');
    }
    e.target.value = '';
}

function renderCards(detections, count) {
    const container = document.getElementById('result-cards');

    if (count === 0) {
        container.innerHTML = '<div style="text-align:center;padding:30px;color:#999;">검출된 알약이 없습니다.</div>';
        return;
    }

    // 공통 증상 배너
    let html = makeCategoryBanner(detections);
    html += `<div class="result-count">${count}개 알약 검출</div>`;

    detections.forEach(det => {
        const confPct = (det.confidence * 100).toFixed(1);
        const confColor = det.confidence >= 0.9 ? '#4ECDC4' : det.confidence >= 0.7 ? '#FFB74D' : '#FF8A80';

        let details = '';
        if (det.info) {
            const rows = [
                ['성분', det.info.material], ['제약사', det.info.company],
                ['구분', det.info.etc_otc],
                ['외형', ((det.info.color1 || '') + ' ' + (det.info.shape || '')).trim()],
                ['각인', [det.info.print_front, det.info.print_back].filter(Boolean).join(' / ')],
                ['설명', det.info.chart],
            ];
            rows.forEach(([label, value]) => {
                if (value) details += `<div class="pill-detail"><span class="pill-label">${label}</span><span class="pill-value">${value}</span></div>`;
            });
        }

        let pillCatId = '';
        if (det.info) {
            for (const [k, v] of Object.entries(pillDB)) {
                if (v.name === det.name) { pillCatId = k; break; }
            }
        }
        const pillImgSrc = pillCatId ? `/static/pills/${pillCatId}.png` : '';
        const pillImg = pillImgSrc ? `<img src="${pillImgSrc}" class="pill-thumb" style="width:52px;height:52px;border-radius:12px;object-fit:cover;flex-shrink:0;" onclick="openPopup('${pillImgSrc}')" onerror="this.style.display='none'">` : '';

        const classNo = det.info ? (det.info.class_no || '') : '';
        const classBadge = classNo ? `<div class="class-badge">${classNo}</div>` : '';

        html += `
        <div class="pill-card">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
                ${pillImg}
                <div style="flex:1;">
                    <span class="pill-name">${det.name}</span>
                    ${classBadge}
                </div>
                <div class="conf-badge" style="background:${confColor};">${Math.round(det.confidence * 100)}%</div>
            </div>
            ${details}
        </div>`;
    });

    // 약상자 저장 버튼
    html += `<button class="btn-save-pillbox" onclick="saveToBox()">💊 약상자에 저장</button>`;

    container.innerHTML = html;
}

// 약상자 저장
function saveToBox() {
    if (lastDetections.length === 0) return;

    const symptoms = new Map();
    const pills = lastDetections.map(det => {
        if (det.info && det.info.class_no) {
            const s = classToSymptom(det.info.class_no);
            symptoms.set(s.label, s);
        }
        return {
            name: det.name,
            confidence: det.confidence,
            info: det.info || null,
        };
    });

    const symptomList = [...symptoms.values()];
    const mainSymptom = symptomList.length > 0 ? symptomList[0] : { emoji: '💊', label: '일반' };

    const entry = {
        id: Date.now(),
        date: new Date().toISOString().split('T')[0],
        symptom: mainSymptom,
        pills: pills,
    };

    const box = getPillBox();
    box.unshift(entry);
    savePillBox(box);

    showToast('💊 약상자에 저장되었습니다');
}

// 약상자 렌더링
function renderPillBox() {
    const container = document.getElementById('pillbox-content');
    const box = getPillBox();

    if (box.length === 0) {
        container.innerHTML = '<div class="empty-state">저장된 약이 없습니다.<br>약을 검출한 후 저장해보세요.</div>';
        return;
    }

    const today = new Date();
    let html = '';

    box.forEach((entry, idx) => {
        const savedDate = new Date(entry.date);
        const dDay = Math.floor((today - savedDate) / (1000 * 60 * 60 * 24));
        const pillNames = entry.pills.map(p => p.name);
        const preview = pillNames.length <= 2 ? pillNames.join(', ') : pillNames.slice(0, 2).join(', ') + ` 외 ${pillNames.length - 2}`;
        const sym = entry.symptom || { emoji: '💊', label: '일반' };

        const hasAlarm = entry.alarm ? true : false;
        const alarmInfo = hasAlarm ? `<div class="pillbox-alarm-info">🔔 하루 ${entry.alarm.times}회 알림</div>` : '';

        html += `
        <div class="pillbox-card">
            <div class="pillbox-header">
                <span class="pillbox-symptom">${sym.emoji} ${sym.label}</span>
                <span class="pillbox-dday">D+${dDay}</span>
            </div>
            <div class="pillbox-date">${entry.date} 저장</div>
            <div class="pillbox-pills">${preview}</div>
            ${alarmInfo}
            <div class="pillbox-actions">
                <button class="pillbox-btn-open" onclick="openBoxEntry(${idx})">열기</button>
                ${hasAlarm
                    ? `<button class="pillbox-btn-open" onclick="turnOffAlarm(${idx})" style="background:#FFE0E0;color:#E74C3C;">🔕 알림끄기</button>`
                    : `<button class="pillbox-btn-open" onclick="openAlarmSetting(${idx})" style="background:#FFF3E0;color:#E65100;">🔔 복약알림</button>`
                }
                <button class="pillbox-btn-delete" onclick="deleteBoxEntry(${idx})">삭제</button>
            </div>
        </div>`;
    });

    container.innerHTML = html;
}

function openBoxEntry(idx) {
    const box = getPillBox();
    const entry = box[idx];
    if (!entry) return;

    const today = new Date();
    const savedDate = new Date(entry.date);
    const dDay = Math.floor((today - savedDate) / (1000 * 60 * 60 * 24));
    const sym = entry.symptom || { emoji: '💊', label: '일반' };

    // 헤더
    document.getElementById('pillbox-detail-header').innerHTML = `
        <div style="padding:14px 20px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:20px;font-weight:700;color:#2D3436;">${sym.emoji} ${sym.label}</span>
                <span class="pillbox-dday">D+${dDay}</span>
            </div>
            <div style="font-size:12px;color:#AAA;margin-top:4px;">${entry.date} 저장 · ${entry.pills.length}개 약품</div>
        </div>
    `;

    // 약 카드 렌더
    const container = document.getElementById('pillbox-detail-content');
    const detections = entry.pills.map(p => ({
        name: p.name,
        confidence: p.confidence,
        info: p.info,
    }));

    let html = makeCategoryBanner(detections);

    detections.forEach(det => {
        const confPct = (det.confidence * 100).toFixed(1);
        const confColor = det.confidence >= 0.9 ? '#4ECDC4' : det.confidence >= 0.7 ? '#FFB74D' : '#FF8A80';

        let details = '';
        if (det.info) {
            const rows = [
                ['성분', det.info.material], ['제약사', det.info.company],
                ['구분', det.info.etc_otc],
                ['외형', ((det.info.color1 || '') + ' ' + (det.info.shape || '')).trim()],
                ['각인', [det.info.print_front, det.info.print_back].filter(Boolean).join(' / ')],
                ['설명', det.info.chart],
            ];
            rows.forEach(([label, value]) => {
                if (value) details += `<div class="pill-detail"><span class="pill-label">${label}</span><span class="pill-value">${value}</span></div>`;
            });
        }

        let pillCatId = '';
        if (det.info) {
            for (const [k, v] of Object.entries(pillDB)) {
                if (v.name === det.name) { pillCatId = k; break; }
            }
        }
        const pillImgSrc = pillCatId ? `/static/pills/${pillCatId}.png` : '';
        const pillImg = pillImgSrc ? `<img src="${pillImgSrc}" class="pill-thumb" style="width:52px;height:52px;border-radius:12px;object-fit:cover;flex-shrink:0;" onclick="openPopup('${pillImgSrc}')" onerror="this.style.display='none'">` : '';

        const classNo = det.info ? (det.info.class_no || '') : '';
        const classBadge = classNo ? `<div class="class-badge">${classNo}</div>` : '';

        html += `
        <div class="pill-card">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
                ${pillImg}
                <div style="flex:1;">
                    <span class="pill-name">${det.name}</span>
                    ${classBadge}
                </div>
                <div class="conf-badge" style="background:${confColor};">${Math.round(det.confidence * 100)}%</div>
            </div>
            ${details}
        </div>`;
    });

    container.innerHTML = html;
    showPage('pillbox-detail');
}

// 프로필
function getProfile() {
    try { return JSON.parse(localStorage.getItem('profile') || '{}'); }
    catch { return {}; }
}
function saveProfileData(data) {
    localStorage.setItem('profile', JSON.stringify(data));
}

function loadProfile() {
    const p = getProfile();
    document.getElementById('mypage-name').textContent = (p.name || '민수') + '님';
    document.getElementById('mypage-email').textContent = p.email || '';
    document.getElementById('mypage-age').textContent = p.age ? p.age + '세' : '';

    const sympContainer = document.getElementById('mypage-symptoms');
    const symptoms = p.symptoms || [];
    sympContainer.innerHTML = symptoms.map(s => `<span class="symptom-tag">🏷 ${s}</span>`).join('');

    // 홈 인사말 반영
    const greetEl = document.querySelector('.greeting-text');
    if (greetEl) greetEl.textContent = `안녕하세요, ${p.name || '민수'}님`;
}

function openEditProfile() {
    const p = getProfile();
    document.getElementById('edit-name').value = p.name || '';
    document.getElementById('edit-email').value = p.email || '';
    document.getElementById('edit-age').value = p.age || '';
    document.getElementById('edit-symptom-custom').value = p.customSymptom || '';

    const checks = document.querySelectorAll('.symptom-check input');
    const symptoms = p.symptoms || [];
    checks.forEach(cb => { cb.checked = symptoms.includes(cb.value); });

    showPage('edit-profile');
}

function saveProfile() {
    const symptoms = [];
    document.querySelectorAll('.symptom-check input:checked').forEach(cb => symptoms.push(cb.value));
    const custom = document.getElementById('edit-symptom-custom').value.trim();
    if (custom) symptoms.push(custom);

    const data = {
        name: document.getElementById('edit-name').value.trim() || '민수',
        email: document.getElementById('edit-email').value.trim(),
        age: document.getElementById('edit-age').value.trim(),
        symptoms: symptoms,
        customSymptom: custom,
    };

    saveProfileData(data);
    loadProfile();
    showToast('✅ 프로필이 저장되었습니다');
    goBack();
}

// 검출 히스토리
function addHistory(detections) {
    if (detections.length === 0) return;
    let history = [];
    try { history = JSON.parse(localStorage.getItem('detectHistory') || '[]'); } catch {}

    const names = detections.map(d => d.name);
    const avgConf = detections.reduce((sum, d) => sum + d.confidence, 0) / detections.length;
    const preview = names.length <= 2 ? names.join(', ') : names.slice(0, 2).join(', ') + ` 외 ${names.length - 2}`;

    history.unshift({
        date: new Date().toISOString().split('T')[0],
        pills: preview,
        conf: Math.round(avgConf * 100),
        count: detections.length,
    });

    if (history.length > 10) history = history.slice(0, 10);
    localStorage.setItem('detectHistory', JSON.stringify(history));
}

function renderHistory() {
    const container = document.getElementById('history-content');
    let history = [];
    try { history = JSON.parse(localStorage.getItem('detectHistory') || '[]'); } catch {}

    if (history.length === 0) {
        container.innerHTML = '<div style="text-align:center;padding:20px;color:#CCC;font-size:13px;">검출 기록이 없습니다.</div>';
        return;
    }

    container.innerHTML = history.slice(0, 10).map(h => `
        <div class="history-item">
            <div>
                <div class="history-date">${h.date}</div>
                <div class="history-pills">${h.pills}</div>
            </div>
            <div class="history-conf">${h.conf}%</div>
        </div>
    `).join('');
}

// 폰트 크기
function setFontSize(size) {
    const frame = document.getElementById('phone-frame');
    frame.style.fontSize = size === 'small' ? '13px' : size === 'large' ? '17px' : '15px';
    document.querySelectorAll('.font-btn').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    localStorage.setItem('fontSize', size);
    showToast('글씨 크기가 변경되었습니다');
}

function clearAllPillBox() {
    localStorage.removeItem('pillbox');
    showToast('🗑️ 약상자가 비워졌습니다');
}

// 홈 복약 일정 (가로 슬라이드)
const FAKE_NOW = { h: 12, m: 26 }; // fake_top 시간 고정

function getScheduleState() {
    try { return JSON.parse(localStorage.getItem('scheduleState') || '{}'); } catch { return {}; }
}
function saveScheduleState(state) {
    localStorage.setItem('scheduleState', JSON.stringify(state));
}

function renderHomeSchedule() {
    const container = document.getElementById('home-schedule');
    const box = getPillBox();
    const labels = ['아침', '점심', '저녁', '오후', '새벽'];
    const state = getScheduleState();

    const alarms = box.filter(e => e.alarm);
    if (alarms.length === 0) {
        container.innerHTML = '<div class="schedule-empty">약상자에서 복약 알림을 설정해보세요</div>';
        return;
    }

    // 전체 스케줄 수집 + 시간순 정렬
    const allSchedules = [];
    alarms.forEach(entry => {
        const pillNames = entry.pills.map(p => p.name);
        const preview = pillNames.length <= 2 ? pillNames.join(', ') : pillNames[0] + ' 외 ' + (pillNames.length - 1);

        entry.alarm.schedules.forEach((time, i) => {
            const [h, m] = time.split(':').map(Number);
            const key = entry.id + '_' + i;
            const done = state[key] === true;
            const nowMin = FAKE_NOW.h * 60 + FAKE_NOW.m;
            const schedMin = h * 60 + m;
            const isPast = schedMin <= nowMin;

            let status = 'safe';
            if (done) status = 'done';
            else if (isPast) status = 'warning';

            allSchedules.push({
                key, time, h, m, preview, label: labels[i] || (i + 1) + '회',
                status, done, isPast, entryId: entry.id,
            });
        });
    });

    allSchedules.sort((a, b) => (a.h * 60 + a.m) - (b.h * 60 + b.m));

    let html = '';
    let firstActiveIdx = -1;

    allSchedules.forEach((s, idx) => {
        const ampm = s.h < 12 ? 'AM' : 'PM';
        const h12 = s.h % 12 || 12;
        const statusMsg = s.done ? '' : s.isPast ? '<div class="schedule-status">⚠️ 복용 시간이 지났습니다</div>' : '';
        const doneBadge = s.done ? '<div class="schedule-done-badge">✓ 복용완료</div>' : '';
        const clickAction = s.done ? '' : `onclick="markDone('${s.key}')"`;

        if (!s.done && firstActiveIdx < 0) firstActiveIdx = idx;

        html += `
        <div class="schedule-card ${s.status}" ${clickAction} id="sched-${idx}">
            <div style="display:flex;align-items:center;">
                <span class="schedule-time">${h12}:${String(s.m).padStart(2, '0')}<br>${ampm}</span>
                <div class="schedule-info">
                    <div class="schedule-pills">${s.preview}</div>
                    <div class="schedule-label">${s.label} 복용</div>
                    ${statusMsg}
                    ${doneBadge}
                </div>
            </div>
        </div>`;
    });

    container.innerHTML = html;

    // 현재 활성 카드로 스크롤 + 슬라이더 초기화
    setTimeout(() => {
        if (firstActiveIdx >= 0) {
            const activeCard = document.getElementById('sched-' + firstActiveIdx);
            if (activeCard) container.scrollLeft = activeCard.offsetLeft - 20;
        }
        initSlider();
    }, 100);
}

// 폴더 탭 전환
function switchFolder(idx, el) {
    document.querySelectorAll('.folder-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.folder-panel').forEach(p => p.classList.remove('active'));
    el.classList.add('active');
    document.getElementById('folder-' + idx).classList.add('active');
}

// 좌우 버튼 스크롤
function scrollSchedule(dir) {
    const slider = document.getElementById('home-schedule');
    if (!slider) return;
    slider.scrollBy({ left: dir * 272, behavior: 'smooth' });
}

// 슬라이더 드래그 (터치 + 마우스)
function initSlider() {
    const slider = document.getElementById('home-schedule');
    if (!slider) return;

    let isDown = false;
    let startX, scrollLeft, startTime, velocity;

    const onStart = (x) => {
        isDown = true;
        dragMoved = false;
        startX = x;
        scrollLeft = slider.scrollLeft;
        startTime = Date.now();
        velocity = 0;
        slider.style.scrollBehavior = 'auto';
        slider.style.cursor = 'grabbing';
    };

    const onMove = (x) => {
        if (!isDown) return;
        const dx = x - startX;
        if (Math.abs(dx) > 5) dragMoved = true;
        const dt = Date.now() - startTime;
        velocity = dx / (dt || 1);
        slider.scrollLeft = scrollLeft - dx;
    };

    const onEnd = () => {
        if (!isDown) return;
        isDown = false;
        slider.style.cursor = 'grab';
        // 관성
        const momentum = velocity * 200;
        slider.style.scrollBehavior = 'smooth';
        slider.scrollLeft -= momentum;
    };

    // 마우스
    slider.addEventListener('mousedown', e => { e.preventDefault(); onStart(e.pageX); });
    slider.addEventListener('mousemove', e => onMove(e.pageX));
    slider.addEventListener('mouseup', onEnd);
    slider.addEventListener('mouseleave', onEnd);

    // 터치
    slider.addEventListener('touchstart', e => onStart(e.touches[0].pageX), { passive: true });
    slider.addEventListener('touchmove', e => onMove(e.touches[0].pageX), { passive: true });
    slider.addEventListener('touchend', onEnd);

    slider.style.cursor = 'grab';
}

let dragMoved = false;

function markDone(key) {
    if (dragMoved) { dragMoved = false; return; }
    showConfirm('💊 복용 완료하시겠습니까?', () => {
        const state = getScheduleState();
        state[key] = true;
        saveScheduleState(state);
        showToast('✅ 복용 완료');
        renderHomeSchedule();
    });
}

function showConfirm(msg, onOk) {
    const popup = document.getElementById('confirm-popup');
    document.getElementById('confirm-msg').textContent = msg;
    const okBtn = document.getElementById('confirm-ok-btn');
    okBtn.onclick = () => { closeConfirm(); onOk(); };
    popup.classList.add('active');
}
function closeConfirm() {
    document.getElementById('confirm-popup').classList.remove('active');
}

// AI 약사
const pharmacists = {
    kim: { name: '김원장 약사', img: '/static/img/kim_won_jang.png', greeting: '안녕하세요. 김원장입니다. 궁금한 점을 꼼꼼히 알려드리겠습니다.' },
    lee: { name: '이준 약사', img: '/static/img/lee_jun.png', greeting: '안녕하세요! 이준 약사입니다. 편하게 물어보세요 😊' },
    park: { name: '박미소 약사', img: '/static/img/park_miso.png', greeting: '어서오세요~ 박미소 약사예요. 어디가 불편하신가요, 어머님?' },
    choi: { name: '최유진 약사', img: '/static/img/choi_yujin.png', greeting: '안녕하세요. 최유진 약사입니다. 정확하게 답변 드리겠습니다.' },
};
let currentPharmacist = null;

// 홈 상단 버튼: 항상 약사 선택 → 선택 후 채팅
function openAiSelect() {
    showPage('ai');
}

// 하단 탭: 약사 있으면 바로 채팅, 없으면 선택
function openAiTab() {
    if (currentPharmacist) {
        document.getElementById('chat-pharmacist-img').src = currentPharmacist.img;
        document.getElementById('chat-pharmacist-name').textContent = currentPharmacist.name;
        if (document.getElementById('chat-messages').children.length === 0) {
            addChatBubble(currentPharmacist.greeting, 'bot');
        }
        showPage('ai-chat');
    } else {
        showPage('ai');
    }
}

function selectPharmacist(key) {
    currentPharmacist = pharmacists[key];
    localStorage.setItem('selectedPharmacist', key);

    // 채팅 헤더
    document.getElementById('chat-pharmacist-img').src = currentPharmacist.img;
    document.getElementById('chat-pharmacist-name').textContent = currentPharmacist.name;
    document.getElementById('chat-messages').innerHTML = '';
    addChatBubble(currentPharmacist.greeting, 'bot');

    // 하단 탭 아이콘 변경 + 컬러 유지
    const tabIcon = document.getElementById('tab-ai-icon');
    tabIcon.src = currentPharmacist.img;
    tabIcon.classList.add('selected');

    // 마이페이지 반영
    document.getElementById('mypage-pharmacist-img').src = currentPharmacist.img;
    document.getElementById('mypage-pharmacist-name').textContent = currentPharmacist.name;

    // 어디서 왔는지에 따라 다른 동작
    const fromPage = pageHistory.length >= 2 ? pageHistory[pageHistory.length - 2] : 'home';

    if (fromPage === 'mypage') {
        showToast(`${currentPharmacist.name}으로 변경되었습니다`);
        showPage('mypage');
    } else {
        showPage('ai-chat');
    }
}

function addChatBubble(text, type) {
    const container = document.getElementById('chat-messages');
    const row = document.createElement('div');
    row.className = 'chat-row ' + type;

    if (type === 'bot' && currentPharmacist) {
        const avatar = document.createElement('img');
        avatar.className = 'chat-avatar';
        avatar.src = currentPharmacist.img;
        row.appendChild(avatar);
    }

    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble ' + type;
    bubble.textContent = text;
    row.appendChild(bubble);

    container.appendChild(row);
    container.scrollTop = container.scrollHeight;
}

let chatHistory = [];

async function sendChat() {
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg) return;
    addChatBubble(msg, 'user');
    chatHistory.push({ type: 'user', text: msg });
    input.value = '';
    input.disabled = true;

    addChatBubble('...', 'bot');
    const loadingBubble = document.getElementById('chat-messages').lastChild;

    const profile = getProfile();
    const pharmacistKey = localStorage.getItem('selectedPharmacist') || 'kim';

    const pillbox = getPillBox();
    const currentMeds = [];
    pillbox.forEach(entry => {
        entry.pills.forEach(p => {
            if (p.name && !currentMeds.includes(p.name)) currentMeds.push(p.name);
        });
    });

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: msg,
                pharmacist: pharmacistKey,
                history: chatHistory.slice(-10),
                user_symptoms: profile.symptoms || [],
                current_meds: currentMeds,
            }),
        });
        const data = await res.json();
        loadingBubble.remove();
        addChatBubble(data.reply, 'bot');
        chatHistory.push({ type: 'bot', text: data.reply });
    } catch (err) {
        loadingBubble.remove();
        addChatBubble('연결에 실패했습니다. 다시 시도해주세요.', 'bot');
    }
    input.disabled = false;
    input.focus();
}

// 복약 알림
let currentAlarmIdx = -1;
let currentAlarmTimes = 3;
const defaultTimes = ['09:00', '13:00', '21:00', '17:00', '07:00'];

function openAlarmSetting(idx) {
    currentAlarmIdx = idx;
    const box = getPillBox();
    const entry = box[idx];

    if (entry && entry.alarm) {
        currentAlarmTimes = entry.alarm.times;
        renderAlarmInputs(entry.alarm.schedules);
    } else {
        currentAlarmTimes = 3;
        renderAlarmInputs(defaultTimes.slice(0, 3));
    }

    updateAlarmTimeBtns();
    showPage('alarm');
}

function setAlarmTimes(n) {
    currentAlarmTimes = n;
    updateAlarmTimeBtns();

    const existing = [];
    document.querySelectorAll('.alarm-time-input').forEach(input => existing.push(input.value));
    while (existing.length < n) existing.push(defaultTimes[existing.length] || '12:00');
    renderAlarmInputs(existing.slice(0, n));
}

function updateAlarmTimeBtns() {
    document.querySelectorAll('.alarm-time-btn').forEach((btn, i) => {
        btn.classList.toggle('active', (i + 1) === currentAlarmTimes);
    });
}

function renderAlarmInputs(schedules) {
    const labels = ['아침', '점심', '저녁', '오후', '새벽'];
    const container = document.getElementById('alarm-time-inputs');
    container.innerHTML = schedules.map((time, i) => `
        <div class="alarm-time-row">
            <span class="alarm-time-label">${labels[i] || (i+1)+'회'}</span>
            <input type="time" class="alarm-time-input" value="${time}">
        </div>
    `).join('');
}

function saveAlarm() {
    const schedules = [];
    document.querySelectorAll('.alarm-time-input').forEach(input => schedules.push(input.value));

    const box = getPillBox();
    if (currentAlarmIdx >= 0 && currentAlarmIdx < box.length) {
        box[currentAlarmIdx].alarm = {
            times: currentAlarmTimes,
            schedules: schedules,
        };
        savePillBox(box);
        showToast('🔔 복약 알림이 설정되었습니다');
        renderHomeSchedule();
        showPage('pillbox');
        renderPillBox();
    }
}


function turnOffAlarm(idx) {
    const box = getPillBox();
    if (box[idx]) {
        delete box[idx].alarm;
        savePillBox(box);
        showToast('🔕 복약 알림이 해제되었습니다');
        renderPillBox();
        renderHomeSchedule();
    }
}

function deleteBoxEntry(idx) {
    const box = getPillBox();
    box.splice(idx, 1);
    savePillBox(box);
    renderPillBox();
    showToast('🗑️ 삭제되었습니다');
}
