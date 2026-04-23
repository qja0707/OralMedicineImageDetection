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

    if (pageHistory[pageHistory.length - 1] !== pageId) pageHistory.push(pageId);

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
function openCamera() { document.getElementById('camera-input').click(); }
function openUpload() { document.getElementById('upload-input').click(); }

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
                <button class="pillbox-btn-open" onclick="openAlarmSetting(${idx})" style="background:#FFF3E0;color:#E65100;">🔔 복약알림</button>
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

    if (history.length > 20) history = history.slice(0, 20);
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

// 홈 복약 일정
function renderHomeSchedule() {
    const container = document.getElementById('home-schedule');
    const box = getPillBox();
    const labels = ['아침', '점심', '저녁', '오후', '새벽'];

    const alarms = box.filter(e => e.alarm);
    if (alarms.length === 0) {
        container.innerHTML = '<div class="schedule-card"><span class="schedule-text" style="color:#BBB;">약상자에서 복약 알림을 설정해보세요</span></div>';
        return;
    }

    let html = '';
    alarms.forEach(entry => {
        const pillNames = entry.pills.map(p => p.name);
        const preview = pillNames.length <= 2 ? pillNames.join(', ') : pillNames[0] + ` 외 ${pillNames.length - 1}`;
        const sym = entry.symptom || { emoji: '💊', label: '' };

        entry.alarm.schedules.forEach((time, i) => {
            const label = labels[i] || (i+1) + '회';
            const h = parseInt(time.split(':')[0]);
            const ampm = h < 12 ? 'AM' : 'PM';
            const displayTime = time;

            html += `
            <div class="schedule-card">
                <span class="schedule-time">${displayTime}<br>${ampm}</span>
                <span class="schedule-text">${preview} - ${label} 복용</span>
                <span class="schedule-arrow">›</span>
            </div>`;
        });
    });

    container.innerHTML = html;
}

// AI약사 탭 (준비중)
function openAiTab() {
    showToast('💬 AI약사 기능은 준비 중입니다');
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


function deleteBoxEntry(idx) {
    const box = getPillBox();
    box.splice(idx, 1);
    savePillBox(box);
    renderPillBox();
    showToast('🗑️ 삭제되었습니다');
}
