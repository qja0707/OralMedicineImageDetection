// 페이지 전환
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById('page-' + pageId).classList.add('active');
    // 스크롤 맨 위로
    document.getElementById('phone-frame').scrollTop = 0;
}

// 카메라
function openCamera() {
    document.getElementById('camera-input').click();
}

// 파일 업로드
function openUpload() {
    document.getElementById('upload-input').click();
}

// 파일 선택 이벤트
document.getElementById('camera-input').addEventListener('change', handleFile);
document.getElementById('upload-input').addEventListener('change', handleFile);

async function handleFile(e) {
    const file = e.target.files[0];
    if (!file) return;

    // 로딩 화면
    showPage('loading');

    // API 호출
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/api/detect', {
            method: 'POST',
            body: formData,
        });
        const data = await res.json();

        if (data.error) {
            alert(data.error);
            showPage('home');
            return;
        }

        // 결과 이미지
        document.getElementById('result-image').src = 'data:image/jpeg;base64,' + data.image;

        // 카드 생성
        renderCards(data.detections, data.count);

        showPage('result');
    } catch (err) {
        alert('검출 실패: ' + err.message);
        showPage('home');
    }

    // 입력 초기화 (같은 파일 다시 선택 가능)
    e.target.value = '';
}

function renderCards(detections, count) {
    const container = document.getElementById('result-cards');

    if (count === 0) {
        container.innerHTML = '<div style="text-align:center;padding:30px;color:#999;">검출된 알약이 없습니다.</div>';
        return;
    }

    let html = `<div class="result-count">${count}개 알약 검출</div>`;

    detections.forEach(det => {
        const confPct = (det.confidence * 100).toFixed(1);
        const confColor = det.confidence >= 0.9 ? '#4ECDC4' : det.confidence >= 0.7 ? '#FFB74D' : '#FF8A80';

        let details = '';
        if (det.info) {
            const rows = [
                ['성분', det.info.material],
                ['제약사', det.info.company],
                ['분류', det.info.class_no],
                ['구분', det.info.etc_otc],
                ['외형', ((det.info.color1 || '') + ' ' + (det.info.shape || '')).trim()],
                ['각인', [det.info.print_front, det.info.print_back].filter(Boolean).join(' / ')],
                ['설명', det.info.chart],
            ];
            rows.forEach(([label, value]) => {
                if (value) {
                    details += `<div class="pill-detail"><span class="pill-label">${label}</span><span class="pill-value">${value}</span></div>`;
                }
            });
        }

        html += `
        <div class="pill-card">
            <div class="pill-header">
                <span class="pill-name">${det.name}</span>
                <span class="pill-conf" style="color:${confColor}">${confPct}%</span>
            </div>
            <div class="conf-bar">
                <div class="conf-fill" style="width:${confPct}%;background:${confColor}"></div>
            </div>
            ${details}
        </div>`;
    });

    container.innerHTML = html;
}
