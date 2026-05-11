// 사용 로그
function appLog(action, detail) {
    const logs = JSON.parse(localStorage.getItem('appLogs') || '[]');
    const now = new Date();
    const ts = now.toLocaleString('ko-KR', { hour12: false });
    logs.push(`[${ts}] ${action}: ${detail || ''}`);
    if (logs.length > 500) logs.splice(0, logs.length - 500);
    localStorage.setItem('appLogs', JSON.stringify(logs));
}
function exportLogs() {
    const logs = JSON.parse(localStorage.getItem('appLogs') || '[]');
    if (logs.length === 0) { showToast('저장된 로그가 없습니다'); return; }
    const text = 'PillScope 사용 로그\n' + '='.repeat(50) + '\n' + logs.join('\n');
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'pillscope_log_' + new Date().toISOString().split('T')[0] + '.txt';
    a.click();
    URL.revokeObjectURL(a.href);
}

// 모델 관리
async function loadModels() {
    try {
        const res = await fetch('/api/models');
        const data = await res.json();
        const select = document.getElementById('model-select');
        const status = document.getElementById('model-status');
        select.innerHTML = data.models.map(m =>
            `<option value="${m}" ${m === data.current ? 'selected' : ''}>${m}</option>`
        ).join('');
        status.textContent = data.models.length > 0
            ? `총 ${data.models.length}개 모델 · 현재: ${data.current}`
            : '모델 없음';
    } catch (e) {
        document.getElementById('model-status').textContent = '모델 목록 로드 실패';
    }
}

async function switchModel() {
    const name = document.getElementById('model-select').value;
    const status = document.getElementById('model-status');
    status.textContent = '모델 교체 중... (최대 1분 소요)';
    try {
        const res = await fetch('/api/models/switch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        const data = await res.json();
        if (data.success) {
            status.textContent = `모델 교체 완료: ${data.current}`;
            showToast(`모델이 ${data.current}로 변경되었습니다`);
            appLog('모델교체', data.current);
        } else {
            status.textContent = `교체 실패: ${data.error}`;
            showToast('모델 교체 실패');
        }
    } catch (e) {
        status.textContent = '모델 교체 실패';
        showToast('모델 교체 실패');
    }
}

// 병용금기 테이블
const contraindicationMap = [
    ['아질렉트정', '졸로푸트정', '치명적 세로토닌 증후군 위험 (절대 병용금기)'],
    ['아질렉트정', '울트라셋이알서방정', '발작 및 세로토닌 증후군 위험 (병용금기)'],
    ['아질렉트정', '보령부스파정', '위험한 혈압 상승 가능 (병용금기)'],
    ['플라빅스정', '에스원엠프정', '에스오메프라졸이 플라빅스 약효를 심각하게 저하'],
    ['플라빅스정', '비모보정', '에스오메프라졸 성분이 혈전 예방 효과 방해'],
    ['마도파정', '아빌리파이정', '도파민 차단으로 파킨슨약 효과 상쇄'],
];

// 제네릭(동일 성분) 테이블
const genericMap = [
    ['리피토정', '아토르바정', '아토르바스타틴'],
    ['리피토정', '리피로우정', '아토르바스타틴'],
    ['리피토정', '아토젯정', '아토르바스타틴'],
    ['아토르바정', '리피로우정', '아토르바스타틴'],
    ['아토르바정', '아토젯정', '아토르바스타틴'],
    ['리피로우정', '아토젯정', '아토르바스타틴'],
    ['크레스토정', '로수젯정', '로수바스타틴'],
    ['크레스토정', '로수바미브정', '로수바스타틴'],
    ['로수젯정', '로수바미브정', '로수바스타틴+에제티미브'],
    ['아토젯정', '로수젯정', '에제티미브'],
    ['아토젯정', '로수바미브정', '에제티미브'],
    ['가바토파정', '동아가바펜틴정', '가바펜틴'],
    ['리리카캡슐', '카발린캡슐', '프레가발린'],
    ['종근당글리아티린', '콜리네이트연질캡슐', '콜린알포세레이트'],
    ['종근당글리아티린', '글리아타민연질캡슐', '콜린알포세레이트'],
    ['종근당글리아티린', '글리틴정', '콜린알포세레이트'],
    ['콜리네이트연질캡슐', '글리아타민연질캡슐', '콜린알포세레이트'],
    ['콜리네이트연질캡슐', '글리틴정', '콜린알포세레이트'],
    ['글리아타민연질캡슐', '글리틴정', '콜린알포세레이트'],
    ['노바스크정', '엑스포지정', '암로디핀'],
    ['노바스크정', '아모잘탄정', '암로디핀'],
    ['노바스크정', '트윈스타정', '암로디핀'],
    ['엑스포지정', '아모잘탄정', '암로디핀'],
    ['엑스포지정', '트윈스타정', '암로디핀'],
    ['아모잘탄정', '트윈스타정', '암로디핀'],
    ['자누비아정', '자누메트정', '시타글립틴'],
    ['자누비아정', '자누메트엑스알', '시타글립틴'],
    ['자누메트정', '자누메트엑스알', '시타글립틴+메트포르민'],
    ['트라젠타정', '트라젠타듀오정', '리나글립틴'],
    ['자누메트정', '트라젠타듀오정', '메트포르민'],
    ['자누메트정', '제미메트서방정', '메트포르민'],
    ['자누메트엑스알', '트라젠타듀오정', '메트포르민'],
    ['자누메트엑스알', '제미메트서방정', '메트포르민'],
    ['트라젠타듀오정', '제미메트서방정', '메트포르민'],
    ['에스원엠프정', '비모보정', '에스오메프라졸'],
    ['써스펜8시간이알서방정', '울트라셋이알서방정', '아세트아미노펜'],
    ['리렉스펜정', '맥시부펜이알정', '덱시부프로펜'],
];

function checkPillboxWarnings(pillNames) {
    const warnings = [];
    for (const [a, b, reason] of contraindicationMap) {
        const hasA = pillNames.some(n => n.includes(a));
        const hasB = pillNames.some(n => n.includes(b));
        if (hasA && hasB) warnings.push({ type: 'danger', a, b, reason });
    }
    for (const [a, b, ingredient] of genericMap) {
        const hasA = pillNames.some(n => n.includes(a));
        const hasB = pillNames.some(n => n.includes(b));
        if (hasA && hasB) warnings.push({ type: 'duplicate', a, b, reason: `동일 성분(${ingredient}) 중복 복용` });
    }
    return warnings;
}

function getAllPillboxPillNames() {
    const box = getPillBox();
    const allNames = [];
    box.forEach(entry => entry.pills.forEach(p => { if (p.name) allNames.push(p.name); }));
    return allNames;
}

function getWarningsForDrug(drugName, allWarnings) {
    return allWarnings.filter(w => drugName.includes(w.a) || drugName.includes(w.b));
}

const warningColors = ['#E53935','#D81B60','#8E24AA','#5E35B1','#3949AB','#1E88E5','#00897B','#F4511E'];

const pillDescMap = {
    '보령부스파정': { effect: '마음을 보호하고 불안을 줄이는 약', caution: '졸릴 수 있으니 운전은 조심하세요' },
    '뮤테란캡슐': { effect: '기관지를 보호하고 가래 빼는 약', caution: '물을 많이 마시면 훨씬 좋아요' },
    '일양하이트린정': { effect: '방광을 보호하고 소변을 돕는 약', caution: '갑자기 일어나면 어지러울 수 있어요' },
    '기넥신에프정': { effect: '혈관을 보호하고 순환을 돕는 약', caution: '수술 전에는 꼭 의사에게 말하세요' },
    '무코스타정': { effect: '위를 보호하고 속쓰림 줄이는 약', caution: '식후에 바로 드시는 게 좋아요' },
    '알드린정': { effect: '뼈를 보호하고 골절을 막는 약', caution: '약 먹고 30분 동안은 눕지 마세요' },
    '뉴로메드정': { effect: '뇌를 보호하고 기억력 돕는 약', caution: '늦은 밤에 먹으면 잠이 깰 수 있어요' },
    '에어탈정': { effect: '관절을 보호하고 통증을 줄이는 약', caution: '속이 쓰릴 수 있으니 식후에 드세요' },
    '리렉스펜정': { effect: '몸을 보호하고 열을 내려주는 약', caution: '술 마시고 이 약을 드시면 안 돼요' },
    '아빌리파이정': { effect: '마음을 보호하고 기분을 맞추는 약', caution: '마음대로 약을 끊으면 절대 안 돼요' },
    '다보타민큐정': { effect: '몸을 보호하고 피로를 줄이는 약', caution: '소변 색이 노랗게 변할 수 있어요' },
    '써스펜8시간이알서방정': { effect: '몸을 보호하고 열을 내려주는 약', caution: '술 마신 날에는 절대 피하세요' },
    '에빅사정': { effect: '뇌를 보호하고 치매를 늦추는 약', caution: '어지러울 수 있으니 넘어짐 조심하세요' },
    '리피토정': { effect: '혈관을 보호하고 기름을 빼는 약', caution: '자몽주스와 함께 드시면 안 돼요' },
    '크레스토정': { effect: '혈관을 보호하고 피 맑게 하는 약', caution: '근육이 아프면 바로 병원에 가세요' },
    '가바토파정': { effect: '신경을 보호하고 통증을 줄이는 약', caution: '임의로 약을 끊으면 발작할 수 있어요' },
    '동아가바펜틴정': { effect: '신경을 보호하고 저림을 줄이는 약', caution: '졸음이 올 수 있으니 운전 조심하세요' },
    '오마코연질캡슐': { effect: '심장을 보호하고 기름을 빼는 약', caution: '생선 알레르기 있으면 조심하세요' },
    '리리카캡슐': { effect: '신경을 보호하고 저림을 줄이는 약', caution: '살이 찌거나 손발이 부을 수 있어요' },
    '종근당글리아티린': { effect: '뇌를 보호하고 기억력 돕는 약', caution: '아침이나 낮에 드시는 게 좋아요' },
    '콜리네이트연질캡슐': { effect: '뇌를 보호하고 기억력 돕는 약', caution: '속이 메스꺼울 땐 식후에 드세요' },
    '트루비타정': { effect: '몸을 보호하고 피로를 줄이는 약', caution: '빈속에 먹으면 속이 쓰릴 수 있어요' },
    '스토가정': { effect: '위를 보호하고 위산을 줄이는 약', caution: '씹거나 부수지 말고 통째로 삼키세요' },
    '노바스크정': { effect: '심장을 보호하고 혈압을 내리는 약', caution: '자몽주스와 함께 드시지 마세요' },
    '마도파정': { effect: '뇌를 보호하고 몸떨림 줄이는 약', caution: '고기 먹은 직후에는 피해서 드세요' },
    '플라빅스정': { effect: '혈관을 보호하고 피떡을 막는 약', caution: '멍이 잘 들 수 있으니 조심하세요' },
    '엑스포지정': { effect: '심장을 보호하고 혈압을 내리는 약', caution: '갑자기 일어나면 어지러울 수 있어요' },
    '아토르바정': { effect: '혈관을 보호하고 기름을 빼는 약', caution: '저녁에 드시는 것이 가장 좋아요' },
    '라비에트정': { effect: '식도를 보호하고 위산을 줄이는 약', caution: '아침 식사 30분 전에 드시면 좋아요' },
    '리피로우정': { effect: '혈관을 보호하고 기름을 빼는 약', caution: '근육이 심하게 아프면 병원 꼭 가세요' },
    '자누비아정': { effect: '혈관을 보호하고 당수치 내리는 약', caution: '어지럽고 식은땀 나면 단것을 드세요' },
    '맥시부펜이알정': { effect: '관절을 보호하고 통증을 줄이는 약', caution: '위가 상할 수 있으니 꼭 식후에 드세요' },
    '놀텍정': { effect: '위를 보호하고 위산을 줄이는 약', caution: '씹거나 부수지 말고 그냥 삼키세요' },
    '자누메트정': { effect: '혈관을 보호하고 당수치 내리는 약', caution: '식사와 함께 드셔야 속이 편합니다' },
    '큐시드정': { effect: '위를 보호하고 소화를 돕는 약', caution: '우유와 함께 드시지 마세요' },
    '아모잘탄정': { effect: '심장을 보호하고 혈압을 내리는 약', caution: '자몽주스와 함께 복용하지 마세요' },
    '트윈스타정': { effect: '심장을 보호하고 혈압을 내리는 약', caution: '약을 미리 까두면 녹으니 주의하세요' },
    '카나브정': { effect: '심장을 보호하고 혈압을 내리는 약', caution: '어지러울 수 있으니 천천히 일어나세요' },
    '울트라셋이알서방정': { effect: '신경을 보호하고 통증을 줄이는 약', caution: '간이 상할 수 있으니 절대 술 먹지 마요' },
    '졸로푸트정': { effect: '마음을 보호하고 불안을 줄이는 약', caution: '맘대로 끊지 말고 꾸준히 드셔야 해요' },
    '트라젠타정': { effect: '혈관을 보호하고 당수치 내리는 약', caution: '매일 같은 시간에 꾸준히 챙겨 드세요' },
    '비모보정': { effect: '관절을 보호하고 통증을 줄이는 약', caution: '쪼개거나 씹지 말고 통째로 삼키세요' },
    '레일라정': { effect: '연골을 보호하고 통증을 줄이는 약', caution: '속이 쓰릴 수 있으니 식후에 복용하세요' },
    '리바로정': { effect: '혈관을 보호하고 기름을 빼는 약', caution: '이유 없이 근육이 아프면 꼭 병원 가요' },
    '트라젠타듀오정': { effect: '혈관을 보호하고 당수치 내리는 약', caution: '설사를 할 수 있으니 식후에 드세요' },
    '아질렉트정': { effect: '뇌를 보호하고 몸떨림 줄이는 약', caution: '감기약 먹기 전에 의사에게 꼭 물어봐요' },
    '자누메트엑스알': { effect: '혈관을 보호하고 당수치 내리는 약', caution: '약 껍질이 변으로 나와도 정상입니다' },
    '글리아타민연질캡슐': { effect: '뇌를 보호하고 기억력 돕는 약', caution: '캡슐을 터뜨리지 말고 그대로 넘기세요' },
    '신바로정': { effect: '뼈를 보호하고 통증을 줄이는 약', caution: '위가 아플 수 있으니 식후에 꼭 드세요' },
    '에스원엠프정': { effect: '식도를 보호하고 위산을 줄이는 약', caution: '아침 밥 먹기 30분 전에 꼭 드세요' },
    '글리틴정': { effect: '뇌를 보호하고 기억력 돕는 약', caution: '낮에 드시는 게 밤잠에 방해가 안 돼요' },
    '제미메트서방정': { effect: '혈관을 보호하고 당수치 내리는 약', caution: '식사와 함께 드셔야 부작용이 적어요' },
    '아토젯정': { effect: '혈관을 보호하고 기름을 빼는 약', caution: '자몽이나 자몽주스는 피해주세요' },
    '로수젯정': { effect: '혈관을 보호하고 기름을 빼는 약', caution: '근육이 심하게 아프면 병원에 꼭 가세요' },
    '로수바미브정': { effect: '혈관을 보호하고 기름을 빼는 약', caution: '저녁에 규칙적으로 드셔주세요' },
    '카발린캡슐': { effect: '신경을 보호하고 저림을 줄이는 약', caution: '갑자기 끊으면 안 되니 꾸준히 드세요' },
};

function getPillDesc(name) {
    for (const [key, val] of Object.entries(pillDescMap)) {
        if (name.includes(key)) return val;
    }
    return null;
}


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

    const savedPh = localStorage.getItem('selectedPharmacist');
    if (savedPh && pharmacists[savedPh]) {
        currentPharmacist = pharmacists[savedPh];
        chatHistory = loadChatHistory(savedPh);
        document.getElementById('tab-ai-icon').src = currentPharmacist.img;
        document.getElementById('mypage-pharmacist-img').src = currentPharmacist.img;
        document.getElementById('mypage-pharmacist-name').textContent = currentPharmacist.name;
        document.getElementById('chat-pharmacist-img').src = currentPharmacist.img;
        document.getElementById('chat-pharmacist-name').textContent = currentPharmacist.name;
        restoreChatBubbles();
    }
    const fs = localStorage.getItem('fontSize');
    if (fs) {
        const frame = document.getElementById('phone-frame');
        frame.style.fontSize = fs === 'small' ? '13px' : fs === 'large' ? '17px' : '15px';
    }
});

// 증상 동의어 → 약 이름 키워드 매핑 (Gemini 생성 기반)
const symptomMap = {
    '불안장애': ['보령부스파'], '불안증': ['보령부스파'], '가슴두근거려': ['보령부스파'],
    '초조하고불안해': ['보령부스파'], '긴장완화': ['보령부스파'], '불안': ['보령부스파', '졸로푸트'],
    '기관지염': ['뮤테란'], '객담': ['뮤테란'], '기침나고답답해': ['뮤테란'],
    '목에가래걸려': ['뮤테란'], '가래끓어': ['뮤테란'], '가래': ['뮤테란'], '기침': ['뮤테란'],
    '감기': ['써스펜', '뮤테란', '맥시부펜', '리렉스펜'],
    '몸살': ['써스펜', '맥시부펜', '리렉스펜'], '몸살감기': ['써스펜', '맥시부펜'],
    '전립선비대증': ['일양하이트린'], '오줌발약해': ['일양하이트린'], '잔뇨감': ['일양하이트린'],
    '오줌자주마려워': ['일양하이트린'], '전립선': ['일양하이트린'], '배뇨': ['일양하이트린'],
    '말초동맥질환': ['기넥신'], '혈액순환': ['기넥신', '플라빅스', '오마코'],
    '손발저려': ['기넥신', '동아가바펜틴'], '이명': ['기넥신'], '자꾸깜빡해': ['기넥신', '종근당글리아티린'],
    '어지러움': ['기넥신'],
    '위궤양': ['무코스타', '스토가'], '위염': ['무코스타'], '속쓰려': ['무코스타', '라비에트', '놀텍', '에스원엠프'],
    '소화안돼': ['무코스타', '스토가', '큐시드'], '위보호제': ['무코스타'],
    '속쓰림': ['무코스타', '라비에트', '놀텍', '에스원엠프', '비모보'],
    '배아파': ['스토가', '무코스타', '비모보'], '배아픔': ['스토가', '무코스타', '비모보'],
    '복통': ['스토가', '무코스타', '비모보'], '소화': ['스토가', '무코스타', '큐시드'],
    '골다공증': ['알드린'], '골절예방': ['알드린'], '뼈약해': ['알드린'],
    '뼈시려': ['알드린'], '폐경기뼈건강': ['알드린'],
    '치매예방': ['뉴로메드', '종근당글리아티린', '콜리네이트', '글리아타민', '글리틴'],
    '인지기능장애': ['뉴로메드', '에빅사'], '기억력감퇴': ['뉴로메드', '글리아타민'],
    '깜빡깜빡해': ['뉴로메드', '글리틴'], '건망증심해': ['뉴로메드', '글리틴'],
    '치매': ['에빅사', '글리아타민', '글리틴', '콜리네이트', '종근당글리아티린', '뉴로메드'],
    '기억력': ['글리아타민', '글리틴', '콜리네이트', '기넥신', '뉴로메드'],
    '류마티스관절염': ['에어탈', '비모보'], '근육통': ['에어탈'],
    '허리아파': ['에어탈', '비모보', '신바로', '울트라셋'], '뼈마디쑤셔': ['에어탈', '써스펜', '비모보', '레일라', '신바로'],
    '소염진통제': ['에어탈', '레일라', '신바로'],
    '관절': ['에어탈', '신바로', '레일라', '비모보'], '관절통': ['에어탈', '신바로'],
    '허리': ['신바로', '에어탈', '울트라셋', '비모보'],
    '해열진통제': ['리렉스펜', '써스펜', '맥시부펜'], '생리통': ['리렉스펜'],
    '열나고아파': ['리렉스펜', '맥시부펜'], '몸살기운': ['리렉스펜', '써스펜'],
    '조현병': ['아빌리파이'], '틱장애': ['아빌리파이'], '감정기복심해': ['아빌리파이'], '헛것이보여': ['아빌리파이'],
    '만성피로': ['다보타민큐'], '비타민결핍': ['다보타민큐'], '기운없어': ['다보타민큐', '트루비타'],
    '피곤해죽겠어': ['다보타민큐', '트루비타'], '눈떨려': ['다보타민큐'],
    '비타민': ['트루비타', '다보타민큐'], '피로': ['트루비타', '다보타민큐'],
    '열펄펄나': ['써스펜'], '머리아파': ['써스펜', '맥시부펜', '에어탈'],
    '두통': ['써스펜', '맥시부펜', '에어탈'], '머리아픔': ['써스펜', '맥시부펜'],
    '해열': ['써스펜', '리렉스펜'], '열': ['써스펜'], '발열': ['써스펜'],
    '진통': ['써스펜', '맥시부펜', '에어탈', '울트라셋'],
    '통증': ['써스펜', '에어탈', '울트라셋', '맥시부펜', '리리카'],
    '알츠하이머': ['에빅사'], '자꾸까먹어': ['에빅사', '글리아타민'], '기억이안나': ['에빅사'],
    '고지혈증': ['리피토', '크레스토', '리피로우', '아토르바', '아토젯', '로수젯', '로수바미브', '리바로'],
    '콜레스테롤높아': ['리피토', '리피로우', '로수바미브'], '피가탁해': ['리피토', '리피로우', '로수젯', '로수바미브', '오마코'],
    '콜레스테롤': ['리피토', '크레스토', '리피로우', '아토르바', '아토젯', '로수젯', '로수바미브'],
    '고지혈': ['리피토', '크레스토', '리피로우', '아토르바', '아토젯', '로수젯'],
    '혈관건강': ['크레스토', '플라빅스'], '피맑게해줘': ['크레스토'],
    '뇌전증': ['가바토파', '동아가바펜틴'], '간질': ['가바토파', '동아가바펜틴'],
    '신경통': ['가바토파', '리리카', '동아가바펜틴', '카발린'], '찌릿찌릿해': ['가바토파', '동아가바펜틴', '카발린'],
    '갑자기발작': ['가바토파'], '대상포진': ['동아가바펜틴', '리리카', '카발린'],
    '오메가3': ['오마코'], '오메가': ['오마코'], '피가찐득해': ['오마코'], '혈액순환안돼': ['오마코', '기넥신'],
    '전기에감전된듯아파': ['리리카'], '섬유근육통': ['리리카', '카발린'], '당뇨발저려': ['리리카', '카발린'],
    '기억력떨어져': ['종근당글리아티린', '글리아타민'], '머리가멍해': ['콜리네이트'], '뇌영양제': ['콜리네이트', '글리아타민'],
    '육체피로': ['트루비타'], '눈이피로해': ['트루비타'], '기력회복': ['트루비타'],
    '역류성식도염': ['스토가', '큐시드', '놀텍', '에스원엠프'], '위산과다': ['스토가', '큐시드', '에스원엠프'],
    '신물올라와': ['스토가', '라비에트', '에스원엠프'], '명치가아파': ['스토가'],
    '고혈압': ['노바스크', '엑스포지', '카나브', '아모잘탄', '트윈스타', '일양하이트린'],
    '혈압': ['노바스크', '엑스포지', '카나브', '아모잘탄', '트윈스타'],
    '혈압높아': ['노바스크', '엑스포지', '카나브'], '뒷목당겨': ['노바스크', '엑스포지', '아모잘탄', '카나브'],
    '뒷목뻐근해': ['엑스포지', '트윈스타'], '혈압안떨어져': ['엑스포지', '아모잘탄'],
    '파킨슨': ['마도파', '아질렉트'], '손이떨려': ['마도파', '아질렉트'], '행동이느려져': ['마도파'], '근육이굳어': ['마도파'],
    '뇌졸중예방': ['플라빅스'], '심근경색': ['플라빅스', '오마코'], '동맥경화': ['플라빅스', '아토르바', '아토젯'],
    '피떡생겨': ['플라빅스'], '혈전': ['플라빅스'],
    '나쁜콜레스테롤': ['아토르바', '로수젯'], '피가기름져': ['아토르바', '아토젯'],
    '가슴이타는듯해': ['라비에트'], '신물넘어와': ['라비에트', '놀텍'],
    '당뇨': ['자누비아', '자누메트', '트라젠타', '제미메트', '자누메트엑스알'],
    '혈당': ['자누비아', '자누메트', '트라젠타', '제미메트'], '당수치높아': ['자누비아', '트라젠타듀오'],
    '오줌에거품나': ['자누비아'], '목이자꾸말라': ['자누메트'],
    '무릎이쑤셔': ['맥시부펜', '레일라', '신바로'], '감기몸살': ['맥시부펜', '써스펜'],
    '십이지장궤양': ['놀텍'], '헬리코박터균': ['놀텍'], '위산넘어와': ['놀텍'],
    '더부룩해': ['큐시드'], '명치쓰려': ['큐시드'],
    '혈압이안잡혀': ['아모잘탄'], '뒷목땡겨': ['아모잘탄'],
    '혈압낮춰줘': ['트윈스타'], '혈압치솟아': ['카나브'], '뒷목아파': ['카나브'],
    '수술후너무아파': ['울트라셋'], '강력한진통제': ['울트라셋'], '허리끊어질듯아파': ['울트라셋'],
    '우울': ['졸로푸트', '아빌리파이'], '우울장애': ['졸로푸트'], '공황장애': ['졸로푸트'],
    '강박증': ['졸로푸트'], '우울하고불안해': ['졸로푸트'], '가슴이답답해': ['졸로푸트'],
    '잠이안와': ['보령부스파'], '불면': ['보령부스파'],
    '혈당조절안돼': ['트라젠타'], '밥먹고당올라가': ['트라젠타'],
    '무릎연골아파': ['레일라'], '관절이쑤셔': ['레일라', '신바로', '에어탈'],
    '무릎통증': ['신바로', '레일라'], '뼈마디아파': ['신바로', '비모보'],
    '속이타는듯해': ['에스원엠프'], '소화성궤양': ['에스원엠프'],
    '식후혈당높아': ['제미메트'], '당수치관리': ['제미메트', '트라젠타듀오'],
    '혈관찌꺼기': ['로수젯'], '혈관청소': ['로수바미브'],
    '몸이뻣뻣해져': ['아질렉트'], '손발이떨려': ['아질렉트', '마도파'],
    '공복혈당높아': ['자누메트엑스알'], '소변자주마려워': ['자누메트엑스알', '일양하이트린'],
    '갈증이자꾸나': ['트라젠타듀오'],
    '허리쑤셔': ['비모보'], '속안쓰린진통제': ['비모보'],
};

// 검색
const searchInput = document.getElementById('search-input');
const searchResults = document.getElementById('search-results');

searchInput.addEventListener('input', function() {
    const query = this.value.trim().toLowerCase();
    searchResults.innerHTML = '';
    if (query.length < 1) return;
    if (query.length >= 2) appLog('검색', query);

    const matches = [];
    for (const [id, info] of Object.entries(pillDB)) {
        const searchable = [info.name, info.material, info.company, info.name_en].join(' ').toLowerCase();
        if (searchable.includes(query)) matches.push(info);
    }

    if (matches.length === 0) {
        const symptomKeywords = symptomMap[query] || Object.keys(symptomMap).filter(k => query.includes(k) || k.includes(query)).flatMap(k => symptomMap[k]);
        if (symptomKeywords.length > 0) {
            const unique = [...new Set(symptomKeywords)];
            for (const [id, info] of Object.entries(pillDB)) {
                if (unique.some(kw => info.name.includes(kw)) && !matches.includes(info)) {
                    matches.push(info);
                }
            }
        }
    }

    if (matches.length === 0) {
        searchResults.innerHTML = `<div class="search-empty">검색 결과가 없습니다.<br><a href="javascript:void(0)" onclick="openAiWithSymptom('${query}')" style="color:#4ECDC4;text-decoration:underline;font-size:13px;">AI 약사에게 '${query}' 상담하기</a></div>`;
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
    appLog('페이지이동', pageId);
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
    if (pageId === 'mypage') { loadProfile(); renderHistory(); loadModels(); }
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
            addHistory(data.detections, data.image);
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
document.getElementById('ocr-input').addEventListener('change', handleOcr);

async function handleOcr(e) {
    const file = e.target.files[0];
    if (!file) return;
    appLog('처방전스캔', file.name);
    showPage('loading');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/api/ocr', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.error) { showToast(data.error); showPage('home'); return; }

        if (data.count === 0) {
            showToast('처방전에서 약을 찾지 못했습니다');
            showPage('home');
            return;
        }

        const detections = data.matched.map(m => ({
            name: m.name,
            confidence: 1.0,
            info: m.info,
        }));

        document.getElementById('detect-image-section').style.display = 'none';
        const resultImg = document.getElementById('result-image');
        const resultImgSection = document.getElementById('result-image-section');
        const previewUrl = URL.createObjectURL(file);
        resultImg.src = previewUrl;
        resultImg.style.width = '100%';
        resultImg.style.maxHeight = '200px';
        resultImg.style.objectFit = 'contain';
        resultImg.style.margin = '0 auto 12px';
        resultImg.style.display = 'block';
        resultImg.style.borderRadius = '12px';
        resultImgSection.style.display = 'block';

        lastDetections = detections;
        renderCards(detections, detections.length);

        showPage('result');
        appLog('처방전결과', data.count + '개: ' + data.matched.map(m => m.name).join(', '));
    } catch (err) {
        showToast('처방전 인식 실패: ' + err.message);
        showPage('home');
    }
    e.target.value = '';
}

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
        appLog('검출완료', data.count + '개: ' + data.detections.map(d => d.name).join(', '));
        addHistory(data.detections, data.image);
        renderCards(data.detections, data.count);
        showPage('result');
    } catch (err) {
        appLog('검출실패', err.message);
        alert('검출 실패: ' + err.message);
        showPage('home');
    }
    e.target.value = '';
}

function findGenerics(drugName) {
    const results = [];
    for (const [a, b, ingredient] of genericMap) {
        if (drugName.includes(a) && !drugName.includes(b)) {
            const catId = Object.keys(pillDB).find(k => pillDB[k].name.includes(b));
            if (catId) results.push({ name: pillDB[catId].name, catId, ingredient });
        } else if (drugName.includes(b) && !drugName.includes(a)) {
            const catId = Object.keys(pillDB).find(k => pillDB[k].name.includes(a));
            if (catId) results.push({ name: pillDB[catId].name, catId, ingredient });
        }
    }
    const seen = new Set();
    return results.filter(r => { if (seen.has(r.catId)) return false; seen.add(r.catId); return true; });
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

        const desc = getPillDesc(det.name);
        let descHtml = '';
        if (desc) {
            descHtml = `<div style="margin:10px 0;display:flex;flex-direction:column;gap:8px;">
                <fieldset style="border:1.5px solid #4ECDC4;border-radius:10px;padding:8px 12px 10px;margin:0;">
                    <legend style="font-size:13px;font-weight:700;color:#4ECDC4;padding:0 6px;">💊 약물 효과</legend>
                    <div style="font-size:14px;color:#2D3436;line-height:1.5;">${desc.effect}</div>
                </fieldset>
                <fieldset style="border:1.5px solid #FF8A80;border-radius:10px;padding:8px 12px 10px;margin:0;">
                    <legend style="font-size:13px;font-weight:700;color:#E65100;padding:0 6px;">⚠️ 복용 주의</legend>
                    <div style="font-size:14px;color:#E65100;line-height:1.5;">${desc.caution}</div>
                </fieldset>
            </div>`;
        }
        let details = '';
        if (det.info) {
            const rows = [
                ['성분', det.info.material], ['제약사', det.info.company],
                ['구분', det.info.etc_otc],
                ['외형', ((det.info.color1 || '') + ' ' + (det.info.shape || '')).trim()],
                ['각인', [det.info.print_front, det.info.print_back].filter(Boolean).join(' / ')],
                ['설명', det.info.chart],
            ];
            let detailRows = '';
            rows.forEach(([label, value]) => {
                if (value) detailRows += `<div class="pill-detail"><span class="pill-label">${label}</span><span class="pill-value">${value}</span></div>`;
            });
            if (detailRows) {
                const uid = 'detail_' + Math.random().toString(36).slice(2, 8);
                details = `<div style="margin-top:6px;">
                    <div onclick="document.getElementById('${uid}').style.display=document.getElementById('${uid}').style.display==='none'?'block':'none'" style="cursor:pointer;font-size:12px;color:#999;padding:4px 0;">📋 상세정보 ▼</div>
                    <div id="${uid}" style="display:none;">${detailRows}</div>
                </div>`;
            }
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

        const generics = findGenerics(det.name);
        let genericHtml = '';
        if (generics.length > 0) {
            const chips = generics.map(g =>
                `<span onclick="showPillFromChat('${g.catId}')" style="display:inline-block;background:#E8F5E9;color:#2E7D32;padding:4px 10px;border-radius:12px;font-size:11px;cursor:pointer;margin:2px;">${g.name}</span>`
            ).join('');
            genericHtml = `<div style="margin-top:8px;padding-top:8px;border-top:1px dashed #E0E0E0;">
                <span style="font-size:11px;color:#999;">💰 같은 성분 가성비약</span><br>${chips}</div>`;
        }

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
            ${descHtml}
            ${genericHtml}
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

    const allNames = getAllPillboxPillNames();
    const allWarnings = checkPillboxWarnings(allNames);

    const today = new Date();
    let html = '';

    if (allWarnings.length > 0) {
        let alertHtml = '<div style="background:#FFF0F0;border:2px solid #FF8A80;border-radius:12px;padding:12px;margin:0 16px 14px 16px;">';
        alertHtml += '<div style="font-weight:700;font-size:14px;color:#E53935;margin-bottom:8px;">⚠️ 복용 주의 경고</div>';
        allWarnings.forEach((w, i) => {
            const color = warningColors[i % warningColors.length];
            const icon = w.type === 'danger' ? '🔴' : '🟠';
            alertHtml += `<div style="font-size:12px;margin-bottom:4px;"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${color};margin-right:6px;"></span>${icon} <b>${w.a}</b> ↔ <b>${w.b}</b>: ${w.reason}</div>`;
        });
        alertHtml += '</div>';
        html += alertHtml;
    }

    box.forEach((entry, idx) => {
        const savedDate = new Date(entry.date);
        const dDay = Math.floor((today - savedDate) / (1000 * 60 * 60 * 24));
        const pillNames = entry.pills.map(p => p.name);
        const preview = pillNames.length <= 2 ? pillNames.join(', ') : pillNames.slice(0, 2).join(', ') + ` 외 ${pillNames.length - 2}`;
        const sym = entry.symptom || { emoji: '💊', label: '일반' };

        const entryWarnings = pillNames.some(n => allWarnings.some(w => n.includes(w.a) || n.includes(w.b)));
        const cardStyle = entryWarnings ? 'border:2px solid #FF8A80;background:#FFF8F8;' : '';
        const warningBadge = entryWarnings ? '<span style="color:#E53935;font-weight:700;font-size:13px;"> ⚠️ 주의</span>' : '';

        const hasAlarm = entry.alarm ? true : false;
        const alarmInfo = hasAlarm ? `<div class="pillbox-alarm-info">🔔 하루 ${entry.alarm.times}회 알림</div>` : '';

        html += `
        <div class="pillbox-card" style="${cardStyle}">
            <div class="pillbox-header">
                <span class="pillbox-symptom">${sym.emoji} ${sym.label}${warningBadge}</span>
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

    if (allWarnings.length > 0) {
        showToast(`⚠️ 함께 복용하면 안 되는 약이 ${allWarnings.filter(w=>w.type==='danger').length}건 있습니다!`);
    }
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

        const desc = getPillDesc(det.name);
        let descHtml = '';
        if (desc) {
            descHtml = `<div style="margin:10px 0;display:flex;flex-direction:column;gap:8px;">
                <fieldset style="border:1.5px solid #4ECDC4;border-radius:10px;padding:8px 12px 10px;margin:0;">
                    <legend style="font-size:13px;font-weight:700;color:#4ECDC4;padding:0 6px;">💊 약물 효과</legend>
                    <div style="font-size:14px;color:#2D3436;line-height:1.5;">${desc.effect}</div>
                </fieldset>
                <fieldset style="border:1.5px solid #FF8A80;border-radius:10px;padding:8px 12px 10px;margin:0;">
                    <legend style="font-size:13px;font-weight:700;color:#E65100;padding:0 6px;">⚠️ 복용 주의</legend>
                    <div style="font-size:14px;color:#E65100;line-height:1.5;">${desc.caution}</div>
                </fieldset>
            </div>`;
        }
        let details = '';
        if (det.info) {
            const rows = [
                ['성분', det.info.material], ['제약사', det.info.company],
                ['구분', det.info.etc_otc],
                ['외형', ((det.info.color1 || '') + ' ' + (det.info.shape || '')).trim()],
                ['각인', [det.info.print_front, det.info.print_back].filter(Boolean).join(' / ')],
                ['설명', det.info.chart],
            ];
            let detailRows = '';
            rows.forEach(([label, value]) => {
                if (value) detailRows += `<div class="pill-detail"><span class="pill-label">${label}</span><span class="pill-value">${value}</span></div>`;
            });
            if (detailRows) {
                const uid = 'detail_' + Math.random().toString(36).slice(2, 8);
                details = `<div style="margin-top:6px;">
                    <div onclick="document.getElementById('${uid}').style.display=document.getElementById('${uid}').style.display==='none'?'block':'none'" style="cursor:pointer;font-size:12px;color:#999;padding:4px 0;">📋 상세정보 ▼</div>
                    <div id="${uid}" style="display:none;">${detailRows}</div>
                </div>`;
            }
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

        const generics = findGenerics(det.name);
        let genericHtml = '';
        if (generics.length > 0) {
            const chips = generics.map(g =>
                `<span onclick="showPillFromChat('${g.catId}')" style="display:inline-block;background:#E8F5E9;color:#2E7D32;padding:4px 10px;border-radius:12px;font-size:11px;cursor:pointer;margin:2px;">${g.name}</span>`
            ).join('');
            genericHtml = `<div style="margin-top:8px;padding-top:8px;border-top:1px dashed #E0E0E0;">
                <span style="font-size:11px;color:#999;">💰 같은 성분 가성비약</span><br>${chips}</div>`;
        }

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
            ${descHtml}
            ${genericHtml}
            ${details}
        </div>`;
    });

    const allNames = getAllPillboxPillNames();
    const allWarnings = checkPillboxWarnings(allNames);
    const localNames = detections.map(d => d.name);
    const relevantWarnings = allWarnings.filter(w =>
        localNames.some(n => n.includes(w.a) || n.includes(w.b))
    );

    if (relevantWarnings.length > 0) {
        let warnHtml = '<div style="background:#FFF0F0;border:2px solid #FF8A80;border-radius:12px;padding:12px;margin:0 16px 14px 16px;">';
        warnHtml += '<div style="font-weight:700;font-size:14px;color:#E53935;margin-bottom:8px;">⚠️ 복용 주의 경고</div>';
        relevantWarnings.forEach((w, i) => {
            const color = warningColors[i % warningColors.length];
            const icon = w.type === 'danger' ? '🔴' : '🟠';
            warnHtml += `<div style="font-size:12px;margin-bottom:4px;"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${color};margin-right:6px;"></span>${icon} <b>${w.a}</b> ↔ <b>${w.b}</b>: ${w.reason}</div>`;
        });
        warnHtml += '</div>';
        html = warnHtml + html;
    }

    const cards = html.split('<div class="pill-card">');
    if (cards.length > 1) {
        html = cards[0];
        detections.forEach((det, i) => {
            const drugWarns = getWarningsForDrug(det.name, relevantWarnings);
            let lights = '';
            if (drugWarns.length > 0) {
                drugWarns.forEach(w => {
                    const ci = allWarnings.indexOf(w);
                    const color = warningColors[ci % warningColors.length];
                    lights += `<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:${color};margin-right:3px;" title="${w.reason}"></span>`;
                });
            } else {
                lights = '<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#4CAF50;margin-right:3px;" title="안전"></span>';
            }
            const borderStyle = drugWarns.length > 0 ? 'border-left:4px solid #FF8A80;' : 'border-left:4px solid #4CAF50;';
            html += `<div class="pill-card" style="${borderStyle}"><div style="margin-bottom:4px;">${lights}</div>` + cards[i + 1];
        });
    }

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

    // 건강식품 + 추천정보 렌더링
    renderSupplements(symptoms);
    renderRecommends(symptoms);
}

const SP = '/static/img/supplements/';
const supplementDB = {
    '고혈압': [
        {name: '코큐텐', tag: '심장·혈압 건강', img: SP+'코큐텐.png'},
        {name: '홍국', tag: '콜레스테롤 관리', img: SP+'홍국.png'},
        {name: '은행잎 추출물', tag: '혈행 개선', img: SP+'은행잎.png'},
    ],
    '당뇨': [
        {name: '바나바잎', tag: '혈당 관리', img: SP+'바나바.png'},
        {name: '여주 추출물', tag: '천연 인슐린', img: SP+'여주.png'},
        {name: '상엽 추출물', tag: '식후 혈당 조절', img: SP+'상엽.png'},
    ],
    '간질환': [
        {name: '밀크시슬', tag: '간세포 보호', img: SP+'밀크시슬.png'},
        {name: '씨슬파워', tag: '간 해독 강화', img: SP+'씨슬파워.png'},
        {name: '민들레 추출물', tag: '간 기능 개선', img: SP+'민들레엑스.png'},
    ],
    '신장질환': [
        {name: '키드니포뮬러', tag: '신장 기능 보호', img: SP+'키드니포믈러.png'},
        {name: '네틀포스', tag: '이뇨·신장 건강', img: SP+'네틀포스.png'},
        {name: '코디세핀', tag: '신장 세포 보호', img: SP+'코디세핀.png'},
    ],
    '위장 약함': [
        {name: '프로바이오틱스', tag: '장내 유익균', img: SP+'프로바이오틱스.png'},
        {name: '매스틱', tag: '위점막 보호', img: SP+'매스틱.png'},
        {name: '프리바이오틱스', tag: '유익균 먹이', img: SP+'프리바이오2.png'},
    ],
    '알레르기': [
        {name: '퀘르세틴', tag: '항히스타민 효과', img: SP+'퀘르세틴.png'},
        {name: '퀘르세틴 포스', tag: '알레르기 완화', img: SP+'퀘르세틴-포스.png'},
        {name: '프로바이오틱스', tag: '면역 균형', img: SP+'프로바이오틱스.png'},
    ],
    '임산부': [
        {name: '프리나탈', tag: '엽산·철분 복합', img: SP+'프리나탈.png'},
        {name: '프리나탈코어', tag: '임산부 핵심영양', img: SP+'프리나탈코어.png'},
        {name: '멀티비타민', tag: '종합 영양 보충', img: SP+'멀티비타민.png'},
    ],
    '_default': [
        {name: '멀티비타민', tag: '기본 건강 관리', img: SP+'멀티비타민.png'},
        {name: '프로바이오틱스', tag: '장건강·면역', img: SP+'프로바이오틱스.png'},
        {name: '코큐텐', tag: '항산화·에너지', img: SP+'코큐텐.png'},
        {name: '헛깨 추출물', tag: '간 건강·숙취', img: SP+'헛깨.png'},
        {name: '은행잎 추출물', tag: '혈행·기억력', img: SP+'은행잎.png'},
        {name: '밀크시슬', tag: '간세포 보호', img: SP+'밀크시슬.png'},
    ],
};

const recommendDB = {
    '고혈압': [
        {title: '코큐텐, 혈압약과 함께 드셔보세요', desc: '코큐텐은 혈압약의 부작용인 근육통을 완화하고, 심장 에너지 대사를 도와줍니다. 매일 100mg 권장.'},
        {title: '홍국으로 콜레스테롤을 자연스럽게', desc: '모나콜린K 성분이 나쁜 콜레스테롤(LDL)을 낮춰줍니다. 스타틴 계열 약과 병용 시 주의.'},
    ],
    '당뇨': [
        {title: '바나바잎, 천연 혈당 관리의 시작', desc: '바나바잎의 코로솔산 성분이 식후 혈당 상승을 완만하게 도와줍니다.'},
        {title: '여주가 천연 인슐린이라 불리는 이유', desc: '여주의 카란틴 성분이 인슐린과 유사한 작용으로 혈당 조절에 도움을 줍니다.'},
    ],
    '간질환': [
        {title: '밀크시슬, 간 건강의 대명사', desc: '실리마린 성분이 간세포를 보호하고 재생을 촉진합니다. 음주 전후 섭취도 효과적.'},
        {title: '민들레가 간 해독을 돕는다?', desc: '민들레 추출물은 담즙 분비를 촉진하여 간의 자연 해독 과정을 도와줍니다.'},
    ],
    '신장질환': [
        {title: '신장 건강, 미리 챙기셔야 합니다', desc: '키드니포뮬러는 신장 기능 유지에 필요한 핵심 영양소를 복합 배합하였습니다.'},
        {title: '코디세핀으로 신장 세포 보호', desc: '동충하초 유래 코디세핀이 신장 세포의 산화 스트레스를 줄여줍니다.'},
    ],
    '위장 약함': [
        {title: '유산균이 면역의 70%를 좌우합니다', desc: '장내 유익균 균형이 면역력의 핵심입니다. 100억 CFU 이상의 프로바이오틱스를 추천합니다.'},
        {title: '매스틱검으로 위점막을 지키세요', desc: '그리스 키오스섬의 천연 수지 매스틱이 위벽을 코팅하여 속쓰림을 완화합니다.'},
    ],
    '알레르기': [
        {title: '퀘르세틴, 천연 항히스타민제', desc: '양파·사과에 풍부한 퀘르세틴이 히스타민 분비를 억제하여 알레르기 증상을 완화합니다.'},
        {title: '유산균으로 면역 균형을 잡으세요', desc: '장내 환경 개선이 알레르기 체질 개선의 첫걸음입니다.'},
    ],
    '임산부': [
        {title: '프리나탈, 임신 준비부터 출산까지', desc: '엽산 800mcg + 철분 + DHA를 한 번에. 태아 신경관 발달에 필수적인 영양소입니다.'},
        {title: '임산부 멀티비타민, 꼭 드셔야 하나요?', desc: '임신 중에는 평소보다 철분 2배, 엽산 4배가 필요합니다. 식사만으로는 부족합니다.'},
    ],
    '_default': [
        {title: '멀티비타민, 하루 한 알로 기본 충전', desc: '불규칙한 식사로 부족한 13가지 비타민과 미네랄을 한 번에 채워줍니다.'},
        {title: '유산균이 면역의 70%를 좌우합니다', desc: '장내 유익균 균형이 면역력의 핵심입니다. 100억 CFU 이상의 프로바이오틱스를 추천합니다.'},
        {title: '코큐텐, 30대부터 줄어드는 에너지원', desc: '체내 코큐텐은 30대부터 급감합니다. 심장·피부·에너지 대사에 필수적인 항산화 영양소.'},
        {title: '헛깨, 간이 피로할 때 드세요', desc: '헛깨나무 추출물이 알코올 분해를 도와 숙취 해소와 간 보호에 효과적입니다.'},
        {title: '은행잎이 혈액순환을 돕습니다', desc: '플라보노이드와 테르펜 성분이 말초 혈액순환을 개선하고 기억력 유지에 도움을 줍니다.'},
        {title: '밀크시슬로 간세포를 지키세요', desc: '실리마린 성분이 간세포막을 보호하고 재생을 촉진합니다. 음주 잦은 분께 필수.'},
    ],
};

function renderSupplements(symptoms) {
    const grid = document.getElementById('supplement-grid');
    if (!grid) return;
    let items = [];
    (symptoms || []).forEach(s => {
        const key = Object.keys(supplementDB).find(k => s.includes(k));
        if (key) items.push(...supplementDB[key]);
    });
    if (items.length === 0) items = supplementDB['_default'];
    items = items.slice(0, 6);
    grid.innerHTML = items.map(item => `
        <div class="supplement-card" onclick="openSupplementDetail('${item.name}', '${item.tag}', '${item.img}')">
            <img src="${item.img}" alt="${item.name}">
            <div class="sup-info">
                <div class="sup-name">${item.name}</div>
                <div class="sup-tag">${item.tag}</div>
            </div>
            <span class="sup-arrow">›</span>
        </div>
    `).join('');
}

const articleDB = {
    '_default': {
        title: '간 건강의 든든한 수호자, 밀크씨슬',
        subtitle: '식약처 인정 간 건강 기능성 원료',
        hero: '/static/img/article_milkthistle.png',
        product: {name: '밀크시슬', img: SP+'밀크시슬.png'},
        sections: [
            {icon: '🛡', heading: '간세포 보호 및 재생 촉진', text: '실리마린의 강력한 항산화 작용이 체내 활성산소로부터 간세포 파괴를 막아줍니다. 또한, 단백질 합성을 촉진하여 이미 손상된 간 조직이 빠르게 재생될 수 있도록 돕습니다.'},
            {icon: '⚡', heading: '탁월한 해독 작용과 피로 회복', text: '체내에 들어온 독소나 알코올을 분해하는 간의 필수 해독 기능을 지원합니다. 잦은 야근과 스트레스, 음주로 인해 축적된 현대인들의 만성 피로와 숙취를 개선하는 데 탁월한 효과가 있습니다.'},
            {icon: '💊', heading: '염증 완화', text: '체내 염증 반응을 유발하는 물질의 생성을 억제하여 간염 등 다양한 질환을 예방하는 데 도움을 줄 수 있습니다.'},
        ],
        highlight: "'침묵의 장기'라 불리는 간은 신경 세포가 적어 기능이 절반 이상 저하될 때까지 뚜렷한 증상이 나타나지 않습니다. 식약처로부터 간 건강 기능성을 인정받은 밀크씨슬을 통해 지친 일상 속 활력을 되찾고 간 건강을 미리 챙겨보세요!",
    },
};

function openArticle(title) {
    const article = articleDB['_default'];
    const el = document.getElementById('article-content');
    el.innerHTML = `
        <img class="article-hero" src="${article.hero}" alt="">
        <div class="article-title">${article.title}</div>
        <div class="article-subtitle">${article.subtitle}</div>
        ${article.sections.map(s => `
            <div class="article-section">
                <h3><span class="sec-icon">${s.icon}</span>${s.heading}</h3>
                <p>${s.text}</p>
            </div>
        `).join('')}
        <div class="article-highlight"><p>${article.highlight}</p></div>
        <div class="supplement-card" onclick="openSupplementDetail('${article.product.name}', '', '${article.product.img}')" style="margin:12px 0;">
            <img src="${article.product.img}" alt="">
            <div class="sup-info">
                <div class="sup-name">${article.product.name}</div>
                <div class="sup-tag">제품 상세보기 →</div>
            </div>
        </div>
    `;
    document.getElementById('article-header').textContent = '건강 칼럼';
    showPage('article');
}

function openSupplementDetail(name, tag, img) {
    showPage('supplement-detail');
    document.getElementById('sup-detail-name').textContent = name;
    document.getElementById('sup-detail-tag').textContent = tag;
    document.getElementById('sup-detail-thumb').src = img;
}

function renderRecommends(symptoms) {
    const container = document.getElementById('recommend-cards');
    if (!container) return;
    let items = [];
    (symptoms || []).forEach(s => {
        const key = Object.keys(recommendDB).find(k => s.includes(k));
        if (key) items.push(...recommendDB[key]);
    });
    if (items.length === 0) items = recommendDB['_default'];
    container.innerHTML = items.map(item => `
        <div class="recommend-card" onclick="openArticle('${item.title}')">
            <span class="rec-icon">💊</span>
            <div class="rec-info">
                <div class="rec-title">${item.title}</div>
                <div class="rec-desc">${item.desc}</div>
            </div>
            <span class="rec-arrow">›</span>
        </div>
    `).join('');
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
function addHistory(detections, imgBase64) {
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
        detections: detections,
        image: imgBase64 || '',
    });

    if (history.length > 10) history = history.slice(0, 10);
    localStorage.setItem('detectHistory', JSON.stringify(history));
}

function openHistoryItem(idx) {
    let history = [];
    try { history = JSON.parse(localStorage.getItem('detectHistory') || '[]'); } catch {}
    const item = history[idx];
    if (!item || !item.detections) return;

    if (item.image) {
        document.getElementById('detect-image').src = 'data:image/jpeg;base64,' + item.image;
        document.getElementById('detect-image-section').style.display = 'block';
    } else {
        document.getElementById('detect-image-section').style.display = 'none';
    }
    document.getElementById('result-image-section').style.display = 'none';

    lastDetections = item.detections;
    renderCards(item.detections, item.detections.length);
    showPage('result');
}

function renderHistory() {
    const container = document.getElementById('history-content');
    let history = [];
    try { history = JSON.parse(localStorage.getItem('detectHistory') || '[]'); } catch {}

    if (history.length === 0) {
        container.innerHTML = '<div style="text-align:center;padding:20px;color:#CCC;font-size:13px;">검출 기록이 없습니다.</div>';
        return;
    }

    container.innerHTML = history.slice(0, 10).map((h, i) => `
        <div class="history-item" onclick="openHistoryItem(${i})" style="cursor:pointer;">
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

function openAiWithSymptom(symptom) {
    searchInput.value = '';
    searchResults.innerHTML = '';
    if (currentPharmacist) {
        showPage('ai-chat');
        setTimeout(() => {
            document.getElementById('chat-input').value = symptom;
            sendChat();
        }, 300);
    } else {
        showPage('ai');
    }
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
    saveChatHistory();
    currentPharmacist = pharmacists[key];
    appLog('약사선택', currentPharmacist.name);
    localStorage.setItem('selectedPharmacist', key);
    chatHistory = loadChatHistory(key);

    document.getElementById('chat-pharmacist-img').src = currentPharmacist.img;
    document.getElementById('chat-pharmacist-name').textContent = currentPharmacist.name;
    restoreChatBubbles();

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

function findPillInDB(name) {
    const q = name.replace(/\s/g, '').toLowerCase();
    for (const [id, info] of Object.entries(pillDB)) {
        if (info.name.replace(/\s/g, '').toLowerCase().includes(q) ||
            q.includes(info.name.replace(/\s/g, '').toLowerCase())) {
            return { id, info };
        }
    }
    return null;
}

function renderDrugLinks(text) {
    return text.replace(/\*\*(.+?)\*\*/g, (match, drugName) => {
        const found = findPillInDB(drugName);
        if (found) {
            return `<a href="javascript:void(0)" class="drug-link drug-link-db" onclick="showPillFromChat('${found.id}')">${drugName}</a>`;
        }
        const query = encodeURIComponent(drugName + ' 약 효능 복용법');
        return `<a href="https://www.google.com/search?q=${query}" target="_blank" class="drug-link drug-link-ext">${drugName}</a>`;
    });
}

function showPillFromChat(catId) {
    const info = pillDB[catId];
    if (!info) return;
    showPillDetail(info, catId);
}

function addChatBubble(text, type, supplements) {
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
    if (type === 'bot') {
        bubble.innerHTML = renderDrugLinks(text);
    } else {
        bubble.textContent = text;
    }
    row.appendChild(bubble);

    if (type === 'bot' && supplements && supplements.length > 0) {
        const supWrap = document.createElement('div');
        supWrap.className = 'chat-sup-wrap';
        supWrap.innerHTML = '<div class="chat-sup-label">💊 바로 구입 가능한 건강기능식품</div>';
        const cards = document.createElement('div');
        cards.className = 'chat-sup-cards';
        supplements.forEach(s => {
            cards.innerHTML += `
                <div class="chat-sup-card" onclick="openSupplementDetail('${s.name}', '${s.tag}', '${s.img}')">
                    <img src="${s.img}" alt="${s.name}">
                    <div class="chat-sup-name">${s.name}</div>
                    <div class="chat-sup-tag">${s.tag}</div>
                </div>`;
        });
        supWrap.appendChild(cards);
        row.appendChild(supWrap);
    }

    container.appendChild(row);
    container.scrollTop = container.scrollHeight;
}

let chatHistory = [];

function saveChatHistory() {
    const key = localStorage.getItem('selectedPharmacist') || 'kim';
    localStorage.setItem('chatHistory_' + key, JSON.stringify(chatHistory));
}

function loadChatHistory(pharmacistKey) {
    try {
        return JSON.parse(localStorage.getItem('chatHistory_' + pharmacistKey) || '[]');
    } catch { return []; }
}

function restoreChatBubbles() {
    const container = document.getElementById('chat-messages');
    container.innerHTML = '';
    if (currentPharmacist) {
        addChatBubble(currentPharmacist.greeting, 'bot');
    }
    chatHistory.forEach(h => addChatBubble(h.text, h.type, h.supplements));
}

async function sendChat() {
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg) return;
    appLog('채팅발송', msg);
    addChatBubble(msg, 'user');
    chatHistory.push({ type: 'user', text: msg });
    saveChatHistory();
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
                user_name: profile.name || '',
            }),
        });
        const data = await res.json();
        loadingBubble.remove();
        addChatBubble(data.reply, 'bot', data.supplements || []);
        chatHistory.push({ type: 'bot', text: data.reply, supplements: data.supplements || [] });
        saveChatHistory();
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

// ═══════════════════════════════════════
//  모양으로 약 찾기
// ═══════════════════════════════════════

let selectedShape = '전체';
let selectedColor = '전체';

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('#shape-chips .shape-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            document.querySelectorAll('#shape-chips .shape-chip').forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            selectedShape = chip.dataset.value;
        });
    });
    document.querySelectorAll('#color-chips .color-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            document.querySelectorAll('#color-chips .color-chip').forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            selectedColor = chip.dataset.value;
        });
    });
});

function filterByShape() {
    const printQuery = (document.getElementById('shape-print-input').value || '').trim().toLowerCase();
    const results = [];

    for (const [catId, info] of Object.entries(pillDB)) {
        if (selectedShape !== '전체' && (info.shape || '') !== selectedShape) continue;
        if (selectedColor !== '전체' && (info.color1 || '') !== selectedColor) continue;
        if (printQuery) {
            const front = (info.print_front || '').toLowerCase();
            const back = (info.print_back || '').toLowerCase();
            const name = (info.name || '').toLowerCase();
            if (!front.includes(printQuery) && !back.includes(printQuery) && !name.includes(printQuery)) continue;
        }
        results.push({ catId, info });
    }

    const container = document.getElementById('shape-search-results');

    if (results.length === 0) {
        container.innerHTML = '<div style="text-align:center;padding:30px;color:#999;font-size:14px;">조건에 맞는 알약이 없습니다.</div>';
        return;
    }

    let html = `<div class="shape-result-count">${results.length}개 약품 검색됨</div>`;

    results.forEach(({ catId, info }) => {
        const stamp = [info.print_front, info.print_back].filter(Boolean).join(' / ');
        const chart = info.chart || '';
        const effect = chart ? chart.split('.')[0] + '.' : '';

        html += `
        <div class="search-card" onclick="showPillDetail(pillDB['${catId}'], '${catId}')" style="margin-bottom:10px;">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                <img src="/static/pills/${catId}.png" style="width:52px;height:52px;border-radius:14px;object-fit:cover;box-shadow:0 2px 6px rgba(0,0,0,0.08);" onerror="this.style.display='none'">
                <div style="flex:1;">
                    <div class="search-card-name">${info.name}</div>
                    <div class="search-card-sub">${info.company || ''}</div>
                    <div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:5px;">
                        <span style="font-size:10px;color:#666;background:#F1F3F5;padding:2px 8px;border-radius:8px;">${info.color1 || ''} ${info.shape || ''}</span>
                        ${stamp ? `<span style="font-size:10px;color:#4ECDC4;background:#E8F8F0;padding:2px 8px;border-radius:8px;">${stamp}</span>` : ''}
                    </div>
                </div>
            </div>
            ${effect ? `<div style="font-size:12px;color:#666;line-height:1.5;padding:8px 10px;background:#F8F9FA;border-radius:10px;border-left:3px solid #4ECDC4;">${effect}</div>` : ''}
        </div>`;
    });

    container.innerHTML = html;
    appLog('모양검색', `${selectedShape}/${selectedColor}/${printQuery} → ${results.length}건`);
}
