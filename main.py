import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes
import io
import re
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import pytesseract

# ═══════════════════════════════════════════════════════════════
#  ✅ ANSWER KEYS — Update these when officially released
#     Format: 100 answers in order Q1→Q100, values: 'A','B','C','D'
# ═══════════════════════════════════════════════════════════════

ANSWER_KEYS = {
    "A1": [
        # Q1  -  Q10
        'A','B','C','D','A','B','C','D','A','B',
        # Q11 -  Q20
        'C','D','A','B','C','D','A','B','C','D',
        # Q21 -  Q30
        'A','B','C','D','A','B','C','D','A','B',
        # Q31 -  Q40
        'C','D','A','B','C','D','A','B','C','D',
        # Q41 -  Q50
        'A','B','C','D','A','B','C','D','A','B',
        # Q51 -  Q60
        'C','D','A','B','C','D','A','B','C','D',
        # Q61 -  Q70
        'A','B','C','D','A','B','C','D','A','B',
        # Q71 -  Q80
        'C','D','A','B','C','D','A','B','C','D',
        # Q81 -  Q90
        'A','B','C','D','A','B','C','D','A','B',
        # Q91 - Q100
        'C','D','A','B','C','D','A','B','C','D',
    ],
    "B1": [
        # Q1  -  Q10
        'B','C','D','A','B','C','D','A','B','C',
        # Q11 -  Q20
        'D','A','B','C','D','A','B','C','D','A',
        # Q21 -  Q30
        'B','C','D','A','B','C','D','A','B','C',
        # Q31 -  Q40
        'D','A','B','C','D','A','B','C','D','A',
        # Q41 -  Q50
        'B','C','D','A','B','C','D','A','B','C',
        # Q51 -  Q60
        'D','A','B','C','D','A','B','C','D','A',
        # Q61 -  Q70
        'B','C','D','A','B','C','D','A','B','C',
        # Q71 -  Q80
        'D','A','B','C','D','A','B','C','D','A',
        # Q81 -  Q90
        'B','C','D','A','B','C','D','A','B','C',
        # Q91 - Q100
        'D','A','B','C','D','A','B','C','D','A',
    ],
    "C1": [
        # Q1  -  Q10
        'C','D','A','B','C','D','A','B','C','D',
        # Q11 -  Q20
        'A','B','C','D','A','B','C','D','A','B',
        # Q21 -  Q30
        'C','D','A','B','C','D','A','B','C','D',
        # Q31 -  Q40
        'A','B','C','D','A','B','C','D','A','B',
        # Q41 -  Q50
        'C','D','A','B','C','D','A','B','C','D',
        # Q51 -  Q60
        'A','B','C','D','A','B','C','D','A','B',
        # Q61 -  Q70
        'C','D','A','B','C','D','A','B','C','D',
        # Q71 -  Q80
        'A','B','C','D','A','B','C','D','A','B',
        # Q81 -  Q90
        'C','D','A','B','C','D','A','B','C','D',
        # Q91 - Q100
        'A','B','C','D','A','B','C','D','A','B',
    ],
    "D1": [
        # Q1  -  Q10
        'D','A','B','C','D','A','B','C','D','A',
        # Q11 -  Q20
        'B','C','D','A','B','C','D','A','B','C',
        # Q21 -  Q30
        'D','A','B','C','D','A','B','C','D','A',
        # Q31 -  Q40
        'B','C','D','A','B','C','D','A','B','C',
        # Q41 -  Q50
        'D','A','B','C','D','A','B','C','D','A',
        # Q51 -  Q60
        'B','C','D','A','B','C','D','A','B','C',
        # Q61 -  Q70
        'D','A','B','C','D','A','B','C','D','A',
        # Q71 -  Q80
        'B','C','D','A','B','C','D','A','B','C',
        # Q81 -  Q90
        'D','A','B','C','D','A','B','C','D','A',
        # Q91 - Q100
        'B','C','D','A','B','C','D','A','B','C',
    ],
}

# ─────────────────────────── Page Config ───────────────────────────
st.set_page_config(
    page_title="PGCET OMR Checker",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────── CSS ───────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; text-align: center;
        margin-bottom: 2rem; box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .main-header h1 { color: #e94560; font-size: 2.2rem; margin: 0; }
    .main-header p  { color: #a8b2d8; margin: 0.5rem 0 0; font-size: 1.1rem; }
    .step-card {
        background: #16213e; border: 1px solid #0f3460;
        border-radius: 10px; padding: 1.5rem; margin-bottom: 1rem;
    }
    .step-label {
        background: #e94560; color: white; border-radius: 50%;
        width: 28px; height: 28px; display: inline-flex;
        align-items: center; justify-content: center;
        font-weight: bold; font-size: 0.9rem; margin-right: 0.5rem;
    }
    .score-box {
        background: linear-gradient(135deg, #0f3460, #16213e);
        border: 2px solid #e94560; border-radius: 16px;
        padding: 2rem; text-align: center; margin: 1rem 0;
    }
    .score-big { font-size: 4rem; font-weight: 900; color: #e94560; line-height: 1; }
    .score-sub { color: #a8b2d8; font-size: 1rem; margin-top: 0.5rem; }
    .info-chip {
        display: inline-block; background: #0f3460; color: #a8b2d8;
        border-radius: 20px; padding: 4px 14px; font-size: 0.85rem; margin: 3px;
    }
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #e94560, #c73652);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-weight: 600; font-size: 1rem; width: 100%;
    }
    div[data-testid="stButton"] > button:hover { opacity: 0.9; }
    .key-status-ok   { background:#1a472a; color:#52c41a; padding:6px 14px; border-radius:8px; font-weight:bold; }
    .key-status-wait { background:#4a3000; color:#faad14; padding:6px 14px; border-radius:8px; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Header ───────────────────────────
st.markdown("""
<div class="main-header">
  <h1>📝 PGCET OMR Answer Checker</h1>
  <p>MBA &amp; MCA Entrance Exam · 100 Questions · Versions A1 B1 C1 D1</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────── Session State ───────────────────────────
for key in ["omr_answers", "student_info", "results"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ═══════════════════════════════════════════════════════════════
#  OMR PROCESSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def pdf_to_images(pdf_bytes):
    images = convert_from_bytes(pdf_bytes, dpi=300)
    return [np.array(img.convert("RGB")) for img in images]


def enhance_image(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    return sharpened


def extract_text_from_image(img_np):
    pil_img = Image.fromarray(img_np)
    text = pytesseract.image_to_string(pil_img, config='--psm 6')
    return text


def parse_student_info(ocr_text):
    info = {"name": "", "reg_no": ""}
    lines = [l.strip() for l in ocr_text.split('\n') if l.strip()]
    for line in lines:
        reg_match = re.search(r'\b(\d{6,12})\b', line)
        if reg_match and not info["reg_no"]:
            info["reg_no"] = reg_match.group(1)
        if re.match(r'^[A-Za-z\s\.]+$', line) and len(line) > 4 and not info["name"]:
            if not any(kw in line.lower() for kw in ['version','set','exam','date','roll','reg','serial']):
                info["name"] = line.title()
    return info


def detect_bubbles_in_row(row_img, num_options=4):
    h, w = row_img.shape
    cell_w = w // num_options
    fill_ratios = []
    for i in range(num_options):
        cell = row_img[:, i * cell_w:(i + 1) * cell_w]
        _, binary = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ratio = np.sum(binary == 255) / binary.size
        fill_ratios.append(ratio)
    max_ratio = max(fill_ratios)
    if max_ratio < 0.08:
        return -1
    sorted_ratios = sorted(fill_ratios, reverse=True)
    if len(sorted_ratios) > 1 and sorted_ratios[0] > sorted_ratios[1] * 1.4:
        return fill_ratios.index(max_ratio)
    return fill_ratios.index(max_ratio)


def detect_omr_answers(img_np, num_questions=100, num_options=4):
    enhanced = enhance_image(img_np)
    h, w = enhanced.shape
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    top_crop    = int(h * 0.20)
    bottom_crop = int(h * 0.90)
    left_crop   = int(w * 0.30)
    right_crop  = int(w * 0.85)
    bubble_region = binary[top_crop:bottom_crop, left_crop:right_crop]
    region_h, region_w = bubble_region.shape
    row_height = max(1, region_h // num_questions)
    answers = []
    for q in range(num_questions):
        y1 = q * row_height
        y2 = (q + 1) * row_height
        row = bubble_region[y1:y2, :]
        if row.size == 0:
            answers.append(-1)
            continue
        answers.append(detect_bubbles_in_row(row, num_options))
    return answers


def answers_to_letters(answer_indices):
    mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', -1: '-'}
    return [mapping.get(a, '-') for a in answer_indices]


# ═══════════════════════════════════════════════════════════════
#  RESULT CALCULATION
# ═══════════════════════════════════════════════════════════════

def calculate_results(student_answers, key_answers):
    results = []
    correct = wrong = skipped = 0
    for i, (sa, ka) in enumerate(zip(student_answers, key_answers), start=1):
        if sa == '-' or sa == '':
            status = 'skipped'; marks = 0; skipped += 1
        elif sa == ka:
            status = 'correct'; marks = 1; correct += 1
        else:
            status = 'wrong';   marks = 0; wrong   += 1
        results.append({'q': i, 'student': sa, 'key': ka, 'status': status, 'marks': marks})
    return results, correct, wrong, skipped


# ═══════════════════════════════════════════════════════════════
#  RESULT PDF GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_result_pdf(student_info, results, correct, wrong, skipped, version):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                             rightMargin=15*mm, leftMargin=15*mm,
                             topMargin=15*mm, bottomMargin=15*mm)
    story = []
    styles = getSampleStyleSheet()

    header_style = ParagraphStyle('header', fontSize=18, fontName='Helvetica-Bold',
                                   alignment=TA_CENTER, textColor=colors.HexColor('#e94560'), spaceAfter=4)
    sub_style    = ParagraphStyle('sub', fontSize=11, fontName='Helvetica',
                                   alignment=TA_CENTER, textColor=colors.HexColor('#333333'), spaceAfter=2)
    story.append(Paragraph("PGCET ENTRANCE EXAMINATION", header_style))
    story.append(Paragraph("Official OMR Answer Report", sub_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#e94560')))
    story.append(Spacer(1, 8))

    info_data = [
        ["Student Name",    ":", student_info.get("name",   "N/A"), "Version / Set",   ":", version],
        ["Registration No.",":", student_info.get("reg_no", "N/A"), "Total Questions", ":", "100"],
    ]
    info_table = Table(info_data, colWidths=[38*mm, 5*mm, 55*mm, 38*mm, 5*mm, 35*mm])
    info_table.setStyle(TableStyle([
        ('FONTNAME',  (0,0), (-1,-1), 'Helvetica'),
        ('FONTNAME',  (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',  (3,0), (3,-1), 'Helvetica-Bold'),
        ('FONTSIZE',  (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.HexColor('#f0f4ff'), colors.white]),
        ('BOX',       (0,0), (-1,-1), 0.5, colors.grey),
        ('INNERGRID', (0,0), (-1,-1), 0.25, colors.lightgrey),
        ('PADDING',   (0,0), (-1,-1), 5),
        ('VALIGN',    (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 10))

    pct = round((correct / 100) * 100, 1)
    if pct >= 70:   grade = "A (Excellent)"
    elif pct >= 50: grade = "B (Good)"
    elif pct >= 35: grade = "C (Pass)"
    else:           grade = "D (Below Average)"

    score_data = [
        ["TOTAL SCORE", "CORRECT", "WRONG", "SKIPPED", "PERCENTAGE", "GRADE"],
        [
            Paragraph(f'<font size="20" color="#e94560"><b>{correct}/100</b></font>', styles['Normal']),
            Paragraph(f'<font size="14" color="#52c41a"><b>{correct}</b></font>',     styles['Normal']),
            Paragraph(f'<font size="14" color="#ff4d4f"><b>{wrong}</b></font>',        styles['Normal']),
            Paragraph(f'<font size="14" color="#faad14"><b>{skipped}</b></font>',      styles['Normal']),
            Paragraph(f'<font size="14"><b>{pct}%</b></font>',                         styles['Normal']),
            Paragraph(f'<font size="11"><b>{grade}</b></font>',                        styles['Normal']),
        ]
    ]
    score_table = Table(score_data, colWidths=[30*mm]*6)
    score_table.setStyle(TableStyle([
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,0), 8),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#f0f4ff')),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('BOX',        (0,0), (-1,-1), 1, colors.HexColor('#0f3460')),
        ('INNERGRID',  (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
        ('ROWHEIGHT',  (0,0), (-1,0), 18),
        ('ROWHEIGHT',  (0,1), (-1,1), 30),
        ('PADDING',    (0,0), (-1,-1), 5),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Detailed Answer Analysis</b>",
                            ParagraphStyle('head2', fontSize=11, fontName='Helvetica-Bold',
                                           spaceAfter=6, textColor=colors.HexColor('#1a1a2e'))))

    COLS = 5
    header_row = ["Q#", "Your Ans", "Key Ans", "Result"] * COLS
    rows = [header_row]
    chunk = []
    for r in results:
        color_tag = {'correct': '#52c41a', 'wrong': '#ff4d4f', 'skipped': '#faad14'}[r['status']]
        symbol    = {'correct': '✓', 'wrong': '✗', 'skipped': '–'}[r['status']]
        result_cell = Paragraph(
            f'<font color="{color_tag}"><b>{symbol}</b></font>', styles['Normal'])
        chunk.append([str(r['q']), r['student'], r['key'], result_cell])
        if len(chunk) == COLS:
            row = []
            for c in chunk: row += c
            rows.append(row)
            chunk = []
    if chunk:
        row = []
        for c in chunk: row += c
        while len(row) < COLS * 4: row += ["", "", "", ""]
        rows.append(row)

    col_w = [8*mm, 14*mm, 14*mm, 12*mm] * COLS
    ans_table = Table(rows, colWidths=col_w, repeatRows=1)
    ts = [
        ('FONTNAME',   (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 7),
        ('BACKGROUND', (0,0), (-1,0),  colors.HexColor('#0f3460')),
        ('TEXTCOLOR',  (0,0), (-1,0),  colors.white),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('BOX',        (0,0), (-1,-1), 0.5, colors.grey),
        ('INNERGRID',  (0,0), (-1,-1), 0.25, colors.lightgrey),
        ('ROWHEIGHT',  (0,0), (-1,-1), 10),
    ]
    for i in range(1, len(rows)):
        bg = colors.HexColor('#f9f9f9') if i % 2 == 0 else colors.white
        ts.append(('BACKGROUND', (0,i), (-1,i), bg))
    ans_table.setStyle(TableStyle(ts))
    story.append(ans_table)
    story.append(Spacer(1, 10))

    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 4))
    footer_style = ParagraphStyle('footer', fontSize=7, alignment=TA_CENTER, textColor=colors.grey)
    story.append(Paragraph(
        "This is a computer-generated report. Official marks are subject to verification by the examining authority.",
        footer_style))
    story.append(Paragraph(
        f"PGCET OMR Checker  |  Reg No: {student_info.get('reg_no','N/A')}  |  Version: {version}",
        footer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📋 How to Use")
    st.markdown("""
1. **Upload OMR PDF** – Student's scanned OMR copy
2. **Enter Student Details** – Name & Reg Number
3. **Select Version** – A1 / B1 / C1 / D1
4. **Analyze** – Click to compare & score
5. **Download PDF** – Printable result report
    """)
    st.markdown("---")
    st.markdown("### ℹ️ Exam Info")
    st.info("📌 100 Questions | MCQ | 1 mark each | Max 100 marks")
    st.markdown("---")
    st.markdown("### 🔑 Answer Key Status")
    # Show which versions have real keys loaded
    st.markdown("""
> ⚠️ **Placeholder keys loaded.**
> Update `ANSWER_KEYS` in code once official keys are released.
    """)
    st.markdown("---")
    st.markdown("### 🎯 Grade Scale")
    st.markdown("""
- 🟢 **A** – 70+ (Excellent)
- 🔵 **B** – 50–69 (Good)
- 🟡 **C** – 35–49 (Pass)
- 🔴 **D** – Below 35
    """)


# ═══════════════════════════════════════════════════════════════
#  MAIN UI
# ═══════════════════════════════════════════════════════════════
col1, col2 = st.columns([1, 1], gap="large")

# ── STEP 1: OMR Upload ──
with col1:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("#### <span class='step-label'>1</span> Upload Student OMR PDF", unsafe_allow_html=True)

    omr_file = st.file_uploader("Upload OMR PDF", type=["pdf"],
                                  key="omr_upload", label_visibility="collapsed")

    if omr_file:
        st.success("✅ OMR PDF uploaded!")
        with st.spinner("🔍 Enhancing & reading OMR..."):
            try:
                pdf_bytes = omr_file.read()
                images    = pdf_to_images(pdf_bytes)
                enhanced_preview = enhance_image(images[0])
                st.image(enhanced_preview, caption="Enhanced OMR (Page 1)",
                         use_container_width=True, clamp=True)

                ocr_text  = extract_text_from_image(images[0])
                auto_info = parse_student_info(ocr_text)
                st.session_state._auto_info = auto_info

                raw_answers    = detect_omr_answers(images[0])
                letter_answers = answers_to_letters(raw_answers)
                st.session_state.omr_answers = letter_answers

                answered = sum(1 for a in letter_answers if a != '-')
                st.markdown(f"**Detected:** {answered} answered · {100-answered} skipped")
            except Exception as e:
                st.error(f"Error processing OMR: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ── STEP 2: Student Details + Version ──
with col2:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("#### <span class='step-label'>2</span> Student Details & Version", unsafe_allow_html=True)

    auto = getattr(st.session_state, '_auto_info', {})
    student_name = st.text_input("👤 Student Name",
                                   value=auto.get("name", ""),
                                   placeholder="Enter student full name")
    student_reg  = st.text_input("🔢 Registration Number",
                                   value=auto.get("reg_no", ""),
                                   placeholder="e.g. 2024MBA001")

    version = st.selectbox("📋 Answer Version / Set",
                             ["A1", "B1", "C1", "D1"], index=0)

    # Show key preview for selected version
    key = ANSWER_KEYS[version]
    st.markdown(f"**Key preview for {version}:** "
                + "  ".join([f"Q{i+1}:{a}" for i, a in enumerate(key[:10])]) + "  ...")

    st.session_state.student_info = {
        "name":    student_name,
        "reg_no":  student_reg,
        "version": version
    }
    st.markdown('</div>', unsafe_allow_html=True)


# ── STEP 3: Analyze ──
st.markdown("---")
st.markdown("#### <span class='step-label'>3</span> Analyze & Generate Score", unsafe_allow_html=True)

if st.button("🔍 Analyze OMR & Calculate Score", use_container_width=True):
    if not st.session_state.omr_answers:
        st.warning("⚠️ Please upload an OMR PDF first.")
    else:
        key_answers = ANSWER_KEYS[version]
        with st.spinner("Comparing answers..."):
            results, correct, wrong, skipped = calculate_results(
                st.session_state.omr_answers, key_answers)
            st.session_state.results = {
                "details": results, "correct": correct,
                "wrong": wrong, "skipped": skipped
            }

# ── STEP 4: Results ──
if st.session_state.results:
    r       = st.session_state.results
    correct = r["correct"]; wrong = r["wrong"]
    skipped = r["skipped"]; details = r["details"]
    pct     = round((correct / 100) * 100, 1)

    st.markdown("---")
    st.markdown("### 🏆 Results")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="score-box"><div class="score-big">{correct}</div><div class="score-sub">✅ Score / 100</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="score-box"><div class="score-big" style="color:#52c41a">{correct}</div><div class="score-sub">✅ Correct</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="score-box"><div class="score-big" style="color:#ff4d4f">{wrong}</div><div class="score-sub">❌ Wrong</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="score-box"><div class="score-big" style="color:#faad14">{skipped}</div><div class="score-sub">⬜ Skipped</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📋 Answer-by-Answer Breakdown")

    DISPLAY_COLS = 5
    header = ["Q#", "Your Ans", "Key Ans", "Status"] * DISPLAY_COLS
    table_rows = [header]
    chunk = []
    for row in details:
        symbol = {'correct': '✅', 'wrong': '❌', 'skipped': '⬜'}[row['status']]
        chunk.append([f"Q{row['q']}", row['student'], row['key'], symbol])
        if len(chunk) == DISPLAY_COLS:
            flat = []
            for c in chunk: flat += c
            table_rows.append(flat)
            chunk = []
    if chunk:
        flat = []
        for c in chunk: flat += c
        while len(flat) < DISPLAY_COLS * 4: flat += ["", "", "", ""]
        table_rows.append(flat)

    df = pd.DataFrame(table_rows[1:], columns=table_rows[0])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Download PDF ──
    st.markdown("---")
    st.markdown("#### 📄 Download Printable Result PDF")

    si = st.session_state.student_info or {}
    if st.button("📥 Generate & Download PDF Report", use_container_width=True):
        with st.spinner("Generating PDF..."):
            pdf_bytes = generate_result_pdf(si, details, correct, wrong, skipped, version)
            reg = si.get("reg_no", "student")
            st.download_button(
                label="⬇️ Download PDF — Click to Save & Print",
                data=pdf_bytes,
                file_name=f"PGCET_Result_{reg}_{version}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            st.success("✅ PDF ready! Open it in any PDF viewer to print.")
