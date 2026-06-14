import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes
import json
import io
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import pytesseract

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
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .main-header h1 { color: #e94560; font-size: 2.2rem; margin: 0; }
    .main-header p  { color: #a8b2d8; margin: 0.5rem 0 0; font-size: 1.1rem; }

    .step-card {
        background: #16213e;
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .step-label {
        background: #e94560;
        color: white;
        border-radius: 50%;
        width: 28px; height: 28px;
        display: inline-flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 0.9rem;
        margin-right: 0.5rem;
    }

    .score-box {
        background: linear-gradient(135deg, #0f3460, #16213e);
        border: 2px solid #e94560;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .score-big { font-size: 4rem; font-weight: 900; color: #e94560; line-height: 1; }
    .score-sub { color: #a8b2d8; font-size: 1rem; margin-top: 0.5rem; }

    .correct-tag   { background: #1a472a; color: #52c41a; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }
    .wrong-tag     { background: #4a1515; color: #ff4d4f; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }
    .skipped-tag   { background: #2d2d1a; color: #faad14; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }

    .info-chip {
        display: inline-block;
        background: #0f3460;
        color: #a8b2d8;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.85rem;
        margin: 3px;
    }
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #e94560, #c73652);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-weight: 600; font-size: 1rem;
        width: 100%;
    }
    div[data-testid="stButton"] > button:hover { opacity: 0.9; }
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
for key in ["omr_answers", "student_info", "key_answers", "results"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ═══════════════════════════════════════════════════════════════
#  OMR PROCESSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def pdf_to_images(pdf_bytes):
    images = convert_from_bytes(pdf_bytes, dpi=300)
    return [np.array(img.convert("RGB")) for img in images]


def enhance_image(img_np):
    """Enhance OMR image for better bubble detection."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    return sharpened


def extract_text_from_image(img_np):
    """Use OCR to extract student info from OMR image."""
    pil_img = Image.fromarray(img_np)
    text = pytesseract.image_to_string(pil_img, config='--psm 6')
    return text


def parse_student_info(ocr_text):
    """Parse student name, registration number, and version from OCR text."""
    info = {"name": "", "reg_no": "", "version": ""}
    lines = [l.strip() for l in ocr_text.split('\n') if l.strip()]

    for line in lines:
        # Registration number patterns
        reg_match = re.search(r'\b(\d{6,12})\b', line)
        if reg_match and not info["reg_no"]:
            info["reg_no"] = reg_match.group(1)

        # Version detection
        ver_match = re.search(r'\b([ABCD]1)\b', line, re.IGNORECASE)
        if ver_match and not info["version"]:
            info["version"] = ver_match.group(1).upper()

        # Name: line with mostly alphabetic content
        if re.match(r'^[A-Za-z\s\.]+$', line) and len(line) > 4 and not info["name"]:
            if not any(kw in line.lower() for kw in ['version', 'set', 'exam', 'date', 'roll', 'reg', 'serial']):
                info["name"] = line.title()

    return info


def detect_bubbles_in_row(row_img, num_options=4):
    """
    Detect which bubble is filled in a single question row.
    Returns: 0=A, 1=B, 2=C, 3=D, -1=no answer / multiple
    """
    h, w = row_img.shape
    cell_w = w // num_options
    fill_ratios = []

    for i in range(num_options):
        cell = row_img[:, i * cell_w:(i + 1) * cell_w]
        _, binary = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ratio = np.sum(binary == 255) / binary.size
        fill_ratios.append(ratio)

    max_ratio = max(fill_ratios)
    if max_ratio < 0.08:   # Nothing filled
        return -1

    # Check if only one bubble is significantly darker than others
    sorted_ratios = sorted(fill_ratios, reverse=True)
    if len(sorted_ratios) > 1 and sorted_ratios[0] > sorted_ratios[1] * 1.4:
        return fill_ratios.index(max_ratio)

    return fill_ratios.index(max_ratio)  # Best guess


def detect_omr_answers(img_np, num_questions=100, num_options=4):
    """
    Main OMR detection: find the answer grid and read each row.
    Returns list of 100 answers (0-3 = A-D, -1 = skipped)
    """
    enhanced = enhance_image(img_np)
    h, w = enhanced.shape

    # Binary threshold
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find horizontal contours / rows (question rows)
    # Typical OMR: bubble area is roughly 50-80% width, centered
    # We divide the bubble region into 100 equal rows

    # Crop to expected answer region (remove header ~20%, footer ~10%)
    top_crop    = int(h * 0.20)
    bottom_crop = int(h * 0.90)
    left_crop   = int(w * 0.30)
    right_crop  = int(w * 0.85)

    bubble_region = binary[top_crop:bottom_crop, left_crop:right_crop]
    region_h, region_w = bubble_region.shape

    row_height = region_h // num_questions
    if row_height < 1:
        row_height = 1

    answers = []
    for q in range(num_questions):
        y1 = q * row_height
        y2 = (q + 1) * row_height
        row = bubble_region[y1:y2, :]
        if row.size == 0:
            answers.append(-1)
            continue
        ans = detect_bubbles_in_row(row, num_options)
        answers.append(ans)

    return answers


def answers_to_letters(answer_indices):
    """Convert 0-3 indices to A,B,C,D or '-' for skipped."""
    mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', -1: '-'}
    return [mapping.get(a, '-') for a in answer_indices]


# ═══════════════════════════════════════════════════════════════
#  KEY ANSWER PARSING
# ═══════════════════════════════════════════════════════════════

def parse_key_pdf(pdf_bytes, version):
    """
    Parse answer key PDF. Expects text like:
    1.A  2.B  3.C ... or tables.
    Returns dict {version: [100 answers]}
    """
    import pdfplumber
    key_answers = {}

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text() or ""

    # Try to find all 4 versions
    versions = ["A1", "B1", "C1", "D1"]
    for ver in versions:
        # Look for version section
        pattern = rf'{ver}.*?(?={"|".join(v for v in versions if v != ver)}|$)'
        match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
        section = match.group(0) if match else full_text

        # Extract Q->Answer pairs: "1.A", "1) A", "1 A", "1.B" etc.
        pairs = re.findall(r'(\d{1,3})[.):\s]+([ABCD])', section, re.IGNORECASE)
        if pairs:
            ans_dict = {int(q): a.upper() for q, a in pairs}
            answers_list = [ans_dict.get(i, '-') for i in range(1, 101)]
            if sum(1 for a in answers_list if a != '-') >= 20:  # At least 20 found
                key_answers[ver] = answers_list

    # If only one version found, assign to all
    if not key_answers and full_text:
        pairs = re.findall(r'(\d{1,3})[.):\s]+([ABCD])', full_text, re.IGNORECASE)
        if pairs:
            ans_dict = {int(q): a.upper() for q, a in pairs}
            answers_list = [ans_dict.get(i, '-') for i in range(1, 101)]
            for ver in versions:
                key_answers[ver] = answers_list

    return key_answers


def parse_key_manual(text_input):
    """Parse manually entered key: '1.A 2.B 3.C ...' or one per line."""
    pairs = re.findall(r'(\d{1,3})[.):\s]+([ABCD])', text_input, re.IGNORECASE)
    if not pairs:
        return None
    ans_dict = {int(q): a.upper() for q, a in pairs}
    return [ans_dict.get(i, '-') for i in range(1, 101)]


# ═══════════════════════════════════════════════════════════════
#  RESULT CALCULATION
# ═══════════════════════════════════════════════════════════════

def calculate_results(student_answers, key_answers):
    """
    Compare student answers with key.
    Returns: list of dicts per question + summary
    """
    results = []
    correct = wrong = skipped = 0

    for i, (sa, ka) in enumerate(zip(student_answers, key_answers), start=1):
        if sa == '-' or sa == '':
            status = 'skipped'
            marks = 0
            skipped += 1
        elif sa == ka:
            status = 'correct'
            marks = 1
            correct += 1
        else:
            status = 'wrong'
            marks = 0
            wrong += 1
        results.append({
            'q': i,
            'student': sa,
            'key': ka,
            'status': status,
            'marks': marks
        })

    return results, correct, wrong, skipped


# ═══════════════════════════════════════════════════════════════
#  RESULT PDF GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_result_pdf(student_info, results, correct, wrong, skipped, version):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=15*mm, leftMargin=15*mm,
        topMargin=15*mm, bottomMargin=15*mm
    )
    story = []
    styles = getSampleStyleSheet()

    # ── Header ──
    header_style = ParagraphStyle('header', fontSize=18, fontName='Helvetica-Bold',
                                   alignment=TA_CENTER, textColor=colors.HexColor('#e94560'),
                                   spaceAfter=4)
    sub_style    = ParagraphStyle('sub', fontSize=11, fontName='Helvetica',
                                   alignment=TA_CENTER, textColor=colors.HexColor('#333333'),
                                   spaceAfter=2)
    story.append(Paragraph("PGCET ENTRANCE EXAMINATION", header_style))
    story.append(Paragraph("Official OMR Answer Report", sub_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#e94560')))
    story.append(Spacer(1, 8))

    # ── Student Info Table ──
    info_data = [
        ["Student Name", ":", student_info.get("name", "N/A"),
         "Version / Set", ":", version],
        ["Registration No.", ":", student_info.get("reg_no", "N/A"),
         "Total Questions", ":", "100"],
    ]
    info_table = Table(info_data, colWidths=[38*mm, 5*mm, 55*mm, 38*mm, 5*mm, 35*mm])
    info_table.setStyle(TableStyle([
        ('FONTNAME',  (0,0), (-1,-1), 'Helvetica'),
        ('FONTNAME',  (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',  (3,0), (3,-1), 'Helvetica-Bold'),
        ('FONTSIZE',  (0,0), (-1,-1), 9),
        ('BACKGROUND',(0,0), (-1,-1), colors.HexColor('#f5f5f5')),
        ('ROWBACKGROUNDS',(0,0),(-1,-1),[colors.HexColor('#f0f4ff'), colors.HexColor('#ffffff')]),
        ('BOX',       (0,0), (-1,-1), 0.5, colors.grey),
        ('INNERGRID', (0,0), (-1,-1), 0.25, colors.lightgrey),
        ('PADDING',   (0,0), (-1,-1), 5),
        ('VALIGN',    (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 10))

    # ── Score Summary ──
    total_marks = correct
    pct = round((correct / 100) * 100, 1)

    if pct >= 70:   grade, gcol = "A (Excellent)", colors.HexColor('#52c41a')
    elif pct >= 50: grade, gcol = "B (Good)",      colors.HexColor('#1890ff')
    elif pct >= 35: grade, gcol = "C (Pass)",      colors.HexColor('#faad14')
    else:           grade, gcol = "D (Below Average)", colors.HexColor('#ff4d4f')

    score_data = [
        ["TOTAL SCORE", "CORRECT", "WRONG", "SKIPPED", "PERCENTAGE", "GRADE"],
        [
            Paragraph(f'<font size="20" color="#e94560"><b>{total_marks}/100</b></font>', styles['Normal']),
            Paragraph(f'<font size="14" color="#52c41a"><b>{correct}</b></font>', styles['Normal']),
            Paragraph(f'<font size="14" color="#ff4d4f"><b>{wrong}</b></font>',   styles['Normal']),
            Paragraph(f'<font size="14" color="#faad14"><b>{skipped}</b></font>', styles['Normal']),
            Paragraph(f'<font size="14"><b>{pct}%</b></font>', styles['Normal']),
            Paragraph(f'<font size="11"><b>{grade}</b></font>', styles['Normal']),
        ]
    ]
    score_table = Table(score_data, colWidths=[30*mm]*6)
    score_table.setStyle(TableStyle([
        ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,0), 8),
        ('BACKGROUND',  (0,0), (-1,0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR',   (0,0), (-1,0), colors.white),
        ('BACKGROUND',  (0,1), (-1,1), colors.HexColor('#f0f4ff')),
        ('ALIGN',       (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('BOX',         (0,0), (-1,-1), 1, colors.HexColor('#0f3460')),
        ('INNERGRID',   (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
        ('ROWHEIGHT',   (0,0), (-1,0), 18),
        ('ROWHEIGHT',   (0,1), (-1,1), 30),
        ('PADDING',     (0,0), (-1,-1), 5),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 12))

    # ── Answer-by-Answer Table ──
    story.append(Paragraph("<b>Detailed Answer Analysis</b>",
                            ParagraphStyle('head2', fontSize=11, fontName='Helvetica-Bold',
                                           spaceAfter=6, textColor=colors.HexColor('#1a1a2e'))))

    # Build rows in groups of 5 cols (Q | Your | Key | Result) × 5
    COLS = 5
    header_row = []
    for _ in range(COLS):
        header_row += ["Q#", "Your Ans", "Key Ans", "Result"]

    rows = [header_row]
    chunk = []
    for r in results:
        color_tag = {'correct': '#52c41a', 'wrong': '#ff4d4f', 'skipped': '#faad14'}[r['status']]
        result_cell = Paragraph(
            f'<font color="{color_tag}"><b>{"✓" if r["status"]=="correct" else ("✗" if r["status"]=="wrong" else "–")}</b></font>',
            styles['Normal']
        )
        chunk.append([str(r['q']), r['student'], r['key'], result_cell])
        if len(chunk) == COLS:
            row = []
            for c in chunk:
                row += c
            rows.append(row)
            chunk = []

    if chunk:  # Remaining
        row = []
        for c in chunk:
            row += c
        # Pad empty
        while len(row) < COLS * 4:
            row += ["", "", "", ""]
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
    # Alternating row bg
    for i in range(1, len(rows)):
        bg = colors.HexColor('#f9f9f9') if i % 2 == 0 else colors.white
        ts.append(('BACKGROUND', (0,i), (-1,i), bg))

    ans_table.setStyle(TableStyle(ts))
    story.append(ans_table)
    story.append(Spacer(1, 10))

    # ── Footer ──
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 4))
    footer_style = ParagraphStyle('footer', fontSize=7, alignment=TA_CENTER,
                                   textColor=colors.grey)
    story.append(Paragraph(
        "This is a computer-generated report. Official marks are subject to verification by the examining authority.",
        footer_style
    ))
    story.append(Paragraph(
        f"Generated by PGCET OMR Checker  |  Reg No: {student_info.get('reg_no','N/A')}  |  Version: {version}",
        footer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR – INSTRUCTIONS
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📋 How to Use")
    st.markdown("""
1. **Upload OMR PDF** – Student's scanned OMR copy  
2. **Select Version** – A1 / B1 / C1 / D1  
3. **Upload Key** – Answer key PDF (or enter manually)  
4. **Analyze** – Click to compare & score  
5. **Download PDF** – Printable result report  
    """)
    st.markdown("---")
    st.markdown("### ℹ️ Exam Info")
    st.info("📌 100 Questions | MCQ | 1 mark each | Max 100 marks")
    st.markdown("---")
    st.markdown("### 🎯 Score Legend")
    st.markdown("""
- 🟢 **A** – 70+ marks (Excellent)  
- 🔵 **B** – 50–69 (Good)  
- 🟡 **C** – 35–49 (Pass)  
- 🔴 **D** – Below 35  
    """)


# ═══════════════════════════════════════════════════════════════
#  MAIN UI – STEP 1: OMR Upload
# ═══════════════════════════════════════════════════════════════
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("#### <span class='step-label'>1</span> Upload Student OMR PDF", unsafe_allow_html=True)

    omr_file = st.file_uploader("Upload OMR PDF", type=["pdf"], key="omr_upload",
                                  label_visibility="collapsed")

    version = st.selectbox("📋 Select Answer Version / Set",
                            ["A1", "B1", "C1", "D1"],
                            index=0)

    # Manual student info override
    with st.expander("✏️ Enter / Override Student Details"):
        manual_name   = st.text_input("Student Name", placeholder="e.g. Rahul Kumar")
        manual_reg    = st.text_input("Registration Number", placeholder="e.g. 2024MBA001")

    if omr_file:
        st.success("✅ OMR PDF uploaded!")
        with st.spinner("🔍 Enhancing & reading OMR..."):
            try:
                pdf_bytes = omr_file.read()
                images    = pdf_to_images(pdf_bytes)

                # Show enhanced preview
                enhanced_preview = enhance_image(images[0])
                st.image(enhanced_preview, caption="Enhanced OMR (Page 1)", use_container_width=True,
                         clamp=True)

                # OCR for student info
                ocr_text  = extract_text_from_image(images[0])
                auto_info = parse_student_info(ocr_text)

                student_info = {
                    "name":   manual_name  if manual_name  else auto_info.get("name", ""),
                    "reg_no": manual_reg   if manual_reg   else auto_info.get("reg_no", ""),
                    "version": version
                }
                st.session_state.student_info = student_info

                # Detect answers
                raw_answers = detect_omr_answers(images[0])
                letter_answers = answers_to_letters(raw_answers)
                st.session_state.omr_answers = letter_answers

                # Show parsed info
                st.markdown(f"""
<span class='info-chip'>👤 {student_info['name'] or 'Name not detected'}</span>
<span class='info-chip'>🔢 {student_info['reg_no'] or 'Reg not detected'}</span>
<span class='info-chip'>📋 Version {version}</span>
""", unsafe_allow_html=True)

                # Show detected answers preview
                answered = sum(1 for a in letter_answers if a != '-')
                skipped  = 100 - answered
                st.markdown(f"**Detected:** {answered} answered · {skipped} skipped")

            except Exception as e:
                st.error(f"Error processing OMR: {e}")
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  STEP 2: Answer Key
# ═══════════════════════════════════════════════════════════════
with col2:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("#### <span class='step-label'>2</span> Upload / Enter Answer Key", unsafe_allow_html=True)

    key_mode = st.radio("Key Input Method", ["Upload Key PDF", "Enter Manually"],
                         horizontal=True, label_visibility="collapsed")

    if key_mode == "Upload Key PDF":
        key_file = st.file_uploader("Upload Answer Key PDF", type=["pdf"],
                                     key="key_upload", label_visibility="collapsed")
        if key_file:
            with st.spinner("Parsing answer key..."):
                try:
                    key_bytes   = key_file.read()
                    all_keys    = parse_key_pdf(key_bytes, version)
                    if version in all_keys:
                        st.session_state.key_answers = all_keys[version]
                        st.success(f"✅ Key loaded for Version {version} – {sum(1 for a in all_keys[version] if a!='-')} answers found")
                    elif all_keys:
                        # Use first available
                        first_ver = list(all_keys.keys())[0]
                        st.session_state.key_answers = all_keys[first_ver]
                        st.warning(f"Version {version} not found; using {first_ver}")
                    else:
                        st.error("Could not parse key. Try manual entry.")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.markdown("Enter answers as `1.A 2.B 3.C ...` or one per line `1) A`")
        key_text = st.text_area("Answer Key (Q.Answer format)",
                                  placeholder="1.A 2.B 3.C 4.D 5.A ...\nor\n1) A\n2) B\n3) C",
                                  height=200)
        if st.button("Parse Key", key="parse_key_btn"):
            parsed = parse_key_manual(key_text)
            if parsed:
                st.session_state.key_answers = parsed
                found = sum(1 for a in parsed if a != '-')
                st.success(f"✅ {found} answers parsed!")
            else:
                st.error("Could not parse. Use format: 1.A 2.B 3.C ...")

    # Preview key
    if st.session_state.key_answers:
        with st.expander("👁️ Preview Answer Key"):
            preview = " · ".join([f"Q{i+1}:{a}" for i, a in
                                   enumerate(st.session_state.key_answers[:20])])
            st.code(preview + " ...")

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  STEP 3: ANALYZE
# ═══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("#### <span class='step-label'>3</span> Analyze & Generate Report", unsafe_allow_html=True)

analyze_btn = st.button("🔍 Analyze OMR & Calculate Score", use_container_width=True)

if analyze_btn:
    if not st.session_state.omr_answers:
        st.warning("⚠️ Please upload an OMR PDF first.")
    elif not st.session_state.key_answers:
        st.warning("⚠️ Please provide the answer key.")
    else:
        with st.spinner("Comparing answers..."):
            results, correct, wrong, skipped = calculate_results(
                st.session_state.omr_answers,
                st.session_state.key_answers
            )
            st.session_state.results = {
                "details": results, "correct": correct,
                "wrong": wrong, "skipped": skipped
            }

# ═══════════════════════════════════════════════════════════════
#  STEP 4: RESULTS DISPLAY
# ═══════════════════════════════════════════════════════════════
if st.session_state.results:
    r       = st.session_state.results
    correct = r["correct"]
    wrong   = r["wrong"]
    skipped = r["skipped"]
    details = r["details"]
    pct     = round((correct / 100) * 100, 1)

    st.markdown("---")
    st.markdown("### 🏆 Results")

    # Score cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="score-box"><div class="score-big">{correct}</div><div class="score-sub">✅ Correct</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="score-box"><div class="score-big" style="color:#ff4d4f">{wrong}</div><div class="score-sub">❌ Wrong</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="score-box"><div class="score-big" style="color:#faad14">{skipped}</div><div class="score-sub">⬜ Skipped</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="score-box"><div class="score-big">{pct}%</div><div class="score-sub">📊 Percentage</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Detailed table
    st.markdown("#### 📋 Answer-by-Answer Breakdown")

    # Build display table
    DISPLAY_COLS = 5
    header = []
    for _ in range(DISPLAY_COLS):
        header += ["Q#", "Your Ans", "Key Ans", "Status"]

    table_rows = [header]
    chunk = []
    for row in details:
        status_html = {
            'correct': "✅",
            'wrong':   "❌",
            'skipped': "⬜"
        }[row['status']]
        chunk.append([f"Q{row['q']}", row['student'], row['key'], status_html])
        if len(chunk) == DISPLAY_COLS:
            flat = []
            for c in chunk:
                flat += c
            table_rows.append(flat)
            chunk = []

    if chunk:
        flat = []
        for c in chunk:
            flat += c
        while len(flat) < DISPLAY_COLS * 4:
            flat += ["", "", "", ""]
        table_rows.append(flat)

    # Display using st.table
    import pandas as pd
    cols_labels = []
    for i in range(DISPLAY_COLS):
        cols_labels += [f"Q#", f"Yours", f"Key", f"✓/✗"]

    df = pd.DataFrame(table_rows[1:], columns=table_rows[0])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Generate & Download PDF ──
    st.markdown("---")
    st.markdown("#### 📄 Download Result Report (Printable PDF)")

    si = st.session_state.student_info or {}
    # Allow override before PDF
    with st.expander("✏️ Confirm Student Details for PDF"):
        pdf_name   = st.text_input("Name for PDF",   value=si.get("name", ""),   key="pdf_name")
        pdf_reg    = st.text_input("Reg No for PDF", value=si.get("reg_no", ""), key="pdf_reg")
        pdf_ver    = st.selectbox("Version for PDF", ["A1","B1","C1","D1"],
                                   index=["A1","B1","C1","D1"].index(version), key="pdf_ver")

    if st.button("📥 Generate Printable PDF Report", use_container_width=True):
        with st.spinner("Generating PDF..."):
            final_info = {"name": pdf_name, "reg_no": pdf_reg, "version": pdf_ver}
            pdf_bytes  = generate_result_pdf(
                final_info, details, correct, wrong, skipped, pdf_ver
            )
            st.download_button(
                label="⬇️ Click to Download PDF",
                data=pdf_bytes,
                file_name=f"PGCET_Result_{pdf_reg or 'student'}_{pdf_ver}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            st.success("✅ PDF ready! Click the button above to download & print.")
