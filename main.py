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
from reportlab.lib.enums import TA_CENTER
import pytesseract
import base64

# ═══════════════════════════════════════════════════════════════
#   ✅ ANSWER KEYS — Provide Master Key per Version Set
# ═══════════════════════════════════════════════════════════════
ANSWER_KEYS = {
    "A1": ['-'] * 100,
    "B1": ['-'] * 100,
    "C1": ['-'] * 100,
    "D1": ['-'] * 100,
}

# ═══════════════════════════════════════════════════════════════
#   🎯 LEVEL 5: IMMUTABLE COORDINATE MAP TEMPLATE (1200 x 1650 Canvas)
# ═══════════════════════════════════════════════════════════════
def generate_pgcet_template_map():
    """
    Generates precision fixed-coordinate mapping arrays for KEA PGCET layout.
    Tracks 4 parallel vertical tracks of 25 questions each across a 1200x1650 canvas.
    """
    template_map = {}
    
    # Precise anchor starts for columns 1-25, 26-50, 51-75, 76-100
    col_x_starts = [125, 410, 695, 980]
    row_y_start = 760
    row_gap = 32.5
    bubble_gap = 31
    
    for block in range(4):
        x_base = col_x_starts[block]
        for row in range(25):
            q_num = (block * 25) + row + 1
            y_center = int(row_y_start + (row * row_gap))
            
            # Map four horizontal options (A, B, C, D) coordinates
            options_coords = []
            for opt in range(4):
                x_center = int(x_base + (opt * bubble_gap))
                options_coords.append((x_center, y_center))
                
            template_map[q_num] = options_coords
            
    return template_map

QUESTION_MAP = generate_pgcet_template_map()

st.set_page_config(page_title="PGCET Industrial Template Engine", page_icon="📝", layout="wide")

# ─────────────────────────── Image Asset Loading ───────────────────────────
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return ""

img_base64 = get_base64_image("kea_banner.jpg")

# ─────────────────────────── CSS Styling ───────────────────────────
header_bg = f'background: linear-gradient(rgba(15,23,42,0.75), rgba(15,23,42,0.9)), url("data:image/jpeg;base64,{img_base64}") center/cover;' if img_base64 else 'background: linear-gradient(135deg, #1a1a2e, #0f3460);'
st.markdown(f"<style>.main-header {{{header_bg} padding: 3rem; text-align: center; border-radius: 12px; margin-bottom: 2rem; border: 1px solid #e94560;}} .main-header h1 {{color: white; margin:0;}} .main-header p {{color: white; background: #e94560; display:inline-block; padding:4px 16px; border-radius:20px; margin-top:10px;}} .step-card {{background: #16213e; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; border: 1px solid #0f3460;}} .score-box {{background: #0f3460; padding: 1.5rem; text-align: center; border-radius: 12px; border: 1px solid #e94560;}}</style>", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>📝 PGCET Industrial Template Engine</h1><p>Level 1-5 Production-Grade Coordinate Registration Architecture</p></div>', unsafe_allow_html=True)

# ─────────────────────────── Session States ───────────────────────────
for key in ["omr_answers", "student_info", "results"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ═══════════════════════════════════════════════════════════════
#  🔥 THE FIVE LEVELS COMPUTER VISION ARCHITECTURE
# ═══════════════════════════════════════════════════════════════

def pdf_to_images(pdf_bytes):
    images = convert_from_bytes(pdf_bytes, dpi=300)
    return [np.array(img.convert("RGB")) for img in images]


# LEVEL 1: Perspective Correction Transformation
def four_point_transform(image, pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    
    # Establish a rigid master digital viewport matrix canvas
    maxWidth, maxHeight = 1200, 1650
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


# LEVEL 2 & 3: Page Contour Isolation & Adaptive Threshold Tracking
def process_and_align_sheet(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)

    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    paper = None

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            paper = approx
            break

    # If the edge tracker fails due to low edge contrast, fall back on direct dimensions layout normalization
    if paper is None:
        H, W = img_np.shape[:2]
        paper = np.array([[0, 0], [W, 0], [W, H], [0, H]])

    warped_color = four_point_transform(img_np, paper.reshape(4, 2))
    warped_gray = cv2.cvtColor(warped_color, cv2.COLOR_RGB2GRAY)
    
    # Clean up ink payloads with specialized adaptive tracking
    thresh = cv2.adaptiveThreshold(
        warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 25, 15
    )
    
    return warped_color, thresh


# LEVEL 4 & 5: Fixed Template Mapping & Density Extractions
def extract_answers_via_template(warped_color, thresh):
    detected_letters = []
    mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    canvas = warped_color.copy()

    for q in range(1, 101):
        scores = []
        for option in range(4):
            cx, cy = QUESTION_MAP[q][option]
            
            # Extract localized Region of Interest (ROI) bounding circle
            roi = thresh[cy-11:cy+11, cx-11:cx+11]
            score = cv2.countNonZero(roi)
            scores.append(score)

        sorted_scores = sorted(scores, reverse=True)
        # Verify if bubble filling exceeds a standard target ink-mass density threshold
        if sorted_scores[0] > 140:
            # Enforce a strict difference gap margin between filled and empty selections
            if (sorted_scores[0] - sorted_scores[1]) > 50:
                chosen_opt = scores.index(sorted_scores[0])
                detected_letters.append(mapping[chosen_opt])
                
                # Highlight recognized selections on the UI image canvas
                tx, ty = QUESTION_MAP[q][chosen_opt]
                cv2.circle(canvas, (tx, ty), 12, (233, 69, 96), 3)
                continue
                
        detected_letters.append('-')

    return detected_letters, canvas


def extract_text_from_image(img_np):
    return pytesseract.image_to_string(Image.fromarray(img_np), config='--oem 3 --psm 11')


def parse_student_info(ocr_text):
    info = {"name": "", "reg_no": ""}
    lines = [l.strip() for l in ocr_text.split('\n') if l.strip()]
    for line in lines:
        reg_match = re.search(r'\b(\d{9})\b', line)
        if reg_match:
            info["reg_no"] = reg_match.group(1)
            break
    for line in lines:
        if re.match(r'^[A-Za-z\s\.]+$', line) and len(line) > 4 and not info["name"]:
            if not any(kw in line.lower() for kw in ['version','set','exam','date','roll','reg','barcode','authority','karnataka']):
                info["name"] = line.title()
    return info


def calculate_results(student_answers, key_answers):
    results = []
    correct = wrong = skipped = 0
    for i, (sa, ka) in enumerate(zip(student_answers, key_answers), start=1):
        if sa == '-': status, marks, skipped = 'skipped', 0, skipped + 1
        elif sa == ka: status, marks, correct = 'correct', 1, correct + 1
        else: status, marks, wrong = 'wrong', 0, wrong + 1
        results.append({'q': i, 'student': sa, 'key': ka, 'status': status, 'marks': marks})
    return results, correct, wrong, skipped


def generate_result_pdf(student_info, results, correct, wrong, skipped, version):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=15*mm, leftMargin=15*mm, topMargin=15*mm, bottomMargin=15*mm)
    story, styles = [], getSampleStyleSheet()
    
    h_style = ParagraphStyle('H', fontSize=18, fontName='Helvetica-Bold', alignment=TA_CENTER, textColor=colors.HexColor('#e94560'), spaceAfter=4)
    sub = ParagraphStyle('S', fontSize=11, fontName='Helvetica', alignment=TA_CENTER, spaceAfter=10)
    story.append(Paragraph("PGCET ENTRANCE EXAMINATION REPORT", h_style))
    story.append(Paragraph("Industrial Template Coordinate Mapping System", sub))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#e94560'), spaceAfter=10))
    
    info_data = [
        ["Candidate Name", ":", student_info.get("name", "DIVAKARA"), "Key Set Code", ":", version],
        ["Registration No", ":", student_info.get("reg_no", "249171118"), "Evaluated Qs", ":", "100 Items"]
    ]
    t = Table(info_data, colWidths=[35*mm, 5*mm, 55*mm, 35*mm, 5*mm, 45*mm])
    t.setStyle(TableStyle([('FONTNAME', (0,0), (-1,-1), 'Helvetica'), ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'), ('FONTNAME', (3,0), (3,-1), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,-1), 9.5), ('PADDING', (0,0), (-1,-1), 4)]))
    story.append(t)
    story.append(Spacer(1, 15))
    
    score_data = [
        ["FINAL NET SCORE", "CORRECT METRICS", "INCORRECT DETECTED", "SKIPPED MARKS"],
        [f"{correct} / 100", str(correct), str(wrong), str(skipped)]
    ]
    stb = Table(score_data, colWidths=[45*mm]*4)
    stb.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1a2e')), ('TEXTCOLOR', (0,0), (-1,0), colors.white), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,1), (-1,1), 14), ('TEXTCOLOR', (0,1), (0,1), colors.HexColor('#e94560')), ('PADDING', (0,0), (-1,-1), 6), ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#1a1a2e'))]))
    story.append(stb)
    story.append(Spacer(1, 15))
    
    COLS = 5
    rows = [["Q#", "Ans", "Key", "Res"] * COLS]
    chunk = []
    for r in results:
        sym = '✓' if r['status'] == 'correct' else '✗' if r['status'] == 'wrong' else '–'
        chunk.append([str(r['q']), r['student'], r['key'], sym])
        if len(chunk) == COLS:
            row = []
            for c in chunk: row += c
            rows.append(row)
            chunk = []
            
    mat_table = Table(rows, colWidths=[8*mm, 13*mm, 13*mm, 12*mm] * COLS)
    mat_table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f3460')), ('TEXTCOLOR', (0,0), (-1,0), colors.white), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTSIZE', (0,0), (-1,-1), 7.5), ('INNERGRID', (0,0), (-1,-1), 0.25, colors.lightgrey), ('BOX', (0,0), (-1,-1), 0.5, colors.grey)]))
    story.append(mat_table)
    
    doc.build(story)
    buffer.seek(0)
    return buffer.read()

# ═══════════════════════════════════════════════════════════════
#  RENDER APPLICATION LAYOUT VIEWPORT
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🛠️ Configuration")
    version = st.selectbox("📋 Answer Key Set Code", ["A1", "B1", "C1", "D1"], index=1)

col1, col2 = st.columns([1, 1], gap="large")
run_analysis = False

with col1:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("#### Step 1: Upload Student OMR PDF")
    omr_file = st.file_uploader("Upload OMR Sheet PDF", type=["pdf"], label_visibility="collapsed")
    
    if omr_file:
        st.success("✅ OMR File Received Successfully!")
        if "last_file" not in st.session_state or st.session_state.last_file != omr_file.name:
            with st.spinner("🚀 Performing Template Matrix Registration Analysis..."):
                try:
                    raw_images = pdf_to_images(omr_file.read())
                    
                    # Run Levels 1 to 3: Perspective Normalization
                    warped_color, thresh = process_and_align_sheet(raw_images[0])
                    
                    # Run Levels 4 & 5: Static Structural Mapping Engine
                    answers, verification_canvas = extract_answers_via_template(warped_color, thresh)
                    
                    st.session_state.processed_img = verification_canvas
                    st.session_state.omr_answers = answers
                    
                    # Profile Information Data Mining
                    ocr_text = extract_text_from_image(raw_images[0])
                    st.session_state.student_info = parse_student_info(ocr_text)
                    
                    st.session_state.last_file = omr_file.name
                    run_analysis = True
                except Exception as e:
                    st.error(f"Execution processing error encountered: {e}")
                    
        if "processed_img" in st.session_state:
            st.image(st.session_state.processed_img, caption="Aligned Frame with Fixed Coordinates Verification Dots", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("#### Step 2: Validated Metadata Identity Profiles")
    
    c_info = st.session_state.student_info if st.session_state.student_info is not None else {"name": "", "reg_no": ""}
    s_name = st.text_input("👤 Full Candidate Name", value=c_info.get("name") if c_info.get("name") else "DIVAKARA")
    s_reg = st.text_input("🔢 Parsed Registration ID Sequence", value=c_info.get("reg_no") if c_info.get("reg_no") else "249171118")
    
    if st.session_state.student_info is not None:
        if st.session_state.student_info.get("name") != s_name or st.session_state.student_info.get("reg_no") != s_reg:
            st.session_state.student_info = {"name": s_name, "reg_no": s_reg}
            run_analysis = True
    else:
        st.session_state.student_info = {"name": s_name, "reg_no": s_reg}
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.omr_answers is not None:
    st.markdown("---")
    st.markdown("#### 📝 Diagnostic Data Editor Framework Overrides")
    edit_df = pd.DataFrame({"Question Item": list(range(1, 101)), "Recognized Matrix Bubble": st.session_state.omr_answers})
    edited_df = st.data_editor(edit_df, use_container_width=True, hide_index=True, height=180, column_config={"Question Item": st.column_config.NumberColumn(disabled=True)})
    
    if st.session_state.omr_answers != edited_df["Recognized Matrix Bubble"].tolist():
        st.session_state.omr_answers = edited_df["Recognized Matrix Bubble"].tolist()
        run_analysis = True

if st.session_state.omr_answers is not None:
    if run_analysis or st.session_state.results is None:
        res, c, w, s = calculate_results(st.session_state.omr_answers, ANSWER_KEYS[version])
        st.session_state.results = {"details": res, "correct": c, "wrong": w, "skipped": s}

if st.session_state.results is not None:
    r = st.session_state.results
    si = st.session_state.student_info if st.session_state.student_info is not None else {"name": s_name, "reg_no": s_reg}
    
    pdf_report_bytes = generate_result_pdf(si, r["details"], r["correct"], r["wrong"], r["skipped"], version)
    
    st.markdown("---")
    st.download_button(label="📥 DOWNLOAD COMPREHENSIVE PERFORMANCE VERIFICATION REPORT (PDF)", data=pdf_report_bytes, file_name=f"PGCET_Advanced_Report_{si.get('reg_no')}.pdf", mime="application/pdf", use_container_width=True)
    
    st.markdown("### 📊 Metrics Analytical Scoreboard Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="score-box"><h2 style="color:#e94560; margin:0;">{r["correct"]}</h2><small>Final Net Marks</small></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="score-box"><h2 style="color:#52c41a; margin:0;">{r["correct"]}</h2><small>Correct Evaluated</small></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="score-box"><h2 style="color:#ff4d4f; margin:0;">{r["wrong"]}</h2><small>Incorrect Detected</small></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="score-box"><h2 style="color:#faad14; margin:0;">{r["skipped"]}</h2><small>Omitted Items</small></div>', unsafe_allow_html=True)
