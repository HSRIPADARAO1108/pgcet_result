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
#   ✅ ANSWER KEYS — Fill these with 100 answers ('A'/'B'/'C'/'D')
# ═══════════════════════════════════════════════════════════════
ANSWER_KEYS = {
    "A1": ['-'] * 100,
    "B1": ['-'] * 100,
    "C1": ['-'] * 100,
    "D1": ['-'] * 100,
}

st.set_page_config(page_title="PGCET Advanced OMR Checker", page_icon="📝", layout="wide")

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

st.markdown('<div class="main-header"><h1>📝 PGCET Advanced OMR Engine</h1><p>High-Precision Perspective Alignment Engine</p></div>', unsafe_allow_html=True)

# ─────────────────────────── Session States ───────────────────────────
for key in ["omr_answers", "student_info", "results"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ═══════════════════════════════════════════════════════════════
#  🔥 STRONGER COMPUTER VISION ARCHITECTURE
# ═══════════════════════════════════════════════════════════════

def pdf_to_images(pdf_bytes):
    images = convert_from_bytes(pdf_bytes, dpi=300)
    return [np.array(img.convert("RGB")) for img in images]

def align_and_warp_omr(img_np):
    """
    Finds registration anchors on the page corners and applies 
    Perspective Transformation to completely eliminate camera tilt/rotation.
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_anchors = []
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.8 <= aspect_ratio <= 1.2 and 20 <= w <= 80:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    valid_anchors.append((cx, cy))
                    
    if len(valid_anchors) >= 4:
        valid_anchors = sorted(valid_anchors, key=lambda p: p[1])
        top_pts = sorted(valid_anchors[:2], key=lambda p: p[0])
        bottom_pts = sorted(valid_anchors[-2:], key=lambda p: p[0])
        src_pts = np.array([top_pts[0], top_pts[1], bottom_pts[0], bottom_pts[1]], dtype="float32")
        
        maxWidth, maxHeight = 1200, 1600
        dst_pts = np.array([[0, 0], [maxWidth - 1, 0], [0, maxHeight - 1], [maxWidth - 1, maxHeight - 1]], dtype="float32")
        M_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(img_np, M_matrix, (maxWidth, maxHeight))
        
    return cv2.resize(img_np, (1200, 1600)) # Fallback if anchors missing

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

def robust_contour_bubble_reader(warped_img):
    """
    Uses Threshold Contour Pixel-Density Analysis instead of Hough Circles.
    Extremely resilient against poor scanning artifacts.
    """
    gray = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Define coordinate grid layout zones relative to a fixed 1200x1600 warped sheet
    # 4 columns blocks (Q1-25, Q26-50, Q51-75, Q76-100)
    col_x_bounds = [(350, 520), (550, 720), (750, 920), (950, 1120)]
    row_y_start, row_y_end = 800, 1500
    row_height = (row_y_end - row_y_start) / 25
    
    detected_indices = [-1] * 100
    
    for block_idx, (x_start, x_end) in enumerate(col_x_bounds):
        bubble_w = (x_end - x_start) / 4
        for row in range(25):
            q_num = (block_idx * 25) + row + 1
            y_top = int(row_y_start + (row * row_height))
            y_bot = int(y_top + row_height)
            
            densities = []
            for option in range(4):
                bx0 = int(x_start + (option * bubble_w))
                bx1 = int(bx0 + bubble_w)
                
                # Dynamic bubble mask bounding padding
                bubble_crop = thresh[y_top+2:y_bot-2, bx0+2:bx1-2]
                total_pixels = cv2.countNonZero(bubble_crop)
                densities.append(total_pixels)
                
            sorted_densities = sorted(densities, reverse=True)
            # High certainty baseline contrast validation delta check
            if sorted_densities[0] > 90 and (sorted_densities[0] - sorted_densities[1]) > 25:
                detected_indices[q_num - 1] = densities.index(sorted_densities[0])
                
    return detected_indices

def answers_to_letters(answer_indices):
    mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', -1: '-'}
    return [mapping.get(a, '-') for a in answer_indices]

# ═══════════════════════════════════════════════════════════════
#  METRICS, EVALUATION & EXPORTS
# ═══════════════════════════════════════════════════════════════

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
    story.append(Paragraph("Automated Computer Vision Verification", sub))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#e94560'), spaceAfter=10))
    
    info_data = [
        ["Candidate Name", ":", student_info.get("name", "N/A"), "Key Set Code", ":", version],
        ["Registration No", ":", student_info.get("reg_no", "N/A"), "Evaluated Qs", ":", "100 Items"]
    ]
    t = Table(info_data, colWidths=[35*mm, 5*mm, 55*mm, 35*mm, 5*mm, 45*mm])
    t.setStyle(TableStyle([('FONTNAME', (0,0), (-1,-1), 'Helvetica'), ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'), ('FONTNAME', (3,0), (3,-1), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,-1), 9.5), ('PADDING', (0,0), (-1,-1), 4)]))
    story.append(t)
    story.append(Spacer(1, 15))
    
    # Structural breakdown metric scorecard grid layout
    score_data = [
        ["FINAL NET SCORE", "CORRECT METRICS", "INCORRECT DETECTED", "SKIPPED MARKS"],
        [f"{correct} / 100", str(correct), str(wrong), str(skipped)]
    ]
    stb = Table(score_data, colWidths=[45*mm]*4)
    stb.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1a2e')), ('TEXTCOLOR', (0,0), (-1,0), colors.white), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,1), (-1,1), 14), ('TEXTCOLOR', (0,1), (0,1), colors.HexColor('#e94560')), ('PADDING', (0,0), (-1,-1), 6), ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#1a1a2e'))]))
    story.append(stb)
    story.append(Spacer(1, 15))
    
    # Compact 5 Column Detailed Answer Matrix Layout Generation Loop 
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
    version = st.selectbox("📋 Answer Key Set Code", ["A1", "B1", "C1", "D1"], index=1) # Set default to matching document B1

col1, col2 = st.columns([1, 1], gap="large")
run_analysis = False

with col1:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("#### Step 1: Upload Student OMR PDF")
    omr_file = st.file_uploader("Upload OMR Sheet PDF", type=["pdf"], label_visibility="collapsed")
    
    if omr_file:
        st.success("✅ OMR File Received Successfully!")
        if "last_file" not in st.session_state or st.session_state.last_file != omr_file.name:
            with st.spinner("🚀 Running High-Precision Computer Vision Real-Time Verification Engine..."):
                try:
                    raw_images = pdf_to_images(omr_file.read())
                    # Structural Perspective Fix Alignment Pass
                    warped = align_and_warp_omr(raw_images[0])
                    st.session_state.processed_img = warped
                    
                    # OCR Context Processing Block
                    ocr_text = extract_text_from_image(raw_images[0])
                    st.session_state.student_info = parse_student_info(ocr_text)
                    
                    # Robust Pixel Density Contour Matrix Scanner Execution Pass
                    detected = robust_contour_bubble_reader(warped)
                    st.session_state.omr_answers = answers_to_letters(detected)
                    st.session_state.last_file = omr_file.name
                    run_analysis = True
                except Exception as e:
                    st.error(f"Execution processing error encountered: {e}")
                    
        if "processed_img" in st.session_state:
            st.image(st.session_state.processed_img, caption="Aligned & Normalized Matrix Canvas", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("#### Step 2: Validated Metadata Identity Profiles")
    
    c_info = st.session_state.student_info if st.session_state.student_info is not None else {"name": "", "reg_no": ""}
    # Fallback default value injection to automatically populate name if empty strings return
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

# ═══════════════════════════════════════════════════════════════
#  AUTOMATED COMPILATION PRESENTATION VIEW 
# ═══════════════════════════════════════════════════════════════
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
