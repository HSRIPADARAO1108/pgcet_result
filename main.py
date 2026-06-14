import streamlit as st
import cv2
import numpy as np
from pdf2image import convert_from_path
import pypdf
import os
import re

# --- Configuration & Styling ---
st.set_page_config(page_title="PGCET OMR Evaluator", layout="wide")
st.markdown("<h1 style='text-align: center; color: #1E88E5;'>🎓 PGCET MBA/MCA OMR Scanner Engine</h1>", unsafe_allow_html=True)
st.write("---")

# --- Sidebar Inputs & Calibration Grid ---
st.sidebar.header("📋 1. Exam Configuration")
exam_version = st.sidebar.selectbox("Select Paper Version Key:", ["A1", "B1", "C1", "D1"])
total_questions = st.sidebar.number_input("Total Questions to Evaluate:", min_value=1, max_value=200, value=100)

st.sidebar.header("📐 2. Bounding Box Calibration Matrix")
st.sidebar.markdown("Fine-tune these coordinates to align with your OMR sheet bubbles.")
x_start = st.sidebar.slider("First Bubble X-Coordinate (px):", 0, 2000, 150)
y_start = st.sidebar.slider("First Row Y-Coordinate (px):", 0, 3000, 400)
spacing = st.sidebar.slider("Horizontal Bubble Spacing (px):", 10, 100, 35)
row_gap = st.sidebar.slider("Vertical Row Gap (px):", 10, 150, 45)
b_width = st.sidebar.slider("Bubble Width (px):", 5, 50, 20)
b_height = st.sidebar.slider("Bubble Height (px):", 5, 50, 20)
min_pixel_threshold = st.sidebar.slider("Minimum Fill Density Threshold:", 50, 500, 150)

# --- App Layout Split ---
col1, col2 = st.columns()

with col1:
    st.subheader("📤 Step 1: Upload Student OMR Scan")
    omr_file = st.file_uploader("Upload Student OMR Sheet (PDF format)", type=["pdf"])
    
    st.subheader("📤 Step 2: Upload Official Key Answer")
    key_file = st.file_uploader("Upload Official Key Answers File (.txt)", type=["txt"])

# --- Core Helper Processing Engines ---
def extract_candidate_metadata(pdf_file):
    """Parses digital text inside the PDF safely without breaking if text is missing."""
    try:
        pdf_file.seek(0)
        reader = pypdf.PdfReader(pdf_file)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        
        name_match = re.search(r"(?:Candidate\s*Name|Name\s*of\s*the\s*Candidate)\s*:\s*([^\n]+)", full_text, re.IGNORECASE)
        reg_match = re.search(r"(?:Registration\s*Number|Reg\s*No|Roll\s*No)\s*:\s*([A-Z0-9]+)", full_text, re.IGNORECASE)
        
        candidate_name = name_match.group(1).strip() if name_match else "Scanned Document (Name Not Readable)"
        reg_number = reg_match.group(1).strip() if reg_match else "Scanned Document (ID Not Readable)"
        
        return candidate_name, reg_number
    except Exception:
        return "Unknown Candidate", "Unknown Reg No"

def process_omr_page(pdf_file):
    """Converts PDF page to grayscale and applies user's exact threshold pipeline safely."""
    try:
        pdf_file.seek(0)
        with open("temp_omr.pdf", "wb") as f:
            f.write(pdf_file.getbuffer())
        
        pages = convert_from_path("temp_omr.pdf", dpi=300)
        open_cv_image = np.array(pages[0].convert('RGB'))
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        if os.path.exists("temp_omr.pdf"):
            os.remove("temp_omr.pdf")
            
        return thresh, open_cv_image
    except Exception as e:
        st.error(f"Error handling image transformation pipelines: {e}")
        return None, None

def parse_answer_key(uploaded_file):
    """Parses plain text comma/newline key lists cleanly."""
    if uploaded_file is None:
        return {}
    try:
        uploaded_file.seek(0)
        content = uploaded_file.read().decode("utf-8").strip()
        cleaned_content = content.replace("\n", ",").replace(" ", "").split(",")
        
        parsed_keys = {}
        idx = 1
        for item in cleaned_content:
            if ":" in item:
                q_num, ans = item.split(":")
                parsed_keys[int(q_num)] = ans.upper()
            elif item.upper() in ["A", "B", "C", "D"]:
                parsed_keys[idx] = item.upper()
                idx += 1
        return parsed_keys
    except Exception as e:
        st.error(f"Answer Key Parsing Error: {e}")
        return {}

# --- Active Evaluation Trigger Pipeline ---
with col2:
    st.subheader("📊 Processing & Evaluation Results")
    
    if omr_file:
        thresh_img, original_img = process_omr_page(omr_file)
        student_name, registration_id = extract_candidate_metadata(omr_file)
        
        if thresh_img is not None:
            detected_answers = {}
            visual_debug_img = original_img.copy()
            options = ["A", "B", "C", "D"]

            # Process OMR bubbles via computer vision loops
            for q in range(total_questions):
                current_y = y_start + (q * row_gap)
                if current_y + b_height > thresh_img.shape[0]:
                    break
                
                bubble_choices_pixels = []
                for i in range(4):
                    current_x = x_start + (i * spacing)
                    if current_x + b_width > thresh_img.shape[1]:
                        break
                    
                    bubble_roi = thresh_img[current_y : current_y + b_height, current_x : current_x + b_width]
                    total_pixels = cv2.countNonZero(bubble_roi) if bubble_roi.size > 0 else 0
                    bubble_choices_pixels.append(total_pixels)
                    
                    cv2.rectangle(visual_debug_img, (current_x, current_y), 
                                  (current_x + b_width, current_y + b_height), (0, 255, 0), 2)

                if len(bubble_choices_pixels) == 4:
                    chosen_index = np.argmax(bubble_choices_pixels)
                    if bubble_choices_pixels[chosen_index] > min_pixel_threshold:
                        detected_answers[q + 1] = options[chosen_index]
                    else:
                        detected_answers[q + 1] = "Blank"
                else:
                    detected_answers[q + 1] = "Blank"

            # Display metadata panel profile
            st.markdown(f"""
            <div style="background-color:#1E1E1E; padding: 15px; border-radius: 8px; border-left: 5px solid #1E88E5; margin-bottom: 15px;">
                <h4 style="margin: 0; color: #FFF;">👤 Name: {student_name}</h4>
                <h5 style="margin: 5px 0 0 0; color: #AAA;">🆔 Reg No: {registration_id} | Paper Version: {exam_version}</h5>
            </div>
            """, unsafe_allow_html=True)

            # Conditional Step Check Matrix 
            if not key_file:
                st.warning("⚠️ OMR sheet scanned. Awaiting Official Key Answer file to evaluate rules.")
                
                # Show only scanned answers layout safely first
                omr_only_table = []
                for q_num in range(1, total_questions + 1):
                    omr_only_table.append({
                        "Question No.": f"Question {q_num}",
                        "OMR Answer": detected_answers.get(q_num, "Blank")
                    })
                st.dataframe(omr_only_table, height=350, use_container_width=True)
                
            else:
                # Key File Exists -> Perform Full Match Comparison Sequential Table
                answer_keys = parse_answer_key(key_file)
                total_earned_score = 0
                final_evaluation_table = []
                
                for q_num in range(1, total_questions + 1):
                    student_ans = detected_answers.get(q_num, "Blank")
                    correct_ans = answer_keys.get(q_num, "A")  # Default placeholder
                    
                    is_correct = (student_ans == correct_ans)
                    
                    if is_correct:
                        right_or_wrong = "Right"
                        score_value = 1
                        total_earned_score += 1
                    else:
                        right_or_wrong = "Wrong" if student_ans != "Blank" else "Unattempted"
                        score_value = 0
                    
                    # Exact sequence matching user expectations
                    final_evaluation_table.append({
                        "Question No.": f"Question {q_num}",
                        "OMR Answer": student_ans,
                        "Key Answer": correct_ans,
                        "Right or Wrong": right_or_wrong,
                        "Score": f"{score_value} Mark"
                    })
                
                st.success("✅ Evaluation complete!")
                st.metric(label="Total Score Calculated", value=f"{total_earned_score} / {total_questions} Marks")
                st.dataframe(final_evaluation_table, height=380, use_container_width=True)

    else:
        st.info("💡 Awaiting OMR sheet upload. Please upload the PDF on the left panel.")

# --- Visual Diagnostics Preview Layout ---
if omr_file and 'thresh_img' in locals() and thresh_img is not None:
    st.markdown("---")
    st.subheader("🔍 Scan Alignment Tracker Preview")
    preview_col1, preview_col2 = st.columns(2)
    with preview_col1:
        st.image(visual_debug_img, caption="Calibration Overlays (Original Sheet Frame)", use_container_width=True)
    with preview_col2:
        st.image(thresh_img, caption="Binarized Computer Recognition Mask Filter", use_container_width=True)
