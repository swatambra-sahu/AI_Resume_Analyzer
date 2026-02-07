import streamlit as st
import PyPDF2
import io
import os
import pandas as pd
import joblib
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ==============================
# 🌍 Environment Setup
# ==============================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ==============================
# 🎨 Streamlit Page Configuration
# ==============================
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

# ==============================
# 💅 Custom Styling
# ==============================
st.markdown("""
<style>
   .stApp {
       background: linear-gradient(135deg, #E3F2FD, #F3E5F5);
       font-family: 'Inter', sans-serif;
   }
   .title {
       text-align: center;
       color: #1A237E;
       font-size: 2.5em;
       font-weight: bold;
   }
   .subtitle {
       text-align: center;
       color: #512DA8;
       font-size: 1.1em;
       margin-bottom: 2em;
   }
   .card {
       background-color: white;
       border-radius: 20px;
       padding: 25px;
       box-shadow: 0 4px 15px rgba(0,0,0,0.1);
       margin-top: 20px;
   }
   div.stButton > button:first-child {
       background-color: #3949AB;
       color: white;
       border-radius: 12px;
       font-weight: 600;
       padding: 0.6em 1.2em;
   }
   div.stButton > button:hover {
       background-color: #5C6BC0;
   }
   .score-box {
       background: #E8F5E9;
       border-left: 6px solid #2E7D32;
       padding: 15px;
       border-radius: 10px;
   }
   .missing-box {
       background: #FFF3E0;
       border-left: 6px solid #EF6C00;
       padding: 15px;
       border-radius: 10px;
   }
</style>
""", unsafe_allow_html=True)

# ==============================
# 📘 Page Header
# ==============================
st.markdown('<h1 class="title">📄 AI Resume Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI + ML-powered resume matching and GPT-based review 🔍</p>', unsafe_allow_html=True)

# ==============================
# 📂 File Upload and Inputs
# ==============================
col1, col2 = st.columns([1, 1])
with col1:
    uploaded_file = st.file_uploader("📤 Upload your resume (PDF or TXT):", type=["pdf", "txt"])
with col2:
    job_role = st.text_input("🎯 Enter your target job role:")

analyze = st.button("🚀 Analyze Resume")
retrain = st.button("🔄 Retrain Model")

# ==============================
# 📄 PDF/TXT Extraction
# ==============================
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text.strip()
    except Exception:
        st.error("❌ Unable to extract text from the PDF. Ensure it is not scanned or image-based.")
        return ""

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")

# ==============================
# 🧠 Model Loading / Training
# ==============================
@st.cache_resource
def load_or_train_model(force_retrain=False):
    csv_path = "datasets/resume_data.csv"
    model_path = "resume_match_model.pkl"

    if os.path.exists(model_path) and not force_retrain:
        vectorizer, data, clf, label_encoder = joblib.load(model_path)
        return vectorizer, data, clf, label_encoder

    if not os.path.exists(csv_path):
        st.error(f"❌ Dataset not found at `{csv_path}`. Please include 'Job_Role' and 'Skills' columns.")
        st.stop()

    data = pd.read_csv(csv_path)
    data.dropna(subset=["Job_Role", "Skills"], inplace=True)

    if "Job_Role" not in data.columns or "Skills" not in data.columns:
        st.error("❌ CSV must contain 'Job_Role' and 'Skills' columns.")
        st.stop()

    st.info("⚙️ Training model with improved pipeline...")

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(data["Skills"].astype(str))
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data["Job_Role"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    joblib.dump((vectorizer, data, clf, label_encoder), model_path)
    st.success("✅ Model trained and saved successfully!")
    return vectorizer, data, clf, label_encoder

vectorizer, data, clf, label_encoder = load_or_train_model()

if retrain:
    vectorizer, data, clf, label_encoder = load_or_train_model(force_retrain=True)

# ==============================
# 🤖 ML Matching
# ==============================
def predict_match_score(resume_text, job_role):
    job_role = job_role.strip().lower()
    if job_role not in data["Job_Role"].str.lower().values:
        return None, f"⚠️ Job role '{job_role}' not found in dataset. Please update resume_data.csv."

    job_index = data[data["Job_Role"].str.lower() == job_role].index[0]
    job_skills = data.loc[job_index, "Skills"]

    tfidf_resume = vectorizer.transform([resume_text])
    tfidf_job = vectorizer.transform([job_skills])

    similarity = cosine_similarity(tfidf_resume, tfidf_job)[0][0]
    score = round(similarity * 100, 2)

    missing_skills = [s.strip() for s in job_skills.split(",") if s.strip().lower() not in resume_text.lower()]
    return score, missing_skills

# ==============================
# 🚀 Main Execution
# ==============================
if analyze and uploaded_file:
    try:
        with st.spinner("🔍 Extracting text and analyzing..."):
            resume_text = extract_text_from_file(uploaded_file)

        if not resume_text.strip():
            st.error("❌ Could not extract text. The file may be empty or image-based.")
            st.stop()

        # --- ML Match ---
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🤖 Machine Learning Match Score")

        if job_role:
            score, missing = predict_match_score(resume_text, job_role)
            if score is not None:
                st.markdown(f"<div class='score-box'><b>Match Score:</b> {score}/100 ✅</div>", unsafe_allow_html=True)
                if missing:
                    st.markdown(f"<div class='missing-box'><b>Missing Skills:</b> {', '.join(missing)}</div>", unsafe_allow_html=True)
            else:
                st.warning(missing)
        else:
            st.info("ℹ️ Enter a job role to compute ML match score.")
        st.markdown('</div>', unsafe_allow_html=True)

        # --- GPT Feedback ---
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧠 AI Feedback (OpenAI GPT)")

        prompt = f"""
        You are an expert resume reviewer. 
        Evaluate the following resume for {job_role if job_role else "general roles"} and provide:
        1. Clarity, formatting, and structure analysis.
        2. Strengths and weaknesses.
        3. Skill and experience relevance.
        4. Concrete suggestions for improvement.
        Resume:
        {resume_text}
        
        Provide a structured analysis with actionable recommendations and ATS score (out of 100)."""

        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional resume analyst with HR experience."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        st.markdown(response.choices[0].message.content)
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
