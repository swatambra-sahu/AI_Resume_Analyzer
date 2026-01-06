# AI_Resume_Analyzer
Machine Learning–powered Streamlit app that analyzes PDF/TXT resumes using TF-IDF, Logistic Regression, and cosine similarity for ATS match scoring and skill-gap detection. Integrates OpenAI GPT to deliver structured, role-specific resume feedback through a modern interactive UI.

AI Resume Analyzer

Machine Learning + GPT-powered Resume Matching & Feedback System

The AI Resume Analyzer is an intelligent web application built using Streamlit, Machine Learning, and GPT-based evaluation to analyze user resumes, match them with job roles, and provide detailed improvement suggestions.

It extracts text from a resume, computes the similarity with industry-standard skill sets using ML, and generates expert-level resume feedback using GPT.

⭐ Key Features

🔍 1. Resume Text Extraction (PDF / TXT)

Extracts text from uploaded PDF and TXT files.
Handles multi-page resumes.
Detects unreadable or scanned PDFs.

🤖 2. ML-Based Resume–Job Matching

Uses TF-IDF vectorization + Logistic Regression.
Compares resume content with job-role skill requirements.
Outputs:
Match score (0–100)
Missing skills
Job role validation

🧠 3. GPT-Powered Resume Review

Structure & clarity analysis
Strengths and weaknesses
ATS compatibility score
Missing improvements
Actionable recommendations

🔄 4. Retrainable ML Model

Allows instant retraining using your dataset.
Automatically loads saved model for faster use.

🎨 5. Modern UI (Streamlit + CSS)

Gradient background
Shadowed cards
Highlighted skill sections
Clean and professional design


🛠 Tech Stack

Component	Technology
Frontend	Streamlit, Custom CSS
Backend	Python
ML	TF-IDF, Logistic Regression, scikit-learn
NLP	OpenAI GPT (gpt-4o-mini)
Storage	CSV dataset, joblib model serialization
File Parsing	PyPDF2
