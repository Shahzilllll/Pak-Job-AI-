import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import PyPDF2
import re

# ----------------- Hide Streamlit default footer and menu -----------------
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ----------------- Page Config -----------------
st.set_page_config(
    page_title="Pak Job AI",
    layout="wide"
)

# ----------------- Load Data & Models -----------------
@st.cache_data(show_spinner=True)
def load_data_and_models():
    df = pd.read_csv("cleaned_data.csv")
    model = joblib.load("salary_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    tfidf_matrix = joblib.load("tfidf_matrix.pkl")
    le_city = joblib.load("le_city.pkl")
    le_dept = joblib.load("le_dept.pkl")
    gap_df = pd.read_csv("student_skill_gap.csv")
    roadmap_df = pd.read_csv("learning_roadmap.csv")
    return df, model, tfidf, tfidf_matrix, le_city, le_dept, gap_df, roadmap_df

df, model, tfidf, tfidf_matrix, le_city, le_dept, gap_df, roadmap_df = load_data_and_models()

# ----------------- Helper Functions -----------------
def clean_jd(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", "", text).lower()
    return re.sub(r"[^a-zA-Z\s]", "", text)

def normalize_skill(s):
    return s.strip().lower()

def is_skill_matched(a, b):
    return SequenceMatcher(None, a, b).ratio() > 0.8

def missing_skills(student_skills, market_skills):
    return [
        m for m in market_skills
        if not any(is_skill_matched(s, m) for s in student_skills)
    ]

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    return " ".join(page.extract_text() or "" for page in reader.pages)

def extract_text(file):
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    return ""

def extract_skills_from_text(text, market_skills):
    words = set(text.lower().replace("\n", " ").split())
    return [skill for skill in market_skills if skill in words]

gap_df["skill"] = gap_df["skill"].apply(normalize_skill)
roadmap_df["skill"] = roadmap_df["skill"].apply(normalize_skill)
market_skills = gap_df["skill"].tolist()

# ----------------- App Layout -----------------
st.title("🚀 Pakistan Job Market: AI Dashboard")
st.caption("AI-powered insights, job matching, salary prediction & skill gap analysis")

menu = [
    "📊 Market Insights",
    "🤖 Job Recommendation",
    "💰 Salary Estimator",
    "📉 Skill Gap Analyzer"
]
choice = st.sidebar.radio("Navigation", menu)

# ----------------- Market Insights -----------------
if choice == "📊 Market Insights":
    st.header("📊 Market Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Hiring Cities")
        st.bar_chart(df["City"].value_counts().head(10))

    with col2:
        st.subheader("Department Distribution")
        fig = px.pie(df, names="Department", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

# ----------------- Job Recommendation -----------------
elif choice == "🤖 Job Recommendation":
    st.header("🤖 AI Job Finder")
    user_input = st.text_input("Enter your skills (e.g., Python, SQL, Data Analysis)")

    if st.button("Find Jobs"):
        if user_input.strip():
            query_vec = tfidf.transform([clean_jd(user_input)])
            scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
            top_idx = scores.argsort()[-5:][::-1]

            st.subheader("Top Job Matches")
            for i in top_idx:
                st.info(
                    f"*{df.iloc[i]['Job Name']}*  \n"
                    f"{df.iloc[i]['Company Name']} — {df.iloc[i]['City']}"
                )
        else:
            st.warning("Please enter your skills.")

# ----------------- Salary Estimator -----------------
elif choice == "💰 Salary Estimator":
    st.header("💰 Salary Predictor")
    col1, col2, col3 = st.columns(3)

    with col1:
        city = st.selectbox("City", sorted(df["City"].unique()))
    with col2:
        dept = st.selectbox("Department", sorted(df["Department"].unique()))
    with col3:
        exp = st.slider("Experience (Years)", 0, 20, 2)

    if st.button("Predict Salary"):
        city_code = le_city.transform([city])[0]
        dept_code = le_dept.transform([dept])[0]
        salary = model.predict([[exp, city_code, dept_code]])[0]
        st.success(f"💵 Estimated Salary: Rs. {int(salary):,}")

# ----------------- Skill Gap Analyzer -----------------
elif choice == "📉 Skill Gap Analyzer":
    st.header("📉 AI Skill Gap Analyzer")

    uploaded_file = st.file_uploader("Upload your CV (PDF or TXT)", type=["pdf", "txt"])
    manual_input = st.text_input("Or enter skills manually (comma separated)")

    student_skills = []

    if uploaded_file:
        text = extract_text(uploaded_file)
        student_skills = extract_skills_from_text(text, market_skills)

    if manual_input:
        for s in manual_input.split(","):
            s = normalize_skill(s)
            if s and s not in student_skills:
                student_skills.append(s)

    if student_skills:
        gaps = missing_skills(student_skills, market_skills)
        gap_score = 100 - int(len(gaps) / len(market_skills) * 100)

        st.metric(
            "Skill Readiness Score",
            f"{gap_score}/100",
            f"{len(gaps)} skills missing" if gaps else "Market-ready"
        )

        st.subheader("📌 Missing Skills")
        if gaps:
            st.write(", ".join(gaps[:5]))
            if len(gaps) > 5:
                with st.expander("Show more"):
                    st.write(", ".join(gaps[5:]))
        else:
            st.success("You’re aligned with current market demand.")

        st.subheader("🛠 Learning Roadmap")
        roadmap = roadmap_df[roadmap_df["skill"].isin(gaps)]
        if not roadmap.empty:
            st.dataframe(roadmap.reset_index(drop=True))
        else:
            st.info("Focus on refining existing skills.")
    else:
        st.info("Upload a CV or enter at least one skill.")
