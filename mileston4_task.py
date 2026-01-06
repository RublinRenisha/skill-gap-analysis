import streamlit as st
import pdfplumber
import docx
import pandas as pd


st.title("Skill Gap Analyzer")
st.write("This app compares Resume skills with Job Description skills.")

st.sidebar.title("Navigation")
st.sidebar.write("Upload Files")
st.sidebar.write("Skill Analysis")
st.sidebar.write("Reports")


resume_file = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx", "txt"]
)

job_file = st.file_uploader(
    "Upload Job Description",
    type=["pdf", "docx", "txt"]
)


def read_text(file):
    text = ""

    if file.type == "text/plain":
        text = file.read().decode("utf-8")

    elif file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"

    else:
        text = "Unsupported file format."

    return text


if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if "job_text" not in st.session_state:
    st.session_state.job_text = ""


if resume_file:
    resume_text = read_text(resume_file)
    st.session_state.resume_text = resume_text
    st.write(resume_text[:300])

if job_file:
    job_text = read_text(job_file)
    st.session_state.job_text = job_text
    st.write(job_text[:300])


process = st.button("Analyze Skills")

if process:
    if not resume_file or not job_file:
        st.error("Please upload both Resume and Job Description.")
    else:
        st.subheader("Resume Preview")
        st.text(st.session_state.resume_text[:300])

        st.subheader("Job Description Preview")
        st.text(st.session_state.job_text[:300])

        
        resume_skills = {"python", "sql", "machine learning"}
        job_skills = {"python", "sql", "nlp", "deep learning"}

        matched_skills = resume_skills & job_skills
        missing_skills = job_skills - resume_skills

        match_percentage = (len(matched_skills) / len(job_skills)) * 100
        st.metric("Skill Match %", f"{match_percentage:.2f}%")

        st.subheader("Matched Skills")
        st.write(list(matched_skills))

        st.subheader("Missing Skills")
        st.write(list(missing_skills))

        
        chart_data = {
            "Matched Skills": len(matched_skills),
            "Missing Skills": len(missing_skills)
        }
        st.bar_chart(chart_data)

        
        data = {
            "Skill": list(job_skills),
            "Similarity Score": [0.9 if s in matched_skills else 0.0 for s in job_skills]
        }

        df = pd.DataFrame(data)
        st.table(df)

        
        csv = df.to_csv(index=False)

        st.download_button(
            label="Download Skill Gap Report",
            data=csv,
            file_name="skill_gap_report.csv",
            mime="text/csv"
        )

