from flask import Flask, render_template, request, send_file
import spacy
import matplotlib.pyplot as plt
import pdfplumber, docx
from spacy.matcher import PhraseMatcher
from collections import Counter
from io import BytesIO
import pandas as pd
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

skills_list = [
    "python","java","c","c++","c#","r","javascript","typescript","go","ruby","php",
    "machine learning","deep learning","artificial intelligence","nlp",
    "tensorflow","keras","pytorch","scikit-learn",
    "pandas","numpy","matplotlib","seaborn",
    "sql","mysql","postgresql","mongodb",
    "html","css","react","angular","vue","nodejs","express","django","flask",
    "aws","azure","docker","kubernetes","ci/cd","git","github",
    "communication","debugging","unit testing"
]

skill_categories = {

    "Programming Languages": [
        "python","java","c","c++","c#","r",
        "javascript","typescript","go","ruby",
        "php","swift","kotlin","scala","perl",
        "matlab","bash","shell scripting"
    ],

    "Data Science, ML & AI": [
        "machine learning","deep learning","artificial intelligence",
        "supervised learning","unsupervised learning",
        "natural language processing","nlp",
        "computer vision","time series analysis",
        "feature engineering","model evaluation",
        "data preprocessing","statistical modeling"
    ],

    "ML / AI Libraries & Frameworks": [
        "tensorflow","keras","pytorch","scikit-learn",
        "xgboost","lightgbm","catboost",
        "huggingface","transformers",
        "spacy","nltk","opencv"
    ],

    "Data Analysis & Visualization": [
        "pandas","numpy","scipy",
        "matplotlib","seaborn","plotly",
        "tableau","power bi","excel",
        "google sheets","data visualization"
    ],

    "Databases & Big Data": [
        "sql","mysql","postgresql","sqlite",
        "oracle","mongodb","cassandra",
        "redis","elasticsearch",
        "hadoop","spark","hive",
        "big data","data warehousing"
    ],

    "Web Development": [
        "html","css","react","angular","vue",
        "nodejs","express","django",
        "flask","fastapi",
        "rest api","graphql",
        "bootstrap","tailwind css"
    ],

    "Cloud & DevOps": [
        "aws","azure","google cloud",
        "ec2","s3","lambda",
        "docker","kubernetes",
        "ci/cd","jenkins",
        "gitlab ci","terraform",
        "ansible","linux","unix"
    ],

    "Software Engineering & Tools": [
        "git","github","gitlab","bitbucket",
        "jira","confluence",
        "unit testing","integration testing",
        "debugging","code review",
        "agile","scrum","waterfall",
        "object oriented programming",
        "design patterns"
    ],

    "Cybersecurity & Networking": [
        "cybersecurity","network security",
        "ethical hacking","penetration testing",
        "cryptography","firewalls",
        "authentication","authorization",
        "owasp","risk assessment"
    ],

    "Testing & QA": [
        "manual testing","automation testing",
        "selenium","cypress",
        "test cases","test planning",
        "bug tracking","quality assurance"
    ],

    "Management & Business": [
        "project management","product management",
        "stakeholder management","risk management",
        "resource planning","budgeting",
        "business analysis","requirements gathering",
        "process improvement","documentation"
    ],

    "Soft Skills": [
        "communication","verbal communication",
        "written communication",
        "teamwork","collaboration",
        "leadership","decision making",
        "problem solving","critical thinking",
        "time management","adaptability",
        "creativity","conflict resolution",
        "presentation skills"
    ],

    "Academic & Research": [
        "research","literature survey",
        "technical writing","report writing",
        "data interpretation","analysis"
    ]
}


matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
matcher.add("SKILLS", [nlp.make_doc(s) for s in skills_list])

def extract_text(file):
    if file.filename.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file) as pdf:
            for p in pdf.pages:
                text += p.extract_text() or ""
        return text
    elif file.filename.endswith(".docx"):
        doc = docx.Document(file)
        return " ".join(p.text for p in doc.paragraphs)
    return file.read().decode("utf-8")

def extract_skill_counts(text):
    doc = nlp(text.lower())
    matches = matcher(doc)
    return Counter(doc[s:e].text for _, s, e in matches)

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        resume = request.files["resume"]
        jd = request.form["jd"]

        resume_c = extract_skill_counts(extract_text(resume))
        jd_c = extract_skill_counts(jd)
        jd_c = dict(jd_c)


        full, partial, missing = [], [], []
        for s in jd_c:
            if s in resume_c:
                ratio = resume_c[s] / jd_c[s]
                full.append(s) if ratio >= 0.8 else partial.append(s)
            else:
                missing.append(s)

        match_percent = round(
            (len(full) + 0.5*len(partial)) / len(jd_c) * 100, 2
        )

        os.makedirs("static/charts", exist_ok=True)

        plt.figure(figsize=(5,5))
        plt.pie(
            [len(full), len(partial), len(missing)],
            labels=["Fully Matched","Partially Matched","Missing"],
            autopct="%1.1f%%",
            colors=["#4CAF50","#FFC107","#F44336"],
            labeldistance=1.1
        )
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig("static/charts/overall.png", bbox_inches="tight")
        plt.close()

        skills = list(jd_c.keys())

        resume_vals = [1 if s in resume_c else 0 for s in skills]
        jd_vals = [1] * len(skills)

        plt.figure(figsize=(10, 5))

        for i, skill in enumerate(skills):
            plt.plot([resume_vals[i], jd_vals[i]], [i, i], color="#D3D3D3", linewidth=2)
            plt.scatter(resume_vals[i], i, color="red", s=120)
            plt.scatter(jd_vals[i], i, color="green", s=120)

        plt.yticks(range(len(skills)), [s.title() for s in skills])
        plt.xticks([0, 1], ["Resume", "Job Description"])
        plt.title("Skill Comparison")
        plt.grid(axis="x", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig("static/charts/lollipop.png", bbox_inches="tight")
        plt.close()

        category_charts = []

        for category, skills in skill_categories.items():
            jd_skills = [s for s in skills if s in jd_c]

            if not jd_skills:
                continue

            matched = len([s for s in jd_skills if s in resume_c])
            missing_count = len(jd_skills) - matched


            plt.figure(figsize=(2.5, 2.5))
            plt.pie(
                [matched, missing_count],
                autopct="%1.0f%%",
                startangle=90,
                colors=["#4CAF50", "#E0E0E0"],
                textprops={'fontsize': 16, 'weight': 'bold'}
            )

            plt.axis("equal")

            filename = f"static/charts/{category.replace(' ','_')}.png"
            plt.savefig(filename, bbox_inches="tight")
            plt.close()

            category_charts.append({
                "name": category,
                "image": filename
            })

        max_len = max(len(full), len(partial), len(missing))
        df = pd.DataFrame({
            "Fully Matched": full + [""]*(max_len-len(full)),
            "Partially Matched": partial + [""]*(max_len-len(partial)),
            "Missing": missing + [""]*(max_len-len(missing))
        })
        df.to_csv("static/report.csv", index=False)

        
        try:
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer)
            styles = getSampleStyleSheet()
            content = []

            content.append(Paragraph("Skill Gap Analysis Report", styles["Title"]))
            content.append(Paragraph(f"Overall Match: {match_percent}%", styles["Normal"]))

            if os.path.exists("static/charts/overall.png"):
                content.append(Paragraph("Overall Skill Match Distribution", styles["Heading2"]))
                content.append(Image("static/charts/overall.png", width=220, height=220))

            content.append(Paragraph("JD Category-wise Skill Match", styles["Heading2"]))
            for cat in category_charts:
                if os.path.exists(cat["image"]):
                    content.append(Paragraph(cat["name"], styles["Heading3"]))
                    content.append(Image(cat["image"], width=160, height=160))

            if os.path.exists("static/charts/lollipop.png"):
                content.append(Paragraph("Skill-by-Skill Comparison (Resume vs JD)", styles["Heading2"]))
                content.append(Image("static/charts/lollipop.png", width=420, height=240))

            content.append(Paragraph("Fully Matched Skills", styles["Heading2"]))
            content.append(Paragraph(", ".join(full) if full else "None", styles["Normal"]))

            content.append(Paragraph("Partially Matched Skills", styles["Heading2"]))
            content.append(Paragraph(", ".join(partial) if partial else "None", styles["Normal"]))

            content.append(Paragraph("Missing Skills", styles["Heading2"]))
            content.append(Paragraph(", ".join(missing) if missing else "None", styles["Normal"]))

            doc.build(content)
            buffer.seek(0)

            with open("static/report.pdf", "wb") as f:
                f.write(buffer.read())

        except Exception as e:
            print("PDF generation skipped:", e)


        return render_template(
            "index.html",
            analyzed=True,
            match=match_percent,
            full=full[:5],
            partial=partial[:5],
            missing=missing[:5],
            category_charts=category_charts
        )

    return render_template("index.html", analyzed=False)

@app.route("/download/pdf")
def download_pdf():
    return send_file("static/report.pdf", as_attachment=True)

@app.route("/download/csv")
def download_csv():
    return send_file("static/report.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
