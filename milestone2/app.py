from flask import Flask, render_template
import PyPDF2

from nlp.spacy_extractor import extract_skills_spacy
from nlp.bert_extractor import extract_skills_bert
from nlp.skill_merger import merge_skills
from nlp.visualization import generate_skill_chart

app = Flask(__name__)

# Milestone 1 output PDF path
PDF_PATH = "../milestone1/uploads/task resume.pdf"

def extract_text_from_pdf(path):
    text = ""
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

@app.route("/")
def index():
    resume_text = extract_text_from_pdf(PDF_PATH)

    spacy_skills = extract_skills_spacy(resume_text)
    bert_skills = extract_skills_bert(resume_text)

    final_skills = merge_skills(spacy_skills, bert_skills)

    generate_skill_chart(
        final_skills["tech_count"],
        final_skills["soft_count"]
    )

    return render_template("index.html", skills=final_skills)

if __name__ == "__main__":
    app.run(debug=True)
