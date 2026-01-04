from flask import Flask, render_template
import PyPDF2

from nlp.skill_gap import skill_gap_analysis
from nlp.visualization import generate_charts

app = Flask(__name__)


RESUME_PATH = "../milestone1/uploads/task resume.pdf"
JD_PATH = "../milestone1/uploads/job description.pdf"

def extract_text_from_pdf(path):
    text = ""
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

@app.route("/")
def index():
    resume_text = extract_text_from_pdf(RESUME_PATH)
    jd_text = extract_text_from_pdf(JD_PATH)

    result = skill_gap_analysis(resume_text, jd_text)

    generate_charts(result)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
