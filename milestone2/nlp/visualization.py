import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

def generate_skill_chart(tech_count, soft_count):
    labels = ["Technical Skills", "Soft Skills"]
    values = [tech_count, soft_count]

    plt.figure()
    plt.pie(values, labels=labels, autopct="%1.1f%%")
    plt.title("Skill Distribution")

    plt.savefig("static/skill_chart.png")
    plt.close()

