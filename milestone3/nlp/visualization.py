import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_charts(result):
    os.makedirs("static/charts", exist_ok=True)

    labels = ["Matched", "Partial", "Missing"]
    values = [
        len(result["matched"]),
        len(result["partial"]),
        len(result["missing"])
    ]

    plt.figure(figsize=(5, 5))
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title("Skill Match Overview")
    plt.tight_layout()
    plt.savefig("static/charts/overview.png")
    plt.close()

    
    if result["similarity_matrix"]:
        plt.figure(figsize=(10, 6))  

        sns.heatmap(
            result["similarity_matrix"],
            xticklabels=result["jd_skills"],
            yticklabels=result["resume_skills"],
            cmap="YlGnBu",
            annot=False,
            linewidths=0.5
        )

        plt.title("Similarity Matrix", fontsize=16, pad=15)
        plt.xlabel("Job Description Skills", fontsize=12)
        plt.ylabel("Resume Skills", fontsize=12)

        
        plt.tight_layout(rect=[0.15, 0.05, 1, 1])

        plt.savefig("static/charts/heatmap.png", dpi=300)
        plt.close()

