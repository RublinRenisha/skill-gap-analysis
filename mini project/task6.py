def count_skills_in_text(skills, text):
    text_lower = text.lower().split()
    skill_counts = {}
    for skill in skills:
        count = text_lower.count(skill.lower())
        skill_counts[skill] = count
    return skill_counts
skills = ["Python","Java","C","GIT","Eclipse","Rapid Miner","Apache Mahout","Hive","MapReduce","MongoDB","Excel","SAS","R"]
paragraph="Data Science Major in Foundational courses in Mathematics, Computing and Statistics Part of the Maths & Statistics Society for 3 years Took additional courses in Big Data Ecosysters and Data Visualisation Developed a company-wide digitized filing system with Python and Java Developed internal analytics and data science services to tackle rising internal displacement in the Middle East Sourcing data for the UN from MongoDB with Rapid Miner and sorting it with Python scripts and Excel "
counts = count_skills_in_text(skills, paragraph)
for skill, count in counts.items():
    print(f"{skill} appears {count} times")