skills = ["Python","Java","C","GIT","Eclipse","Rapid Miner","Apache Mahout","Hive","MapReduce","MongoDB","Excel","SAS","R"]

data_ml = []
web = []
software = []
other = []

for skill in skills:
    s = skill.lower()

    if s in ['python', 'r', 'rapid miner', 'apache mahout', 'hive', 'mapreduce', 'sas']:
        data_ml.append(skill)

    elif s in ['html', 'css', 'javascript', 'react', 'angular', 'vue']:
        web.append(skill)

    elif s in ['java', 'c', 'c++', 'c#', 'ruby', 'swift', 'kotlin', 'eclipse', 'git']:
        software.append(skill)

    else:
        other.append(skill)

print("data/ML:", data_ml)
print("web:", web)
print("software:", software)
print("other:", other)
