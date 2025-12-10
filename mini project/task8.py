tools = ["Python","Java","C","GIT","Eclipse","Rapid Miner","Apache Mahout","Hive","MapReduce","MongoDB","Excel","SAS","R"]

programming_languages = ['python', 'java', 'c', 'c++', 'r']
databases = ['mongodb', 'mysql', 'postgresql', 'oracle', 'sqlite']
frameworks = ['django', 'flask', 'react', 'angular', 'spring']

count_programming = 0
count_database = 0
count_framework = 0
count_other = 0

for tool in tools:
    t = tool.lower()
    if t in programming_languages:
        count_programming += 1
    elif t in databases:
        count_database += 1
    elif t in frameworks:
        count_framework += 1
    else:
        count_other += 1

print("Programming Languages:", count_programming)
print("Databases:", count_database)
print("Frameworks:", count_framework)
print("Other:", count_other)
