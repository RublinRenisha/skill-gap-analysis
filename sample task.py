
print('Hello SkillGapAi')
#o/p: Hello SkillGapAi


name='Rublin Renisha'
print(name)
#Rublin Renisha


age=25
city='Chennai'
skill='Python'
print(age,city,skill)
# 25 Chennai Python


value=str(100)
print(type(value))
# <class 'str'>


text='Python basics'
print(text.upper())
# PYTHON BASICS


print(len('SkillGapAI'))
# 10

colors=['red','blue','green']
print(colors[1])
# blue


fruits=['apple','banana']
fruits.append('orange')
print(fruits)
# ['apple', 'banana', 'orange']

items=['apple','banana','orange']
items.remove('banana')
print(items)
#['apple', 'orange']

nums=[1,2,3]
t=tuple(nums)
print(t)
#(1, 2, 3)

resume={'name':'john','age':25,'skills':['Pyhton','SQL']}
print(resume['skills'])
#['Pyhton', 'SQL']

marks=50
print('pass'if marks>40 else 'Fail')
#pass

num=7
print('Even' if num%2==0 else 'Odd')
# Odd

for i in range(1,11):
    print(i)
#1
#2
#3
#4
#5
#6
#7
#8
#9
#10

i=5
while i>=1:
    print(i)
    i-=1
#5
#4
#3
#2
#1


def say_hello():
    print('Hello')
say_hello()
# Hello

def add(a,b):
    return a+b
print(add(1,2))
# 3

import math
print(math.pi)
#3.141592653589793

#(venv) PS C:\Users\USER\OneDrive\Desktop\Infosys internship>