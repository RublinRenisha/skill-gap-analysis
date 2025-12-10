import re
def extract_experience(text):
    pattern = r'(\d+)\s*(?:years|year|yrs|yr)'
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    
    if matches:
        total_experience = sum(int(num) for num in matches)
        print(f"Experience detected : {total_experience} years")
    else:
        print("Experience detected : 0 years")
text = "2 years of experience in data scientist and 1 year as data analyst"
extract_experience(text)