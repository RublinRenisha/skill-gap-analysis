summary="Entry-Level Data Scientist who is keen to contribute to a cause. I am a young, energetic, and geeky individual whose desire to learn is endless."
word_count=len(summary.split())

import re
words = re.findall(r'\b\w+\b', summary.lower())
unique_count = len(set(words))

most_repeated = max(set(words), key=words.count)

print("total words:",word_count)
print("Unique words:",unique_count)
print("most repeated:",words.count(most_repeated))