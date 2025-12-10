text="email is danbowen@gmail.com"
text=text.split()
found=False
for i in text:
    if '@' in i:
        print(i)
        found=True
if not found:
    print("No email detected")