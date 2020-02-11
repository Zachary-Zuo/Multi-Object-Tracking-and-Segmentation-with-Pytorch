lines = open("text.txt").readlines()

# remove comments (lines starting with #) (like python)
lines = [l if not l.strip().startswith("#") else "\n" for l in lines]
s = "".join(lines)
print(s)