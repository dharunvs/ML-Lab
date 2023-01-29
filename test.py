import random

with open("weightedX.csv", 'r') as f:
      a=f.read().split("\n")[:-1]

# print(a)

print(round(random.random()*10))
s= ""


for i in a:
      s+= "" + str(float(i)+round(random.random()*10)) + "\n"

with open("weightedY.csv", "w") as f:
      f.write(s)
      