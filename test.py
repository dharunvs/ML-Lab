import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

a = [["11", "12"]
      ["22", "13"]
      ["33", "14"]
      ["44", "15"]]

from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer() 
a = cv.fit_transform(a) 

print(a)    