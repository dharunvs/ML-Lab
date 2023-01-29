import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("survey-lung-cancer.csv") 
df['LUNG_CANCER']=df['LUNG_CANCER'].map({'YES':1,'NO':0})

df['GENDER']=df['GENDER'].map({'M':1,'F':0})
print(df.head())
print("\n")
print(df.tail())

from sklearn.model_selection import train_test_split

X = df.iloc[:]. values
y = df.iloc[:, -1]. values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# print("\n",y_pred)
np.set_printoptions(precision=2)
for i in range(len(y_pred)):
    print(y_test[i], y_pred[i])

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
print(cm)