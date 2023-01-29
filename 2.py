'''Detecting Spam mails using Support vector machine'''

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("spam.csv") 

from sklearn.model_selection import train_test_split

X = df["Message"].values 
y = df["Category"].values 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer() 
X_train = cv.fit_transform(X_train) 
X_test = cv.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test) 
print(y_pred)

from sklearn.metrics import confusion_matrix, accuracy_score 

print(confusion_matrix(y_test, y_pred)) 
print(accuracy_score(y_test, y_pred))
