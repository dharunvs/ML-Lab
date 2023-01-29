'''Choose best machine learning algorithm to 
implement online fraud detection'''

import numpy as np
import pandas as pd

dataset = pd.read_csv('PS_20174392719_1491204439457_log.csv')
X=np.array(dataset[["type","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]])
y = np.array(dataset["isFraud"])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,0] = le.fit_transform(X[:,0])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state = 0)
classifier1.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier
classifier2 = DecisionTreeClassifier(random_state = 0)
classifier2.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
classifier3 = RandomForestClassifier(random_state = 0)
classifier3.fit(X_train, y_train)

y_pred1 = classifier1.predict(X_test)
y_pred2 = classifier2.predict(X_test)
y_pred3 = classifier3.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score

print(confusion_matrix(y_test, y_pred1))
print(accuracy_score(y_test, y_pred1))

print(confusion_matrix(y_test, y_pred2))
print(accuracy_score(y_test, y_pred2))

print(confusion_matrix(y_test, y_pred3))
print(accuracy_score(y_test, y_pred3))
