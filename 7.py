'''Implement sentiment analysis using random 
forest optimization algorithm'''

import pandas as pd

dataset = pd.read_csv("Restaurant_Reviews.csv")

corpus = []
for i in range(0,1000):
    review = dataset['Review'][i]
    review = review.lower()
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer(max_features=1500) 
X = cv.fit_transform(corpus)
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test) 

print(y_pred)

from sklearn.metrics import confusion_matrix, accuracy_score

print(confusion_matrix(y_test, y_pred)) 
print(accuracy_score(y_test, y_pred))
