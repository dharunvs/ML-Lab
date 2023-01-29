from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

neigh = KNeighborsRegressor()

dfx = pd.read_csv('weightedX.csv')
dfy = pd.read_csv('weightedY.csv')

X = np.array(dfx.values)
Y = np.array(dfy.values)

plt.scatter(X, Y)

X_test = np.linspace(-5, 12.5,10)
Y_test = []

fit = neigh.fit(dfx, dfy)

for i in X_test:
    Y_test.append(fit.predict([[i]]))

plt.scatter(X_test, Y_test, color="red")
plt.show()