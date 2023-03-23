import pandas_datareader as pdr
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
key="0d7d08b1ef617c881687c625f95bd9b9af0a12e8"
df= pdr.get_data_tiingo("AAPL",start='2021-01-25', end='2021-07-29', api_key=key)

print(df.head(5))

X = df.iloc[:, 3:4].values
Y = df.iloc[:, 0].values
print(X)
print(Y)



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
print(X_train)
print("X_test is:")
print(X_test)
print(y_train)
print("y-test is:")
print(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train[:, 0].reshape(-1, 1))
X_test = sc.transform(X_test[:, 0].reshape(-1, 1))
print(X_train)
print(X_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

with open("model.pkl", "wb") as f:
    pickle.dump(regressor, f)

print(len(X_train))
print(len(y_train))
print(y_pred)
# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# # # Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()