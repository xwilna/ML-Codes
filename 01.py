import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

diabetes =load_diabetes()

X = diabetes.data
y = diabetes.target

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)

model = LinearRegression(n_jobs=-1)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

rmse = root_mean_squared_error(y_test, y_pred)
print(rmse)

plt.plot( y_test, 'b', label='Real Data')
plt.plot( y_pred, 'r', label='Predicted Data')

plt.legend()
plt.show()
