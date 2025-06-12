import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Reading Data from sauce
df = pd.read_csv("/home/ellah/ros2_ws/src/linear_regression_pkg/linear_regression_pkg/boston_housing.csv",sep=",", header=None)
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

X = df.drop('MEDV', axis=1).values
y = df['MEDV'].values

# Data preprocessing
X = df.iloc[:, :-1].values
y = df.iloc[:, 0].values

#split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Print confirmation
print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

#plotting the line of regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
line = regressor.coef_[0] * X_test[:, 0] + regressor.intercept_

plt.scatter(X_test[:, 0], y_test)
plt.plot(X_test[:, 0], line, color='red')
plt.show()

count = df.B.value_counts()

count.plot(kind='pie',autopct='%1.1f%%')
plt.title('Category Distribution')

plt.savefig("B_class_dist.pdf",dpi=100)

plt.show()

#Making Prediction
#testing
print(X_test)
    #model Prediction
y_pred = regressor.predict(X_test)

# Comparing actual results to the predicted model result
df = pd.DataFrame({'Actual': y_test, 'predicted': y_pred})
df

# Ploting the bar graph to predict the difference between the actual and predicted value
df.head(20).plot(kind='bar', figsize=(8,4))
plt.grid(which='major', linewidth=0.5, color='red')
plt.grid(which='minor', linewidth=0.2, color='blue')
plt.show()


