import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# dataset = pd.read_csv('Position_Salaries.csv')
dataset = pd.read_csv('../Data.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Simple Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Polynomial Regression
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# # Linear Results

# # Real Results
# plt.scatter(X, y, color='red')
#
# plt.plot(X, lin_reg.predict(X), color='blue')
# plt.title('Truth or Bluff (Linear Regression)'))
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

# # Polynomial Results
# # Smoothing
# X_grid = np.arange(min(X), max(X), 0.1)
# X_grid = X_grid.reshape((len(X_grid), 1))
#
# # Real Results
# plt.scatter(X, y, color='red')
#
# plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
# plt.title('Truth or Bluff (Polynomial Regression)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

# predicting with linear
print('Linear Prediction:')
print(lin_reg.predict([[6.5]]))

# predict with polynomial
print('Polynomial Prediction:')
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))
