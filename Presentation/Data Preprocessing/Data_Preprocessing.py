import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Import the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values # Location, Age, Salary
y = dataset.iloc[:, -1].values # Purchased

# print(X)
# print(y)

# Taking care of the missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# print(X)

# print('-------------------------------------')

# encode categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# remainder='passthrough': we want to keep the columns that won't be affected by the transformation
X = np.array(ct.fit_transform(X))

# print(y)
# print('-------------------------')

le = LabelEncoder()
y = le.fit_transform(y)

# print(y)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Show:
# print(X_train)
# print('--------------------------------')
# print(X_test)
# print('--------------------------------')
# print(y_train)
# print('--------------------------------')
# print(y_test)

# print(X_test)
# Feature Scaling
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
# print(X_test)