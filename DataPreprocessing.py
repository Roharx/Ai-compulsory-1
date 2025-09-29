import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import dataset
# --------------------------------
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
# [:, :-1] means, we're taking all the rows ([:] = beginning:end)
# and every row except the last ([:-1] = beginning:end-1)
# when we have : in an array, we have a range which goes from value to value
# if the range is left empty, then by default, it will be: [start:end]
y = dataset.iloc[:, -1].values
# here, we're taking every row but only the last column (not a range from column to column) so the value is -1 (end-1)

# Show:
# print(X)
# print(y)

# Taking care of missing data
# --------------------------------
# To have the least amount of impact, we're replacing them with the avg. of the column's data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3]) # Age (2nd) & salary (3rd) column, string (1st column) has no mean
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Show:
# print(X)

# Encoding categorical data
# --------------------------------
# Encoding the independent variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# remainder='passthrough': we want to keep the columns that won't be affected by the transformation
X = np.array(ct.fit_transform(X))

# Show:
# print(X)

# Encoding the dependent variable
le = LabelEncoder()
y = le.fit_transform(y)

# Show:
# print(y)

# Feature scaling
# --------------------------------
# It should be applied after splitting the training and test set:
# Test set = new observations, like future data we'd be getting
# Feature scaling consists of scaling all the vars/features to make sure they all take values in the same scale

# Reason: the test set should be a "brand new set of data", not scaled with our current set of data

# Splitting the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# test_size=0.2: 80%-20% train-test
# random factors will happen since it will split the data randomly, to have the same results, random_state can be
# given a value

# Show:
# print(X_train)
# print('--------------------------------')
# print(X_test)
# print('--------------------------------')
# print(y_train)
# print('--------------------------------')
# print(y_test)

# Feature scaling
sc = StandardScaler()
# We're not applying feature scaling to the dummy variables (countries) since they are already in the range
# of our feature scaling values. Applying feature scaling to them might increase the performance a little
# but it's so small, it is negligible.
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
# We're selecting from the 4th column till the last (0, 1 and 2 are our dummy variables for the country names)

X_test[:, 3:] = sc.transform(X_test[:, 3:])

# Show:
print(X_train)
print(X_test)