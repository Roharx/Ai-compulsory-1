import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# --------------------------------Data Preprocessing----------------------------------
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1]. values

# print(X)
# print(y)

# encoding male/female to 1/0
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# print(X)

# onehot encoding countries
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# print(X)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# scaling everything, absolutely necessary on everything for deep learning
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# --------------------------------Building the ANN----------------------------------
# init
ann = tf.keras.models.Sequential()
# add input layer & first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) # 6 nodes (experiment), rectifier activation function
# add 2nd hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# add output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # we need to predict 0 or 1, but we also get % w/ sigmoid
# for non-binary, we need softmax

# --------------------------------Training the ANN----------------------------------
# compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# adam: stochastic,
# binary_crossentropy: because our outcome is binary (1 or 0), non-binary: categorical_crossentropy

# training
ann.fit(X_train, y_train, batch_size=32, epochs=100)

# --------------------------------Homework: predict a single result----------------------------------
# France, cs: 600, male, 40, 3 years, 60k usd, products: 2, credit card: yes, active: yes, est. income: 50000
# prediction = ann.predict(sc.transform([[
#     1.0, 0.0, 0.0, # France
#     600,
#     1, # Male
#     40,
#     3,
#     60000,
#     2,
#     1,
#     1,
#     50000
# ]]))
#
# print(prediction > 0.5)
# --------------------------------Predict test set results----------------------------------
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)