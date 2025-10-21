import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------Data Preprocessing----------------------------------
dataset = pd.read_excel('Folds5x2_pp.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1]. values

# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# --------------------------------Building the ANN----------------------------------
# init
ann = tf.keras.models.Sequential()
# add input layer & first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) # 6 nodes (experiment), rectifier activation function
# add 2nd hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# add output layer
ann.add(tf.keras.layers.Dense(units=1))
# regression: no activation

# --------------------------------Training the ANN----------------------------------
# compiling the ANN
ann.compile(optimizer='adam', loss='mean_squared_error')
# adam: stochastic

# training
ann.fit(X_train, y_train, batch_size=32, epochs=100)
# --------------------------------Predict test set results----------------------------------
y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")