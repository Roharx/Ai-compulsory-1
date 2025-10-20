import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Classifiers models for testing:
from sklearn.linear_model import LogisticRegression # 0.775
from sklearn.naive_bayes import GaussianNB # 0.73

# tsv: read_csv with delimiter '\t', we also have quotes in the reviews -> quoting=3, so it ignores normal quotes
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
nltk.download('stopwords') # stopwords are the non-important words like: and, or etc.
corpus = []

# stemming: taking only the root of the word that indicates enough of what it means:
# "Oh I loved this restaurant!" -> loved -> love
# without stemming we would have 1 column for loved and a separate one for love as well

# Cleaning data:
# 1.: removing punctuations (',', '!', etc.)
# 2.: replacing capital letters to lower (A is different from a, it would be an extra character)
# 3.: splitting different elements of the review into different words
# 4.: stemming & omitting stopwords

for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) # 1
    review = review.lower() # 2
    review = review.split() # 3
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if word not in set(all_stopwords)] # 4
    review = ' '.join(review)
    corpus.append(review)

# print(corpus)

cv = CountVectorizer(max_features=1500) # 1500 most frequent words, can experiment with lower values as well
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# print(len(X[0]))

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training the Naive Bayes model on the Training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
