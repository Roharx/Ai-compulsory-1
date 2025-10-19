import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori

# Nothing changes from apriori, except for displaying info
# ---------------------------------------------------------
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

# Converting pandas frame to list to feed it to apriori, it needs a list
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# print(transactions[:5])

rules = apriori(transactions = transactions,
                min_support =0.003,
                min_confidence=0.2,
                min_lift=3,
                min_length=2,
                max_length=2 # cause buy 1, get 1 free rule in the shop exercise
)

results = list(rules)
# print(results)

# Organizing results into a pandas dataframe
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))
resultsInDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side',
                                                               'Right Hand Side',
                                                               'Support'])

print(resultsInDataFrame.nlargest(n = 10, columns = 'Support'))

