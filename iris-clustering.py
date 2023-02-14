import pandas as pd
import numpy as np

#loading data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, header = None)
print(df)

#assigning column names
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'Species']
print(df)

#assinging zero to every 5th element
for i in range(len(df)):
    if i%5==0:
        df.loc[i, "sepal_length"]=0


print(df)

#ignoring last column cause it's categorical
df = df.drop(['Species'], axis=1)
print(df)

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

X = df.loc[:,['sepal_width', 'petal_length', 'petal_width']]
Y = df.sepal_length

#splitting data into training and testing
skip_rows = df.index.isin([1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19,21,22,23,24,26,27,28,29,31,32,33,34,36,37,38,39,41,42,43,44,46,47,48,49,51,52,53,54,56,57,58,59,61,62,63,64,66,67,68,69,71,72,73,74,76,77,78,79,81,82,83,84,86,87,88,89,91,92,93,94,96,97,98,99,101,102,103,104,106,107,108,109,111,112,113,114,116,117,118,119,121,122,123,124,126,127,128,129,131,132,133,134,136,137,138,139,141,142,143,144,146,147,148,149])


for index in range(len(df)):
        X_train = df.loc[skip_rows, ['sepal_width', 'petal_length', 'petal_width']]

        X_test = df.loc[::5, ['sepal_width', 'petal_length', 'petal_width']]

        Y_train = df.loc[skip_rows, ['sepal_length']]

        Y_test = df.loc[::5, ['sepal_length']]

print(X_train)
print(X_test)
print(Y_train)
print(Y_test)


#predicting missing values using DecisionTreeRegressor
dt = DecisionTreeRegressor()

dt.fit(X_train, Y_train)
Y_pred = dt.predict(X_test)

print(Y_pred)

#putting the predicted values back in the data frame
y=0
for i in range(len(df)):
    if i%5==0:
        df.loc[i, "sepal_length"]=Y_pred[y]
        y=y+1

print(df)

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#reducing features into 2D through PCA for kmeans
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(df)
pca_data = pd.DataFrame(pca.transform(df))
pca_data.head()

#using kmeans for clustering
Kmeans = KMeans(n_clusters=3)
Kmeans.fit(pca_data)
centroids = Kmeans.cluster_centers_

plt.scatter(pca_data[0], pca_data[1], c = Kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()


#kmeans with normalization
normalized_df = df
import scipy.stats as stats
stats.zscore(normalized_df)

print(normalized_df)


pca = PCA(2)
pca.fit(df)
pca_data = pd.DataFrame(pca.transform(normalized_df))
print(pca_data.head())

Kmeans = KMeans(n_clusters=3).fit(pca_data)
centroids = Kmeans.cluster_centers_

plt.scatter(pca_data[0], pca_data[1], c = Kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()
