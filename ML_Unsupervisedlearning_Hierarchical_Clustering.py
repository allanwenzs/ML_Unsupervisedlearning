
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ds= pd.read_csv('Mall_Customers.csv')
x=ds.iloc[:,2:].values
print(x)
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customer')
plt.ylabel('Euclidean Distance')
plt.show()
from sklearn.cluster import AgglomerativeClustering
hierachicalClustering= AgglomerativeClustering(n_clusters=5, linkage='ward')
y_hierarchicalClustering = hierachicalClustering.fit_predict(x)
#ploting
plt.scatter(x[y_hierarchicalClustering == 0, 0], x[y_hierarchicalClustering == 0, 1], s = 10, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_hierarchicalClustering == 1, 0], x[y_hierarchicalClustering == 1, 1], s = 10, c = 'blue',label = 'Cluster 2')
plt.scatter(x[y_hierarchicalClustering == 2, 0], x[y_hierarchicalClustering== 2, 1], s = 10, c = 'green',label = 'Cluster 3')
plt.scatter(x[y_hierarchicalClustering == 3, 0], x[y_hierarchicalClustering == 3, 1], s =10, c = 'orange',label = 'Cluster 4')
plt.scatter(x[y_hierarchicalClustering == 4, 0], x[y_hierarchicalClustering == 4, 1], s =10, c = 'yellow',label = 'Cluster 5')
plt.title('Clustering of customers')
plt.xlabel('Annual Income cash')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
