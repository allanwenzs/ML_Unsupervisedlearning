
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('Mall_CustomersKmeansclusteringData.csv')
x = df.iloc[:, [3,4]].values

print(x)
plt.scatter(x[:,0],x[:,1])
plt.title('Income vs Spending')

plt.xlabel('spending')
plt.ylabel('K')
plt.show()
kmeans = KMeans(n_clusters= 5)
y_pred = kmeans.fit_predict(x)
print(y_pred)
x = np.column_stack((x, y_pred))
print(x)
kmeans.cluster_centers_

#seperating cluster
x1 = x[x[:,2]==0]
x2 = x[x[:,2]==1]
x3 = x[x[:,2]==2]
x4 = x[x[:,2]==3]
x5 = x[x[:,2]==4]
#presenting the graph
plt.scatter(x1[:, 0], x1[:, 1], color='green', label='Cluster 1')
plt.scatter(x2[:, 0], x2[:, 1], color='red', label='Cluster 2')
plt.scatter(x3[:, 0], x3[:, 1], color='blue', label='Cluster 3')
plt.scatter(x4[:, 0], x4[:, 1], color='orange', label='Cluster 4')
plt.scatter(x5[:, 0], x5[:, 1], color='cyan', label='Cluster 5')
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, marker="*", color ="purple", label="Centroid")
plt.xlabel("Annual income")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
sse = []
for i in range(1, 11):
 kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
 kmeans.fit(x)
 sse.append(kmeans.inertia_)
 print(sse)

plt.plot(range(1, 11), sse)
plt.title("The elbow method Updated")
plt.xlabel("Number of cluster")
plt.ylabel("Summ of the square error ")
plt.show()
