import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(0)
data_size = 400

mu1 = [2, 2]
sigma1 = [[0.9, -0.0255], [-0.0255, 0.9]]

mu2 = [5, 5]
sigma2 = [[0.5, 0], [0, 0.3]]

mu3 = [-2, -2]
sigma3 = [[1, 0], [0, 0.9]]

mu4 = [-4, 8]
sigma4 = [[0.8, 0], [0, 0.6]]

data1 = np.random.multivariate_normal(mu1, sigma1, data_size)
data2 = np.random.multivariate_normal(mu2, sigma2, data_size)
data3 = np.random.multivariate_normal(mu3, sigma3, data_size)
data4 = np.random.multivariate_normal(mu4, sigma4, data_size)
synthetic_data = np.vstack((data1, data2, data3, data4))

plt.scatter(synthetic_data[:, 0], synthetic_data[:, 1], cmap='viridis')
plt.title('Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()

wcss = []
for i in range(1, 21):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(synthetic_data)
    wcss.append(kmeans.inertia_)

optimal_num_clusters = 4
kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=0)
kmeans.fit(synthetic_data)
labels = kmeans.labels_

plt.plot(range(1, 21), wcss, marker='o')
plt.title('Elbow Method to find optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS Score')
plt.xticks(range(1, 21))
plt.grid(True)
plt.show()

markers = ['o', 'x', 'D', '^']

for cluster_id in set(labels):
    cluster_data = synthetic_data[labels == cluster_id]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], marker=markers[cluster_id], label=f'Cluster {cluster_id + 1}')

plt.title('Clustered Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.legend() 
plt.show()
