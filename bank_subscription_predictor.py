import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

## Reading data
df_bank = pd.read_csv('bank+marketing/bank-additional/bank-additional-full.csv', sep=';')
df_bank.head()
print(df_bank.describe().T)
df_bank_dummy = pd.get_dummies(df_bank)
print(df_bank_dummy.head())

#  Scaling the data
scaler = StandardScaler()
df_bank_scaled = scaler.fit_transform(df_bank_dummy)
df_bank_scaled[0]

# Perform k-means clustering
k_means = KMeans(n_clusters=2, random_state=42) # Random state for reproducibility
clusters = k_means.fit_predict(df_bank_scaled)
plt.scatter(df_bank_scaled[:, 0], df_bank_scaled[:, 1], c=clusters)
# The red cross is the  coordinates of the cluster centroids
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], marker='x', s=150, c='r')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.savefig('k_means_clustering.png')
plt.clf()

# Evaluate the clustering
print(f'Inertia (Sum of Squared Distances): {k_means.inertia_:.2f}')
ssd = []
for k in range(1,5):
    model = KMeans(n_clusters=k)
    model.fit(df_bank_scaled)
    ssd.append(model.inertia_) ## SSD Point to cluster centers
plt.plot(range(1,5),ssd,"o--")
plt.title("Elbow Range")
plt.savefig('elbow_range.png')