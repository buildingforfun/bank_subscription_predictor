# Bank subscription predictor

We want to predict if someone will subscribe to new offer or not.

The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y).

## 1. Data collection

The project sources data from: https://archive.ics.uci.edu/dataset/222/bank+marketing.
I've followed along this kaggle post for reference: https://www.kaggle.com/code/berkayalan/unsupervised-learning-clustering-complete-guide/notebook#Clustering


We can use the one-hot encoding method to convert categorical variables into numeric data. This will create a new column for each category of each feature. As we are dealing with distance metric, we will apply scaling. We scale the features to prevent certain features from dominating the distance calculation due to their larger numeric ranges. The standardization process ensures that all features have a mean of 0 and a standard deviation of 1. Ensures all features contribute equally to the distance calculation, regardless of their original scales or units

## 3. Training the model

The n_clusters parameter will try to cluster the data into two groups. The scaledd ata is then fed to the model by using fit(). After fitting, you can use the model to predict new data points by using predict. The K-means algorithm is an iterative clustering algorithm that aims to partition a dataset into K distinct clusters:

1. **Initialization**
   - The algorithm starts by randomly selecting K data points from the dataset as the initial cluster centroids (centers).

2. **Assignment Step**
   - Each data point in the dataset is assigned to the nearest centroid based on a distance metric, typically the Euclidean distance.
   - This step creates K clusters, where each data point belongs to the cluster with the nearest centroid.

3. **Update Step**
   - After assigning all data points to clusters, the centroids are updated by calculating the mean of all data points in each cluster.
   - The new centroids are the means of the clusters formed in the previous step.

4. **Iteration**
   - Steps 2 and 3 are repeated until the centroids stop changing significantly or a maximum number of iterations is reached.
   - In each iteration, data points are reassigned to the nearest new centroid, and centroids are recomputed based on the new cluster memberships.

5. **Convergence**
   - The algorithm converges when the centroids stop changing significantly between iterations or when the maximum number of iterations is reached.
   - The final centroids represent the centers of the K clusters, and each data point is assigned to the cluster with the nearest centroid.

The goal of the K-means algorithm is to minimize the sum of squared distances between each data point and its assigned centroid, known as the inertia or within-cluster sum of squares (WCSS). This is achieved by iteratively reassigning data points to the nearest centroids and recomputing the centroids until convergence.

## 4. Evaluate the model
The inertia (sum of squared distances of samples to their closest cluster center) of the k-means model can be used as a measure of the model

How to find optimal number of clusters (K)?
Elbow Curve method - We run the model across a range of clusters, calculate average distances to the centroid across all data points and use the number where there is the greatest drop.
