import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

#Read previously generated data
user_table = pd.read_csv("transformed_data/user_table_full.csv") 
data = user_table.iloc[:,1:] #numeric values; first colum is User_ID

#Channel lists
channels = list(data) #all
main_channels = ['A', 'B', 'E', 'G', 'H', 'I'] #most relevant; see other script
secondary_channels = list(set(channels) - set(main_channels))

#Clustering - KMeans

from sklearn.cluster import KMeans
from sklearn import metrics 

#Void to determine optimal number of clusters 
#Using silhouette method / Harabasz Score / Davies-Bouldin index
#Performs clustering to 2,...,max_n cluster and comptes scores
#Prints optimal number of clusters according to the three methods
def optimal_n_clusters(max_n):
	sil_score_vector = [] 
	har_score_vector = []
	dav_score_vector = []

	for i in range(2, max_n + 1):
		kmeans_users = KMeans(n_clusters = i, random_state = 1).fit(data)
		labels = kmeans_users.labels_
		#bigger is better (highest = 1)
		silhouette = metrics.silhouette_score(data, labels, metric = "euclidean") 
		sil_score_vector.append(silhouette)
		#bigger is better
		harabasz = metrics.calinski_harabasz_score(data, labels) 
		har_score_vector.append(harabasz)
		#smaller is better (lowest = 0)
		davies =  metrics.davies_bouldin_score(data, labels) 
		dav_score_vector.append(davies)

	best_sil = np.argmax(sil_score_vector) + 2
	best_har = np.argmax(har_score_vector) + 2
	best_dav = np.argmin(dav_score_vector) + 2
	print("Optimal n. of clusters (Silhouette) :", best_sil)
	print("Optimal n. of clusters (Calinski-Harabasz) :", best_har)
	print("Optimal n. of clusters (Davies-Bouldin) :", best_dav)

#optimal_n_clusters(5)
#Running this function shows optimal n_clusters = 2 for Silhouette and Calinski-Harabasz
#									 			= 3 for Davies-Bouldin


#Perform KMeans with two clusters
kmeans_users_2 = KMeans(n_clusters = 2, random_state = 1).fit(data)
labels = kmeans_users_2.labels_

#Add cluster labels as last column
data.insert(data.shape[1], "Cluster", labels)

#PCA for 2D visual representation of cluster
#(and data reduction for possible further analysis)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Standardize values to perform PCA, which is scale-sensitive
x = data.values #as array
x = StandardScaler().fit_transform(x)

#Perform PCA
pca = PCA(n_components = 2) #2D
pr_comp_x = pca.fit_transform(x) #principal components
pr_comp_df = pd.DataFrame(data = pr_comp_x, columns = ["STD_PC1", "STD_PC2"] ) #PC dataframe

#Proportion of variance explained with two principal components
#print("Explained variance ratio: ", pca.explained_variance_ratio_.sum())
#Very poor ~12% -> Good only for visual representation

#Add cluster lables to PCA dataframe
pr_comp_df.insert(pr_comp_df.shape[1], "Cluster", labels)
#Separate Values
pc_0 = pr_comp_df[pr_comp_df["Cluster"] == 0]
pc_1 = pr_comp_df[pr_comp_df["Cluster"] == 1]

#Build adequate Random Sample for plots
n_0 = len(pc_0.index)#samples in cluster 0
n_1 = len(pc_1.index) #samples in cluster 1
prop_0 = n_0/(n_0 + n_1) #proportion of type-0 -> ~97.3%

sample_size = 1000 #plot sample size
sample_size_0 = int(np.ceil(prop_0 * sample_size)) #type 0 samples
sample_size_1 = sample_size - sample_size_0 #type 1 samples 
np.random.seed(1) #reproducibility
choice_0 = np.random.choice(range(0, n_0), size = sample_size_0)
choice_1 = np.random.choice(range(0,n_1), size = sample_size_1)

sample_0 = pc_0.iloc[choice_0]
sample_1 = pc_1.iloc[choice_1]

plt.scatter(sample_0["STD_PC1"], sample_0["STD_PC2"], marker ='.', color = 'turquoise')
plt.scatter(sample_1["STD_PC1"], sample_1["STD_PC2"], marker ='.', color = 'red')
plt.legend(("Cluster 0 (Average Spending)", "Cluster 1 (High Spending)"))
plt.title("Channel-based Spending Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

#Cluster Analisys
#Aggregate secondary channels for analysis 
data["Secondary"] = data[secondary_channels].sum(axis = 1)

cluster_0 = data[data["Cluster"] == 0]
cluster_1 = data[data["Cluster"] == 1]

cluster_0 = cluster_0[main_channels + ["Secondary"]]
cluster_1 = cluster_1[main_channels + ["Secondary"]]

#Proportion of revenue due to clusters
revenue_0 = cluster_0.sum().sum()
revenue_1 = cluster_1.sum().sum()
prop_0 = np.round(revenue_0/(revenue_0 + revenue_1), 3)
prop_1 = 1 - prop_0
#Cluster 0: 84.7%; Cluster 1: 15.3%

'''
#Summary info
#Clearly shows cluster_0 can be considered that of "average-spending" users,
#while cluster_1 is "high-spending". Note variance tends to be much higher for
#Type-1 users.
cluster_0.describe().to_csv("result_data/cluster_0_summary.csv")
cluster_1.describe().to_csv("result_data/cluster_1_summary.csv")
'''

#Channel Influence per cluster plot
#See difference in influential channels
#Colors - using search dictionary to have same colors for same channels
col_list = ["cornflowerblue", "orange", "limegreen", "orangered",
 			"mediumpurple", "chocolate", "skyblue"]
c_dict = dict(zip(main_channels + ["Secondary"], col_list))

#Extract most influential channels per cluster
inf_chans_0 = cluster_0.idxmax(axis = 1).value_counts()
inf_chans_1 = cluster_1.idxmax(axis = 1).value_counts()

#Pie Plots
inf_chans_0.plot.pie(colors = [c_dict[c] for c in inf_chans_0.index.values ])
plt.axes().set_ylabel('Channel Influence')
plt.title("Most influential channels for type-0 users (Average Spending)")
plt.legend()
plt.show()

inf_chans_1.plot.pie(colors = [c_dict[c] for c in inf_chans_1.index.values ])
plt.axes().set_ylabel('Channel Influence')
plt.title("Most influential channels for type-1 users (High Spending)")
plt.legend()
plt.show()

'''
#Power Bi data generation
inf_chans_bi = pd.concat([inf_chans_0, inf_chans_1], axis = 1)
inf_chans_bi.to_csv("powerbi_data/channel_inf_cluster.csv")
sample_0.to_csv("powerbi_data/cluster_0_pc.csv")
sample_1.to_csv("powerbi_data/cluster_1_pc.csv")
'''
