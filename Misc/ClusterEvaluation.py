import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import community
from sklearn.utils.graph import graph_laplacian
from sklearn.utils.arpack import eigsh
from sklearn.manifold.spectral_embedding_ import _set_diag
import networkx as nx

def elbow_method(tfidf_representation):
	# Elbow method to get the value of #clusters
	X = np.array(tfidf_representation)
	distortions = []
	K = range(1,10)
	for k in K:
		kmeanModel = KMeans(n_clusters=k).fit(X)
		kmeanModel.fit(X)
		distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

	# Plot the elbow
	plt.plot(K, distortions, 'bx-')
	plt.xlabel('k')
	plt.ylabel('Distortion')
	plt.title('The Elbow Method showing the optimal k')
	plt.show()

def modularity_maximization(G):
	
	#first compute the best partition
	partition = community.best_partition(G)

	#drawing
	size = float(len(set(partition.values())))
	pos = nx.spring_layout(G)
	count = 0.
	for com in set(partition.values()) :
		count = count + 1.
		list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
		nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,node_color = str(count / size))

	nx.draw_networkx_edges(G,pos, alpha=0.5)
	plt.show()


def predict_k(affinity_matrix):
	
	normed_laplacian, dd = graph_laplacian(affinity_matrix, normed=True, return_diag=True)
	laplacian = _set_diag(normed_laplacian, 1,norm_laplacian=True)

	n_components = affinity_matrix.shape[0] - 1

	eigenvalues, eigenvectors = eigsh(-laplacian, k=n_components, which="LM", sigma=1.0, maxiter=5000)
	eigenvalues = -eigenvalues[::-1]  # Reverse and sign inversion.

	max_gap = 0
	gap_pre_index = 0
	for i in range(1, eigenvalues.size):
		gap = eigenvalues[i] - eigenvalues[i - 1]
		if gap > max_gap:
			max_gap = gap
			gap_pre_index = i - 1

	k = gap_pre_index + 1

	return k