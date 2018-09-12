import math,random
import subprocess
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm,inv
from scipy.sparse.linalg import eigsh
from tfidf import *
from edge_weighing import *
from gap_statistics import *
from cluster_evaluation import *
from sklearn.cluster import KMeans
from sklearn import metrics,preprocessing
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import community
from optparse import OptionParser
from entropy import *
from sklearn.manifold.spectral_embedding_ import _graph_is_connected,_graph_connected_component
from sklearn.decomposition import PCA

class Point:
	#An point in n dimensional space
	def __init__(self, coords):
	#coords - A list of values, one per dimension
		self.coords = coords
		self.n = len(coords)

	def __repr__(self):
		return str(self.coords)

class Cluster:
	#A set of points and their centroid

	def __init__(self, points):
	#points - A list of point objects

		if len(points) == 0: 
			raise Exception("ILLEGAL: empty cluster")
		# The points that belong to this cluster
		self.points = points

		# The dimensionality of the points in this cluster
		self.n = points[0].n
		self.length = 1
	
		# Assert that all points are of the same dimensionality
		for p in points:
			if p.n != self.n: raise Exception("ILLEGAL: wrong dimensions")
	
		# Set up the initial centroid (this is usually based off one point)
		self.centroid = self.calculateCentroid()

	def __repr__(self):
	#String representation of this object
		return str(self.points)

	def update(self, points):
	#Returns the distance between the previous centroid and the new after
	#recalculating and storing the new centroid.
		old_centroid = self.centroid
		self.points = points
		self.centroid = self.calculateCentroid()
		shift = getDistance(old_centroid.coords, self.centroid.coords) 
		return shift

	def calculateCentroid(self):
	#Finds a virtual center point for a group of n-dimensional points
		numPoints = len(self.points)
		# Get a list of all coordinates in this cluster
		coords = [p.coords for p in self.points]
		# Reformat that so all x's are together, all y'z etc.
		unzipped = zip(*coords)
		# Calculate the mean for each dimension
		centroid_coords = [math.fsum(dList)/numPoints for dList in unzipped]

		return Point(centroid_coords)


def cosine_similarity(vector1, vector2):
	dot_product = sum(p*q for p,q in zip(vector1, vector2))
	magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
	if not magnitude:
		return 0
	return float(dot_product)/magnitude	

def getDistance(a, b):

	#Euclidean distance between two n-dimensional points.
	#Note: This can be very slow and does not scale well
	#if a.n != b.n:
	#    raise Exception("ILLEGAL: non comparable points")
	#print len(a)
	ret = reduce(lambda x,y: x + pow((a[y]-b[y]), 2),range(len(a)),0.0)
	return math.sqrt(ret)

def kmeans(points, k, cutoff):

	# Pick out k random points to use as our initial centroids
	initial = random.sample(points, k)

	# Create k clusters using those centroids
	clusters = [Cluster([p]) for p in initial]

	# Loop through the dataset until the clusters stabilize
	loopCounter = 0
	while True:
		# Create a list of lists to hold the points in each cluster
		lists = [ [] for c in clusters]
		clusterCount = len(clusters)

		# Start counting loops
		loopCounter += 1
		# For every point in the dataset ...
		for p in points:
			# Get the distance between that point and the centroid of the first
			# cluster.
			smallest_distance = getDistance(p.coords, clusters[0].centroid.coords)

			# Set the cluster this point belongs to
			clusterIndex = 0

			# For the remainder of the clusters ...
			for i in range(clusterCount - 1):
				# calculate the distance of that point to each other cluster's
				# centroid.
				distance = getDistance(p.coords, clusters[i+1].centroid.coords)
				# If it's closer to that cluster's centroid update what we
				# think the smallest distance is, and set the point to belong
				# to that cluster
				if distance < smallest_distance:
					smallest_distance = distance
					clusterIndex = i+1
			lists[clusterIndex].append(p)

		# Set our biggest_shift to zero for this iteration
		biggest_shift = 0.0

		# As many times as there are clusters ...
		for i in range(clusterCount):
			# Calculate how far the centroid moved in this iteration
			shift = clusters[i].update(lists[i])
			clusters[i].length = len(lists[i])
			# Keep track of the largest move from all cluster centroid updates
			biggest_shift = max(biggest_shift, shift)

		# If the centroids have stopped moving much, say we're done!
		if biggest_shift < cutoff:
			#print "Converged after %s iterations" % loopCounter
			break
	return clusters


def similarity_matrix(points):
	S = []
	for i in range(0, len(points)):
		row = []
		for j in range(0, len(points)):
				# scaled pairwise comparison
			if i == j:
				row.append(0)
			else:
				row.append(getDistance(points[i],points[j]))
		S.append(row)
	return S

def adjacency_matrix(points,knn,S):
	A = []
	if knn < len(points):
		for i in range(0, len(S)):
			A.append([])
			row = S[i][:]
			#row = np.array(row)[0].tolist()
			row = sorted(row,reverse=True)
			row = row[:knn]
			#print type(row)
			#print S[0][0].shape
			for j in range(0, len(S[i])):
				if S[i][j] in row:
					A[i].append(S[i][j])
				else:
					A[i].append(0)
	else:
		A = S
	return A

def diag_degree_matrix(A):
	D = []
	for j in range(0, len(A)):
		D.append([])
		dj = 0
		index = 0
		for i in range(0, len(A[j])):
			D[j].append(0)
			if i == j:
				index = i
			dj += A[j][i]
		D[j][index] = dj
	return D

if __name__ == "__main__":
	
	documents = []
	filename = sys.argv[1]
	user_msgs = cluster_user_msgs(filename)
	for user in user_msgs.keys():
		documents.append(user_msgs[user])

	tfidf_representation = pd.DataFrame(np.array(tfidf(documents)))
	print tfidf_representation
	user_entropy = pd.DataFrame(np.array(get_entropy_features(filename)))
	feature_vectors = pd.concat([tfidf_representation,user_entropy],axis=1)
	features = preprocessing.StandardScaler().fit_transform(feature_vectors.values)
	print features
	#elbow_method(feature_vectors.values)
	users = user_msgs.keys()
	
	our_tfidf_comparisons = similarity_based_approach(features)

	G = nx.Graph()
	for i in range(len(our_tfidf_comparisons)):
		if our_tfidf_comparisons[i][0] > 0.0:
			G.add_edge(users[our_tfidf_comparisons[i][1]],users[our_tfidf_comparisons[i][2]],weight=our_tfidf_comparisons[i][0])
	
	#modularity_maximization(G)
	#A = nx.adjacency_matrix(G)
	print nx.is_connected(G)
	components = nx.connected_components(G)
	#print [len(c) for c in sorted(components, key=len, reverse=True)]
	#print len(components)
	largest_component = max(components,key=len)
	#print largest_component
	subgraph = G.subgraph(largest_component)
	diameter = nx.diameter(subgraph)
	print "Network diameter of largest component:" + str(diameter)

	S = nx.to_numpy_matrix(G)
	A = adjacency_matrix(feature_vectors.values,5,S.tolist())
	D = diag_degree_matrix(A)
	#print predict_k(A)
	#print A
	'''
	pos = nx.spring_layout(G)
	nx.draw_networkx_nodes(G,pos,node_size=1000)
	nx.draw_networkx_edges(G,pos,width=0.5)
	# labels
	nx.draw_networkx_labels(G,pos,font_size=7,font_family='sans-serif')
	plt.axis('off')
	plt.savefig("weighted_graph.png") # save as png
	plt.show()
	'''
	#print _graph_is_connected(A)
	temp = []
	for p in D:
		temp.append([np.array(element) for element in p])
	D = np.array(temp)
	D = np.matrix(D)
	sqrt_D = sqrtm(D)
	inv_D = inv(sqrt_D)
	invsqrt_D = sqrtm(inv_D)
	#I = np.identity(D.shape[0])
	L = np.dot(np.dot(invsqrt_D,A),sqrt_D)
	normalized_L = preprocessing.normalize(L)
	#print normalized_L
	k = 5
	#print L.shape[0]
	eig_vals, eig_vecs = eigsh(normalized_L,k=L.shape[0]-1)
	print eig_vals
	#print eig_vecs
	#for i in range(len(eig_vals)):
	#	if eig_vals[i] == 0.0:
	#		print eig_vecs[:,i]
	#print np.iscomplex(eig_vals)
	#x_labels = range(1,42)
	#y_labels = eig_vals
	#plt.plot(x_labels,y_labels,'ro')
	#plt.savefig('Eigen_values.png')
	#plt.show()
	eig_pairs = [((eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
	# Sort the (eigenvalue, eigenvector) tuples from high to low
	eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
	eig_vals_sum = 0.0
	for pair in eig_pairs:
		eig_vals_sum += pair[0]
	k = 0
	k_eig_vals_sum = 0.0
	while k_eig_vals_sum < 0.95*eig_vals_sum:
		k_eig_vals_sum += eig_pairs[k][0]
		k += 1
	print k
	
	vec = np.array([ eig_pairs[i][1] for i in range(k)])
	vec = np.transpose(vec)
	vec = list(vec)
	temp = []
	#print "first k eigen vectors"
	for p in vec:
		temp.append([float(elements) for elements in p])
	vec = temp
	#print type(vec[i])
	points = []
	for i in range(len(vec)):
		points.append(Point(vec[i]))
	'''
	
	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(feature_vectors)
	plt.scatter(principalComponents[:, 0], principalComponents[:, 1])
	plt.show()
	'''
	#print feature_vectors
	
	k, gapdf = optimalK(features, nrefs=5, maxClusters=15)
	print 'Optimal k is: ', k
	plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
	plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
	plt.grid(True)
	plt.xlabel('Cluster Count')
	plt.ylabel('Gap Value')
	plt.title('Gap Values by Cluster Count')
	plt.show()
	'''
	num_clusters = 6
	opt_cutoff = 0.5
	clusters = kmeans(points,num_clusters,opt_cutoff)
	for i in range(num_clusters):
		print "cluster No:" + str(i) + ' ' + "#points:" + str(clusters[i].length)
	'''