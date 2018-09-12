import math
from sklearn import manifold
import numpy as np
import pylab as pl
from scipy.sparse import linalg, eye
from pyamg import smoothed_aggregation_solver
from scikits.learn import neighbors


def cosine_similarity(vector1, vector2):
	dot_product = sum(p*q for p,q in zip(vector1, vector2))
	magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
	if not magnitude:
		return 0
	return dot_product/magnitude

def similarity_based_approach(features):
	our_tfidf_comparisons = []
	for count_0,doc_0 in enumerate(features):
		for count_1,doc_1 in enumerate(features):
			our_tfidf_comparisons.append((cosine_similarity(doc_0,doc_1),count_0,count_1))
	return our_tfidf_comparisons

def locally_linear_embedding(X, n_neighbors, out_dim, tol=1e-6, max_iter=200):
	W = neighbors.kneighbors_graph(X, n_neighbors=n_neighbors, mode='barycenter')

	# M = (I-W)' (I-W)
	A = eye(*W.shape, format=W.format) - W
	A = (A.T).dot(A).tocsr()

	# initial approximation to the eigenvectors
	X = np.random.rand(W.shape[0], out_dim)
	ml = smoothed_aggregation_solver(A, symmetry='symmetric')
	prec = ml.aspreconditioner()

	# compute eigenvalues and eigenvectors with LOBPCG
	eigen_values, eigen_vectors = linalg.lobpcg(
		A, X, M=prec, largest=False, tol=tol, maxiter=max_iter)

	index = np.argsort(eigen_values)
	return eigen_vectors[:, index], np.sum(eigen_values)

# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
	x_min, x_max = np.min(X, 0), np.max(X, 0)
	X = (X - x_min) / (x_max - x_min)

	pl.figure()
	ax = pl.subplot(111)
	for i in range(digits.data.shape[0]):
		pl.text(X[i, 0], X[i, 1], str(digits.target[i]),
				color=pl.cm.Set1(digits.target[i] / 10.),
				fontdict={'weight': 'bold', 'size': 9})

	if hasattr(offsetbox, 'AnnotationBbox'):
		# only print thumbnails with matplotlib > 1.0
		shown_images = np.array([[1., 1.]])  # just something big
		for i in range(digits.data.shape[0]):
			dist = np.sum((X[i] - shown_images) ** 2, 1)
			if np.min(dist) < 4e-3:
				# don't show points that are too close
				continue
			shown_images = np.r_[shown_images, [X[i]]]
			imagebox = offsetbox.AnnotationBbox(
				offsetbox.OffsetImage(digits.images[i], cmap=pl.cm.gray_r),
				X[i])
			ax.add_artist(imagebox)
	pl.xticks([]), pl.yticks([])
	if title is not None:
		pl.title(title)
'''
n_samples, n_features = 2000, 3
n_turns, radius = 1.2, 1.0
rng = np.random.RandomState(0)
t = rng.uniform(low=0, high=1, size=n_samples)
data = np.zeros((n_samples, n_features))

# generate the 2D spiral data driven by a 1d parameter t
max_rot = n_turns * 2 * np.pi
data[:, 0] = radius = t * np.cos(t * max_rot)
data[:, 1] = radius = t * np.sin(t * max_rot)
data[:, 2] = rng.uniform(-1, 1.0, n_samples)
manifold = np.vstack((t * 2 - 1, data[:, 2])).T.copy()
colors = manifold[:, 0]

# rotate and plot original data
sp = pl.subplot(211)
U = np.dot(data, [[-.79, -.59, -.13],
				  [ .29, -.57,  .75],
				  [-.53,  .56,  .63]])
sp.scatter(U[:, 1], U[:, 2], c=colors)
sp.set_title("Original data")


print "Computing LLE embedding"
n_neighbors, out_dim = 12, 2
X_r, cost = locally_linear_embedding(data, n_neighbors, out_dim)

plot_embedding(X_r)
#sp = pl.subplot(212)
#sp.scatter(X_r[:,0], X_r[:,1], c=colors)
#sp.set_title("LLE embedding")
#pl.show()
'''

