import math,random,requests,json
import subprocess
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm,inv
from scipy.sparse.linalg import eigsh
from tfidf import *
import matplotlib.pyplot as plt
import community,urllib2
from optparse import OptionParser
from entropy import *
from sklearn import preprocessing
from sklearn.manifold.spectral_embedding_ import _graph_is_connected,_graph_connected_component
from sklearn.decomposition import PCA
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go
from twitch import TwitchClient
import networkx as nx
from collections import OrderedDict,Counter
import json,time
from conversations import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from os import listdir
from os.path import isfile,join,isdir
from UserClustering import *
from sklearn.semi_supervised import label_propagation


def cosine_similarity(vector1, vector2):
	dot_product = sum(p*q for p,q in zip(vector1, vector2))
	magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
	if not magnitude:
		return 0
	return float(dot_product)/magnitude	


def plot_similarity(features,labels,rotation):
	#corr = np.inner(features,features)
	#print corr.shape
	corr = np.empty([len(labels),len(labels)])
	for count_0,feature_0 in enumerate(features):
		for count_1,feature_1 in enumerate(features):
			corr[count_0][count_1] = cosine_similarity(feature_0,feature_1)
	#corr = np.array(corr)		
	sns.set(font_scale=0.5)
	fig,g = plt.subplots(1,1,figsize=(20,20))
	sns.heatmap(corr,ax=g,xticklabels=labels,yticklabels=labels,vmin=0,vmax=1,cmap="YlOrRd")
	g.set_xticklabels(labels, rotation=rotation)
	g.set_yticklabels(labels,rotation=0)
	g.set_title("Similarity measures betweeen users")
	plt.show()


def get_user_mentions_features(per_user_mentions):
	n = len(per_user_mentions.keys())
	#print n
	features = np.zeros((n,n))
	user_mentions = OrderedDict()
	for user in per_user_mentions.keys():
		freq = Counter(per_user_mentions[user])
		if user not in user_mentions.keys():
			user_mentions[user] = freq
	#print user_mentions
	for user in per_user_mentions.keys():
		i = per_user_mentions.keys().index(user)
		for user_mention in user_mentions[user].keys():
			if user_mention in per_user_mentions.keys():
				j = per_user_mentions.keys().index(user_mention)
				features[i][j] = user_mentions[user][user_mention] 
	return features

def similarity_based_approach(features,users):
	user_comparisons = list()
	for count_0,feature_0 in enumerate(features):
		for count_1,feature_1 in enumerate(features):
			pairwise_comparison = tuple()
			pairwise_comparison = (users[count_0],users[count_1],cosine_similarity(feature_0,feature_1))
			user_comparisons.append(pairwise_comparison)
	return user_comparisons

def get_channel_followers(path,channel_name):
	'''
	headers = {'Client-ID':'r3i67jv87x49xeuggnjbwts540kjnw','Accept':'application/vnd.twitchtv.v5+json'}
	urlf1 = 'https://api.twitch.tv/kraken/users/'
	urlf2 = '/follows/channels/'

	urlu1 = 'https://api.twitch.tv/kraken/users?login='
	r = requests.get(urlu1+channel_name,headers=headers)
	c = r.json()['users'][0]['_id']

	follows = []

	for i in users:
		r = requests.get(urlu1+i,headers=headers)
		u = r.json()['users'][0]['_id']
		r = requests.get(urlf1+u+urlf2+c,headers=headers)
		if 'error' in r.json():
			continue
		else:
			follows.append(i)
	'''
	#print channel_name	
	follows = set()
	for file in listdir(path):
		#print file
		if file[13:-2] == channel_name:
			#print "entered"
			with open(join(path,file),'r') as f:
				lines = f.readlines()
				users = lines[1].split(' ')
				#print users
				for user in users:
					follows.add(user)
			f.close()
	return list(follows)

def users_metadata(users):
	#print len(users)
	#print users
	username = "maharani_ryp"
	client_id = "ykvnzoa5e28awecbj52yiva0f6dgme"
	token = "cpecc5vo2jeb7ag5zsauchvggrk2rl"
	client = TwitchClient(client_id=client_id,oauth_token=token)
	users_info = []
	for i in range(0,len(users),50):
		print "i value:"+ str(i)
		print "users chunk len:" + str(len(users[i:i+50]))
		temp =  client.users.translate_usernames_to_ids(users[i:i+50])
		print "len of temp:" + str(len(temp))
		users_info = users_info + temp
		print len(users_info)
		time.sleep(1)
	user_app = []	
	for user in users_info:
		print '{} : {}'.format(user.name,user.id)
		user_app.append(user.name)
	for user in users:
		if user not in user_app:
			print user
	features = np.empty([len(users),3])
	for i in range(len(users_info)):
		url = 'https://api.twitch.tv/helix/users/follows?to_id='+str(users_info[i].id)
		headers = {'Client-ID': 'ykvnzoa5e28awecbj52yiva0f6dgme'}
		req = urllib2.Request(url,headers=headers)
		response = urllib2.urlopen(req)
		respData = json.loads(response.read())
		features[i][1]=respData['total'] #No of users following the user
		url = 'https://api.twitch.tv/helix/users/follows?from_id='+str(users_info[i].id)
		req = urllib2.Request(url,headers=headers)
		response = urllib2.urlopen(req)
		respData = json.loads(response.read())
		features[i][0] = respData['total'] #No of Users being followed by the user
		headers = {'Accept': 'application/vnd.twitchtv.v5+json','Client-ID': 'ykvnzoa5e28awecbj52yiva0f6dgme'}		
		url = 'https://api.twitch.tv/kraken/users/'+str(users_info[i].id)+'/follows/channels'
		req = urllib2.Request(url,headers=headers)
		response = urllib2.urlopen(req)
		respData = json.loads(response.read())
		features[i][2] = respData['_total'] # No of channels been followed by user 
		time.sleep(10)
	return features


def run_model(model,alg_name,X_train,y_train,X_test,y_test):
	model.fit(np.array(X_train),np.array(y_train).ravel())
	y_pred = model.predict(np.array(X_test))
	accuracy = accuracy_score(np.array(y_test),y_pred)*100
	print "Classifier:" + alg_name + ' ' + "Accuracy:" + str(accuracy)

def getAllFilesRecursive(path,certified_reals,certified_new_reals):
	for file in listdir(path):
		filename = join(path,file)
		if (file.split('#')[1].split('database')[0] in certified_new_reals or file.split('#')[1].split('database')[0] in certified_reals) and 'b<r' not in str(filename):
			lt = file.split('#')
			real_file = "../Data/Real Data/"+str(lt[0])+'/#'+str(lt[1])+'.txt'
			if 'random1' in str(lt[2]):
				boted_file = "../Data/Bot Data/#dip_7777database_random1.txt"
				main(filename,real_file,boted_file)
			if 'random2' in str(lt[2]):
				boted_file = "../Data/Bot Data/#dip_7777database_random2.txt"
				main(filename,real_file,boted_file)
			if 'chatterscontrolled' in str(lt[2]):
				boted_file = "../Data/Bot Data/#dip_7777database_chatterscontrolled.txt"
				main(filename,real_file,boted_file)

def get_final_features(feature_vectors,considerd_users_index):
	X = []
	for i in range(len(feature_vectors)):
		if i in considerd_users_index:
			X.append(feature_vectors.iloc[i,:].values.tolist())
	return pd.DataFrame(X)


def data_labelprop(X,real_users_index,bot_users_index):
	label_X = []
	label_Y = []

	for index in range(len(X)):
		#if index in considerd_users_index:
		if index in real_users_index:
			label_X.append(X.iloc[index,:].values.tolist())
			label_Y.append(0)
		elif index in bot_users_index:
			label_X.append(X.iloc[index,:].values.tolist())
			label_Y.append(1)
		else:
			label_X.append(X.iloc[index,:].values.tolist())
			label_Y.append(-1)
	return label_X,label_Y

def label_data(X,real_users_index,bot_users_index):
	X_train = []
	Y_train = []
	X_test = []
	for index in real_users_index:
		X_train.append(X.iloc[index,:].values.tolist())
		Y_train.append(0)
	for index in bot_users_index:
		X_train.append(X.iloc[index,:].values.tolist())
		Y_train.append(1)
	for i in range(len(X)):
		if i not in real_users_index and i not in bot_users_index:
			X_test.append(X.iloc[i,:].values.tolist())
	return X_train,X_test,Y_train

def get_no_user_msgs(filename):
	no_user_msgs = OrderedDict()
	with open(filename,'r') as f:
		lines = f.readlines()
		for line in lines:
			user = str(line.split(',"u":')[1].split(',"e":')[0].replace('"','')).lower()
			if user not in no_user_msgs.keys():
				no_user_msgs[user] = 1
			else:
				no_user_msgs[user] += 1
	return no_user_msgs

def main(merged_filename,real_file,boted_file):
	
	documents = []
	print merged_filename
	users_info = getUserIMDMessages(merged_filename)
	print users_info.keys()
	print "total users:" + str(len(users_info.keys()))
	considerd_users_index = []
	index = 0
	for user in users_info.keys():
		if users_info[user]['m'] > 1:
			considerd_users_index.append(index)
		index += 1
	real_users_index,bot_users_index,real_users,bot_users = labeling_data(merged_filename,real_file,boted_file)
	similar_users = OrderedDict()
	real_users_cnt = 0
	bot_users_cnt = 0
	per_user_mentions = OrderedDict()
	spell = SpellChecker()
	slang = SlangNormalization()
	slang.readfile('slang.txt')
	rr = RepeatReplacer()
	lemmatiser = WordNetLemmatizer()
	stop_words = set(stopwords.words('english'))
	chats_info = []
	users_list = []
	with open(merged_filename,'r') as f:
		lines = f.readlines()
		for line in lines:
			chat_info = {}
			user = str(line.split(',"u":')[1].split(',"e":')[0].replace('"','')).lower()
			message = format_line(str(line.split(',"m":')[1].split(',"nm":')[0].replace('"','')))
			chat_info['user'] = user
			if user not in users_list:
				users_list.append(user)
				if user in real_users:
					real_users_cnt += 1
				else:
					bot_users_cnt += 1
			message_tokens = tokenize(message)
			if user not in per_user_mentions.keys():
				per_user_mentions[user] = [token for token in message_tokens if token in users_list]
			message = ' '.join(token.lower() for token in tokenize(message))
			normalized_msg = slang.translator(message)
			normalized_msg = ' '.join([lemmatiser.lemmatize(spell.replace(rr.replace(token.lower())).decode('utf-8'),pos="v") for token in normalized_msg.split(' ') if token not in stop_words and token!=""])
			chat_info['msgs'] = normalized_msg
			chats_info.append(chat_info)
	f.close()
	#print len(chats_info)

	user_msgs = OrderedDict()
	for i in range(len(chats_info)):
		if chats_info[i]['user'] not in user_msgs.keys():
			user_msgs[chats_info[i]['user']] = ''
			user_msgs[chats_info[i]['user']] += chats_info[i]['msgs'] + ' '
		else:
			user_msgs[chats_info[i]['user']] += chats_info[i]['msgs'] + ' '
	labels = []
	for user in user_msgs.keys():
		documents.append(user_msgs[user])
		
	users = user_msgs.keys()
	print users 
	users_dict = getUserIMDMessages(merged_filename)
	#print len(users.keys())
	user_chats_ft = get_chats_features(users_dict)
	#print users_list
	user_chats_ft = pd.DataFrame(user_chats_ft)
	user_imd_bins = pd.DataFrame(get_IMD_features(users_dict))
	user_features = pd.concat([user_chats_ft,user_imd_bins],axis=1)
	tfidf_representation = tfidf(documents)
	tfidf_features = pd.DataFrame(np.array(tfidf_representation))
	#user_mentions_features = pd.DataFrame(user_mentions_features)
	#print get_no_user_msgs(merged_filename).values
	no_user_msgs_features = pd.DataFrame(get_no_user_msgs(merged_filename).values())
	user_entropy = pd.DataFrame(np.array(get_entropy_features(merged_filename)))
	#CCE_features = pd.DataFrame(get_user_cce_features(merged_filename))
	#print "user Entropy features"
	#print per_user_mentions
	#print "user mention features"
	user_mentions_features = pd.DataFrame(get_user_mentions_features(per_user_mentions))
	#print user_mentions_features.values.tolist()
	
	#metadata_features = pd.DataFrame(users_metadata(users_list))
	conversation_features = pd.DataFrame(ConversationalFeatures(chats_info))
	#print "conversational features"
	feature_vectors = pd.concat([user_entropy,user_mentions_features,no_user_msgs_features,conversation_features],axis=1)
	feature_vectors = get_final_features(feature_vectors,considerd_users_index)
	final_features = pd.concat([user_features,feature_vectors],axis=1)
	#print "shape of features:" + str(np.array(feature_vectors).shape)
	channel_followers = get_channel_followers('../followers_cnt/',real_file.split('#')[1].split('database')[0])
	#real_users_index = set(real_users_index)
	users = set(users)
	for user in channel_followers:
		if user in users:
			real_users_index.append(list(users).index(user))
	print real_users_index
	
	print "# real users:" + str(real_users_cnt) + ' ' + "labelled real users:" + str(len(real_users_index))
	print '# bot users:' + str(bot_users_cnt) + ' ' + "labelled bot users:" + str(len(bot_users_index))
	#x_train,x_test,y_train = label_data(feature_vectors,real_users_index,bot_users_index)
	#print len(x_train[0])
	#print type(metadata_features)
	#print "shape of train features:" + str(np.array(x_train).shape)
	print "#considerd users:" + str(len(considerd_users_index))
	label_X,label_Y = data_labelprop(final_features,real_users_index,bot_users_index)
	print len(label_X),len(label_Y)
	# Learn with LabelSpreading
	label_spread = label_propagation.LabelSpreading(kernel='rbf', alpha=0.8)
	label_spread.fit(label_X, label_Y)
	output_labels = label_spread.transduction_
	print output_labels
	print accuracy_score(np.array(label_Y),output_labels)*100
	#print feature_vectors.isnull().any()
	#print metadata_features.values
	#print feature_vectors.columns[feature_vectors.isnull().any()].tolist()
	#print feature_vectors.isnull().sum()
	'''
	#labels = pd.DataFrame(labels)
	x_train = pd.DataFrame(x_train)
	y_train = pd.DataFrame(y_train)
	x_test = pd.DataFrame(x_test)
	X_train,X_test,y_train,y_test = train_test_split(x_train,y_train,test_size = 0.3,random_state=100)

	#-------Decision Tree-----------

	model = tree.DecisionTreeClassifier(criterion='entropy',max_depth=5)
	run_model(model,"Decision Tree",X_train,y_train,X_test,y_test)

	#-------Random Forest-----------

	model = RandomForestClassifier(n_estimators=10)
	run_model(model,"Random Forest",X_train,y_train,X_test,y_test)
	

	#-------xgboost-----------------

	model = XGBClassifier()
	run_model(model,"XGBoost",X_train,y_train,X_test,y_test)
	

	#-------SVM Classifier-----------

	model = SVC()
	run_model(model,"SVM Classifier",X_train,y_train,X_test,y_test)

	#-------Nearest Classifier-------

	model = neighbors.KNeighborsClassifier()
	run_model(model,"Nearest Neighbors Classifier",X_train,y_train,X_test,y_test)

	#-------SGD Classifier-----------

	model = OneVsRestClassifier(SGDClassifier())
	run_model(model,"SGD Classifier",X_train,y_train,X_test,y_test)

	#-------Gaussian NB--------------

	model = GaussianNB()
	run_model(model,"Gaussian NB",X_train,y_train,X_test,y_test)

	#-------NN-MLP-------------------

	model = MLPClassifier()
	run_model(model,"NN-MLP",X_train,y_train,X_test,y_test)
'''
with open('../Data/Real Data/certified real.txt','r') as f:
	certified_reals = f.readlines()
certified_reals = [real.strip('\n') for real in certified_reals]
f.close()
with open('../Data/Real Data/certified real for new data.txt') as f:
	certified_new_reals = f.readlines()
certified_new_reals = [real.strip('\n') for real in certified_new_reals]
f.close()
#getAllFilesRecursive('../Data/Real Data/Merged_Data',certified_reals,certified_new_reals)	
main('../Data/Real Data/Merged_Data/data#gbonbomdatabase_new#dip_7777database_random1_1.txt',
	'../Data/Real Data/data/#gbonbomdatabase_new.txt','../Data/Bot Data/#dip_7777database_random1.txt')

	
	#features = preprocessing.StandardScaler().fit_transform(feature_vectors.values)
'''
	print "obtained all features"
	plot_similarity(feature_vectors.values,user_msgs.keys(),90)
	user_comparisons = similarity_based_approach(feature_vectors.values,users)
	#print sorted(our_tfidf_comparisons,reverse=True)
	for i in range(len(user_comparisons)):
		if user_comparisons[i][0] not in similar_users.keys():
			similar_users[user_comparisons[i][0]] = []
			if user_comparisons[i][2] > 0.0:
				similar_users[user_comparisons[i][0]].append(user_comparisons[i][1])
		else:
			if user_comparisons[i][2] > 0.0:
				similar_users[user_comparisons[i][0]].append(user_comparisons[i][1])
	print similar_users
	G = nx.Graph()
	count = 0
	for i in range(len(user_comparisons)):
		if user_comparisons[i][0] > 0.0:
			G.add_edge(users[user_comparisons[i][0]],users[user_comparisons[i][1]],weight=user_comparisons[i][2])
			count += 1
	print "# Non zero weights: " + str(count)
	#modularity_maximization(G)
	#A = nx.adjacency_matrix(G)
	print nx.is_connected(G)
	components = nx.connected_components(G)
	print [len(c) for c in sorted(components, key=len, reverse=True)]
	largest_component = max(components,key=len)
	print largest_component
	subgraph = G.subgraph(largest_component)
	diameter = nx.diameter(subgraph)
	print "Network diameter of largest component:" + str(diameter)

	#S = nx.to_numpy_matrix(G)
	#A = adjacency_matrix(feature_vectors.values,5,S.tolist())
	#D = diag_degree_matrix(A)
	#print predict_k(A)
	#print A
	
	pos = nx.spring_layout(G)
	nx.draw_networkx_nodes(G,pos,node_size=1000)
	nx.draw_networkx_edges(G,pos,width=0.5)
	# labels
	nx.draw_networkx_labels(G,pos,font_size=7,font_family='sans-serif')
	plt.axis('off')
	plt.savefig("weighted_graph.png") # save as png
	plt.show()
'''
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
	#print eig_vals
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
	#print k
	
	vec = np.array([ eig_pairs[i][1] for i in range(k)])
	vec = np.transpose(vec)
	vec = list(vec)
	temp = []
	#print "first k eigen vectors"
	for p in vec:
		temp.append([float(elements) for elements in p])
	vec = temp
	#print vec
	points = []
	for i in range(len(vec)):
		points.append(Point(vec[i]))
'''
'''
	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(feature_vectors)
	plt.scatter(principalComponents[:, 0], principalComponents[:, 1])
	plt.show()
'''
	#print feature_vectors
'''
	points = np.array([np.array(point) for point in points])
	points.reshape(features.shape[0],k)
	print points.shape
	print features.shape
	print features[0]
	print points[0]
	k, gapdf = optimalK(points, nrefs=5, maxClusters=15)
	print 'Optimal k is: ', k
	plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
	plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
	plt.grid(True)
	plt.xlabel('Cluster Count')
	plt.ylabel('Gap Value')
	plt.title('Gap Values by Cluster Count')
	plt.show()
'''
'''
	num_clusters = 6
	opt_cutoff = 0.5
	clusters = kmeans(points,num_clusters,opt_cutoff)
	for i in range(num_clusters):
		print "cluster No:" + str(i) + ' ' + "#points:" + str(clusters[i].length)
'''