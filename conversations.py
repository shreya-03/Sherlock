from math import sqrt
import numpy as np
from scipy.linalg import norm
from scipy.sparse import csc_matrix
from tfidf import *
import networkx as nx
from collections import OrderedDict,Counter
import csv
import seaborn as sns
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords


class HITS(object):

	def __init__(self,A,epsilon=0.0001):
		self.A = A
		self.epsilon = epsilon
		self.authority_scores = np.empty([A.shape[0],1])
		self.hub_scores = np.empty([A.shape[1],1])

	def update(self):
		# Normalizing the authorities and hubs vector by their L2 norm
		M,N = self.A.shape
		#print M,N
		auth0 = (1.0/sqrt(M)) * np.ones([M, 1])
		hubs0 = (1.0/sqrt(N)) * np.ones([N, 1])
		#print hubs0.shape
		self.authority_scores = self.A.dot(hubs0)
		self.hub_scores = self.A.transpose().dot(auth0)

		# Normalizing auth and hub by their L2 norm
		self.authority_scores = (1.0/norm(self.authority_scores, 2))*(self.authority_scores)
		self.hub_scores = (1.0/norm(self.hub_scores, 2))*(self.hub_scores)

		# Calculating the hub and authority vectors until convergence
		while((norm(self.authority_scores-auth0, 2) > self.epsilon) or (norm(self.hub_scores-hubs0, 2) > self.epsilon)):
			auth0 = self.authority_scores
			hubs0 = self.hub_scores
			self.authority_scores = self.A.dot(hubs0)
			self.hub_scores = self.A.transpose().dot(auth0)

			self.authority_scores = (1.0/norm(self.authority_scores, 2))*(self.authority_scores)
			self.hub_scores = (1.0/norm(self.hub_scores, 2))*(self.hub_scores)

		# Printing the values of hubs and authorities vectors
		#print "authority vector is ", "\n", self.authority_scores
		#print "hubs vector is ", "\n", self.hub_scores


class UsersConversations(object):

	def __init__(self,chats_info):
		self.msgs_per_user = self.cluster_users_msgs(chats_info)
		self.documents = [self.msgs_per_user[user][:-1] for user in self.msgs_per_user.keys()]
		#self.normalized_documents = DocumentNormalization(self.documents)
		self.tokenized_documents = [d.split(' ' ) for d in self.documents]
		self.terms_per_user = []
		self.avg_terms = self.user_average_terms()
		self.no_users = len(self.msgs_per_user.keys())
		self.idf_values = self.inverse_user_frequencies()
		self.adjacency_matrix = np.zeros([len(self.msgs_per_user.keys()),len(self.idf_values.keys())])

	def cluster_users_msgs(self,chats_info):
		user_msgs = OrderedDict()
		for i in range(len(chats_info)):
			if chats_info[i]['user'] not in user_msgs.keys():
				user_msgs[chats_info[i]['user']] = ''
				user_msgs[chats_info[i]['user']] += chats_info[i]['msgs'] + ' '
			else:
				user_msgs[chats_info[i]['user']] += chats_info[i]['msgs'] + ' '
		return user_msgs
				
	def inverse_user_frequencies(self):
		idf_values = OrderedDict()
		all_tokens_set = set([item for sublist in self.tokenized_documents for item in sublist])
		#print "users total tokens:" + str(len(list(all_tokens_set)))
		for tkn in all_tokens_set:
			contains_token = map(lambda doc: tkn in doc, self.tokenized_documents)
			idf_values[tkn] = 1 + math.log(len(self.tokenized_documents)/(sum(contains_token)))
		return idf_values

	def sublinear_term_frequency(self,term,tokenized_document):
		count = tokenized_document.count(term)
		if count == 0:
			return 0
		return 1 + math.log(count)

	def user_average_terms(self):
		sum_terms = 0
		for document in self.tokenized_documents:
			sum_terms += len(list(set(document)))
			self.terms_per_user.append(len(list(set(document))))
		return sum_terms/len(self.tokenized_documents)

	def AssignWeights(self):
		for user_index,user in enumerate(self.msgs_per_user.keys()):
			for term_index,term in enumerate(self.idf_values.keys()):
				freq = self.sublinear_term_frequency(term,self.tokenized_documents[user_index])
				no_terms = self.terms_per_user[user_index]
				ifreq = self.idf_values[term]
				self.adjacency_matrix[user_index][term_index] = (freq/(freq+0.5+(1.5*no_terms/self.avg_terms))) * (math.log((self.no_users+0.5)/ifreq)/math.log(self.no_users+1))

class SessionsConversations(object):

	def __init__(self,chats_info,k):
		self.msgs_per_session = self.cluster_k_msgs(chats_info,k)
		self.documents = [self.msgs_per_session[i]['msgs'] for i in range(len(self.msgs_per_session))]
		#self.normalized_documents = DocumentNormalization(self.documents)
		self.tokenized_documents = [d.split(' ') for d in self.documents]
		self.terms_per_session = []
		self.avg_terms = self.user_average_terms()
		self.no_users = [len(list(self.msgs_per_session[i]['users'])) for i in range(len(self.msgs_per_session))]
		self.idf_values = self.inverse_user_frequencies()
		self.adjacency_matrix = np.zeros([len(self.msgs_per_session),len(self.idf_values.keys())])

	def cluster_k_msgs(self,chats_info,k):
		session_msgs = []
		for i in range(0,len(chats_info),k):
			session_info = {}
			session_msg = ''
			session_info['users'] = set()
			session = chats_info[i:i+k]
			for chat_info in session:
				session_info['users'].add(chat_info['user'])
				session_msg += chat_info['msgs'] + ' '
			#message_tokens = tokenize(session_msg)
			#session_info['msgs'] = ''
			session_info['msgs'] = session_msg[:-1]
			session_msgs.append(session_info)
		return session_msgs

	def inverse_user_frequencies(self):
		idf_values = OrderedDict()
		all_tokens_set = set([item for sublist in self.tokenized_documents for item in sublist])
		#print "sessions total tokens:" + str(len(list(all_tokens_set)))
		for tkn in all_tokens_set:
			contains_token = map(lambda doc: tkn in doc, self.tokenized_documents)
			idf_values[tkn] = 1 + math.log(len(self.tokenized_documents)/(sum(contains_token)))
		return idf_values

	def sublinear_term_frequency(self,term,tokenized_document):
		count = tokenized_document.count(term)
		if count == 0:
			return 0
		return 1 + math.log(count)

	def user_average_terms(self):
		sum_terms = 0
		for document in self.tokenized_documents:
			sum_terms += len(list(set(document)))
			self.terms_per_session.append(len(list(set(document))))
		return sum_terms/len(self.tokenized_documents)

	def AssignWeights(self):
		for i in range(len(self.msgs_per_session)):
			for term_index,term in enumerate(self.idf_values.keys()):
				freq = self.sublinear_term_frequency(term,self.tokenized_documents[i])
				no_terms = self.terms_per_session[i]
				ifreq = self.idf_values[term]
				self.adjacency_matrix[i][term_index] = (freq/(freq+0.5+(1.5*no_terms/self.avg_terms))) * (math.log((self.no_users[i]+0.5)/ifreq)/math.log(self.no_users[i]+1))

def term_feature_scores(user_hub_scores,session_hub_scores,user_vocab,session_vocab,alpha):
	feature_scores = np.empty([user_hub_scores.shape[0],1])
	for index,term in enumerate(user_vocab.keys()):
		#print user_hub_scores[index][0]
		#print session_hub_scores[session_vocab.keys().index(term)][0]
		feature_scores[index][0] = alpha*user_hub_scores[index][0] + (1-alpha)*session_hub_scores[session_vocab.keys().index(term)][0]
	return feature_scores

def ConversationalFeatures(chats_info):
	
	epsilon = 0.0001 		
	conversations = UsersConversations(chats_info)
	conversations.AssignWeights()
	users_A = conversations.adjacency_matrix
	users_hits = HITS(users_A,epsilon=epsilon)
	users_hits.update()
	sessionconvo = SessionsConversations(chats_info,10)
	sessionconvo.AssignWeights()
	#print "Sessions:" + str(sessionconvo.idf_values.keys())
	sessions_A = sessionconvo.adjacency_matrix
	sessions_hits = HITS(sessions_A,epsilon=epsilon)
	sessions_hits.update()
	#print set(sessionconvo.idf_values.keys()).intersection(set(conversations.idf_values.keys()))
	feature_scores = term_feature_scores(users_hits.hub_scores,sessions_hits.hub_scores,conversations.idf_values,sessionconvo.idf_values,0.5)
	user_term_feature_scores = np.zeros([users_A.shape[0],users_A.shape[1]])
	for doc_index,document in enumerate(conversations.tokenized_documents):
		for term_index,term in enumerate(conversations.idf_values.keys()):
			if term in document:
				user_term_feature_scores[doc_index][term_index] = feature_scores[term_index][0]

	features = np.empty([users_A.shape[0],users_A.shape[0]])
	for id_0,feature_0 in enumerate(user_term_feature_scores.tolist()):
		for id_1,feature_1 in enumerate(user_term_feature_scores.tolist()):
			user_0 = conversations.msgs_per_user.keys()[id_0]
			user_1 = conversations.msgs_per_user.keys()[id_1]
			deg_0 = 0
			deg_1 = 0
			deg_01 = 0
			for i in range(len(sessionconvo.msgs_per_session)):
				if user_0 in list(sessionconvo.msgs_per_session[i]['users']) and user_1 in list(sessionconvo.msgs_per_session[i]['users']):
					deg_01 += 1
					deg_1 += 1
					deg_0 += 1
				else:
					if user_0  in list(sessionconvo.msgs_per_session[i]['users']):
						deg_0 += 1
					if user_1 in list(sessionconvo.msgs_per_session[i]['users']):
						deg_1 += 1
			features[id_0][id_1] = cosine_similarity(feature_0,feature_1)* ((deg_01*(deg_0+deg_1))/float(2*deg_0*deg_1))
	#print features
	#plot_similarity(features,conversations.msgs_per_user.keys(),90)
	return features

#ConversationalFeatures(sys.argv[1])