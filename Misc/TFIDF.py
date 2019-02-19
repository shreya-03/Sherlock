from __future__ import division
import string
import math
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer,CountVectorizer
import re,sys
from ipy_table import *
#from SpectralClustering import *
import matplotlib.pyplot as plt
import networkx as nx
from collections import OrderedDict

def format_line(line):
	line = re.sub("\'s","",line)
	line = re.sub("\'ve","have",line)
	line = re.sub("n\'t","not",line)
	line = re.sub("\'ll","will",line)
	line = re.sub("\'m","am",line)
	line = re.sub("\'d","would",line)
	line = re.sub("\'re","are",line)
	line = re.sub("\. ","",line)
	line = re.sub("&gt",">",line)
	line = re.sub("&lt","<",line)
	line = line.lower()
	return line

def tokenize(message):
	tokens = re.split(r'\s|[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]',message)
	tokens = filter(lambda tok: tok.strip(),tokens)
	return tokens

def jaccard_similarity(query, document):
	intersection = set(query).intersection(set(document))
	union = set(query).union(set(document))
	return len(intersection)/len(union)

def term_frequency(term, tokenized_document):
	return tokenized_document.count(term)

def sublinear_term_frequency(term, tokenized_document):
	count = tokenized_document.count(term)
	if count == 0:
		return 0
	return 1 + math.log(count)

def augmented_term_frequency(term, tokenized_document):
	max_count = max([term_frequency(t, tokenized_document) for t in tokenized_document])
	return (0.5 + ((0.5 * term_frequency(term, tokenized_document))/max_count))

def inverse_document_frequencies(tokenized_documents):
	idf_values = {}
	all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
	for tkn in all_tokens_set:
		contains_token = map(lambda doc: tkn in doc, tokenized_documents)
		idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
	return idf_values

def tfidf(documents):
	tokenized_documents = [tokenize(d) for d in documents]
	idf = inverse_document_frequencies(tokenized_documents)
	tfidf_documents = []
	for document in tokenized_documents:
		doc_tfidf = []
		for term in idf.keys():
			tf = sublinear_term_frequency(term, document)
			doc_tfidf.append(tf * idf[term])
		tfidf_documents.append(doc_tfidf)
	return tfidf_documents

def cluster_user_msgs(filename):
	user_msgs = OrderedDict()
	with open(filename,'r') as f:
		lines = f.readlines()
		for line in lines:
			user = str(line.split(',"u":')[1].split(',"e":')[0].replace('"',''))
			if user not in user_msgs.keys():
				user_msgs[user] = ''
				user_msgs[user] += ''.join(format_line(str(line.split(',"m":')[1].split(',"nm":')[0].replace('"',''))))
			else:
				user_msgs[user] += ''.join(format_line(str(line.split(',"m":')[1].split(',"nm":')[0].replace('"',''))))
	f.close()
	return user_msgs

def build_lexicon(messages):
	lexicon = set()
	for message in messages:
		lexicon.update([word for word in tokenize(message)])
	return lexicon

def cosine_similarity(vector1, vector2):
	dot_product = sum(p*q for p,q in zip(vector1, vector2))
	magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
	if not magnitude:
		return 0
	return dot_product/magnitude


	