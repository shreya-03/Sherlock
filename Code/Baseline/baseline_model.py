import sys,time
import math,random,requests,json
import community,urllib2
from twitch import TwitchClient
from collections import OrderedDict,Counter
from os import listdir
from os.path import isfile,join,isdir
from ConditionalEntropy import *
import json
import numpy as np 
import pandas as pd 
from UserBins import map_user_timestamp_bins,map_user_msg_bins
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
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
import operator

def cluster_user_timestamps_msgs(merged_filename,real_filename):
	real_users = set()
	with open(real_filename,'r') as f:
		lines = f.readlines()
		for line in lines:
			real_users.add(str(line.split(',"u":')[1].split(',"e":')[0].replace('"','')))
	f.close()
	#print "real users listed"
	users = {}
	with open(merged_filename,'r') as f:
		lines = f.readlines()
		for line in lines:
			user = str(line.split(',"u":')[1].split(',"e":')[0].replace('"',''))
			if user in users.keys():
				users[user]['t'].append(int(line.split('"t":')[1].split(',"u":')[0].replace('"','')))
				users[user]['m'].append(len(line.split(',"m":')[1].split(',"nm":')[0].split(' ')))
			else:
				users[user] = {}
				users[user]['t'] = []
				users[user]['m'] = []
				users[user]['t'].append(int(line.split('"t":')[1].split(',"u":')[0].replace('"','')))
				users[user]['m'].append(len(line.split(',"m":')[1].split(',"nm":')[0].split(' ')))
				if user in real_users:
					users[user]['b'] = 'no'
				else:
					users[user]['b'] = 'yes'
	#print "labelled users"
	return users

def cluster_test_user_timestamps_msgs(filename):
	users = {}
	with open(filename,'r') as f:
		lines = f.readlines()
		for line in lines:
			user = str(line.split(',"u":')[1].split(',"e":')[0].replace('"',''))
			if user in users.keys():
				users[user]['t'].append(int(line.split('"t":')[1].split(',"u":')[0].replace('"','')))
				users[user]['m'].append(len(line.split(',"m":')[1].split(',"nm":')[0].split(' ')))
			else:
				users[user] = {}
				users[user]['t'] = []
				users[user]['m'] = []
				users[user]['t'].append(int(line.split('"t":')[1].split(',"u":')[0].replace('"','')))
				users[user]['m'].append(len(line.split(',"m":')[1].split(',"nm":')[0].split(' ')))
				if line.split(',"b":')[1] == "b":
					users[user]['b'] = 'yes'
				else:
					users[user]['b'] = 'no'
	return users

def select_test_users(users):
	selected_users = dict()
	for user in users.keys():
		if len(users[user]['t']) >= 6:	
			if user not in selected_users.keys():
				selected_users[user] = {}
				selected_users[user]['t'] = []
				selected_users[user]['m'] = []
				#user_imd[user].append(0)
				for i in range(len(users[user]['t'])-1):
					selected_users[user]['t'].append(users[user]['t'][i+1]-users[user]['t'][i])
				selected_users[user]['m'] = users[user]['m']
				selected_users[user]['b'] = users[user]['b']
				#user_imd[user]['t'] = user_imd[user]['t']*3
				#selected_users[user]['b'] = users[user]['b']
	return selected_users

def select_users(users):
	selected_users = dict()
	for user in users.keys():
		if len(users[user]['t']) >= 6:	
			if user not in selected_users.keys():
				selected_users[user] = {}
				selected_users[user]['t'] = []
				selected_users[user]['m'] = []
				#user_imd[user].append(0)
				for i in range(len(users[user]['t'])-1):
					selected_users[user]['t'].append(users[user]['t'][i+1]-users[user]['t'][i])
				selected_users[user]['m'] = users[user]['m']
				#user_imd[user]['t'] = user_imd[user]['t']*3
				selected_users[user]['b'] = users[user]['b']
	return selected_users

def get_imd_cce(selected_users):
	cce_imds = []
	for user in selected_users.keys():
		#print "u:" + user +' ' + "b:" + user_imd[user]['b']
		bins = map_user_timestamp_bins(selected_users[user]['t'])
		s = ''.join(str(b) for b in bins)
		tree = Trie()
		MIN_CCE = sys.float_info.max
		for length in range(1,len(s)+1):
			for i in range(len(s)-length+1):
				tree.AddString(s[i:i+length]) 
			tree.LevelSum()
			tree.AssignNodeProbability()
			tree.ConditionalEntropy()
			#print len(tree.conditionalentropy)
			imd_cce = tree.conditionalentropy[length-1] + ((float(tree.leveluniquepatterns[length-1])/tree.levelsum[length-1])*100)*tree.conditionalentropy[0]
			if MIN_CCE < imd_cce:
				imd_cce = MIN_CCE
				break
		cce_imds.append(imd_cce)
	return cce_imds

def get_msgLength_cce(selected_users):
	cce_msgLengths = []
	for user in selected_users.keys():
		bins = map_user_msg_bins(selected_users[user]['m'])
		s = ''.join(str(b) for b in bins)
		tree = Trie()
		MIN_CCE = sys.float_info.max
		for length in range(1,len(s)+1):
			for i in range(len(s)-length+1):
				tree.AddString(s[i:i+length]) 
			tree.LevelSum()
			tree.AssignNodeProbability()
			tree.ConditionalEntropy()
			#print len(tree.conditionalentropy)
			msg_len_cce = tree.conditionalentropy[length-1] + ((float(tree.leveluniquepatterns[length-1])/tree.levelsum[length-1])*100)*tree.conditionalentropy[0]
			if MIN_CCE < msg_len_cce:
				msg_len_cce = MIN_CCE
				break
		#print "msg lengths entropy:" + str(msg_len_cce)
		#print "avg entropy:" + str((imd_cce + msg_len_cce)/2)
		cce_msgLengths.append(msg_len_cce)
	return cce_msgLengths

def assign_labels(selected_users):
	labels = []
	for user in selected_users.keys():
		labels.append(selected_users[user]['b'])
	return labels

def getAllFilesRecursilvely(path):
	train_features = {'cce_imds':[],'cce_ml':[],'labels':[]}
	r_train_count = 0
	cc_train_count = 0
	for f in listdir(path):
		if isfile(join(path,f)):
			merged_filename = join(path,f)
			print merged_filename
			lt = f.split('#')
			real_file = "../Real Data/"+str(lt[0])+'/#'+str(lt[1])+'.txt'
			users = cluster_user_timestamps_msgs(merged_filename,real_file)
			selected_users = select_users(users)
			if 'random' in str(f) and r_train_count <= 15: 
				train_features['cce_imds'].extend(get_imd_cce(selected_users))
				train_features['cce_ml'].extend(get_msgLength_cce(selected_users))
				train_features['labels'].extend(assign_labels(selected_users))
				r_train_count += 1
			if 'chatterscontrolled' in str(f) and cc_train_count <= 15:
				train_features['cce_imds'].extend(get_imd_cce(selected_users))
				train_features['cce_ml'].extend(get_msgLength_cce(selected_users))
				train_features['labels'].extend(assign_labels(selected_users))
				cc_train_count += 1
			'''
			if 'chatterscontrolled' in str(f) and test_count <= 10:
				test_features['cce_imds'].extend(get_imd_cce(selected_users))
				test_features['cce_ml'].extend(get_msgLength_cce(selected_users))
				test_features['labels'].extend(assign_labels(selected_users))
				test_count += 1
			'''
			if r_train_count > 15 and cc_train_count > 15:
				break
			#print "train count:" + str(train_count) + ' ' + "test count:" + str(test_count)
	return train_features

def run_model(model,alg_name,X_train,y_train,X_test,y_test):
	model.fit(X_train,y_train)
	y_pred = model.predict(X_test)
	#return y_pred
	accuracy = accuracy_score(y_test,y_pred)*100
	print "Classifier:" + alg_name + ' ' + "Accuracy:" + str(accuracy)
	#print "Precision Recall fscore:" + str(precision_recall_fscore_support(y_test,y_pred,average='weighted'))

def main():
	'''
	users = cluster_user_timestamps_msgs(sys.argv[1],sys.argv[2])
	selected_users = select_users(users)
	cce_imds = get_imd_cce(selected_users)
	cce_msgLengths = get_msgLength_cce(selected_users)
	labels = []
	for i in range(len(selected_users.keys())):
		user = selected_users.keys()[i]
		print "u: " + str(user) + ' ' + "cce_imd: " + str(cce_imds[i]) + ' ' + "cce_ml: " + str(cce_msgLengths[i])
		labels.append(selected_users[user]['b'])
	'''
	'''
	test_features = {'cce_imds':[],'cce_ml':[]}
	index_dict = {}
	index = 0
	with open('../bots.txt','r') as f:
		lines = f.readlines()
		for line in lines:
			if line.startswith('../'):
				filename = '../Real Data/' + line.split('#')[0].split('/')[3] + '/#' +line.split('#')[1].split(' ')[0].replace('\n','')
				print filename
				users = cluster_test_user_timestamps_msgs(filename)
				selected_users = select_test_users(users)
				test_features['cce_imds'].extend(get_imd_cce(selected_users))
				test_features['cce_ml'].extend(get_msgLength_cce(selected_users))
				index_dict[filename] = index
				index += len(selected_users.keys())
				#print index
	f.close()
	with open('../jan_bots.txt','r') as f:
		lines = f.readlines()
		for line in lines:
			if line.startswith('Downloads'):
				filename = '../data_Jan/#' + line.split('#')[1].replace('\n','')
				print filename
				users = cluster_test_user_timestamps_msgs(filename)
				selected_users = select_test_users(users)
				test_features['cce_imds'].extend(get_imd_cce(selected_users))
				test_features['cce_ml'].extend(get_msgLength_cce(selected_users))
				index_dict[filename] = index
				index += len(selected_users.keys())
				#print index
	f.close()				
	'''
	'''
	train_features = getAllFilesRecursilvely(sys.argv[1])
	df1_train = pd.DataFrame({'cce_imds':train_features['cce_imds']})
	df2_train = pd.DataFrame({'cce_msgLengths':train_features['cce_ml']})
	df3_train = pd.DataFrame({'labels':train_features['labels']})
	X_train = pd.concat([df1_train,df2_train,df3_train],axis=1)
	'''
	df = pd.read_csv('./train_features.csv')
	'''
	#test_features = {'cce_imds':[],'cce_ml':[]}
	#train_features = pd.read_csv('train_features.csv',delimiter=',')
	'''
	X_train = df.iloc[:,1:3]
	Y_train = df.iloc[:,3:4]
	filename = '../../Sample Data/sample_chatlog_real.txt'
	print filename
	users = cluster_test_user_timestamps_msgs(filename)
	selected_users = select_test_users(users)
	test_features = {'cce_imds':[],'cce_ml':[],'labels':[]}
	test_features['cce_imds'].extend(get_imd_cce(selected_users))
	test_features['cce_ml'].extend(get_msgLength_cce(selected_users))
	test_features['labels'].extend(assign_labels(selected_users))
	df1_test = pd.DataFrame({'cce_imds':test_features['cce_imds']})
	df2_test = pd.DataFrame({'cce_msgLengths':test_features['cce_ml']})
	X_test = pd.concat([df1_test,df2_test],axis=1)
	Y_test = pd.DataFrame({'labels':test_features['labels']})
	'''
	model = tree.DecisionTreeClassifier(criterion='entropy',max_depth=5)
	run_model(model,"Decision Tree",X_train,Y_train,X_test,Y_test)

	#-------Random Forest-----------

	model = RandomForestClassifier(n_estimators=10)
	run_model(model,"Random Forest",X_train,Y_train,X_test,Y_test)
	'''
	
	#-------xgboost-----------------
	model = XGBClassifier()
	run_model(model,"XGBoost",X_train,Y_train,X_test,Y_test)
	'''
	real_count = 0
	bot_count = 0
	for i in range(len(y_pred)):
		if y_pred[i] == 'yes':
			bot_count += 1
		else:
			real_count += 1
	print "#real users:" + str(real_count) + " " + "#bot users:" + str(bot_count)
	'''
	'''
	#print y_pred
	sorted_index_dict = sorted(index_dict.items(), key=operator.itemgetter(1))
	print sorted_index_dict
	#indexes_list = sorted_index_dict.values()
	#print indexes_list
	#files_list = sorted_index_dict.keys()
	for idx,tup in enumerate(sorted_index_dict):
		real_count = 0
		bot_count = 0
		print tup[0]
		if idx < len(sorted_index_dict)-1:
			for i in range(tup[1],sorted_index_dict[idx+1][1]):
				if y_pred[i] == 'yes':
					bot_count += 1
				else:
					real_count += 1
			print "#real users:" + str(real_count) + " " + "#bot users:" + str(bot_count)
		else:
			for i in range(tup[1],index):
				if y_pred[i] == 'yes':
					bot_count += 1
				else:
					real_count += 1
			print "#real users:" + str(real_count) + " " + "#bot users:" + str(bot_count)
	'''
	'''
	#-------SVM Classifier-----------

	model = SVC()
	run_model(model,"SVM Classifier",X_train,Y_train,X_test,Y_test)

	#-------Nearest Classifier-------

	model = neighbors.KNeighborsClassifier()
	run_model(model,"Nearest Neighbors Classifier",X_train,Y_train,X_test,Y_test)

	#-------SGD Classifier-----------

	model = OneVsRestClassifier(SGDClassifier())
	run_model(model,"SGD Classifier",X_train,Y_train,X_test,Y_test)

	#-------Gaussian NB--------------

	model = GaussianNB()
	run_model(model,"Gaussian NB",X_train,Y_train,X_test,Y_test)

	#-------NN-MLP-------------------

	model = MLPClassifier()
	run_model(model,"NN-MLP",X_train,Y_train,X_test,Y_test)
	'''
main()

