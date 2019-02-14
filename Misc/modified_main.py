import math,random,requests,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from entropy import *
from collections import OrderedDict,Counter
import json,time
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from os import listdir
from os.path import isfile,join,isdir
from modified_UserClustering import *
from sklearn.semi_supervised import label_propagation

plt.rcParams['axes.facecolor'] = 'white'

def plot_entropy_feature(feature_vectors,considered_users_index,real_users,bot_users,users):
	X = []
	for i in range(feature_vectors.shape[0]):
		if i in considered_users_index:
			X.append(feature_vectors.iloc[i,:].values.tolist())
	print pd.DataFrame(X).iloc[:,1].values
	X = np.array(X)
	real_X = []
	real_Y = []
	bot_X = []
	bot_Y = []
	print X.shape[0]
	for i,index in enumerate(considered_users_index):
		if users[index] in real_users:
			real_X.append(X[i][0])
			real_Y.append(X[i][1])
		else:
			bot_X.append(X[i][0])
			bot_Y.append(X[i][1])
	print len(real_X),len(bot_X)
	plt.scatter(real_X,real_Y,c='red',s=7,label='real')
	plt.scatter(bot_X,bot_Y,c='blue',s=7,label='bot')
	plt.legend(loc='upper left')
	plt.title('entropy features')
	plt.show()

def get_channel_followers(path,channel_name):
	
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

def run_model(model,alg_name,X_train,y_train,X_test,y_test):
	model.fit(np.array(X_train),np.array(y_train).ravel())
	y_pred = model.predict(np.array(X_test))
	accuracy = accuracy_score(np.array(y_test),y_pred)*100
	print "Classifier:" + alg_name + ' ' + "Accuracy:" + str(accuracy)

def getAllFilesRecursive(path,certified_reals,certified_new_reals):
	count = 0
	acc = []
	for file in listdir(path):
		print file
		filename = join(path,file)
		# if (file.split('#')[1].split('database')[0] in certified_new_reals or file.split('#')[1].split('database')[0] in certified_reals) and 'b<r' not in str(filename):
		lt = file.split('#')
		real_file = "../Data/Real Data/"+str(lt[0])+'/#'+str(lt[1])+'.txt'
		try:
			if 'random1' in str(lt[2]):
				boted_file = "../Data/Bot Data/#dip_7777database_random1.txt"
				ret = main(filename,real_file,boted_file)
				if ret != -1:
					acc.append(ret)
					count += 1
			if 'random2' in str(lt[2]):
				boted_file = "../Data/Bot Data/#dip_7777database_random2.txt"
				ret = main(filename,real_file,boted_file)
				if ret != -1:
					acc.append(ret)
					count += 1
			if 'chatterscontrolled' in str(lt[2]):
				boted_file = "../Data/Bot Data/#dip_7777database_chatterscontrolled.txt"
				ret = main(filename,real_file,boted_file)
				if ret != -1:
					acc.append(ret)
					count += 1
			if 'organicgrowth' in str(lt[2]):
				boted_file = "../Data/Bot Data/#dip_7777database_organicgrowth.txt"
				ret = main(filename,real_file,boted_file)
				if ret != -1:
					acc.append(ret)
					count += 1
		except Exception as e:
			print e
			continue
	print (sum(acc)/count)

def get_final_features(feature_vectors,considered_users_index):
	X = []
	for i in range(len(feature_vectors)):
		if i in considered_users_index:
			X.append(feature_vectors.iloc[i,:].values.tolist())
	return pd.DataFrame(X)


def data_labelprop(X,real_users_index,bot_users_index):
	label_X = []
	label_Y = []

	for index in range(len(X)):
		#if index in considered_users_index:
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

def label_data_classifier(X,real_users_index,bot_users_index,users,considered_users_index,real_users,bot_users):
	X_train = []
	Y_train = []
	X_test = []
	Y_test = []
	users = list(users)
	for index in range(len(X)):
		if index in real_users_index: 
			X_train.append(X.iloc[index,:].values.tolist())
			Y_train.append(0)
		elif index in bot_users_index:
			X_train.append(X.iloc[index,:].values.tolist())
			Y_train.append(1)
		else:
			X_test.append(X.iloc[index,:].values.tolist())
			if users[index] in real_users:
				Y_test.append(0)
			else:
				Y_test.append(1)
	return X_train,X_test,Y_train,Y_test

def readjust(label_X,label_Y,uw_ft,user_entropy):
	uw_ft = [i[0] for i in uw_ft.values]
	user_entropy = [i[0] for i in user_entropy.values]
	data = [label_X[i] for i in range(len(label_X)) if label_Y[i] == 1]
	# data = (sorted(data,key=lambda tup:(tup[0],tup[1])))
	xs = [i[0] for i in data]
	xs = sorted(xs)
	ys = [i[1] for i in data]
	ys = sorted(ys)
	quant = (int)(0.9 * (float)(len(xs)))
	print(xs[0],xs[-1],ys[0],ys[-1])
	try:
		print(xs[len(xs)-quant], xs[quant], ys[len(ys)-quant], ys[quant])
		xmean = mean(xs)
		ymean = mean(ys)
		xmarg, ymarg = 5, 100
		if xs[quant] - xs[len(xs)-quant] > 5 and ys[quant] - ys[len(ys)-quant] < 60:
			print("read1")
			xmarg, ymarg = 5, 20
		elif xs[quant] - xs[len(xs)-quant] < 5 or ys[quant] - ys[len(ys)-quant] > 150:
			print("read2")
			xmarg, ymarg = 5, 200
	except Exception as e:
		xmean, ymean = mean(xs), mean(ys)
		xmarg, ymarg = 5, 100
	xmin, xmax = xmean, xmean
	ymin, ymax = ymean, ymean
	xmin -= xmarg
	xmax += xmarg
	ymin -= ymarg
	ymax += ymarg
	print(xmin, xmax, ymin, ymax)
	uw_select,ue_select = [],[]
	for i in range(len(label_X)):
		print(label_X[i][0],label_X[i][1],label_Y[i],i)
		if label_Y[i] == 1:
			uw_select.append(uw_ft[i])
			ue_select.append(user_entropy[i])
		if label_X[i][0] > xmin and label_X[i][0] < xmax and label_X[i][1] > ymin and label_X[i][1] < ymax:
			if label_Y[i] == 0:
				print label_X[i]
				label_Y[i] = 1
	for i in range(len(label_X)):
		if label_X[i][0] > xmin and label_X[i][0] < xmax and label_X[i][1] > ymin and label_X[i][1] < ymax:
			if label_Y[i] != 1:
				if uw_ft[i] in uw_select and user_entropy[i] in ue_select:
					label_Y[i] = 1
					print label_X[i], i, "changed"
			else:
				if (uw_ft[i] not in uw_select) and (user_entropy[i] not in ue_select):
					label_Y[i] = -1
					print label_X[i], i, "changed back"
	uw_many = [i for i in uw_ft if uw_ft.count(i) > 5]
	ue_many = [i for i in user_entropy if user_entropy.count(i) > 5]
	for i in range(len(label_X)):
		if uw_ft[i] in uw_many and user_entropy[i] in ue_many:
			label_Y[i] = 1
			print "outside change"
	# print(uw_select)
	# print(ue_select)
	return label_X, label_Y


def main(merged_filename,real_file,boted_file):
	
	documents = []
	print merged_filename
	users_info = getUserIMDMessages(merged_filename)
	#print users_info.keys()
	#print "total users:" + str(len(users_info.keys()))
	considered_users_index = []
	index = 0
	for user in users_info.keys():
		if users_info[user]['m'] > 1:
			considered_users_index.append(index)
		index += 1
	real_users_index,bot_users_index,real_users,bot_users = labeling_data(merged_filename,real_file,boted_file)
	
	labels = []
	users = users_info.keys()
	for user in users_info.keys():
		if users.index(user) in considered_users_index: 
			if user in real_users:
				labels.append(0)
			else:
				labels.append(1)
	
	users_dict = users_info

	## For no of messages per user feature
	user_chats_ft = get_chats_features(users_dict)
	user_chats_ft = pd.DataFrame(user_chats_ft)

	# User windows
	uw_dict = user_windows(merged_filename)
	uw_ft = []
	for user in users_dict.keys():
		if users_dict[user]['m'] > 1:
			uw_ft.append(uw_dict[user])
	uw_ft = pd.DataFrame(uw_ft)
	
	## For per user imds features 
	user_imd_bins = pd.DataFrame(get_IMD_features(users_dict))
	
	# Feature of Entropy of imds of a user
	user_entropy = pd.DataFrame(np.array(get_entropy_features(merged_filename)))
	
	## to get the features of users with no of messages > 1
	user_entropy = get_final_features(user_entropy,considered_users_index)
	
	# print(uw_ft, user_entropy)

	## plot entropy feature of each user 
	#plot_entropy_feature(user_entropy,considered_users_index,real_users,bot_users,users)
	
	final_features = pd.concat([user_chats_ft,user_imd_bins],axis=1)
	# final_features = pd.concat([user_chats_ft,user_imd_bins,uw_ft,user_entropy],axis=1)
	
	real_X = []
	real_Y = []
	bot_X = []
	bot_Y = []
	list_user_fts = final_features.values.tolist()
	#print len(list_user_fts)
	for i,index in enumerate(considered_users_index):
		if users[index] in real_users:
			real_X.append(list_user_fts[i][0])
			real_Y.append(list_user_fts[i][1])
		else:
			bot_X.append(list_user_fts[i][0])
			bot_Y.append(list_user_fts[i][1])
	real_tup = [(real_X[i],real_Y[i]) for i in range(len((real_X)))]
	bot_tup = [(bot_X[i],bot_Y[i]) for i in range(len((bot_X)))]
	#real_tup = sorted(real_tup)
	#bot_tup = sorted(bot_tup)
	#print real_tup
	#print bot_tup
	#plt.scatter(real_X,real_Y,c='red',s=7)
	#plt.scatter(bot_X,bot_Y,c='blue',s=7)
	#plt.axvline(x=initavg_f1,c='yellow')
	#lt.axhline(y=initavg_f2,c='green')
	#plt.title(filename)
	#plt.show()

	## Cluster datapoints on desired new set of features 
	# f1 = final_features.iloc[:,0].values
	# f2 = final_features.iloc[:,1].values
	# gt_X = np.array(list(zip(f1,f2)))
	# Xmeans = XMeans(kmax=7)
	# Xmeans.fit(list(gt_X))
	# XMeanslabels = Xmeans.labels_
	# plot_graph(gt_X,real_X,real_Y,bot_X,bot_Y,XMeanslabels)
	channel_followers = get_channel_followers('../followers_cnt/',real_file.split('#')[1].split('database')[0])
	#real_users_index = set(real_users_index)
	users = set(users)
	for user in channel_followers:
		if user in users:
			real_users_index.append(list(users).index(user))
	print real_users_index
	#plot_graph(final_features,real_users_index,bot_users_index)
	
	print "#considered users:" + str(len(considered_users_index))
	
	label_X,label_Y = data_labelprop(final_features,real_users_index,bot_users_index)
	print len(label_X),len(label_Y)
	orig_labelX, orig_labelY = label_X[:], label_Y[:]
	label_X, label_Y = readjust(label_X,label_Y,uw_ft,user_entropy)
	real_X = []
	real_Y = []
	bot_X = []
	bot_Y = []
	for i in range(len(label_X)):
		if label_Y[i] == 0:
			real_X.append(final_features.iloc[i,0])
			real_Y.append(final_features.iloc[i,1])
		elif label_Y[i] == 1:
			bot_X.append(final_features.iloc[i,0])
			bot_Y.append(final_features.iloc[i,1])
	plt.scatter(real_X,real_Y,c='blue',s=25,label='real')
	plt.scatter(bot_X,bot_Y,c='red',s=25,label='bot')
	#plt.tick_params(labelsize=16)
	#hfont = {'fontname':'Helvetica'}
	plt.rc('font', family='sans-serif')
	plt.rc('xtick', labelsize='x-large')
	plt.rc('ytick', labelsize='x-large')
	plt.xlabel('Number  of  messages  per  user',fontsize='x-large')
	plt.ylabel('Mean  IMD  per  user',fontsize='x-large')
	#plt.title('Seed labels',fontsize='large',fontweight='bold')
	plt.legend(loc='lower right',prop={'size':12})
	plt.show()
	# Learn with LabelSpreading
	label_spread = label_propagation.LabelSpreading(kernel='rbf', alpha=0.6)
	label_spread.fit(label_X, label_Y)
	output_labels = label_spread.transduction_
	label_spread.fit(orig_labelX, orig_labelY)
	orig_output_labels = label_spread.transduction_
	
	# print output_labels
	# print labels
	pred_real_X = []
	pred_real_Y = []
	pred_bot_X = []
	pred_bot_Y = []
	for i in range(len(output_labels)):
		if output_labels[i] == 0:
			pred_real_X.append(final_features.iloc[i,0])
			pred_real_Y.append(final_features.iloc[i,1])
		else:
			pred_bot_X.append(final_features.iloc[i,0])
			pred_bot_Y.append(final_features.iloc[i,1])

	plt.xlim(0,80)
	plt.ylim(0,1500)
	plt.scatter(pred_real_X,pred_real_Y,c='blue',s=25,label='real')
	plt.scatter(pred_bot_X,pred_bot_Y,c='red',s=25,label='bot')
	#plt.tick_params(labelsize=16)
	plt.legend(loc='upper right',prop={'size':12})
	#hfont = {'fontname':'Helvetica'}
	plt.rc('font', family='sans-serif')
	plt.rc('xtick', labelsize='x-large')
	plt.rc('ytick', labelsize='x-large')
	plt.xlabel('Number  of  messages  per  user',fontsize='x-large')
	plt.ylabel('Mean  IMD  per  user',fontsize='x-large')
	#plt.title('Final labels after propagation',fontsize='large',fontweight='bold')
	plt.show()

	orig_tot,orig_cor = 0,0
	total,correct = 0,0
	for i in range(len(output_labels)):
		if label_Y[i] == -1:
			if labels[i] == output_labels[i]:
				correct += 1
			else:
				print label_X[i],i
			total += 1
		if orig_labelY[i] == -1:
			if labels[i] == orig_output_labels[i]:
				orig_cor += 1
			orig_tot += 1
	print correct,total
	print (float(correct)/total)*100
	print accuracy_score(np.array(labels),output_labels)*100
	return accuracy_score(np.array(labels),output_labels)*100
	print orig_cor,orig_tot
	print (float(orig_cor)/orig_tot)*100
	print accuracy_score(np.array(labels),orig_output_labels)*100
	
	# x_train,x_test,y_train,y_test = label_data_classifier(final_features,real_users_index,bot_users_index,users,considered_users_index,real_users,bot_users)
	# X_train = pd.DataFrame(x_train)
	# y_train = pd.DataFrame(y_train)
	# X_test = pd.DataFrame(x_test)
	# y_test = pd.DataFrame(y_test)
	

	# #-------Nearest Classifier-------

	# model = neighbors.KNeighborsClassifier()
	# run_model(model,"Nearest Neighbors Classifier",X_train,y_train,X_test,y_test)

	# #-------NN-MLP-------------------

	# model = MLPClassifier()
	# run_model(model,"NN-MLP",X_train,y_train,X_test,y_test)

with open('../Data/Real Data/certified real.txt','r') as f:
	certified_reals = f.readlines()
certified_reals = [real.strip('\n') for real in certified_reals]
f.close()
with open('../Data/Real Data/certified real for new data.txt') as f:
	certified_new_reals = f.readlines()
certified_new_reals = [real.strip('\n') for real in certified_new_reals]
f.close()
# getAllFilesRecursive('../Data/Real Data/Merged_Data',certified_reals,certified_new_reals)	
#getAllFilesRecursive('../Data/Real Data/Merged_Data/',certified_reals,certified_new_reals)
#main('../Data/Real Data/Merged_Data/data#prophdawgdatabase_new#dip_7777database_random1_1.txt',
#	'../Data/Real Data/data/#prophdawgdatabase_new.txt','../Data/Bot Data/#dip_7777database_random1.txt')
main('../Data/Real Data/Merged_Data/10-15 Viewers#aristineklexisdatabase_new#dip_7777database_random1_1.txt',
	'../Data/Real Data/10-15 Viewers/#aristineklexisdatabase_new.txt','../Data/Bot Data/#dip_7777database_random1.txt')
#callmain('../Data/Real Data/35-40 Viewers/#alcanti_database_new.txt')
# main('../Data/Real Data/Merged_Data/25-30 Viewers#shyflavordatabase_new#dip_7777database_chatterscontrolled_5.txt',
#  	'../Data/Real Data/25-30 Viewers/#shyflavordatabase_new.txt','../Data/Bot Data/#dip_7777database_chatterscontrolled.txt')
# main('../Data/Real Data/Merged_Data/data#gbonbomdatabase_new#dip_7777database_random1_1.txt',
#  	'../Data/Real Data/data/#gbonbomdatabase_new.txt','../Data/Bot Data/#dip_7777database_random1.txt')
# main('../Data/Real Data/Merged_Data/data#soarcarldatabase_new#dip_7777database_chatterscontrolled_7.txt',
#  	'../Data/Real Data/data/#soarcarldatabase_new.txt','../Data/Bot Data/#dip_7777database_chatterscontrolled.txt')
# main('../Data/Real Data/Merged_Data/data#nightfall369database_new#dip_7777database_random1_1.txt',
#  	'../Data/Real Data/data/#nightfall369database_new.txt','../Data/Bot Data/#dip_7777database_random1.txt')
# main('../Data/Real Data/Merged_Data/data#mtsack_officialdatabase_new#dip_7777database_chatterscontrolled_8.txt',
#  	'../Data/Real Data/data/#mtsack_officialdatabase_new.txt','../Data/Bot Data/#dip_7777database_chatterscontrolled.txt')
# main('../Data/Real Data/Merged_Data/data#gunslayerdatabase_new#dip_7777database_random2_1.txt',
# 	'../Data/Real Data/data/#gunslayerdatabase_new.txt','../Data/Bot Data/#dip_7777database_random2.txt')
