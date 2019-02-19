import sys
import numpy as np
import pandas as pd
from math import floor,ceil
from collections import OrderedDict,Counter
from matplotlib import pyplot as plt
from statistics import mean
from os import listdir
from os.path import isfile,join,isdir
from XMeans import *

plt.rcParams['figure.figsize'] = (16,9)
#plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'

def getUserIMDMessages(filename):
	users = OrderedDict()
	with open(filename,'r') as f:
		#count = 1
		lines = f.readlines()
		for line in lines:
			#print count
			user = str(line.split(',"u":')[1].split(',"e":')[0].replace('"',''))
			if user in users.keys():
				users[user]['t'].append(int(line.split('"t":')[1].split(',"u":')[0].replace('"','')))
				users[user]['m'] += 1
			else:
				users[user] = {}
				users[user]['t'] = []
				users[user]['m'] = 1
				users[user]['t'].append(int(line.split('"t":')[1].split(',"u":')[0].replace('"','')))
			#count += 1
	f.close()
	return users

def get_chats_features(users):
	user_chats_ft = []
	for user in users.keys():
		if users[user]['m'] > 1:
			user_chats_ft.append(users[user]['m'])
	return user_chats_ft

def ts_sim(filename):
	# path = os.getcwd()+filename
	# filename = path
	# print(filename)
	user_msgs = []
	users = OrderedDict()
	with open(filename,'r') as f:
		lines = f.readlines()
		for line in lines:
			user = str(line.split(',"u":')[1].split(',"e":')[0].replace('"',''))
			ts = str(line.split(',"u":')[0].split(':')[1].replace('"',''))
			user_msgs.append((user,int(ts)))
			if user not in users.keys():
				users[user] = len(users)
	f.close()
	return users, user_msgs

def user_windows(filename):
	users, user_msgs = ts_sim(filename)
	user_dict = {i:0 for i in users}
	window_len = 15000

	start_time = user_msgs[0][1]
	end_time = user_msgs[-1][1]

	j = 0
	for i in range(start_time,end_time,window_len):
		cur_start = i
		cur_end = i + window_len
		counts = 0
		this_window = set()
		while j < len(user_msgs):
			if user_msgs[j][1] >= cur_start and user_msgs[j][1] <= cur_end and (user_msgs[j][0] not in this_window):
				this_window.add(user_msgs[j][0])
				user_dict[user_msgs[j][0]] += 1
				j += 1
			elif user_msgs[j][1] < cur_start or user_msgs[j][1] > cur_end:
				break
			else:
				j += 1
	user_dict = {i[0]:i[1] for i in sorted(user_dict.items(), key = lambda kv: kv[1],reverse=True)}
	return user_dict

def user_intermessage_delay(users):
	user_imd = OrderedDict()
	for user in users.keys():
		if len(users[user]['t']) > 1:
			user_imd[user] = []
			for i in range(len(users[user]['t'])-1):
				user_imd[user].append(users[user]['t'][i+1]-users[user]['t'][i])
	return user_imd

def get_IMD_features(users):
	user_imd = user_intermessage_delay(users)
	#print user_imd.keys()
	imds_list = []
	for user in user_imd.keys():
		imds_list.extend(user_imd[user])
	imds_list = np.array(imds_list)
	bins = [i for i in range(0,np.max(imds_list),1000)]
	user_bins = []
	for user in user_imd.keys():
		bin_nos = np.digitize(user_imd[user],bins)
		user_bins.append(floor(mean(bin_nos)))
	return user_bins

def get_user_names(filename):
	user_names = set()
	with open(filename,'r') as f:
		lines = f.readlines()
		for line in lines:
			user = str(line.split(',"u":')[1].split(',"e":')[0].replace('"',''))
			user_names.add(user)
	return list(user_names)

def eucleadian_dist(a,b,ax=1):
	return np.linalg.norm(a-b,axis=ax)

def plot_graph(X,real_X,real_Y,bot_X,bot_Y,labels):
	#fig,(ax1,ax2) = plt.subplots(1,2,sharey=True)
	#fig.suptitle(filename)
	print labels
	plt.scatter(real_X,real_Y,c='blue',s=25,label='real')
	plt.scatter(bot_X,bot_Y,c='red',s=25,label='bot')
	#plt.tick_params(labelsize=16)
	#hfont = {'fontname':'Helvetica'}
	plt.rc('font', family='sans-serif')
	plt.rc('xtick', labelsize='x-large')
	plt.rc('ytick', labelsize='x-large')
	plt.xlabel('Number  of  messages  per  user',fontsize='x-large')
	plt.ylabel('Mean  IMD  per  user',fontsize='x-large')
	plt.legend(loc='upper right',prop={'size':12})
	#plt.title('User original labels in 3rd quadrant',fontsize='large',fontweight='bold')
	plt.show()
	plt.ylim(200,260)
	plt.scatter(X[:,0], X[:,1], c=labels, cmap=plt.cm.Dark2,s=25)
	plt.rc('font', family='sans-serif')
	plt.rc('xtick', labelsize='x-large')
	plt.rc('ytick', labelsize='x-large')
	#plt.tick_params(labelsize=16)
	#hfont = {'fontname':'Helvetica'}
	plt.xlabel('Number  of  messages  per  user',fontsize='x-large')
	plt.ylabel('Mean  IMD  per  user',fontsize='x-large')
	#plt.title('Labels after clustering',fontsize='large',fontweight='bold')
	#plt.axvline(x=avg_f1,c='yellow')
	#plt.axhline(y=avg_f2,c='green')
	plt.show()	

def labeling_data(filename,realfile,botfile):
	
	real_users = get_user_names(realfile)
	bot_users = get_user_names(botfile)
	users = getUserIMDMessages(filename)
	users_list = users.keys()
	user_chats_ft= get_chats_features(users)
	user_chats_ft = pd.DataFrame(user_chats_ft)
	
	# due to poor code
	# uw_dict = user_windows(filename)
	# uw_ft = []
	# for user in users.keys():
	# 	if users[user]['m'] > 1:
	# 		uw_ft.append(uw_dict[user])
	# user_chats_ft = pd.DataFrame(uw_ft)

	user_imd_bins = pd.DataFrame(get_IMD_features(users))
	user_features = pd.concat([user_chats_ft,user_imd_bins],axis=1)
	real_X = []
	real_Y = []
	bot_X = []
	bot_Y = []
	list_user_fts = user_features.values.tolist()
	for i in range(len(list_user_fts)):
		if users_list[i] in real_users:
			real_X.append(list_user_fts[i][0])
			real_Y.append(list_user_fts[i][1])
		else:
			bot_X.append(list_user_fts[i][0])
			bot_Y.append(list_user_fts[i][1])
	initavg_f1 = floor(mean([ft[0] for ft in list_user_fts]))
	initavg_f2 = floor(mean([ft[1] for ft in list_user_fts]))

	five0 = len([1 for ft in list_user_fts if ft[0] > initavg_f1])
	list_user_fts1 = sorted(list_user_fts,key=lambda x:x[0])
	list_user_fts1 = list_user_fts1[:len(list_user_fts1)-five0]
	list_user_fts1 = sorted(list_user_fts1,key=lambda x:x[1])
	five1 = len([1 for ft in list_user_fts1 if ft[1] > initavg_f2])
	list_user_fts1 = list_user_fts1[:len(list_user_fts1)-five1]
	# if list size becomes zero, uncomment
	# avg_f1, avg_f2 = 1,1
	# if len(list_user_fts1) > 0:
	
 	avg_f1 = floor(mean([ft[0] for ft in list_user_fts1]))
	avg_f2 = floor(mean([ft[1] for ft in list_user_fts1]))
	#print "shifted f1 avg:" + str(avg_f1)
	#print "shifted f2 avg:" + str(avg_f2)
	selected_bot_user_fts = []
	selected_bot_user_index_map = []
	for i in range(len(list_user_fts)):
		if list_user_fts[i][0] >= avg_f1 and list_user_fts[i][1] >= avg_f2:
			selected_bot_user_fts.append(list_user_fts[i])
			selected_bot_user_index_map.append((len(selected_bot_user_fts)-1,users.keys().index(users_list[i])))
	selected_bot_user_index_map = dict(selected_bot_user_index_map)
	new_bot_user_features = pd.DataFrame(selected_bot_user_fts)

	#------Plot initial real and bot distribution on #msgs and imds feature space------

	plt.rc('font', family='sans-serif')
	plt.rc('xtick', labelsize='x-large')
	plt.rc('ytick', labelsize='x-large')
	plt.xlim(0,80)
	plt.ylim(0,1500)
	plt.scatter(bot_X,bot_Y,c='red',s=20,label='bot',marker='o')
	plt.scatter(real_X,real_Y,c='blue',s=20,label='real',marker='o')
	plt.xlabel('Number  of  messages  per  user',fontsize='x-large')
	plt.ylabel('Mean  IMD  per  user',fontsize='x-large')
	plt.legend(loc='upper right',prop={'size':12})
	plt.axvline(x=initavg_f1,c='#FF6833',lw=2)
	plt.axhline(y=initavg_f2,c='green',lw=2)
	plt.axvline(x=avg_f1,c='#FF6833',linestyle='--',lw=2,dashes=(5, 5))
	plt.axhline(y=avg_f2,c='green',linestyle='--',lw=2,dashes=(5, 5))
	#plt.tick_params(labelsize=16)
	#hfont = {'fontname':'Helvetica'}
	#plt.title('Original label distribution',fontsize='large',fontweight='bold')
	plt.show()

	
	f1 = new_bot_user_features.iloc[:,0].values
	f2 = new_bot_user_features.iloc[:,1].values
	gt_X = np.array(list(zip(f1,f2)))
	kmax = 0
	if len(f1) > 80:
		kmax = 8
	else:
		kmax = 4
	print(kmax)
	Xmeans = XMeans(kmax=kmax)
	Xmeans.fit(list(gt_X))
	labels = Xmeans.labels_

	real_X = []
	real_Y = []
	bot_X = []
	bot_Y = []
	#print len(selected_bot_user_fts)
	#print len(users_list)
	for i in range(len(selected_bot_user_fts)):
		#print i,selected_bot_user_index_map[i]
		if users_list[i] in real_users:
			real_X.append(selected_bot_user_fts[i][0])
			real_Y.append(selected_bot_user_fts[i][1])
		else:
			bot_X.append(selected_bot_user_fts[i][0])
			bot_Y.append(selected_bot_user_fts[i][1])
	plot_graph(gt_X,real_X,real_Y,bot_X,bot_Y,labels)

	
	## Getting index of botted users
	bot_user_cluster_dict = {}
	for i in range(Xmeans.n_clusters):
		bot_user_cluster_dict[i] = []
	gt_labels = Xmeans.labels_
	for i in range(len(gt_labels)):
		bot_user_cluster_dict[gt_labels[i]].append(i)
	max_user_cluster_cnt = 0
	for i in bot_user_cluster_dict.keys():
		cnt_user = len(bot_user_cluster_dict[i])
		if max_user_cluster_cnt < cnt_user:
			max_user_cluster_cnt = cnt_user
			gt_max_cluster_index = i 
	bot_users_index = []
	for index in bot_user_cluster_dict[gt_max_cluster_index]:
		bot_users_index.append(selected_bot_user_index_map[index])

	## Getting index of real users
	selected_real_user_fts = []
	selected_real_user_index_map = []
	for i in range(len(list_user_fts)):
		if list_user_fts[i][0] < avg_f1 and list_user_fts[i][1] < avg_f2:
			selected_real_user_fts.append(list_user_fts[i])
			selected_real_user_index_map.append((len(selected_real_user_fts)-1,users.keys().index(users_list[i])))
	selected_real_user_index_map = dict(selected_real_user_index_map)
	real_users_index = []
	for index in selected_real_user_index_map.keys():
		real_users_index.append(selected_real_user_index_map[index])

	return real_users_index,bot_users_index,real_users,bot_users
