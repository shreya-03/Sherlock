import sys,math
import seaborn as sns
from collections import Counter,OrderedDict
import operator
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile,join,isdir

fit_results = OrderedDict()

#----------Function returns dictionary of users with list of their corresponding imds----------- 

def user_intermessage_delay(users):
	user_imd = dict()
	for user in users.keys():
		if user not in user_imd.keys():
			user_imd[user] = []
			for i in range(len(users[user])-1):
				user_imd[user].append(users[user][i+1]-users[user][i])
	return user_imd

#---------Function returns dictionary of users and the values as the list of timestamp at which message has been made--------

def cluster_user_timestamps(filename):
	users = {}
	with open(filename,'r') as f:
		lines = f.readlines()
		for line in lines:
			user = str(line.split(',"u":')[1].split(',"e":')[0].replace('"',''))
			if user in users.keys():
				users[user].append(int(line.split('"t":')[1].split(',"u":')[0].replace('"','')))
			else:
				users[user] = []
				users[user].append(int(line.split('"t":')[1].split(',"u":')[0].replace('"','')))
	f.close()
	return users

#---------Function returns list of all imds and dictionary of users with their corresponding imds----------

def get_data(filename):
	users = cluster_user_timestamps(filename)
	user_imds = user_intermessage_delay(users)
	imds_list = []
	for user in user_imds.keys():
		imds_list.extend(user_imds[user])
		
	return imds_list,user_imds

#--------Function returns the list of all users in a stream-------------

def get_user_names(filename):
	user_names = set()
	with open(filename,'r') as f:
		lines = f.readlines()
		for line in lines:
			user = str(line.split(',"u":')[1].split(',"e":')[0].replace('"',''))
			user_names.add(user)
	return list(user_names)

#-----------Function stores the #real users and #bots respectively for corresponding bin number--------------

def group_users(real_users,user_imds,bins,min_imd,max_imd):
	real_users_count_bin = np.zeros([len(bins),],dtype='int')
	bot_users_count_bin = np.zeros([len(bins),],dtype='int')
	for user in user_imds.keys():
		if user in real_users:
			for imd in user_imds[user]:
				bin_no = (imd-min_imd)/1000
				real_users_count_bin[bin_no] += 1
		else:
			for imd in user_imds[user]:
				bin_no = (imd-min_imd)/1000
				bot_users_count_bin[bin_no] += 1
	print "real users bin count"
	print real_users_count_bin
	print "bot users bin count"
	print bot_users_count_bin

	# plots of figure 2 of the paper
	plot_stacked_barplot(real_users_count_bin,bot_users_count_bin,bins)

#---------Function plots barplot for real users and bots on the count of each in particular bin----------

def plot_stacked_barplot(real_users_count_bin,bot_users_count_bin,bins):
	barwidth = 1000
	bins = bins[:5000]
	p1 = plt.bar(np.array(bins),real_users_count_bin,color='b',width=barwidth)
	p2 = plt.bar(np.array(bins),bot_users_count_bin,color='r',bottom=real_users_count_bin,width=barwidth)
	plt.legend((p1[0],p2[0]),('real','bots'))
	plt.show()

def plot_ccdf(data,ax):
	sorted_vals = np.sort(np.unique(data))
	ccdf = np.zeros(len(sorted_vals))
	n = float(len(data))
	for i,val in enumerate(sorted_vals):
		ccdf[i] = np.sum(data>=val)/n
	ax.plot(sorted_vals,ccdf,"-")
'''
def getAllFilesRecursive(path):
	for f in listdir(path):
		if isfile(join(path,f)) and 'database' in str(f) and 'organicgrowth.txt' not in str(f):
			#if str(f) not in fit_results.keys():
				#fit_results[str(f)] = stream_best_curve_fit(join(path,f))
	dirs = [d for d in listdir(path) if isdir(join(path,d))]
	for d in dirs:
		files_in_d = getAllFilesRecursive(join(path,d))
		if files_in_d:
			for f in files_in_d:
				if 'database' in str(f) and 'organicgrowth' not in str(f):
					#fit_results[str(f)] = stream_best_curve_fit(join(path,f))
'''
def stream_best_curve_fit(filename):
	user_imds_list,user_imds = get_data(filename)
	no_bins = len([i for i in range(np.min(user_imds_list),np.max(user_imds_list),1000)])
	if no_bins > 0:
		f = Fitter(user_imds_list,verbose=False,bins=no_bins)
	else:
		f = Fitter(user_imds_list,verbose=False,bins=1)
	f.fit() 
	return f.get_best()

def get_real_users(filename):
	user_names = set()
	with open(filename,'r') as f:
		lines = f.readlines()
		for line in lines:
			user = str(line.split(',"u":')[1].split(',"e":')[0].replace('"',''))
			if line.split(',"b":')[1].split('}')[0] == "\"nb\"":
				user_names.add(user)
	return list(user_names)

# a stacked barplot for the sample_chatlog_chatbotted.txt file
user_imds_list,user_imds = np.array(get_data('../../Sample Data/sample_chatlog_chatbotted.txt'))
real_users = get_real_users('../../Sample Data/sample_chatlog_chatbotted.txt')
bins = [i for i in range(0,np.max(user_imds_list),1000)]
group_users(real_users,user_imds,bins,np.min(user_imds_list),np.max(user_imds_list))

# barplot for the sample_chatlog_real.txt file
user_imds_list,user_imds = np.array(get_data('../../Sample Data/sample_chatlog_real.txt'))
real_users = get_user_names('../../Sample Data/sample_chatlog_real.txt')
bins = [i for i in range(0,np.max(user_imds_list),1000)]
plt.hist(user_imds_list,bins=bins)
plt.show()