from UserBins import map_user_msg_bins,map_user_timestamp_bins
from itertools import groupby
from collections import Counter,OrderedDict
import math,sys

#---------Function returns dictionary with keys as users and values as list of imds and length of each message by a user----------

def user_imds_msglengths(users):
	user_imd_ml = OrderedDict()
	for user in users.keys():
		if user not in user_imd_ml.keys():
			user_imd_ml[user] = {}
			user_imd_ml[user]['t'] = []
			user_imd_ml[user]['m'] = []
			if len(users[user]['t']) == 1:
				user_imd_ml[user]['t'] = users[user]['t']
				user_imd_ml[user]['m'] = users[user]['m']
			for i in range(len(users[user]['t'])-1):
				user_imd_ml[user]['t'].append(users[user]['t'][i+1]-users[user]['t'][i])
				user_imd_ml[user]['m'] = users[user]['m']
	return user_imd_ml

#---------Function returns dictionary with keys as users and values as list of timestamps and length of a mesaage for each user------------

def get_user_data(filename):
	users = OrderedDict()
	with open(filename,'r') as f:
		lines = f.readlines()
		for line in lines:
			user = str(line.split(',"u":')[1].split(',"e":')[0].replace('"','')) 
			if user not in users.keys():
				users[user] = {}
				users[user]['t'] = []
				users[user]['m'] = []
				users[user]['t'].append(int(line.split('"t":')[1].split(',"u":')[0].replace('"','')))
				users[user]['m'].append(len(line.split(',"m":')[1].split(',"nm":')[0].split(' ')))
			else:
				users[user]['t'].append(int(line.split('"t":')[1].split(',"u":')[0].replace('"','')))
				users[user]['m'].append(len(line.split(',"m":')[1].split(',"nm":')[0].split(' ')))
	f.close()
	return users

#--------Function returns first level entropy for each user-----------

def calculate_entropy(bins_list):
	counter = Counter(bins_list)
	sum = 0
	for value in counter.values():
		sum += value
	entropy = 0.0
	for value in counter.values():
		entropy += -1 * (float(value)/sum)* math.log((float(value)/sum))
	return entropy

#--------Function returns entropies of message lengths and imds of each user----------- 

def get_entropy_features(filename):
	users = get_user_data(filename)
	user_imd_ml = user_imds_msglengths(users)
	#print user_imd_ml
	#user_entropy = OrderedDict()
	user_entropy = []
	for user in user_imd_ml.keys():
		if len(user_imd_ml[user]['t']) >= 5:
			user_bins = map_user_timestamp_bins(user_imd_ml[user]['t'])
			user_msg_bins = map_user_msg_bins(user_imd_ml[user]['m'])
		else:
			user_bins = []
			user_msg_bins = []
			for i in range(len(user_imd_ml[user]['t'])):
				user_bins.append(i);
				user_msg_bins.append(i)
		#if user not in user_entropy.keys():
		entropies = []
 		entropies.append(calculate_entropy(user_bins))
		entropies.append(calculate_entropy(user_msg_bins))
		user_entropy.append(entropies)
	return user_entropy

