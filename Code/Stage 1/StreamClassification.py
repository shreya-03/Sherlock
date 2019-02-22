import sys,time,math
import bisect
from math import floor
import numpy as np
from collections import OrderedDict,Counter
from os import listdir
from os.path import isfile,join,isdir
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import shutil

pd.set_option('display.max_colwidth',-1)
suspicious_files_gt_50msgs = []

real_files_lt = []
cc_files_lt = []
r1_files_lt = []
r2_files_lt = []
og_files_lt = []
cc_test_fts = []
r1_test_fts = []
r2_test_fts = []
og_test_fts = []
real_test_fts = []
train_features = []

def ts_sim(filename):
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

class DistributionChatters(object):

	def __init__(self,filename):
		self.filename = filename
		self.users,self.user_msgs = ts_sim(self.filename)

	def pseudo_vel(self,bs=1):
		users = {}
		maxs = 1
		for i in self.user_msgs:
			if i[0] not in users:
				users[i[0]] = 1
			else:
				users[i[0]] += 1
				maxs = max(maxs,users[i[0]])
		#print users
		if sum(users.values())>50:
			suspicious_files_gt_50msgs.append(self.filename)
		#time.sleep(5)
		bin_size = bs
		arr = [0 for i in range(maxs//bin_size + 1)]
		xs = []
		for i in range(len(arr)+1):
			xs.append(0+(i*bin_size))
		x = [float(xs[i]+xs[i+1])/2 for i in range(len(xs)-1)]
	
		for k,v in users.items():
			arr[v//bin_size] += 1

		sums = sum(arr)
		arr = [float(i)/sums for i in arr]
		newarr = [(arr[i],i) for i in range(len(arr))]
		newarr = sorted(newarr,key=lambda tup:(tup[0],-tup[1]))
		#print newarr
		newarr = newarr[len(arr)-3:]

		# ans = (sum([newarr[i][0] * newarr[i][1] for i in range(len(newarr))]))
		ans = ([newarr[i][0] * newarr[i][1] for i in range(len(newarr))])

		return ans

class UserWindows(object):

	def __init__(self,filename):
		self.filename = filename
		self.users,self.user_msgs = ts_sim(self.filename)

	def user_windows(self):
		user_dict = {i:0 for i in self.users}
		window_len = 15000
		start_time = self.user_msgs[0][1]
		end_time = self.user_msgs[-1][1]
		j = 0
		for i in range(start_time,end_time,window_len):
			cur_start = i
			cur_end = i + window_len
			counts = 0
			this_window = set()
			while j < len(self.user_msgs):
				if self.user_msgs[j][1] >= cur_start and self.user_msgs[j][1] <= cur_end and (self.user_msgs[j][0] not in this_window):
					this_window.add(self.user_msgs[j][0])
					user_dict[self.user_msgs[j][0]] += 1
					j += 1
				elif self.user_msgs[j][1] < cur_start or self.user_msgs[j][1] > cur_end:
					break
				else:
					j += 1
		user_dict = {i[0]:i[1] for i in sorted(user_dict.items(), key = lambda kv: kv[1],reverse=True)}
		final_dict = {}
		maxv, sums = 0, 0
		for k,v in user_dict.items():
			if (v not in final_dict) and (v != 0):
				final_dict[v] = 1
				maxv = max(maxv,v)
				sums += 1
			elif v != 0:
				final_dict[v] += 1
				sums += 1
		for i in range(0,maxv):
			if i not in final_dict:
				final_dict[i] = 0

		for k,v in final_dict.items():
			final_dict[k] = float(v)/sums
		final_dict = {i[0]:i[1] for i in sorted(final_dict.items(), key = lambda kv: kv[0])}
		x = np.array([k for k,v in final_dict.items()])
		y = np.array([v for k,v in final_dict.items()])

		newarr = [(y[i],i) for i in range(len(y))]
		newarr = sorted(newarr,key=lambda tup:(tup[0],-tup[1]))
		newarr = newarr[len(y)-3:]
		#ans = sum([newarr[i][0] * newarr[i][1] for i in range(len(newarr))])
		ans = [newarr[i][0] * newarr[i][1] for i in range(len(newarr))]
		return ans	


class StreamIMDFeature(object):

	def __init__(self,filename):
		self.filename = filename
		self.users_timestamps = self.cluster_user_timestamps()
		self.users_imds = self.user_intermessage_delay()
		self.imd_ft_val,hist,bins,imds_list = self.getimdfeature()
		self.ft_vec = self.quartile_features(hist,bins,imds_list)
	
	def cluster_user_timestamps(self):
		users = {}
		with open(self.filename,'r') as f:
			lines = f.readlines()
			for line in lines:
				user = str(line.split(',"u":')[1].split(',"e":')[0].replace('"',''))
				#print count
				if user in users.keys():
					users[user].append(int(line.split('"t":')[1].split(',"u":')[0].replace('"','')))
				else:
					users[user] = []
					users[user].append(int(line.split('"t":')[1].split(',"u":')[0].replace('"','')))
		f.close()
		return users

	def user_intermessage_delay(self):
		user_imd = dict()
		for user in self.users_timestamps.keys():
			#if len(users[user]['t']) >= 6:
			if user not in user_imd.keys():
				user_imd[user] = []
				for i in range(len(self.users_timestamps[user])-1):
					user_imd[user].append(self.users_timestamps[user][i+1]-self.users_timestamps[user][i])
		return user_imd

	def quartile_features(self,hist,bins,imds_list):
		ft_vec = []

		pre = [hist[0]]
		for i in range(1,len(hist)):
			pre.append(hist[i]+pre[-1])
		scores = [val for i,val in enumerate(hist)]
		temp = bisect.bisect_left(pre, 0.6*float(sum(scores)), lo=0, hi=len(pre))
		ft_vec.append(bins[temp]/1000)
		temp = bisect.bisect_left(pre, 0.7*float(sum(scores)), lo=0, hi=len(pre))
		ft_vec.append(bins[temp]/1000)
		temp = bisect.bisect_left(pre, 0.8*float(sum(scores)), lo=0, hi=len(pre))
		ft_vec.append(bins[temp]/1000)
		temp = bisect.bisect_left(pre, 0.9*float(sum(scores)), lo=0, hi=len(pre))
		ft_vec.append(bins[temp]/1000)

		# print ft_vec

		# plt.hist(imds_list,bins=bins)
		# plt.show()

		return ft_vec

	def getimdfeature(self):
		imds_list = []
		for user in self.users_imds.keys():
			imds_list.extend(self.users_imds[user])
		imds_list = np.array(imds_list)
		bins = [i for i in range(0,np.max(imds_list),1000)]
		bins = bins[:500]
		if len(bins) < 500:
			bins.extend([i for i in range(len(bins)*1000,500000,1000)])

		hist,bins = np.histogram(imds_list,bins,normed=False)
		scores = [val*(i+1) for i,val in enumerate(hist)]
		sum_weights = sum(hist)
		#print "sum of pdfs:"+ str(sum_weights)
		if sum_weights > 0:
			imd_ft_val = sum(scores)/float(sum_weights)
		else:
			imd_ft_val = 0.0
		
		return imd_ft_val,hist,bins,imds_list

def gettotalusers(filename):
	users = set()
	with open(filename,'r') as f:
		lines = f.readlines()
		for line in lines:
			user = str(line.split(',"u":')[1].split(',"e":')[0].replace('"',''))
			users.add(user)
	return len(users)

def get_channel_followers(path,channel_name):
	#print channel_name	
	follows = set()
	for file in listdir(path):
		if file[13:-2] == channel_name:
			#print file
			#print "1st case entered"
			with open(join(path,file),'r') as f:
				lines = f.readlines()
				if len(lines) > 1:
					users = lines[1].split(' ')
					#print users
					for user in users:
						follows.add(user)
			f.close()
		'''
		else:
			if file[13:] == channel_name:
				#print "2nd case entered"
				with open(join(path,file),'r') as f:
					lines = f.readlines()
					users = lines[1].split(' ')
					for user in users:
						follows.add(user)
				f.close()
		'''
	return list(follows)

def getFeatureVecFile(filename,imd_ft_vec,followers,label):
	ft_vec = []
	distr_classob = DistributionChatters(filename)
	userwindows_classob = UserWindows(filename)
	#ft_vec.append(filename)
	ft_vec.extend(distr_classob.pseudo_vel())
	#ft_vec.extend(distr_classob.pseudo_vel())
	ft_vec.extend(userwindows_classob.user_windows())
	ft_vec.extend(imd_ft_vec)
	#ft_vec.append(float(len(followers))/gettotalusers(filename))
	ft_vec.append(label)
	return ft_vec	

def getAllFilesRecursiveinlist(path,certified_reals,certified_new_reals):
	dirs = [d for d in listdir(path) if isdir(join(path,d))]
	for d in dirs:
		for files_in_d in listdir(join(path,d)):
			filename = join(join(path,d),files_in_d)
			if 'dip_7777' in filename:
				if 'chatterscontrolled' in filename:
					cc_files_lt.append(filename)
				elif 'random1' in filename:
					r1_files_lt.append(filename)
				elif 'random2' in filename:
					r2_files_lt.append(filename)
				else:
					og_files_lt.append(filename)
			else:
				if 'database' in files_in_d and (files_in_d.split('#')[1].split('database')[0] in certified_reals or files_in_d.split('#')[1].split('database')[0] in certified_new_reals):
					real_files_lt.append(filename)
	print len(cc_files_lt),len(r1_files_lt),len(r2_files_lt),len(og_files_lt),len(real_files_lt)

def generateTrainTestData(ratio):
	length =  (ratio+0.1)*len(real_files_lt)
	for i in range(len(real_files_lt)):
		if i < length:
			imd_ft = StreamIMDFeature(real_files_lt[i])
			channel_followers = get_channel_followers('../followers_cnt/',real_files_lt[i].split('#')[1].split('database')[0])
			train_features.append(getFeatureVecFile(real_files_lt[i],imd_ft.ft_vec,channel_followers,0))
		else:
			imd_ft = StreamIMDFeature(real_files_lt[i])
			channel_followers = get_channel_followers('../followers_cnt/',real_files_lt[i].split('#')[1].split('database')[0])	
			real_test_fts.append(getFeatureVecFile(real_files_lt[i],imd_ft.ft_vec,channel_followers,0))
	length = ratio*len(cc_files_lt)
	for i in range(len(cc_files_lt)):
		if i < length:
			imd_ft = StreamIMDFeature(cc_files_lt[i])
			channel_followers = get_channel_followers('../followers_cnt/',cc_files_lt[i].split('#')[1].split('database')[0])	
			train_features.append(getFeatureVecFile(cc_files_lt[i],imd_ft.ft_vec,channel_followers,1))
		else:
			imd_ft = StreamIMDFeature(cc_files_lt[i])
			channel_followers = get_channel_followers('../followers_cnt/',cc_files_lt[i].split('#')[1].split('database')[0])	
			cc_test_fts.append(getFeatureVecFile(cc_files_lt[i],imd_ft.ft_vec,channel_followers,1))
	length = ratio*len(r1_files_lt)
	for i in range(len(r1_files_lt)):
		if i < length:
			imd_ft = StreamIMDFeature(r1_files_lt[i])
			channel_followers = get_channel_followers('../followers_cnt/',r1_files_lt[i].split('#')[1].split('database')[0])	
			train_features.append(getFeatureVecFile(r1_files_lt[i],imd_ft.ft_vec,channel_followers,1))
		else:
			imd_ft = StreamIMDFeature(r1_files_lt[i])
			channel_followers = get_channel_followers('../followers_cnt/',r1_files_lt[i].split('#')[1].split('database')[0])
			r1_test_fts.append(getFeatureVecFile(r1_files_lt[i],imd_ft.ft_vec,channel_followers,1))
	length = ratio*len(r2_files_lt)
	for i in range(len(r2_files_lt)):
		if i < length:
			imd_ft = StreamIMDFeature(r2_files_lt[i])
			channel_followers = get_channel_followers('../followers_cnt/',r2_files_lt[i].split('#')[1].split('database')[0])
			train_features.append(getFeatureVecFile(r2_files_lt[i],imd_ft.ft_vec,channel_followers,1))
		else:
			imd_ft = StreamIMDFeature(r2_files_lt[i])
			channel_followers = get_channel_followers('../followers_cnt/',r2_files_lt[i].split('#')[1].split('database')[0])
			r2_test_fts.append(getFeatureVecFile(r2_files_lt[i],imd_ft.ft_vec,channel_followers,1))
	length = ratio*len(og_files_lt)
	for i in range(len(og_files_lt)):
		if i < length:
			imd_ft = StreamIMDFeature(og_files_lt[i])
			channel_followers = get_channel_followers('../followers_cnt/',og_files_lt[i].split('#')[1].split('database')[0])
			train_features.append(getFeatureVecFile(og_files_lt[i],imd_ft.ft_vec,channel_followers,1))
		else:
			imd_ft = StreamIMDFeature(og_files_lt[i])
			channel_followers = get_channel_followers('../followers_cnt/',og_files_lt[i].split('#')[1].split('database')[0])
			og_test_fts.append(getFeatureVecFile(og_files_lt[i],imd_ft.ft_vec,channel_followers,1))

def get_users(filename):
	users = set()
	with open(filename,'r') as f:
		lines = f.readlines()
		for line in lines:
			user = str(line.split(',"u":')[1].split(',"e":')[0].replace('"',''))
			users.add(user)
	f.close()
	return users


def run_model(model,alg_name,X_train,y_train,X_test,y_test):
	model.fit(X_train,y_train)
	y_pred = model.predict(X_test)
	#print y_pred
	print Counter(y_pred)
	accuracy = accuracy_score(y_test,y_pred)*100
	print "Classifier:" + alg_name
	print "Accuracy:" + str(accuracy)
	print "weighted"
	print "Precision Recall fscore:" + str(precision_recall_fscore_support(y_test,y_pred,average='weighted'))
	print "confusion matrix"
	print confusion_matrix(y_test,y_pred)
	return y_pred

'''
with open('../Real Data/certified real.txt','r') as f:
	certified_reals = f.readlines()
certified_reals = [real.strip('\n') for real in certified_reals]
f.close()
with open('../Real Data/certified real for new data.txt') as f:
	certified_new_reals = f.readlines()
certified_new_reals = [real.strip('\n') for real in certified_new_reals]
f.close()
getAllFilesRecursiveinlist('../Real Data/',certified_reals,certified_new_reals)
generateTrainTestData(0.6)
print "training data shape:" + str(np.array(train_features).shape)
#print test_features
print "testing data shape:" + str(np.array(cc_test_fts).shape)
#print features
df = pd.DataFrame(train_features)
#df.to_csv('s1_training_features.csv')
'''
df = pd.read_csv('s1_training_features.csv')
#df_test = pd.DataFrame(real_test_fts)
#df.columns = ['distr_chatters','#user_windows','imds','label']
X_train = df.iloc[:,0:10]
y_train = df.iloc[:,10:11]

# ------Enter test files-----

test_features = []

imd_ft = StreamIMDFeature('../Sample Data/sample_chatlog_chatbotted.txt')
channel_followers = get_channel_followers('../Sample Data/sample_chatlog_chatbotted_followers')
test_features.append(getFeatureVecFile('../Sample Data/sample_chatlog_chatbotted.txt',imd_ft.ft_vec,channel_followers,1))

imd_ft = StreamIMDFeature('../Sample Data/sample_chatlog_real.txt')
channel_followers = get_channel_followers('../Sample Data/sample_chatlog_real_followers')
test_features.append(getFeatureVecFile('../Sample Data/sample_chatlog_real.txt',imd_ft.ft_vec,channel_followers,0))


X_test = pd.DataFrame(test_features).iloc[:,0:10]
y_test = pd.DataFrame(test_features).iloc[:,10:11]
#print X_test.values
#y_test = [1 for i in range(len(og_test_fts))]
#X_train = df.iloc[:,0:10]
#y_train = df.iloc[:,10:11]
#X_test = df_test.iloc[:,0:10]
#y_test = df_test.iloc[:,10:11]
#print df.isnull().any()
#print df['imds']
# Classification models
'''
train_files = []
test_files = []
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,random_state=100)
for file in X_train.iloc[:,0:1].values.tolist():
	train_files.extend(file) 
for file in X_test.iloc[:,0:1].values.tolist():
	test_files.extend(file)
X_train = X_train.iloc[:,1:11]
X_test = X_test.iloc[:,1:11]
print train_files,test_files
'''

#-------Decision Tree-----------

model = tree.DecisionTreeClassifier(criterion='entropy',max_depth=5)
#run_model(model,"Decision Tree",X_train,y_train,X_test,y_test)
run_model(model,"Decision Tree",X_train,y_train,X_test,y_test)

#-------Random Forest-----------

model = RandomForestClassifier(n_estimators=10)
#run_model(model,"Random Forest",X_train,y_train,X_test,y_test)
run_model(model,"Random Forest",X_train,y_train,X_test,y_test)

#-------xgboost-----------------

model = XGBClassifier()
#run_model(model,"XGBoost",X_train,y_train,X_test,y_test)
y_pred = run_model(model,"XGBoost",X_train,y_train,X_test,y_test)
print y_pred

#-------SVM Classifier-----------

model = SVC()
#run_model(model,"SVM Classifier",X_train,y_train,X_test,y_test)
run_model(model,"SVM",X_train,y_train,X_test,y_test)

#-------Nearest Classifier-------

model = neighbors.KNeighborsClassifier()
#run_model(model,"Nearest Neighbors Classifier",X_train,y_train,X_test,y_test)
run_model(model,"Nearest Neighbor Classifier",X_train,y_train,X_test,y_test)
'''
#-------SGD Classifier-----------

model = OneVsRestClassifier(SGDClassifier())
#run_model(model,"SGD Classifier",X_train,y_train,X_test,y_test)
run_model(model,"SGD",X_train,y_train,X_test,y_test,test_files)
#-------Gaussian NB--------------

model = GaussianNB()
#run_model(model,"Gaussian NB",X_train,y_train,X_test,y_test)
run_model(model,"Gaussian NB",X_train,y_train,X_test,y_test,test_files)
'''
#-------NN-MLP-------------------

model = MLPClassifier()
#run_model(model,"NN-MLP",X_train,y_train,X_test,y_test)
run_model(model,"NN-MLP",X_train,y_train,X_test,y_test)

'''
index = int(math.ceil(0.7*len(real_files_lt)))
for i in range(index,len(real_files_lt)):
	filename = real_files_lt[i]
	print filename,y_pred[i-index]
	get_stream_statistics(filename)
'''
