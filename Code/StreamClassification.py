import sys
import numpy as np
from collections import OrderedDict,Counter
from os import listdir
from os.path import isfile,join,isdir
import pandas as pd
import matplotlib.pyplot as plt
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

pd.set_option('display.max_colwidth',-1)

#--------Function returns list of users and list of tuples of user and the timestamp---------

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

#--------Class for obtaining value of #msgs for a stream---------

class DistributionChatters(object):

	def __init__(self,filename):
		self.users,self.user_msgs = ts_sim(filename)

	def pseudo_vel(self,bs=1):
		users = {}
		maxs = 1
		for i in self.user_msgs:
			if i[0] not in users:
				users[i[0]] = 1
			else:
				users[i[0]] += 1
				maxs = max(maxs,users[i[0]])	# maxs stores the maximum number of messages made by users in total

		bin_size = bs 	# setting the bin size to its default value
		arr = [0 for i in range(maxs//bin_size + 1)]
		xs = []
		for i in range(len(arr)+1):
			xs.append(0+(i*bin_size))	# storing the fraction of users * #msgs made by each of them
		x = [float(xs[i]+xs[i+1])/2 for i in range(len(xs)-1)]
	
		for k,v in users.items():
			arr[v//bin_size] += 1

		sums = sum(arr)
		arr = [float(i)/sums for i in arr]
		newarr = [(arr[i],i) for i in range(len(arr))]
		newarr = sorted(newarr,key=lambda tup:(tup[0],-tup[1]))
		newarr = newarr[len(arr)-3:]	# Taking the top 4 values of product of fraction of users and #msgs made by each of them 

		ans = (sum([newarr[i][0] * newarr[i][1] for i in range(len(newarr))]))
		return ans	# Returning the sum of 4 product values obtained above

#----------Class for evaluating the count of windows in which a user appeared in----------

class UserWindows(object):

	def __init__(self,filename):
		self.users,self.user_msgs = ts_sim(filename)

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
		user_dict = {i[0]:i[1] for i in sorted(user_dict.items(), key = lambda kv: kv[1],reverse=True)}		# Dictionary stores the users and #windows it appeared in descending order of #windows
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
		newarr = newarr[len(y)-3:]	# stores the top most 4 value of product of #users and #windows they showed up
		ans = sum([newarr[i][0] * newarr[i][1] for i in range(len(newarr))])	# Sum up all 4 values obtained above
		return ans	# Return the sum value

#---------Class evaluates a value for stream based on imds of users---------  

class StreamIMDFeature(object):

	def __init__(self,filename):
		self.filename = filename
		self.users_timestamps = self.cluster_user_timestamps()
		self.users_imds = self.user_intermessage_delay()
		self.imd_ft_val = self.getimdfeature()
	
	#---------Function returns dictionary of users with list of their corresponding timestamps at which message has been made---------

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

	#---------Function returns the dictionary of users with the list of their corresponding imds---------

	def user_intermessage_delay(self):
		user_imd = dict()
		for user in self.users_timestamps.keys():
			#if len(users[user]['t']) >= 6:
			if user not in user_imd.keys():
				user_imd[user] = []
				for i in range(len(self.users_timestamps[user])-1):
					user_imd[user].append(self.users_timestamps[user][i+1]-self.users_timestamps[user][i])
		return user_imd

	#---------Function returns a value for imd of stream--------

	def getimdfeature(self):
		imds_list = []
		for user in self.users_imds.keys():
			imds_list.extend(self.users_imds[user])
		imds_list = np.array(imds_list)
		bins = [i for i in range(0,np.max(imds_list),1000)]
		bins = bins[:500]
		if len(bins) > 0:
			hist,bins = np.histogram(imds_list,bins,normed=False)
			#print hist
			if len(hist) == 0:
				imd_ft_val = 0.0
			else:
				scores = [val*(i+1) for i,val in enumerate(hist)]
				sum_weights = sum(hist)
				#print "sum of pdfs:"+ str(sum_weights)
				if sum_weights > 0:
					imd_ft_val = sum(scores)/float(sum_weights)	# weighted sum of prodct of imd value and the # users corresponding to the imd 
				else:
					imd_ft_val = 0.0
		else:
			imd_ft_val = 0.0
		return imd_ft_val

#---------Function returns concatenated feature vector of stream----------

def getFeatureVecFile(filename,imd_ft_val,label):
	ft_vec = []
	distr_classob = DistributionChatters(filename)
	userwindows_classob = UserWindows(filename)
	ft_vec.append(distr_classob.pseudo_vel())
	ft_vec.append(userwindows_classob.user_windows())
	ft_vec.append(imd_ft_val)
	ft_vec.append(label)
	return ft_vec	


#--------Function recursively calls all files in the folder and returns feature vectors for each of them---------

def getAllFilesRecursive(path,certified_reals,certified_new_reals):
	features = []
	dirs = [d for d in listdir(path) if isdir(join(path,d))]
	for d in dirs:
		#print d
		if 'Viewers' in str(d):
			for files_in_d in listdir(join(path,d)):
				filename = join(join(path,d),files_in_d)
				if isfile(filename) and 'database' in str(filename):
					#print filename
					#print "real file"
					imd_ft = StreamIMDFeature(filename)
					#print imd_ft.imd_ft_val
					if imd_ft.imd_ft_val != 0.0:
						features.append(getFeatureVecFile(filename,imd_ft.imd_ft_val,0))	#label 0 for real file
		elif 'data' in str(d):
			for files_in_d in listdir(join(path,d)):
				filename = join(join(path,d),files_in_d)
				if isfile(filename) and 'database' in str(filename) and (files_in_d.split('#')[1].split('database')[0] in certified_reals or files_in_d.split('#')[1].split('database')[0] in certified_new_reals):
					#print filename
					#print "new real file"
					imd_ft = StreamIMDFeature(filename)
					#print imd_ft.imd_ft_val
					if imd_ft.imd_ft_val != 0.0:
						features.append(getFeatureVecFile(filename,imd_ft.imd_ft_val,0))	#label 0 for real file
		else:
			for files_in_d in listdir(join(path,d)):
				filename = join(join(path,d),files_in_d)
				if isfile(filename) and 'database' in str(filename):
					#print filename
					#print "bot-ed file"
					imd_ft = StreamIMDFeature(filename)
					#print imd_ft.imd_ft_val
					if imd_ft.imd_ft_val != 0.0:
						features.append(getFeatureVecFile(filename,imd_ft.imd_ft_val,1))	# label 1 for botted file
	return features

#---------Function prints accuracy for the specified classification model for the training and testing dataset---------

def run_model(model,alg_name,X_train,y_train,X_test,y_test):
	model.fit(X_train,y_train)
	y_pred = model.predict(X_test)
	accuracy = accuracy_score(y_test,y_pred)*100
	print "Classifier:" + alg_name + ' ' + "Accuracy:" + str(accuracy)

#imds_ob = StreamIMDFeature('../Data/Real Data/Merged_Data/55-60 Viewers#eloedusdatabase_new#dip_7777database_random1_28.txt')
#print imds_ob.imd_ft_val

with open('../Data/Real Data/certified real.txt','r') as f:
	certified_reals = f.readlines()
certified_reals = [real.strip('\n') for real in certified_reals]
f.close()
with open('../Data/Real Data/certified real for new data.txt') as f:
	certified_new_reals = f.readlines()
certified_new_reals = [real.strip('\n') for real in certified_new_reals]
f.close()
features = getAllFilesRecursive('../Data/Real Data/',certified_reals,certified_new_reals)
print np.array(features).shape
#print features
df = pd.DataFrame(features)
df.columns = ['distr_chatters','#user_windows','imds','label']
X = df.iloc[:,0:3]
y = df.iloc[:,3:4]
#print df.isnull().any()
#print df['imds']
# Classification models

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,random_state=100)

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
