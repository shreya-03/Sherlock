from __future__ import division
import string
import math
import re,sys,os
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()

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

def pseudo_vel(users, user_msgs, bs=1):
	users = {}
	maxs = 1
	for i in user_msgs:
		if i[0] not in users:
			users[i[0]] = 1
		else:
			users[i[0]] += 1
			maxs = max(maxs,users[i[0]])

	bin_size = bs
	arr = [0 for i in range(maxs//bin_size + 1)]
	xs = []
	for i in range(len(arr)+1):
		xs.append(0+(i*bin_size))
	x = [(xs[i]+xs[i+1])/2 for i in range(len(xs)-1)]
	
	for k,v in users.items():
		arr[v//bin_size] += 1

	sums = sum(arr)
	arr = [i/sums for i in arr]
	newarr = [(arr[i],i) for i in range(len(arr))]
	newarr = sorted(newarr,key=lambda tup:(tup[0],-tup[1]))
	newarr = newarr[len(arr)-3:]

	ans = (sum([newarr[i][0] * newarr[i][1] for i in range(len(newarr))]))

	# fig = plt.figure()
	# fig.suptitle('Fraction of users vs. number of msgs')
	# ax = fig.add_subplot(111)
	# ax.set_xlabel('Number of msgs')
	# ax.set_ylabel('Fraction of users')
	# plt.style.use('seaborn')
	# plt.plot(x,arr)
	# plt.show()
	return ans

def user_windows(users,user_msgs):
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
		final_dict[k] = v/sums
	final_dict = {i[0]:i[1] for i in sorted(final_dict.items(), key = lambda kv: kv[0])}
	x = np.array([k for k,v in final_dict.items()])
	y = np.array([v for k,v in final_dict.items()])

	newarr = [(y[i],i) for i in range(len(y))]
	newarr = sorted(newarr,key=lambda tup:(tup[0],-tup[1]))
	newarr = newarr[len(y)-3:]
	ans = sum([newarr[i][0] * newarr[i][1] for i in range(len(newarr))])

	# fig = plt.figure()
	# fig.suptitle('Fraction of users vs. number of windows')
	# ax = fig.add_subplot(111)
	# ax.set_xlabel('Number of windows')
	# ax.set_ylabel('Fraction of users')
	# plt.style.use('seaborn')
	# plt.plot(x,y)
	# plt.show()
	return ans

def main():
	di = "./Real Data"
	# di = "./Merge Bots Real"
	ans = []
	for root, dirs, files in os.walk(di):
		for file in files:
			if (di == "./Real Data" and file[-9] != 'e') or (di == "./Merge Bots Real" and file[-5] == 'h'
				or di == "../Downloads/twitch-master/data2" and file[-9] != 'e'):
				continue
			path = root
			filename = path+'/'+file
			users, user_msgs = ts_sim(filename)
			print(filename)
			# ans.append(pseudo_vel(users, user_msgs))
			ans.append(user_windows(users, user_msgs))
	print(ans)
	# anss = [1 for i in range(len(ans)) if ans[i] < 2.0]
	anss = [1 for i in range(len(ans)) if ans[i] > 2.0]
	print((1 - (len(anss)/len(ans)))*100)

if __name__ == '__main__':
	main()