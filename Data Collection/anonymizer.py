# replaces usernames with "anon" names

import sys
import numpy as np
from collections import OrderedDict,Counter
from os import listdir
from os.path import isfile,join,isdir

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
	return users, user_msgs, lines

def make(users1, users2, lines):
	userc = 0
	userdict = {}
	newlines = []
	for line in lines:
		user = str(line.split(',"u":')[1].split(',"e":')[0].replace('"',''))
		a = line.split(',"u":')
		b = (a[1].split(',"e":'))
		name = b[0].replace('"','')
		if b[0] in userdict:
			b[0] = "\"anon" + str(userdict[b[0]]) + "\""
		else:
			userc += 1
			userdict[b[0]] = userc
			b[0] = "\"anon" + str(userdict[b[0]]) + "\""
		a[1] = b
		lab = ""
		if name in users2:
			lab = ",\"b\":\"nb\"}"
		else:
			lab = ",\"b\":\"b\"}"
		line = a[0] + ",\"u\":" + a[1][0] + ",\"e\":" + a[1][1]
		line = line[:-2]
		line += lab
		newlines.append(line)
	with open('sample_chatlog_real.txt','w') as f:
		for line in newlines:
			f.write(line)
			f.write("\n")
    
    # prints the followers of the channel
	with open('followers_cntalexiaraye_1','r') as f:
		l = f.readlines()[1]
		l = l.split(' ')
		with open('new_followers','w') as ff:
			for i in l:
				temp = "\"" + i + "\""
				if temp in userdict:
					ff.write("anon" + str(userdict[temp]) + " ")
	

def main():
	# chatbotted filename/real filename to be anonymized
	filename1 = "./#alexiarayedatabase_new.txt"
	users1, user_msgs1, lines1 = ts_sim(filename1)
	# corresponding real filename if filename1 is chatbotted
	filename2 = "./#fortnitedenzidatabase_new.txt"
	users2, user_msgs2, lines2 = ts_sim(filename2)
	make(users1, users2, lines1)

if __name__ == '__main__':
	main()