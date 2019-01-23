from math import floor

#--------Function returns list of bin numbers for message lengths keeping the number of points in each bin same---------

def map_user_msg_bins(msg_lengths):
	user_msg_bins = []
	sorted_msg_lengths = sorted(msg_lengths)
	bin_length = int(floor(len(msg_lengths))/5)
	for msg_length in msg_lengths:
		if msg_length >= sorted_msg_lengths[0] and msg_length <= sorted_msg_lengths[bin_length-1]:
			user_msg_bins.append(1)
		elif msg_length >= sorted_msg_lengths[bin_length] and msg_length <= sorted_msg_lengths[2*bin_length-1]:
			user_msg_bins.append(2)
		elif msg_length >= sorted_msg_lengths[2*bin_length] and msg_length <= sorted_msg_lengths[3*bin_length-1]:
			user_msg_bins.append(3)
		elif msg_length >= sorted_msg_lengths[3*bin_length] and msg_length <= sorted_msg_lengths[4*bin_length-1]:
			user_msg_bins.append(4)
		else:
			user_msg_bins.append(5)
	return user_msg_bins

#--------Function returns list of bin numbers for imds keeping the number of points in each bin same----------

def map_user_timestamp_bins(imds):
	user_bins = []
	sorted_imds = sorted(imds)
	bin_length = int(floor(len(imds)/5))
	for imd in imds:
		if imd >= sorted_imds[0] and imd <= sorted_imds[bin_length-1]:
			user_bins.append(1)
		elif imd >= sorted_imds[bin_length] and imd <= sorted_imds[2*bin_length-1]:
			user_bins.append(2)
		elif imd >= sorted_imds[2*bin_length] and imd <= sorted_imds[3*bin_length-1]:
			user_bins.append(3)
		elif imd >= sorted_imds[3*bin_length] and imd <= sorted_imds[4*bin_length-1]:
			user_bins.append(4)
		else:
			user_bins.append(5)
	return user_bins

#---------Function returns dictionary with users and values as lists of messages and timestamps and labels whether the user is bot or real user--------

def cluster_user_timestamps_msgs(filename_annotations,filename):
	users = {}
	count = 1
	with open(filename_annotations,'r') as f:
		lines = f.readlines()
		for line in lines:
			user = str(line.split(',"u":')[1].split(',"e":')[0].replace('"','')) 
			#print count
			if user in users.keys():
				users[user]['t'].append(int(line.split('"t":')[1].split(',"u":')[0].replace('"','')))
				users[user]['m'].append(len(line.split(',"m":')[1].split(',"nm":')[0].split(' ')))
			else:
				users[user] = {}
				users[user]['t'] = []
				users[user]['m'] = []
				users[user]['t'].append(int(line.split('"t":')[1].split(',"u":')[0].replace('"','')))
				users[user]['m'].append(len(line.split(',"m":')[1].split(',"nm":')[0].split(' ')))
				users[user]['b'] = str(line.split(',"b":')[1].split('}')[0].replace('"',''))
			count += 1
	f.close()
	with open(filename,'r') as f:
		lines = f.readlines()
		lines = lines[count+1:]
		for line in lines:
			user = str(line.split(',"u":')[1].split(',"e":')[0].replace('"',''))
			#print count
			if user in users.keys():
				users[user]['t'].append(int(line.split('"t":')[1].split(',"u":')[0].replace('"','')))
				users[user]['m'].append(len(line.split(',"m":')[1].split(',"nm":')[0].split(' ')))
			else:
				users[user] = {}
				users[user]['t'] = []
				users[user]['m'] = []
				users[user]['t'].append(int(line.split('"t":')[1].split(',"u":')[0].replace('"','')))
				users[user]['m'].append(len(line.split(',"m":')[1].split(',"nm":')[0].split(' ')))
				users[user]['b'] = 'no'
			count += 1
	return users

#--------Function returning dictionary of user and the corresponding imds---------

def user_intermessage_delay(users):
	user_imd = dict()
	for user in users.keys():
		#print user
		if users[user]['b'] == 'yes':
			if len(users[user]['t']) >= 6:
				if user not in user_imd.keys():
					user_imd[user] = {}
					user_imd[user]['t'] = []
					user_imd[user]['m'] = []
					#user_imd[user].append(0)
					for i in range(len(users[user]['t'])-1):
						user_imd[user]['t'].append(users[user]['t'][i+1]-users[user]['t'][i])
					user_imd[user]['m'] = users[user]['m']
					#user_imd[user]['t'] = user_imd[user]['t']*3
					user_imd[user]['b'] = 'yes'
		if users[user]['b'] == 'no':
			if len(users[user]['t']) >= 20:
				if user not in user_imd.keys():
					user_imd[user] = {}
					user_imd[user]['t'] = []
					user_imd[user]['m'] = []
					#user_imd[user].append(0)
					for i in range(len(users[user]['t'])-1):
						user_imd[user]['t'].append(users[user]['t'][i+1]-users[user]['t'][i])
					user_imd[user]['m'] = users[user]['m']
					user_imd[user]['b'] = 'no'
	return user_imd

