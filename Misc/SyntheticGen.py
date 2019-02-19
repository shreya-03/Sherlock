import sys
from os import listdir
from os.path import isfile,join,isdir

def merge_files(path,realfile,botpath,botfile):
	merge_file = path + realfile[:-4] + botfile
	with open(join(path,realfile),'r') as f1:
		with open(join(botpath,botfile),'r') as f2:
			with open(merge_file,'w') as fin:
				lines1 = f1.readlines()
				lines2 = f2.readlines()
				base_timestamp1 = int(lines1[0].split('"t":')[1].split(',"u":')[0].replace('"',''))
				base_timestamp2 = int(lines2[0].split('"t":')[1].split(',"u":')[0].replace('"',''))
				line = '{"t":"0"' + ',"u":'+ lines1[0].split(',"u":')[1]
				fin.write(line)
				#fin.write('\n')
				line = '{"t":"0"' + ',"u":'+ lines2[0].split(',"u":')[1]
				fin.write(line)
				#fin.write('\n')
				i = 1
				j = 1
				while i < len(lines1) and j < len(lines2):
					diff_timestamp1 = int(lines1[i].split('"t":')[1].split(',"u":')[0].replace('"',''))-base_timestamp1
					diff_timestamp2 = int(lines2[j].split('"t":')[1].split(',"u":')[0].replace('"',''))-base_timestamp2
					if diff_timestamp1 < diff_timestamp2:
						line = '{"t":'+str(diff_timestamp1) + ',"u":'+ lines1[i].split(',"u":')[1]
						fin.write(line)
						#fin.write('\n')
						i += 1
					elif diff_timestamp1 > diff_timestamp2:
						line = '{"t":'+str(diff_timestamp2) + ',"u":'+ lines2[j].split(',"u":')[1]
						fin.write(line)
						#fin.write('\n')
						j += 1
					else:
						line = '{"t":'+str(diff_timestamp1) + ',"u":'+ lines1[i].split(',"u":')[1]
						fin.write(line)
						#fin.write('\n')
						line = '{"t":'+str(diff_timestamp2)+ ',"u":'+ lines2[j].split(',"u":')[1]
						fin.write(line)
						#fin.write('\n')
						i += 1
						j += 1
				while i < len(lines1):
					diff_timestamp1 = int(lines1[i].split('"t":')[1].split(',"u":')[0].replace('"',''))-base_timestamp1
					line = '{"t":'+str(diff_timestamp1) + ',"u":'+ lines1[i].split(',"u":')[1]
					fin.write(line)
					#fin.write('\n')
					i += 1
				'''
				while j < len(lines2):
					diff_timestamp2 = int(lines2[j].split('"t":')[1].split(',"u":')[0].replace('"',''))-base_timestamp2
					line = '{"t":'+str(diff_timestamp2) + ',"u":'+ lines2[j].split(',"u":')[1]
					fin.write(line)
					#fin.write('\n')
					j += 1
				'''
			#print "i:" + str(i) + ' ' + "j:" + str(j)
	fin.close()
	f1.close()
	f2.close()

#merge_files(sys.argv[1],sys.argv[2])

def getAllFilesRecursive(path,botpath,botfile):
	for f in listdir(path): 
		if isfile(join(path,f)) and 'database' in str(f):
			merge_files(path,f,botpath,botfile)
	dirs = [d for d in listdir(path) if isdir(join(path,d))]
	for d in dirs:
		files_in_d = getAllFilesRecursive(join(path,d),botpath,botfile)
		if files_in_d:
			for f in files_in_d:
				if 'database' in str(f):
					merge_files(join(path,d),f,botpath,botfile)

path = "/home/dell/Documents/research project/twitch/Data/Real Data"
botpath = '/home/dell/Documents/research project/twitch/Data/Bot Data'
botfile = '#dip_7777database_chatterscontrolled.txt'
getAllFilesRecursive(path,botpath,botfile)