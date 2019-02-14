import sys,math
from collections import namedtuple
from Queue import Queue
from UserBins import *

class Node(object):
	def __init__(self,data=None,freq=0,probability=0.0,entropy=0.0):
		self.data = data
		self.freq = freq
		self.probability = probability
		self.entropy = entropy
		#self.conditionalentropy = conditionalentropy
		self.children = [None] * 6;

class Trie(object):
	def __init__(self):
		self.root = self.NewNode(0)
		self.levelsum = [] 
		self.conditionalentropy = []
		self.leveluniquepatterns = []

	def NewNode(self,data):
		Q = Node()
		Q.data = data
		Q.freq = 1
		Q.probability = 0.0
		Q.entropy = 0.0
		#Q.conditionalentropy = 0.0
		return Q

	def AddString(self,s):
		#cur = node()
		cur = self.root
		for i in range(len(s)):
			if cur.children[ord(s[i])-48] == None:
				cur.children[ord(s[i])-48] = self.NewNode(s[i])
			else:
				cur.children[ord(s[i])-48].freq += 1
			cur = cur.children[ord(s[i])-48]

	def LevelOrderTraversal(self):
		if self.root is None:
			return
		q = Queue()
		q.put(self.root)
		while q.empty() == False:
			n = q.qsize()
			while n > 0:
				p = q.get()
				print "value:" + str(p.data) + ' ' + "entropy:" + str(p.entropy)
				for i in range(1,6):
					if p.children[i]:
						q.put(p.children[i])
				n -= 1
			#print '\n'

	def LevelSum(self):
		if self.root is None:
			return
		q = Queue()
		q.put(self.root)
		depth = 0
		while q.empty() == False:
			n = q.qsize()
			Sum = 0
			uniquepatterns = 0
			while n > 0:
				p = q.get()
				Sum += p.freq
				uniquepatterns += 1
				for i in range(1,6):
					if p.children[i]:
						q.put(p.children[i])
				n -= 1
			
			if depth < len(self.levelsum):
				self.levelsum[depth] = Sum
				self.leveluniquepatterns[depth] = uniquepatterns
			else:
				self.levelsum.append(Sum)
				self.leveluniquepatterns.append(uniquepatterns)
			depth += 1

	def AssignNodeProbability(self):
		if self.root is None:
			return
		q = Queue()
		q.put(self.root)
		depth = 1
		while q.empty() == False:
			n = q.qsize()
			while n > 0:
				p = q.get()
				p.probability = float(p.freq)/self.levelsum[depth-1]
				p.entropy = -p.probability*math.log(p.probability)
				for i in range(1,6):
					if p.children[i]:
						q.put(p.children[i])
				n -= 1
			depth += 1

	def ConditionalEntropy(self):
		if self.root is None:
			return
		q = Queue()
		q.put(self.root)
		index = 0
		while q.empty() == False:
			n = q.qsize()
			entropy = 0.0
			while n > 0:
				p = q.get()
				for i in range(1,6):
					if p.children[i]:
						entropy += p.children[i].entropy
						q.put(p.children[i])
				n -= 1
			if index == 0:
				self.conditionalentropy.append(entropy)
			else:
				if q.empty() == False:
					self.conditionalentropy.append(entropy-self.conditionalentropy[index-1])			
			index += 1

if __name__ == "__main__":

	users = cluster_user_timestamps_msgs(sys.argv[1],sys.argv[2])
	user_imd = user_intermessage_delay(users)
	for user in user_imd.keys():
		print "u:" + user +' ' + "b:" + user_imd[user]['b']
		bins = map_user_timestamp_bins(user_imd[user]['t'])
		s = ''.join(str(b) for b in bins)
		tree = Trie()
		MIN_CCE = sys.float_info.max
		for length in range(1,len(s)+1):
			for i in range(len(s)-length+1):
				tree.AddString(s[i:i+length]) 
			tree.LevelSum()
			tree.AssignNodeProbability()
			tree.ConditionalEntropy()
			#print len(tree.conditionalentropy)
			imd_cce = tree.conditionalentropy[length-1] + ((float(tree.leveluniquepatterns[length-1])/tree.levelsum[length-1])*100)*tree.conditionalentropy[0]
			if MIN_CCE < imd_cce:
				imd_cce = MIN_CCE
				break
		#print "imd entropy:" + str(imd_cce)
	
		bins = map_user_msg_bins(user_imd[user]['m'])
		s = ''.join(str(b) for b in bins)
		tree = Trie()
		MIN_CCE = sys.float_info.max
		for length in range(1,len(s)+1):
			for i in range(len(s)-length+1):
				tree.AddString(s[i:i+length]) 
			tree.LevelSum()
			tree.AssignNodeProbability()
			tree.ConditionalEntropy()
			#print len(tree.conditionalentropy)
			msg_len_cce = tree.conditionalentropy[length-1] + ((float(tree.leveluniquepatterns[length-1])/tree.levelsum[length-1])*100)*tree.conditionalentropy[0]
			if MIN_CCE < msg_len_cce:
				msg_len_cce = MIN_CCE
				break
		#print "msg lengths entropy:" + str(msg_len_cce)
		print "avg entropy:" + str((imd_cce + msg_len_cce)/2)
		
#tree.LevelOrderTraversal()


