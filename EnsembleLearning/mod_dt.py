import pandas as pd
import math
import copy
import numpy as np

#decision tree and modified tree

class TreeNode:
	def __init__(self):
		self.feature = None
		self.children = None
		self.depth = -1
		self.is_leaf_node = False
		self.label = None
	
	# functions to set values
	def feat_(self, feature):
		self.feature = feature

	def child(self, children):
		self.children = children

	def dpth(self, depth):
		self.depth = depth

	def leaf_(self, status):
		self.is_leaf_node = status

	def lbl(self, label):
		self.label = label

	# functions to return values
	def leaf(self):
		return self.is_leaf_node

	def depth_(self):
		return self.depth

	def lbl_(self):
		return self.label 


# algorithum implementation

class ID3:
	# Option 0: Entropy, option 1: ME, Option 2: GI
	def __init__(self, option=1, max_depth = 10, subset=2):
		self.option = option
		self.max_depth = max_depth
		self.subset = subset
	
	
	def set_max_depth(self, max_depth):
		self.max_depth = max_depth


	def set_option(self, option):
		self.option = option

	def entro(self, data, label_dict):
		"""
		This function returns entropy for a specific data subsets and a set of labels
		"""
		label_key = list(label_dict.keys())[0]
		label_values = label_dict[label_key]
		if len(data) == 0:
			return 0
		entropy = 0
		for value in label_values:
			p = len(data[data[label_key] == value]) / len(data)
			if p != 0:
				entropy += -p * math.log2(p)
		return entropy
	
	def me_(self, data, label_dict):
		"""
		This function returns ME for a specific data subsets and a set of labels
		"""
		label_key = list(label_dict.keys())[0]
		label_values = label_dict[label_key]
		if len(data) == 0:
			return 0
		max_p = 0
		for value in label_values:
			p = len(data[data[label_key] == value]) / len(data)
			max_p = max(max_p, p)
		return 1 - max_p
		
	
	def gi_(self, data, label_dict):
		"""
		This function returns GI for a specific data subsets and a set of labels
		"""
		label_key = list(label_dict.keys())[0]
		label_values = label_dict[label_key]
		if len(data) == 0:
			return 0
		temp = 0
		for value in label_values:
			p = len(data[data[label_key] == value]) / len(data)
			temp += p**2
		return 1 - temp
	

	def maj_lbl(self,column):
		"""
		This function returns the major label
		"""

		majority_label = column.value_counts().idxmax()

		return majority_label

	def heaur(self):

		if self.option == 0:
			heuristics = self.entro
		if self.option == 1:
			heuristics = self.me_
		if self.option == 2:
			heuristics = self.gi_

		return heuristics


	def max_gain_(self, data, label_dict, features_dict, sampled_features):

		heuristics = self.heaur()
		measure = heuristics(data, label_dict)

		max_gain = float('-inf')
		max_f_name = ''

		for f_name in sampled_features:
			gain = 0
			f_values = features_dict[f_name]
			for val in f_values:
				subset = data[data[f_name] == val]
				p = len(subset) / len(data)
				
				gain += p * heuristics(subset, label_dict)

			# get maximum gain and feature name	
			gain = measure - gain
			if gain > max_gain:
				max_gain = gain
				max_f_name = f_name

		return max_f_name
		

	def feat_split(self, cur_node):
		next_nodes = []
		features_dict = cur_node['features_dict']
		label_dict = cur_node['label_dict']
		dt_node = cur_node['dt_node']
		data = cur_node['data']

		
		label_key = list(label_dict.keys())[0]
		label_values = label_dict[label_key]
		
		if len(data) > 0:
			majority_label = self.maj_lbl(data[label_key])
			
		heuristics = self.heaur()
		measure = heuristics(data, label_dict)

		# check leaf nodes
		if measure == 0 or dt_node.depth_() == self.max_depth or len(features_dict) == 0:
			dt_node.leaf_(True)
			if len(data) > 0:
				dt_node.lbl(majority_label)
			return next_nodes

		
		children = {}

		
		keys = list(features_dict.keys())

		if len(keys) > self.subset:
			sampled_features = np.random.choice(keys, self.subset, replace=False)
		else:
			sampled_features = keys 

		max_f_name = self.max_gain_(data, label_dict, features_dict, sampled_features)
		dt_node.feat_(max_f_name)

		
		rf = copy.deepcopy(features_dict)
		rf.pop(max_f_name, None)
	
		for val in features_dict[max_f_name]:
			child_node = TreeNode()
			child_node.lbl(majority_label)
			child_node.dpth(dt_node.depth_() + 1)
			children[val] = child_node
			primary_node = {'data': data[data[max_f_name] == val],'features_dict': rf, 'label_dict': label_dict, 'dt_node': child_node}
			next_nodes.append(primary_node)
		
		
		dt_node.child(children)
		
		return next_nodes
	   
	
	# construct the decision tree
	def con_tree(self, data, features_dict, label_dict):

		
		import queue
		dt_root = TreeNode()
		dt_root.dpth(0)
		root = {'data': data,'features_dict': features_dict, 'label_dict': label_dict, 'dt_node': dt_root}

		Q = queue.Queue()
		Q.put(root)
		while not Q.empty():
			cur_node = Q.get()
			for node in self.feat_split(cur_node):
				Q.put(node)
		return dt_root
	

	def classify_one(self, dt, data):
		temp = dt
		while not temp.leaf(): 
			temp = temp.children[data[temp.feature]]
		return temp.label

	def prd(self, dt, data):
		return data.apply(lambda row: self.classify_one(dt, row), axis=1)

class MID3:
	def __init__(self, option=1, max_depth = 10):
		self.option = option
		self.max_depth = max_depth
	def set_max_depth(self, max_depth):
		self.max_depth = max_depth
	def set_option(self, option):
		self.option = option
	def entrop(self, data, label_dict, weights):
		label_key = list(label_dict.keys())[0]
		label_values = label_dict[label_key]
		total = np.sum(weights)
		col = np.array(data[label_key].tolist())

		if total == 0:
			return 0
		entropy = 0

		for value in label_values:
			w = weights[col==value]
			p = np.sum(w) / total

			if p != 0:
				entropy += -p * math.log2(p)
		return entropy
	#me of df
	def me_df(self, data, label_dict, weights):
		label_key = list(label_dict.keys())[0]
		label_values = label_dict[label_key]
		total = np.sum(weights)
		col = np.array(data[label_key].tolist())

		if total == 0:
			return 0
		max_p = 0

		for value in label_values:
			w = weights[col==value]
			p = np.sum(w) / total
			max_p = max(max_p, p)
		return 1 - max_p
		
	#gi of df
	def gi_df(self, data, label_dict):
		label_key = list(label_dict.keys())[0]
		label_values = label_dict[label_key]
		total = np.sum(weights)
		col = np.array(data[label_key].tolist())

		if total == 0:
			return 0

		temp = 0
		for value in label_values:
			w = weights[col==value]
			p = np.sum(w) / total
			temp += p**2
		return 1 - temp
	
#maj label
	def maj_lbl(self, data, label_dict, weights):
		label_key = list(label_dict.keys())[0]
		label_values = label_dict[label_key]

		max_sum= float('-inf')
		col = np.array(data[label_key].tolist())

		for value in label_values:
			w = weights[col==value]
			w_sum = np.sum(w)
			if w_sum > max_sum:
				maj_lbl = value
				max_sum = w_sum

		return maj_lbl

	def gain_opt(self):

		if self.option == 0:
			heur = self.entrop
		if self.option == 1:
			heur = self.me_df
		if self.option == 2:
			heur = self.gi_df

		return heur



	def max_gain_(self, data, label_dict, features_dict, weights):

		heur = self.gain_opt()
		measure = heur(data, label_dict, weights)

		total = np.sum(weights)

		max_gain = float('-inf')
		max_f_name = ''

		for f_name, f_values in features_dict.items():
			col = np.array(data[f_name].tolist())
			gain = 0
			for val in f_values:
				w = weights[col==val]
				temp_weights = w 
				p = np.sum(temp_weights) /total
				subset = data[data[f_name] == val]
			
				gain += p * heur(subset, label_dict, temp_weights)

			# get maximum gain and feature name	
			gain = measure - gain
			if gain > max_gain:
				max_gain = gain
				max_f_name = f_name

		return max_f_name
		

	def ft_splt(self, cur_node):
		next_nodes = []
		features_dict = cur_node['features_dict']
		label_dict = cur_node['label_dict']
		dt_node = cur_node['dt_node']
		data = cur_node['data']
		weights = cur_node['weights']

		
		label_key = list(label_dict.keys())[0]
		label_values = label_dict[label_key]
		
		total = sum(weights)
		if total > 0:
			maj_lbl = self.maj_lbl(data, label_dict, weights)
			
		heur = self.gain_opt()
		measure = heur(data, label_dict, weights)

		# check leaf nodes
		if measure == 0 or dt_node.cnt_dpth() == self.max_depth or len(features_dict) == 0:
			dt_node.leaf(True)
			if total > 0:
				dt_node.lbl(maj_lbl)
			return next_nodes

		
		children = {}
		max_f_name = self.max_gain_(data, label_dict, features_dict, weights)
		dt_node.feat(max_f_name)

		# remove the feature that has been splitted on, get remaining features
		rf = copy.deepcopy(features_dict)
		rf.pop(max_f_name, None)
		
		col = np.array(data[max_f_name].tolist())

		for val in features_dict[max_f_name]:
			child_node = TreeNode()
			child_node.lbl(maj_lbl)
			child_node.dpth(dt_node.cnt_dpth() + 1)
			children[val] = child_node
			w = weights[col==val]
			primary_node = {'data': data[data[max_f_name] == val], 'weights': w, 'features_dict': rf, 'label_dict': label_dict, 'dt_node': child_node}
			next_nodes.append(primary_node)
		
		# set chiildren nodes
		dt_node.child_(children)
		
		return next_nodes
	   
	
	# construct the decision tree
	def cons_tree(self, data, features_dict, label_dict, weights):

		# bfs using queue
		import queue
		dt_root = TreeNode()
		dt_root.dpth(0)
		root = {'data': data, 'weights':weights, 'features_dict': features_dict, 'label_dict': label_dict, 'dt_node': dt_root}

		Q = queue.Queue()
		Q.put(root)
		while not Q.empty():
			cur_node = Q.get()
			for node in self.ft_splt(cur_node):
				Q.put(node)
		return dt_root
	

	def classify_one(self, dt, data):
		temp = dt
		while not temp.leaf_(): 
			temp = temp.children[data[temp.feature]]
		return temp.label

	def pred_(self, dt, data):
		return data.apply(lambda row: self.classify_one(dt, row), axis=1)