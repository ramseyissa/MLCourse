import pandas as pd
import math
import copy
import numpy as np
import matplotlib.pyplot as plt 
import sys


# TreeNode class 

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




def num_feat(df, numerical_features):
	for f in numerical_features:
		median_val = df[f].median()
		df[f] = df[f].gt(median_val).astype(int)

	return df 


def bank_df():

	column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
	types = {'age': int, 'job': str, 'marital': str, 'education': str,'default': str,'balance': int,'housing': str,'loan': str,'contact': str,'day': int,'month': str,'duration': int, \
			'campaign': int,'pdays': int,'previous': int,'poutcome': str,'y': str}

	# load train data 
	train_data =  pd.read_csv('~/Courses/Machine_Learning/MLCourse/EnsembleLearning/data/trainbank.csv', names=column_names, dtype=types)
	# load test data 
	test_data =  pd.read_csv('~/Courses/Machine_Learning/MLCourse/EnsembleLearning/data/testbank.csv', names=column_names, dtype=types)

	numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

	train_data = num_feat(train_data, numerical_features)
	test_data = num_feat(test_data, numerical_features)

	features_dict = {}
	features_dict['age'] = [0, 1]
	features_dict['job'] = ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student', 'blue-collar', 'self-employed', 'retired', 'technician', 'services']
	features_dict['marital'] = ['married','divorced','single']
	features_dict['education'] = ['unknown', 'secondary', 'primary', 'tertiary']
	features_dict['default'] = ['yes', 'no']
	features_dict['balance'] = [0, 1]
	features_dict['housing'] = ['yes', 'no']
	features_dict['loan'] = ['yes', 'no']
	features_dict['contact'] = ['unknown', 'telephone', 'cellular']
	features_dict['day'] = [0, 1]
	features_dict['month'] = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
	features_dict['duration'] = [0, 1]
	features_dict['campaign'] = [0, 1]
	features_dict['pdays'] = [0, 1]
	features_dict['previous'] = [0, 1]
	features_dict['poutcome'] = ['unknown', 'other', 'failure', 'success']
	
	label_dict = {}
	label_dict['y'] = ['yes', 'no']

	return features_dict, label_dict, train_data, test_data


# majority error total
def me_tot(data):
  #get data label
  data_lbl = data.columns[-1]
  #get unique vals
  uniq_vals = data[data_lbl].unique()
  uniq_val_lst = []
  for vals in uniq_vals:
    val_cnt = len(data[data_lbl][data[data_lbl] == vals])
    uniq_val_lst.append(val_cnt)
  #for min val in list, divide by entire labels  
  me_tot = min(uniq_val_lst)/len(data[data_lbl])
  return me_tot


#calc me of att
def me_att(data, attr):
  #get label col name
  label = data.columns[-1]
  #get uniq attributes
  uniqu_att = data[attr].unique()
  #uniq labels
  uniq_labls = data[label].unique()
  me = 0
  me_= []
  for vals in uniqu_att:
    for lbl in uniq_labls:
      cnt_att = len(data[attr][data[attr]==vals][data[label]==lbl])
      tot_att = len(data[attr][data[attr]==vals])
      prob = cnt_att/tot_att
      comb_prob = tot_att/len(data)
      me = me + comb_prob*prob
      me_.append(me)
  me_final = sum(me_)
  return me_final

def data_proc(features_dict, label_dict, train_data, test_data, num_subset, label_name):
	T = 5
	train_size, test_size = len(train_data),len(test_data)
	train_errors, test_errors = [0 for x in range(T)], [0 for x in range(T)]

	test_py = np.array([0 for x in range(test_size)])
	train_py = np.array([0 for x in range(train_size)])

	for t in range(T):
		sampled = train_data.sample(frac=0.5, replace=True, random_state=t)
		# build tree 
		dt_generator = ID3(option=0, max_depth=15, subset = num_subset)
			
		dt_construction = dt_generator.con_tree(sampled, features_dict, label_dict)

		# train
		pred_label = dt_generator.prd(dt_construction, train_data)
		pred_label = np.array(pred_label.tolist())

		pred_label[pred_label == 'yes'] = 1 
		pred_label[pred_label == 'no'] = -1
		pred_label = pred_label.astype(int)
		train_py = train_py+pred_label
		pred_label = pred_label.astype(str)
		pred_label[train_py > 0] = 'yes'
		pred_label[train_py <= 0] = 'no'
		train_data['pred_label'] = pd.Series(pred_label)

		train_data['result'] = (train_data[[label_name]].values == train_data[['pred_label']].values).all(axis=1).astype(int)
	
		train_errors[t] = 1 - len(train_data[train_data['result'] == 1]) / train_size
		
		# prd test data 
		pred_label = dt_generator.prd(dt_construction, test_data)
		pred_label = np.array(pred_label.tolist())

		pred_label[pred_label == 'yes'] = 1 
		pred_label[pred_label == 'no'] = -1
		pred_label = pred_label.astype(int)
		test_py = test_py+pred_label
		pred_label = pred_label.astype(str)
		pred_label[test_py > 0] = 'yes'
		pred_label[test_py <= 0] = 'no'
		test_data['pred_label'] = pd.Series(pred_label)

		test_data['result'] = (test_data[['y']].values == test_data[['pred_label']].values).all(axis=1).astype(int)
		
		test_errors[t] = 1 - len(test_data[test_data['result'] == 1]) / test_size

	return train_errors, test_errors



num_subset = 2
features_dict, label_dict, train_data, test_data = bank_df()
train_errors, test_errors = data_proc(features_dict, label_dict, train_data, test_data, num_subset, 'y')
print(test_errors[-1])
# num_subset=[2,4,6]
# save_fig(train_errors, test_errors, num_subset, 'bank_'+str(num_subset)+'.png')


fig, ax = plt.subplots(figsize = (6,4))
ax.plot(train_errors, label='train', c='black', alpha=0.3)
ax.plot(test_errors,  c='purple', label='test',)
ax.set_ylabel('Error')
ax.set_xlabel('Iteration')
ax.set_title("Subset 2")
ax.legend()
plt.show()


num_subset = 4
features_dict, label_dict, train_data, test_data = bank_df()
train_errors, test_errors = data_proc(features_dict, label_dict, train_data, test_data, num_subset, 'y')
print(test_errors[-1])



fig1, ax1 = plt.subplots(figsize = (6,4))
ax1.plot(train_errors, label='train', c='black', alpha=0.3)
ax1.plot(test_errors,  c='purple', label='test',)
ax1.set_ylabel('Error')
ax1.set_xlabel('Iteration')
ax1.set_title("Subset 4")
ax1.legend()
plt.show()


num_subset = 6
features_dict, label_dict, train_data, test_data = bank_df()
train_errors, test_errors = data_proc(features_dict, label_dict, train_data, test_data, num_subset, 'y')
print(test_errors[-1])



fig2, ax2 = plt.subplots(figsize = (6,4))
ax2.plot(train_errors, label='train', c='black', alpha=0.3)
ax2.plot(test_errors,  c='purple', label='test',)
ax2.set_ylabel('Error')
ax2.set_xlabel('Iteration')
ax2.set_title("Subset 6")
ax2.legend()
plt.show()