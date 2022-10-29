import pandas as pd
import math
import copy
import numpy as np
import math 
import matplotlib.pyplot as plt 



#set desired T value
T = 50

###-------------------###

# TreeNode class 

class TreeNode:
	def __init__(self):
		self.feature = None
		self.children = None
		self.depth = -1
		self.is_leaf_node = False
		self.label = None
	
	# functions to set values
	def feat(self, feature):
		self.feature = feature

	def child_(self, children):
		self.children = children

	def dpth(self, depth):
		self.depth = depth

	def leaf(self, status):
		self.is_leaf_node = status

	def lbl(self, label):
		self.label = label

	# functions to return values
	def leaf_(self):
		return self.is_leaf_node

	def cnt_dpth(self):
		return self.depth

	def gt_labels(self):
		return self.label 

class MID3:
	# Option 0: Entropy, option 1: ME, Option 2: GI
	def __init__(self, option=1, max_depth = 10):
		self.option = option
		self.max_depth = max_depth
	
	
	def set_max_depth(self, max_depth):
		self.max_depth = max_depth


	def set_option(self, option):
		self.option = option

	def entrop(self, data, label_dict, weights):
		"""
		This function returns entropy for a specific data subsets and a set of labels
		"""
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
	
	def me_df(self, data, label_dict, weights):
		"""
		This function returns ME for a specific data subsets and a set of labels
		"""
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
		
	
	def gi_df(self, data, label_dict):
		"""
		This function returns GI for a specific data subsets and a set of labels
		"""
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
	

	def maj_lbl(self, data, label_dict, weights):
		"""
		This function returns the major label
		"""
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

	def predict(self, dt, data):
		return data.apply(lambda row: self.classify_one(dt, row), axis=1)


def predt_ins_tree(inst, tree):
  for node in tree.keys():
    prediction = 0
    value = inst[node]
    try:
      tree = tree[node][value]
      if type(tree) is dict:
        prediction = predt_ins_tree(inst, tree)
      else:
        prediction = tree
    except KeyError:
        prediction = mode(vals(tree))
  return prediction

def conv_to_vals(df, numerical_features):
	for f in numerical_features:
		median_val = df[f].median()
		df[f] = df[f].gt(median_val).astype(int)

	return df 

def inf_gain(data, gain):
  infor_gain = []
  if gain=='S':
    for key in data.columns[:-1]:
      infor_gain.append(s_tot(data) - s_att(data, key))
    return data.keys()[:-1][np.argmax(infor_gain)]
  
  elif gain=='me':
    for key in data.columns[:-1]:
      infor_gain.append(me_tot(data) - me_att(data, key))
    return data.keys()[:-1][np.argmax(infor_gain)]

  elif gain=='gi':
    for key in data.columns[:-1]:
      infor_gain.append(gini_tot(data) - gini_attribs(data, key))
    return data.keys()[:-1][np.argmax(infor_gain)]

column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
types = {'age': int, 'job': str, 'marital': str, 'education': str,'default': str,'balance': int,'housing': str,'loan': str,'contact': str,'day': int,'month': str,'duration': int, \
		'campaign': int,'pdays': int,'previous': int,'poutcome': str,'y': str}

# load train data 
train_data =  pd.read_csv('~/Courses/Machine_Learning/MLCourse/EnsembleLearning/data/trainbank.csv', names=column_names, dtype=types)
# # load test data 
test_data =  pd.read_csv('~/Courses/Machine_Learning/MLCourse/EnsembleLearning/data/testbank.csv', names=column_names, dtype=types)

numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

train_data = conv_to_vals(train_data, numerical_features)

test_data = conv_to_vals(test_data, numerical_features)


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



train_size, test_size = len(train_data),len(test_data)
alphas = [0 for x in range(T)]
weights = np.array([1/train_size for x in range(train_size)])
# print(weights)

train_errors, test_errors = [0 for x in range(T)], [0 for x in range(T)]
train_errorsT, test_errorsT = [0 for x in range(T)], [0 for x in range(T)]

test_py = np.array([0 for x in range(test_size)])

train_py = np.array([0 for x in range(train_size)])
for t in range(T):
	dt_generator = MID3(option=0, max_depth=1)
			
	dt_construction = dt_generator.cons_tree(train_data, features_dict, label_dict, weights)

	# train errors
	train_data['pred_label']= dt_generator.predict(dt_construction, train_data)
	train_data['result'] = (train_data[['y']].values == train_data[['pred_label']].values).all(axis=1).astype(int)
	err = 1 - len(train_data[train_data['result'] == 1]) / train_size
	train_errors[t] = err

	# test errors
	test_data['pred_label']= dt_generator.predict(dt_construction, test_data)
	test_data['result'] = (test_data[['y']].values == test_data[['pred_label']].values).all(axis=1).astype(int)
	test_errors[t] = 1 - len(test_data[test_data['result'] == 1]) / test_size
	
	# weighted errors and alphas
	temp = train_data.apply(lambda row: 1 if row['y'] == row['pred_label'] else -1, axis=1)
	temp = np.array(temp.tolist())
	w = weights[temp == -1]
	err = np.sum(w)

	alpha = 0.5 * math.log((1-err)/err)
	alphas[t] = alpha 

	# get new weights 
	weights = np.exp(temp * -alpha) * weights
	total = np.sum(weights)
	weights = weights/total

	#errors of all decision stumps

	pred_label = np.array(train_data['pred_label'].tolist())
	pred_label[pred_label == 'yes'] = 1 
	pred_label[pred_label == 'no'] = -1
	pred_label = pred_label.astype(int)
	train_py = train_py+pred_label*alpha
	pred_label = pred_label.astype(str)
	pred_label[train_py > 0] = 'yes'
	pred_label[train_py <= 0] = 'no'
	train_data['pred_label'] = pd.Series(pred_label)

	train_data['result'] = (train_data[['y']].values == train_data[['pred_label']].values).all(axis=1).astype(int)
	
	train_errorsT[t] = 1 - len(train_data[train_data['result'] == 1]) / train_size


	#  test data 
	
	pred_label = np.array(test_data['pred_label'].tolist())
	pred_label[pred_label == 'yes'] = 1 
	pred_label[pred_label == 'no'] = -1
	pred_label = pred_label.astype(int)
	test_py = test_py+pred_label*alpha
	pred_label = pred_label.astype(str)
	pred_label[test_py > 0] = 'yes'
	pred_label[test_py <= 0] = 'no'
	test_data['pred_label'] = pd.Series(pred_label)

	test_data['result'] = (test_data[['y']].values == test_data[['pred_label']].values).all(axis=1).astype(int)
	
	test_errorsT[t] = 1 - len(test_data[test_data['result'] == 1]) / test_size


print(test_errorsT[-1])

fig, ax = plt.subplots(figsize = (6,4))
ax.plot(train_errors, label='train', c='black', alpha=0.3)
ax.plot(test_errors,  c='purple', label='test',)
ax.set_ylabel('Error')
ax.set_xlabel('Iteration')
ax.set_title("Error per Iteration")
ax.legend()
plt.show()

fig1, ax1 = plt.subplots(figsize = (6,4))
ax1.plot(train_errorsT,  color='black', label='train')
ax1.plot(test_errorsT,  color='purple', label='test')
ax1.set_ylabel('Error')
ax1.set_xlabel('Iteration')
ax1.set_title("Full tree errors")
ax1.legend()
plt.show()