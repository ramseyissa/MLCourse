import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import math
import copy



#define col names and types
column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
types = {'age': int, 'job': str, 'marital': str, 'education': str,'default': str,'balance': int,'housing': str,'loan': str,'contact': str,'day': int,'month': str,'duration': int, \
		'campaign': int,'pdays': int,'previous': int,'poutcome': str,'y': str}



numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']


#bring in bank train/test set
train_df =  pd.read_csv('~/Courses/Machine_Learning/MLCourse/EnsembleLearning/data/trainbank.csv', names=column_names, dtype=types)
#bring in bank test set
test_df =  pd.read_csv('~/Courses/Machine_Learning/MLCourse/EnsembleLearning/data/testbank.csv', names=column_names, dtype=types)

# create tree node for ID3
class TreeNode:
	def __init__(self):
		self.feature = None
		self.children = None
		self.depth = -1
		self.is_leaf_node = False
		self.label = None
	
	# functions to set values
	def featr(self, feature):
		self.feature = feature

	def child_(self, children):
		self.children = children

	def set_depth(self, depth):
		self.depth = depth

	def set_leaf(self, status):
		self.is_leaf_node = status

	def set_label(self, label):
		self.label = label

	# functions to return values
	def lf(self):
		return self.is_leaf_node

	def dpth_(self):
		return self.depth

	def lbl(self):
		return self.label 


# ID3 implementation

class ID3:
	# Option 0: Entropy, option 1: ME, Option 2: GI
	def __init__(self, option=1, max_depth = 10):
		self.option = option
		self.max_depth = max_depth
	
	
	def max_dpth(self, max_depth):
		self.max_depth = max_depth


	def choose_gain(self, option):
		self.option = option

#entropy caculation
	def entr(self, data, lbl_dicti):
		label_key = list(lbl_dicti.keys())[0]
		label_values = lbl_dicti[label_key]
		if len(data) == 0:
			return 0
		entropy = 0
		for value in label_values:
			p = len(data[data[label_key] == value]) / len(data)
			if p != 0:
				entropy += -p * math.log2(p)
		return entropy

	#me of dataset
	def me(self, data, lbl_dicti):
		label_key = list(lbl_dicti.keys())[0]
		label_values = lbl_dicti[label_key]
		if len(data) == 0:
			return 0
		max_p = 0
		for value in label_values:
			p = len(data[data[label_key] == value]) / len(data)
			max_p = max(max_p, p)
		return 1 - max_p
		
	# gini index values 
	def gi(self, data, lbl_dicti):
		label_key = list(lbl_dicti.keys())[0]
		label_values = lbl_dicti[label_key]
		if len(data) == 0:
			return 0
		temp = 0
		for value in label_values:
			p = len(data[data[label_key] == value]) / len(data)
			temp += p**2
		return 1 - temp
	
 
#majority label
	def maj_lbl(self,column):
		majority_label = column.value_counts().idxmax()
		return majority_label

	def iden_heur(self):
		if self.option == 0:
			heuristics = self.entr
		if self.option == 1:
			heuristics = self.me
		if self.option == 2:
			heuristics = self.gi

		return heuristics

#compute max gain
	def maxim_gain(self, data, lbl_dicti, ft_):
		heuristics = self.iden_heur()
		measure = heuristics(data, lbl_dicti)
		max_gain = float('-inf')
		max_f_name = ''
		for f_name, f_values in ft_.items():
			gain = 0
			for val in f_values:
				subset = data[data[f_name] == val]
				p = len(subset.index) / len(data)
				gain += p * heuristics(subset, lbl_dicti)
			gain = measure - gain
			if gain > max_gain:
				max_gain = gain
				max_f_name = f_name

		return max_f_name
		
#calculate best split 
	def optimal_split(self, cur_node):
		node_ = []
		ft_ = cur_node['ft_']
		lbl_dicti = cur_node['lbl_dicti']
		tree_nd = cur_node['tree_nd']
		data = cur_node['data']
		label_key = list(lbl_dicti.keys())[0]
		label_values = lbl_dicti[label_key]
		if len(data) > 0:
			majority_label = self.maj_lbl(data[label_key])
		heuristics = self.iden_heur()
		measure = heuristics(data, lbl_dicti)
		if measure == 0 or tree_nd.dpth_() == self.max_depth or len(ft_) == 0:
			tree_nd.set_leaf(True)
			if len(data) > 0:
				tree_nd.set_label(majority_label)
			return node_
		children = {}
		max_f_name = self.maxim_gain(data, lbl_dicti, ft_)
		tree_nd.featr(max_f_name)
		rf = copy.deepcopy(ft_)
		rf.pop(max_f_name, None)
		for val in ft_[max_f_name]:
			child_node = TreeNode()
			child_node.set_label(majority_label)
			child_node.set_depth(tree_nd.dpth_() + 1)
			children[val] = child_node
			primary_node = {'data': data[data[max_f_name] == val],'ft_': rf, 'lbl_dicti': lbl_dicti, 'tree_nd': child_node}
			node_.append(primary_node)
		tree_nd.child_(children)
		return node_
	   
	
	# construct the decision tree
	def deci_tree(self, data, ft_, lbl_dicti):

		# bfs using queue
		import queue
		dt_root = TreeNode()
		dt_root.set_depth(0)
		root = {'data': data,'ft_': ft_, 'lbl_dicti': lbl_dicti, 'tree_nd': dt_root}

		Q = queue.Queue()
		Q.put(root)
		while not Q.empty():
			cur_node = Q.get()
			for node in self.optimal_split(cur_node):
				Q.put(node)
		return dt_root
	

	def classfy(self, dt, data):
		temp = dt
		while not temp.lf(): 
			temp = temp.children[data[temp.feature]]
		return temp.label

	def predict(self, dt, data):
		return data.apply(lambda row: self.classfy(dt, row), axis=1)

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


#sampled gini att implementation yielded error 
def gini_attribs(data, attr):
  #get label col name
  label = data.columns[-1]
  #get uniq attributes
  uniqu_att = data[attr].unique()
  #uniq labels
  uniq_labls = data[label].unique()
  gi = 0
  gi_0 = []
  for vals in uniqu_att:
    gi_i = 1
    for lbl in uniq_labls:
      cnt_att = len(data[attr][data[attr]==vals][data[label]==lbl])
      tot_att = len(data[attr][data[attr]==vals])
      prob = (cnt_att/tot_att)**2
      gi__ = gi_i + -prob
      comb_prob = tot_att/len(data)
      g_f = gi + comb_prob*gi__
      gi_0.append(g_f)
  gi_final = sum(gi_0)
  return gi_final


def re_assgn_vals(df, numerical_features):
	for f in numerical_features:
		median_val = df[f].median()
		df[f] = df[f].gt(median_val).astype(int)

	return df 

column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
types = {'age': int, 'job': str, 'marital': str, 'education': str,'default': str,'balance': int,'housing': str,'loan': str,'contact': str,'day': int,'month': str,'duration': int, \
		'campaign': int,'pdays': int,'previous': int,'poutcome': str,'y': str}

# orginal implementation yield error
def gini_tot(data):
  data_label = data.columns[-1]
  unique_labl = data[data_label].unique()    
  gi_tot = 0
  for i in unique_labl:
    prob = data[data_label].value_counts()[i]/len(data[data_label])
    gi_tot = gi_tot + prob**2
  gi_final = 1 - gi_tot 
  return gi_final

numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

train_df = re_assgn_vals(train_df, numerical_features)
test_df = re_assgn_vals(test_df, numerical_features)

ft_ = {}
ft_['age'] = [0, 1]
ft_['job'] = ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student', 'blue-collar', 'self-employed', 'retired', 'technician', 'services']
ft_['marital'] = ['married','divorced','single']
ft_['education'] = ['unknown', 'secondary', 'primary', 'tertiary']
ft_['default'] = ['yes', 'no']
ft_['balance'] = [0, 1]
ft_['housing'] = ['yes', 'no']
ft_['loan'] = ['yes', 'no']
ft_['contact'] = ['unknown', 'telephone', 'cellular']
ft_['day'] = [0, 1]
ft_['month'] = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
ft_['duration'] = [0, 1]
ft_['campaign'] = [0, 1]
ft_['pdays'] = [0, 1]
ft_['previous'] = [0, 1]
ft_['poutcome'] = ['unknown', 'other', 'failure', 'success']

lbl_dicti = {}
lbl_dicti['y'] = ['yes', 'no']




#--------SET T ------

T = 50

tain_count, t_size = len(train_df),len(test_df)
train_err, tst_err = [0 for x in range(T)], [0 for x in range(T)]

tstarray = np.array([0 for x in range(t_size)])
trainarray = np.array([0 for x in range(tain_count)])

for t in range(T):

	# sample train data
	train_deci_selc = train_df.sample(frac=0.5, replace=True, random_state = t)

	tree_gen = ID3(option=0, max_depth=15)
			
	dt_construction = tree_gen.deci_tree(train_deci_selc, ft_, lbl_dicti)

	# predict train data
	prd_lbl = tree_gen.predict(dt_construction, train_df)
	prd_lbl = np.array(prd_lbl.tolist())

	prd_lbl[prd_lbl == 'yes'] = 1 
	prd_lbl[prd_lbl == 'no'] = -1
	prd_lbl = prd_lbl.astype(int)
	trainarray = trainarray+prd_lbl
	prd_lbl = prd_lbl.astype(str)
	prd_lbl[trainarray > 0] = 'yes'
	prd_lbl[trainarray <= 0] = 'no'
	train_df['prd_lbl'] = pd.Series(prd_lbl)

	train_df['result'] = (train_df[['y']].values == train_df[['prd_lbl']].values).all(axis=1).astype(int)
	
	train_err[t] = 1 - len(train_df[train_df['result'] == 1]) / tain_count

	# predict test data 
	prd_lbl = tree_gen.predict(dt_construction, test_df)
	prd_lbl = np.array(prd_lbl.tolist())

	prd_lbl[prd_lbl == 'yes'] = 1 
	prd_lbl[prd_lbl == 'no'] = -1
	prd_lbl = prd_lbl.astype(int)
	tstarray = tstarray+prd_lbl
	prd_lbl = prd_lbl.astype(str)
	prd_lbl[tstarray > 0] = 'yes'
	prd_lbl[tstarray <= 0] = 'no'
	test_df['prd_lbl'] = pd.Series(prd_lbl)

	test_df['result'] = (test_df[['y']].values == test_df[['prd_lbl']].values).all(axis=1).astype(int)
	
	tst_err[t] = 1 - len(test_df[test_df['result'] == 1]) / t_size

print(tst_err[-1])
plt.plot(train_err, label = "train",c='purple')
plt.plot(tst_err, label = "test",c='orange')
plt.xlabel('iteration')
plt.ylabel('Error')
plt.title('Bagged Trees')
plt.legend()
plt.show()
# plt.savefig('BDT.png', dpi=300, bbox_inches='tight')