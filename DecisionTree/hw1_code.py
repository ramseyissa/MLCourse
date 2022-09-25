import numpy as np
from statistics import mode
from sklearn.metrics import accuracy_score
from sklearn import metrics
import pandas as pd
import os
import sys



train_car= pd.read_csv('train.csv')
test_car = pd.read_csv('test.csv')
train_bank = pd.read_csv('trainbank.csv')
test_bank = pd.read_csv('testbank.csv')






#entropy of dataset calculation
def s_tot(data):
  #get data label col name
  data_label = data.keys()[-1]
  #get uniq label values
  cnt_uniq = data[data_label].unique()    
  s_tot = 0
  for val in cnt_uniq:
    #calc the prob
    prob = data[data_label].value_counts()[val]/len(data[data_label])
    s_tot = s_tot + -prob * np.log2(prob)
  return np.float64(s_tot)


#entropy of att calculation
def s_att(data, attri):
  #get label value name
  data_label = data.keys()[-1]
  #get uniq att vals
  att_vals = data[attri].unique()
  #get uniq label vals
  label_vals = data[data_label].unique()
  s = 0
  for att in att_vals:
    s_i = 0
    for labl in label_vals:
      cnt_val = len(data[attri][data[attri] == att][data[data_label] == labl])
      cnt_tot = len(data[attri][data[attri] == att])
      #calc prob
      prob = cnt_val/cnt_tot
      # + 0.0001 added to avoid div by zero area 
      s_i = s_i + -prob * np.log2(prob + 0.0001)
    s = s + (cnt_tot/len(data))*s_i
  return np.float64(s)


#calculate ME of total dataset
def me_tot(data):
  #get data label
  data_lbl = data.keys()[-1]
  #get unique vals
  labl_uni, lbl_tot = np.unique(data[data_lbl], return_counts = True)  
  #get max
  cnt_vals = np.amax(lbl_tot)
  probl = cnt_vals/len(data[data_lbl])
  me_tot = 1 - probl
  return np.float64(me_tot)


#calc me of att
def me_att(data, attr):
  label = data.keys()[-1]
  uniqu_vals = data[attr].unique()
  m_e = 0
  for vals_ in uniqu_vals:
    att_cntr, labl_cnt = np.unique(data[label][data[attr] == vals_], return_counts = True)
    cnt_labl = np.amax(labl_cnt)
    tot = len(data[attr][data[attr] == vals_])
    prob = cnt_labl/tot
    me = 1 - prob
    m_e = m_e + (tot/len(data))*me
  return np.float64(m_e)


#get gini index of dataset
def gini_tot(data):
  data_label = data.keys()[-1]
  unique_labl = data[data_label].unique()    
  gi_tot = 1
  for i in unique_labl:
    prob = data[data_label].value_counts()[i]/len(data[data_label])
    gi_tot = gi_tot + -prob**2
  return np.float64(gi_tot)

#get gini att
def gini_attribs(data, att):
  label = data.keys()[-1]
  unique_att = data[att].unique()
  uniqu_val = data[label].unique()
  gini_att = 0
  for atrribs in unique_att:
    gi_i = 1
    for vals_ in uniqu_val:
      cnt_ = len(data[att][data[att] == atrribs][data[label] == vals_])
      total_cnt = len(data[att][data[att] == atrribs])
      prob = cnt_/total_cnt
      gi_i = gi_i + -prob**2
    gini_att = gini_att + (total_cnt/len(data))*gi_i
  return np.float64(gini_att)


#get gain interested in 
def inf_gain(data, gain):
  infor_gain = []
  if gain=='S':
    for key in data.keys()[:-1]:
      infor_gain.append(s_tot(data) - s_att(data, key))
    return data.keys()[:-1][np.argmax(infor_gain)]
  
  elif gain=='me':
    for key in data.keys()[:-1]:
      infor_gain.append(me_tot(data) - me_att(data, key))
    return data.keys()[:-1][np.argmax(infor_gain)]

  elif gain=='gi':
    for key in data.keys()[:-1]:
      infor_gain.append(gini_tot(data) - gini_attribs(data, key))
    return data.keys()[:-1][np.argmax(infor_gain)]


#indexing df
def popu_df(data, att, val):
  return data[data[att] == val].reset_index(drop = True)


#ID3 
def ID3(data, tree = None, gain = 'S', tree_depth=50):
  #first use inf gain specified in gain
  node = inf_gain(data, gain)               
  cnt_att = np.unique(data[node])
  data_label = data.keys()[-1]
  deph = 0
  if tree is None:
    tree = {}
    tree[node] = {}
  for att_cnt in cnt_att:           
    updated_data = popu_df(data,node,att_cnt)
    lbl_val, lbl_cnts = np.unique(updated_data[data_label], return_counts = True)
    if len(lbl_cnts) == 1:
      tree[node][att_cnt] = lbl_val[0]
    else:
      deph = deph + 1
      if deph<tree_depth:
        tree[node][att_cnt] = ID3(updated_data)
      elif deph==tree_depth:
        max_labl = np.where(lbl_cnts == np.amax(lbl_cnts))
        tree[node][att_cnt] = lbl_val[max_labl[0][0]]
  return tree

#stackoverflow solution for nested dict
def vals(x):
    if isinstance(x, dict):
        result = []
        for v in x.values():
            result.extend(vals(v))
        return result
    else:
        return [x]

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

def predt(data, tree):
  label_predicted = []
  for i in range(len(data)):
    inst = data.iloc[i,:]
    prediction = predt_ins_tree(inst, tree)
    label_predicted.append(prediction)
  accur = metrics.accuracy_score(data[data.columns[-1]], label_predicted)
  return accur



#score prediction accuracy for car data 
# tree_car = ID3(train_car)
# Predt_car = predt(train_car,tree_car)
# print(tree_car)
# print(Predt_car)

# tree_car = ID3(train_car,tree_depth=1)
# Predt_car = predt(train_car,tree_car)
# Predt_car_test = predt(test_df,tree_car)

# tree_car2 = ID3(train_df,tree_depth=2)
# Predt_car2 = predt(train_df,tree_car2)
# Predt_car_test2 = predt(test_df,tree_car2)

# tree_car3 = ID3(train_df,tree_depth=2)
# Predt_car3 = predt(train_df,tree_car3)
# Predt_car_test3 = predt(test_df,tree_car3)

# tree_car = ID3(train_df)
# Predt_car = predt(test_df,tree_car)

# tree_car = ID3(train_df,tree_depth=1)
# Predt_car = predt(train_df,tree_car)
# Predt_car_test = predt(test_df,tree_car)

# tree_car2 = ID3(train_df,tree_depth=2)
# Predt_car2 = predt(train_df,tree_car2)
# Predt_car_test2 = predt(test_df,tree_car2)

# tree_car3 = ID3(train_df,tree_depth=2)
# Predt_car3 = predt(train_df,tree_car3)
# Predt_car_test3 = predt(test_df,tree_car3)
# bank Data

# dep = [1, 2, 3, 4, 5, 6]
# train = []
# for i in dep:
#     tree= ID3(train_df, gain = 'gi', tree_depth = i)
#     train_accuracy = predt(train_df, tree)
#     print(tree)
#     train.append(train_accuracy)
#     print(train_accuracy)


# dep = [1, 2, 3, 4, 5, 6]
# train = []
# for i in dep:
#     tree= ID3(train_df, gain = 'gi', tree_depth = i)
#     train_accuracy = predt(test_df, tree)
#     print(tree)
#     train.append(train_accuracy)
#     print(train_accuracy)


# gain = ['S', 'me', 'gi']
# dep = [1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16]
# train = []
# for i in dep:
#     tree= ID3(train_bank, gain = 'gi', tree_depth = i)
#     train_accuracy = predt(train_bank, tree)
#     print(tree)
#     train.append(train_accuracy)
#     print(train_accuracy)


car_train_df = {}
gain = ['S', 'me','gi']
dep = [1, 2,3,4,5,6]
for g in gain:  
    train = []
    for i in dep:
        tree= ID3(train_bank, gain = g, tree_depth = i)
        train_accuracy = predt(test_bank, tree)
        train.append(train_accuracy)
    car_train_df[g] = train