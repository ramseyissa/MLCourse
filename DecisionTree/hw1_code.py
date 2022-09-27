import numpy as np
from statistics import mode
import pandas as pd



#this work was done incollaboration with Hasan Sayeed PhD candidate MSE
#sources used to produce this work were lecture slides, stackoverflow, github, towards data science
#articles  




train_car= pd.read_csv('DecisionTree/train.csv')
test_car = pd.read_csv('DecisionTree/test.csv')
train_bank = pd.read_csv('DecisionTree/trainbank.csv')
test_bank = pd.read_csv('DecisionTree/testbank.csv')






#entropy of dataset calculation
def s_tot(data):
  #get data label col name
  data_label = data.columns[-1]
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
  data_label = data.columns[-1]
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
  data_lbl = data.columns[-1]
  #get unique vals
  labl_uni, lbl_tot = np.unique(data[data_lbl], return_counts = True)  
  #get max
  for i in lbl_tot:
    cnt_vals = np.amax(lbl_tot)
    probl = cnt_vals/len(data[data_lbl])
    me_tot = 1 - probl
  return np.float64(me_tot)


#calc me of att
def me_att(data, attr):
  #get label col name
  label = data.columns[-1]
  #get uniq vals
  uniqu_vals = data[attr].unique()
  m_e = 0
  for vals_ in uniqu_vals:
    #get uniq vals for att
    att_cntr, labl_cnt = np.unique(data[label][data[attr] == vals_], return_counts = True)
    #get max val
    for i in labl_cnt:
      cnt_labl = np.amax(labl_cnt)
      tot = len(data[attr][data[attr] == vals_])
      prob = cnt_labl/tot
      me = 1 - prob
      m_e = m_e + (tot/len(data))*me
  return np.float64(m_e)


#get gini index of dataset
def gini_tot(data):
  data_label = data.columns[-1]
  unique_labl = data[data_label].unique()    
  gi_tot = 1
  for i in unique_labl:
    prob = data[data_label].value_counts()[i]/len(data[data_label])
    gi_tot = gi_tot + -prob**2
  return np.float64(gi_tot)

#get gini att
def gini_attribs(data, att):
  label = data.columns[-1]
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
  else:
    pass


#pop df and index
def popu_df(data, att, val):
  return data[data[att] == val].reset_index(drop = True)


#ID3 
def ID3(data, tree = None, gain = 'S', tree_depth=50):
  #first use inf gain specified in gain
  node = inf_gain(data, gain)
  #get uniq vals               
  cnt_att = np.unique(data[node])
  #get label name 
  data_label = data.keys()[-1]
  #set dep to zero
  deph = 0
  if tree is None:
    # int tree dic
    tree = {}
    tree[node] = {}
  for att_cnt in cnt_att:           
    updated_data = popu_df(data,node,att_cnt)
    #get uniq lbl cnts
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
      else:
        pass
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



#predict f(x) 
def predt(data, tree):
  label_predicted = []
  error = []
  # prd_error = []
  for i in range(len(data)):
    inst = data.iloc[i,:]
    prediction = predt_ins_tree(inst, tree)
    label_predicted.append(prediction)
    #add 1 for error 0 for non-error
    if prediction == data[data.columns[-1]][i]:
      error.append(0)
    elif prediction != data[data.columns[-1]][i]:
      error.append(1)
    else:
      pass
      #pred err 
  pred_error = sum(error)/len(data)
  return pred_error



car_train_df = {}
gain = ['S', 'me','gi']
dep = [1,2,3,4,5,6]
for g in gain:  
    train = []
    for i in dep:
        tree= ID3(train_car, gain = g, tree_depth = i)
        # true_val = ID3(train_bank, gain = g, tree_depth = i)
        true_val = predt(train_car, tree)
        pred_acc = predt(test_car, tree)
        # pred_err = ((true_val -pred_acc )/true_val)*100
        train.append(pred_acc)
    car_train_df[g] = train

df_2b = pd.DataFrame.from_dict(car_train_df)
print(df_2b)


bank_train_df = {}
gain = ['S', 'me','gi']
dep = [1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16]
# dep = [1, 2, 3, 4]
for g in gain:  
    train = []
    for i in dep:
        tree= ID3(train_bank, gain = g, tree_depth = i)
        # true_val = ID3(train_bank, gain = g, tree_depth = i)
        true_val = predt(train_bank, tree)
        pred_acc = predt(test_bank, tree)
        # pred_err = ((true_val -pred_acc )/true_val)*100
        train.append(pred_acc)
    bank_train_df[g] = train
df_3a = pd.DataFrame.from_dict(bank_train_df)
print(df_3a)



#replace items in col with majority val.
#identify the top value in each col

# col_names = train_bank.keys()
# for i in col_names:
#     print(train_bank[i].mode().tolist())
    
#replace poutcome col with second best because unknown is most commmon
#issues with forloop to replace unknown values 

train_bank['age'] = train_bank['age'].replace('unknown','32')
train_bank['job'] = train_bank['job'].replace('unknown','blue-collar')
train_bank['marital'] = train_bank['marital'].replace('unknown','married')
train_bank['education'] = train_bank['education'].replace('unknown','secondary')
train_bank['default'] = train_bank['default'].replace('unknown','no')
train_bank['balance'] = train_bank['balance'].replace('unknown','0')
train_bank['housing'] = train_bank['housing'].replace('unknown','yes')
train_bank['loan'] = train_bank['loan'].replace('unknown','no')
train_bank['contact'] = train_bank['contact'].replace('unknown','cellular')
train_bank['day'] = train_bank['day'].replace('unknown','20')
train_bank['month'] = train_bank['month'].replace('unknown','may')
train_bank['duration'] = train_bank['duration'].replace('unknown','85')
train_bank['campaign'] = train_bank['campaign'].replace('unknown','1')
train_bank['pdays'] = train_bank['pdays'].replace('unknown','-1')
train_bank['previous'] = train_bank['previous'].replace('unknown','0')
train_bank['poutcome'] = train_bank['poutcome'].replace('unknown','failure')
train_bank['y'] = train_bank['y'].replace('unknown','no')

bank_train_df_16 = {}
gain = ['S', 'me','gi']
# dep = [1, 2, 3, 4]
dep = [1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16]
for g in gain:  
    train = []
    for i in dep:
        tree= ID3(train_bank, gain = g, tree_depth = i)
        # true_val = ID3(train_bank, gain = g, tree_depth = i)
        true_val = predt(train_bank, tree)
        pred_acc = predt(test_bank, tree)
        # pred_err = ((true_val -pred_acc )/true_val)*100
        train.append(pred_acc)
    bank_train_df_16[g] = train
df_3b = pd.DataFrame.from_dict(bank_train_df_16)
print(df_3b)

