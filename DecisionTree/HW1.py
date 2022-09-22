import scipy
import pandas as pd
import numpy as np
import os

os.chdir(r'/Users/ramseyissa/Courses/Machine Learning/MLCourse/DecisionTree')
data_path = os.path.join('train.csv')
data = pd.read_csv(data_path)

train_df = pd.read_csv('train.csv')
train_df

#gather unique elements of labels 
label_lst = train_df['label'].unique().tolist()
label_name = train_df.columns[-1]
#set up calculations for Information Gain
def IG(train_df, label_name, label_lst):
    #gather total rows in df 
    tot_rows = train_df.shape[0]
    #set S_tot to zero and begin iteration
    S_tot = 0
    for item in label_lst:
        
    
    # info_gain = -p_*np.log2(p) - p_*np.log2(p)
    
# def ME()