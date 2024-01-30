# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:22:06 2023

@author: sopsla
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm

dpath = "K:/Project/Project 3 - Behavioural studies/1 - norming/data"
alldata = []

# %% path, participant & data loading
#for pp in [1,2,3,4,5]:

 #   fname = f'{dpath}/{pp}_norming_sheet_stiminfo_filled.xlsx'
  #  datafile = pd.ExcelFile(fname)
    
   # df1 = pd.read_excel(datafile, 'sheet1')
    #df2 = pd.read_excel(datafile, 'sheet2')
    #df3 = pd.read_excel(datafile, 'sheet3')
    #df4 = pd.read_excel(datafile, 'sheet4')
    
    #data = pd.concat([df1, df2, df3, df4])
    #data.reset_index(inplace=True)
    
    # convert x-es into numbers
    #rating = np.zeros(len(data))
    #for i,row in data.iterrows():
     #   for column in [1,2,3,4,5,6,7]:
      #      if isinstance(row[column], str):
       #         rating[i] = column
        #        continue
         #   else:
          #      pass
    
#    data['rating'] = rating
 #   data['participant'] = [pp] * len(data)
  #  alldata.append(data)
    
# %% create dataframe
#data = pd.concat(alldata)
#data.reset_index(inplace=True)

# drop irrelevant columns
#data = data.drop(['level_0', 'index','RANDOMIZATION', 'sentence', 1,2,3,4,5,6,7], axis=1)
#data = data.rename(columns={'Unnamed: 6': 'sentence'})

# %% save the dataframe
#data.to_csv(f'{dpath}/norming_data.csv')

# %% read the csv
data = pd.read_csv(f'{dpath}/norming_data.csv')
data = data.drop('Unnamed: 0', axis=1)

# %% outliers
mean = np.mean(data['rating'])
std = np.std(data['rating'])

outlier_range = [mean-2*std, mean+2*std]

# %% plotting the histograms
plt.hist(data.loc[data['cat_surprisal'] == 'high', 'rating'], bins=range(1,8),alpha=0.5)
plt.hist(data.loc[data['cat_surprisal'] == 'low', 'rating'], bins=range(1,8),alpha=0.5)
plt.legend(['High surprisal', 'Low surprisal'], frameon=False)

# %% means over participant per type
data_means = data.groupby(by=['set_number', 'cat_surprisal']).aggregate(mean=pd.NamedAgg('rating', np.mean), std=pd.NamedAgg('rating', np.std))
data_means.reset_index(inplace=True)
data_means = data_means.loc[data_means['cat_surprisal'] != 'none']

# %%
data_means_pp = data.groupby(by=['participant', 'cat_surprisal']).aggregate(mean=pd.NamedAgg('rating', np.mean), std=pd.NamedAgg('rating', np.std))
data_means_pp.reset_index(inplace=True)
data_means_pp = data_means_pp.loc[data_means_pp['cat_surprisal'] != 'none']

# %% plot fast
sns.histplot(data_means, x='mean', hue='cat_surprisal')
plt.legend(['Low surprisal', 'High surprisal'], frameon=False)
sns.despine()
plt.tight_layout()

# %% let's take just the means that are below 4
low_probability_sets = set(data_means.loc[data_means['mean'] <= 4.0, 'set_number'])

# %%
plausible = []
implausible = []
mean_of_4 = []

for set_number in set(data['set_number']):
    for cat_surprisal in ['high', 'low']:
        ratings = data.loc[(data['set_number'] == set_number) & (data['cat_surprisal'] == cat_surprisal)]
        #fig,ax=plt.subplots()
        #sns.histplot(data=ratings, x='rating', hue='participant', bins=range(0,8), ax=ax)
        #ax.set_title(f'{set_number}, {cat_surprisal}')
        
        ratings=ratings['rating']
        ttest = stats.ttest_1samp(ratings, popmean = 4.0)
        if ttest.pvalue < 0.05 and ttest.statistic > 0:
            print(f'{set_number}, {cat_surprisal}: mean = {np.mean(ratings)}; t-value = {ttest.statistic}, p-value = {ttest.pvalue}')
            plausible.append((set_number, cat_surprisal))
        elif ttest.pvalue < 0.05 and ttest.statistic < 0:
            implausible.append((set_number, cat_surprisal))
        else:
            mean_of_4.append((set_number, cat_surprisal))

# %% check all plausible ones that are higher than 3.5
high_surp_plausible_idx = [p[0] for p in plausible if p[1] == 'high']

# select only those
data_plausible = data.loc[data['set_number'].isin(high_surp_plausible_idx)]

# %% now we plot those
sns.histplot(data_plausible, x='rating', hue='cat_surprisal')
plt.legend(['Low surprisal', 'High surprisal'], frameon=False)
sns.despine()
plt.tight_layout()

# %% let's do a t-test 
ttest = stats.ttest_rel(data_plausible.loc[data_plausible['cat_surprisal'] == 'high', 'rating'],
                        data_plausible.loc[data_plausible['cat_surprisal'] == 'low', 'rating'])

# findings: ttest
#Out[52]: Ttest_relResult(statistic=-3.7200965831339703, pvalue=0.00028150103864882386)

#np.mean(data_plausible.loc[data_plausible['cat_surprisal'] == 'high', 'rating'])
#Out[53]: 6.12

#np.mean(data_plausible.loc[data_plausible['cat_surprisal'] == 'low', 'rating'])
#Out[54]: 6.493333333333333

# High surprisal is a bit less plausible than low surprisal, but pretty close

