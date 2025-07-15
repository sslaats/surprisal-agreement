# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:13:07 2023

@author: sopsla
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

#%%
dpath = 'C:/Users/slaats/Documents/MPI/Project/Project 3 - Behavioural studies/5 - metadata'
fname = 'frequency_length_position.csv'

stimuli = pd.read_csv(os.path.join(dpath, fname))

# %% plot high vs low surprisal
fig,ax=plt.subplots()
ax.hist(stimuli.loc[(stimuli['cat_surprisal'] == 'high') & (stimuli['agreement'] == 'correct'), 'surprisal_value'], alpha=0.6)
ax.hist(stimuli.loc[(stimuli['cat_surprisal'] == 'low') & (stimuli['agreement'] == 'correct'), 'surprisal_value'],alpha=0.6)
ax.legend(['High surprisal', 'Low surprisal'], frameon=False)
ax.set_ylabel('count')
ax.set_xlabel('surprisal (bits)')

#%% plot all together
fig,axes = plt.subplots(nrows=2, ncols=2, figsize=(8,6), sharey=True)

ax = axes[0,0]
bins = np.arange(3, 30, 1)

ax.hist(stimuli.loc[(stimuli['cat_surprisal'] == 'low') & (stimuli['agreement'] == 'correct') & (stimuli['word_index'] == 5), 'surprisal_value'], color='green', edgecolor='darkgreen',alpha=0.6, bins=bins)
ax.hist(stimuli.loc[(stimuli['cat_surprisal'] == 'high') & (stimuli['agreement'] == 'correct') & (stimuli['word_index'] == 5), 'surprisal_value'], color='green',hatch='///', edgecolor='black', alpha=0.6, bins=bins)

ax.hist(stimuli.loc[(stimuli['cat_surprisal'] == 'low') & (stimuli['agreement'] == 'incorrect') & (stimuli['word_index'] == 5), 'surprisal_value'], color='red', edgecolor='darkred', alpha=0.6, bins=bins)
ax.hist(stimuli.loc[(stimuli['cat_surprisal'] == 'high') & (stimuli['agreement'] == 'incorrect') & (stimuli['word_index'] == 5), 'surprisal_value'], color='red',hatch='///', edgecolor='black', alpha=0.6, bins=bins)

ax.set_ylabel('Count')
ax.set_xlabel('Surprisal (bits)')
ax.set_title('Surprisal per condition')
ax.text(-1, 15, 'A', fontsize=14)

ax = axes[0,1]
bins = np.arange(1, 7, 0.2)

ax.hist(stimuli.loc[(stimuli['cat_surprisal'] == 'low') & (stimuli['agreement'] == 'correct') & (stimuli['word_index'] == 5), 'word_frequency'], color='green', edgecolor='darkgreen', alpha=0.6, bins=bins)
ax.hist(stimuli.loc[(stimuli['cat_surprisal'] == 'high') & (stimuli['agreement'] == 'correct') & (stimuli['word_index'] == 5), 'word_frequency'], color='green', hatch='///', edgecolor='black', alpha=0.6, bins=bins)

ax.hist(stimuli.loc[(stimuli['cat_surprisal'] == 'low') & (stimuli['agreement'] == 'incorrect') & (stimuli['word_index'] == 5), 'word_frequency'], color='red', edgecolor='darkred',alpha=0.6, bins=bins)
ax.hist(stimuli.loc[(stimuli['cat_surprisal'] == 'high') & (stimuli['agreement'] == 'incorrect') & (stimuli['word_index'] == 5), 'word_frequency'], color='red', hatch='///', edgecolor='black', alpha=0.6, bins=bins)

ax.legend(['Low surp/Correct', 'High surp/Correct',
           'Low surp/Incorrect',  'High surp/Incorrect'], frameon=False)
ax.set_ylabel('Count')
ax.set_xlabel('Word frequency (Zipf)')
ax.set_title('Word frequency per condition')
ax.text(0, 15, 'B', fontsize=14)

ax = axes[1,0]
bins = np.arange(3, 30, 1)
ax.hist(stimuli.loc[(stimuli['list'] == 1)&(stimuli['word_index'] == 5), 'surprisal_value'], color=sns.color_palette('muted')[3], edgecolor=sns.color_palette('dark')[3], alpha=0.6, bins=bins)
ax.hist(stimuli.loc[(stimuli['list'] == 2)&(stimuli['word_index'] == 5), 'surprisal_value'], color=sns.color_palette('muted')[2], edgecolor=sns.color_palette('dark')[2], alpha=0.6, bins=bins)
ax.hist(stimuli.loc[(stimuli['list'] == 3)&(stimuli['word_index'] == 5), 'surprisal_value'], color=sns.color_palette('muted')[1], edgecolor=sns.color_palette('dark')[1], alpha=0.6, bins=bins)
ax.hist(stimuli.loc[(stimuli['list'] == 4)&(stimuli['word_index'] == 5), 'surprisal_value'], color=sns.color_palette('muted')[0], edgecolor=sns.color_palette('dark')[0], alpha=0.6, bins=bins)

ax.set_ylabel('Count')
ax.set_xlabel('Surprisal (bits)')
ax.set_title('Surprisal per list')
ax.text(-1, 15, 'C', fontsize=14)

ax = axes[1,1]
bins = np.arange(1, 7, 0.2)
ax.hist(stimuli.loc[(stimuli['list'] == 1)&(stimuli['word_index'] == 5), 'word_frequency'], color=sns.color_palette('muted')[3], edgecolor=sns.color_palette('dark')[3], alpha=0.6, bins=bins)
ax.hist(stimuli.loc[(stimuli['list'] == 2)&(stimuli['word_index'] == 5), 'word_frequency'], color=sns.color_palette('muted')[2], edgecolor=sns.color_palette('dark')[2], alpha=0.6, bins=bins)
ax.hist(stimuli.loc[(stimuli['list'] == 3)&(stimuli['word_index'] == 5), 'word_frequency'], color=sns.color_palette('muted')[1], edgecolor=sns.color_palette('dark')[1], alpha=0.6, bins=bins)
ax.hist(stimuli.loc[(stimuli['list'] == 4)&(stimuli['word_index'] == 5), 'word_frequency'], color=sns.color_palette('muted')[0], edgecolor=sns.color_palette('dark')[0], alpha=0.6, bins=bins)
ax.legend(['List 1', 'List 2', 'List 3', 'List 4'], frameon=False)
ax.set_ylabel('Count')
ax.set_xlabel('Word frequency (Zipf)')
ax.set_title('Word frequency per list')
ax.text(0, 15, 'D', fontsize=14)

plt.tight_layout()
sns.despine()

#fig.savefig(f'{dpath}/stimulus_statistics.svg')

# %% ttests wf and surprisal

# two-way anova for surprisal and wf
import statsmodels.api as sm
from statsmodels.formula.api import ols

surp_model = ols('surprisal_value ~ C(agreement) + C(cat_surprisal) + C(agreement):C(cat_surprisal)', data=stimuli.loc[stimuli['word_index'] == 5]).fit()
surp_result = sm.stats.anova_lm(surp_model, type=3)
print(surp_result)

# %% print the means
surpmeans = stimuli.loc[stimuli['word_index']==5].groupby(by=['agreement', 'cat_surprisal']).aggregate(mean=pd.NamedAgg('surprisal_value', np.mean), std=pd.NamedAgg('surprisal_value', np.std))

# %% wf
wf_model = ols('word_frequency ~ C(agreement) + C(cat_surprisal) + C(agreement):C(cat_surprisal)', data=stimuli.loc[stimuli['word_index'] == 5]).fit()
wf_result = sm.stats.anova_lm(wf_model, type=2)
print(wf_result)


# %% test differences between lists
# no surprisal differences between lists
print(stats.f_oneway(stimuli.loc[(stimuli['list'] == 1)&(stimuli['word_index'] == 5), 'surprisal_value'],
                                 stimuli.loc[(stimuli['list'] == 2)&(stimuli['word_index'] == 5), 'surprisal_value'],
                                 stimuli.loc[(stimuli['list'] == 3)&(stimuli['word_index'] == 5), 'surprisal_value'],
                                 stimuli.loc[(stimuli['list'] == 4)&(stimuli['word_index'] == 5), 'surprisal_value']))

# no word frequency differences between lists
print(stats.f_oneway(stimuli.loc[(stimuli['list'] == 1)&(stimuli['word_index'] == 5), 'word_frequency'],
                                 stimuli.loc[(stimuli['list'] == 2)&(stimuli['word_index'] == 5), 'word_frequency'],
                                 stimuli.loc[(stimuli['list'] == 3)&(stimuli['word_index'] == 5), 'word_frequency'],
                                 stimuli.loc[(stimuli['list'] == 4)&(stimuli['word_index'] == 5), 'word_frequency']))

#%% t-tests [old]
print(stats.ttest_ind(stimuli.loc[(stimuli['cat_surprisal'] == 'high')&(stimuli['word_index'] == 5), 'word_frequency'] ,
                                  stimuli.loc[(stimuli['cat_surprisal'] == 'low') & (stimuli['word_index'] == 5), 'word_frequency']))
# no difference

print(stats.ttest_ind(stimuli.loc[(stimuli['cat_surprisal'] == 'high')&(stimuli['word_index'] == 5), 'surprisal_value'] ,
                                  stimuli.loc[(stimuli['cat_surprisal'] == 'low') & (stimuli['word_index'] == 5), 'surprisal_value']))
# very significant
# %% separate plots 
# lists
fig,ax=plt.subplots(figsize=(6,4))
bins = np.arange(3, 30, 1)
ax.hist(stimuli.loc[(stimuli['list'] == 1)&(stimuli['word_index'] == 5), 'surprisal_value'], color=sns.color_palette('muted')[3], edgecolor=sns.color_palette('dark')[3], alpha=0.6, bins=bins)
ax.hist(stimuli.loc[(stimuli['list'] == 2)&(stimuli['word_index'] == 5), 'surprisal_value'], color=sns.color_palette('muted')[2], edgecolor=sns.color_palette('dark')[2], alpha=0.6, bins=bins)
ax.hist(stimuli.loc[(stimuli['list'] == 3)&(stimuli['word_index'] == 5), 'surprisal_value'], color=sns.color_palette('muted')[1], edgecolor=sns.color_palette('dark')[1], alpha=0.6, bins=bins)
ax.hist(stimuli.loc[(stimuli['list'] == 4)&(stimuli['word_index'] == 5), 'surprisal_value'], color=sns.color_palette('muted')[0], edgecolor=sns.color_palette('dark')[0], alpha=0.6, bins=bins)
ax.legend(['List 1', 'List 2', 'List 3', 'List 4'], frameon=False)
ax.set_ylabel('count')
ax.set_xlabel('surprisal (bits)')
plt.tight_layout()
sns.despine()
fig.savefig(f'{dpath}/surprisal_per_list.svg')

fig,ax=plt.subplots(figsize=(6,4))
bins = np.arange(1, 7, 0.2)
ax.hist(stimuli.loc[(stimuli['list'] == 1)&(stimuli['word_index'] == 5), 'word_frequency'], color=sns.color_palette('muted')[3], edgecolor=sns.color_palette('dark')[3], alpha=0.6, bins=bins)
ax.hist(stimuli.loc[(stimuli['list'] == 2)&(stimuli['word_index'] == 5), 'word_frequency'], color=sns.color_palette('muted')[2], edgecolor=sns.color_palette('dark')[2], alpha=0.6, bins=bins)
ax.hist(stimuli.loc[(stimuli['list'] == 3)&(stimuli['word_index'] == 5), 'word_frequency'], color=sns.color_palette('muted')[1], edgecolor=sns.color_palette('dark')[1], alpha=0.6, bins=bins)
ax.hist(stimuli.loc[(stimuli['list'] == 4)&(stimuli['word_index'] == 5), 'word_frequency'], color=sns.color_palette('muted')[0], edgecolor=sns.color_palette('dark')[0], alpha=0.6, bins=bins)
ax.legend(['List 1', 'List 2', 'List 3', 'List 4'], frameon=False)
ax.set_ylabel('count')
ax.set_xlabel('Word frequency (Zipf)')
plt.tight_layout()
sns.despine()
fig.savefig(f'{dpath}/WF_per_list.svg')

# surprisal
fig,ax=plt.subplots(figsize=(6,4))
bins = np.arange(3, 30, 1)

ax.hist(stimuli.loc[(stimuli['cat_surprisal'] == 'low') & (stimuli['agreement'] == 'correct') & (stimuli['word_index'] == 5), 'surprisal_value'], color='green', edgecolor='darkgreen',alpha=0.6, bins=bins)
ax.hist(stimuli.loc[(stimuli['cat_surprisal'] == 'low') & (stimuli['agreement'] == 'incorrect') & (stimuli['word_index'] == 5), 'surprisal_value'], color='red', edgecolor='darkred', alpha=0.6, bins=bins)

ax.hist(stimuli.loc[(stimuli['cat_surprisal'] == 'high') & (stimuli['agreement'] == 'correct') & (stimuli['word_index'] == 5), 'surprisal_value'], color='green',hatch='///', edgecolor='black', alpha=0.6, bins=bins)
ax.hist(stimuli.loc[(stimuli['cat_surprisal'] == 'high') & (stimuli['agreement'] == 'incorrect') & (stimuli['word_index'] == 5), 'surprisal_value'], color='red',hatch='///', edgecolor='black', alpha=0.6, bins=bins)

ax.legend(['Low surprisal [correct]', 'Low surprisal [incorrect]',
           'High surprisal [correct]',  'High surprisal [incorrect]'], frameon=False)
ax.set_ylabel('Count')
ax.set_xlabel('Surprisal (bits)')
plt.tight_layout()
sns.despine()
fig.savefig(f'{dpath}/surprisal_per_condition.svg')

# word frequency
fig,ax=plt.subplots(figsize=(6,4))
bins = np.arange(1, 7, 0.2)

ax.hist(stimuli.loc[(stimuli['cat_surprisal'] == 'low') & (stimuli['agreement'] == 'correct') & (stimuli['word_index'] == 5), 'word_frequency'], color='green', edgecolor='darkgreen', alpha=0.6, bins=bins)
ax.hist(stimuli.loc[(stimuli['cat_surprisal'] == 'low') & (stimuli['agreement'] == 'incorrect') & (stimuli['word_index'] == 5), 'word_frequency'], color='red', edgecolor='darkred',alpha=0.6, bins=bins)

ax.hist(stimuli.loc[(stimuli['cat_surprisal'] == 'high') & (stimuli['agreement'] == 'correct') & (stimuli['word_index'] == 5), 'word_frequency'], color='green', hatch='///', edgecolor='black', alpha=0.6, bins=bins)
ax.hist(stimuli.loc[(stimuli['cat_surprisal'] == 'high') & (stimuli['agreement'] == 'incorrect') & (stimuli['word_index'] == 5), 'word_frequency'], color='red', hatch='///', edgecolor='black', alpha=0.6, bins=bins)

ax.legend(['Low surprisal [correct]', 'Low surprisal [incorrect]',
           'High surprisal [correct]',  'High surprisal [incorrect]'], frameon=False)
ax.set_ylabel('count')
ax.set_xlabel('Word frequency (Zipf)')

plt.tight_layout()
sns.despine()
fig.savefig(f'{dpath}/WF_per_condition.svg')
