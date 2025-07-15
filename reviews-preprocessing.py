# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:37:15 2023

@author: sopsla
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import statsmodels
import scipy.stats as stats
import seaborn as sns

import patsy
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf

# %% get the stimulus information
stim_info = pd.read_csv('//nasac-faculty.isis.unige.ch/MEDECINE_HOME_PAPAT/Neufo/slaats/MPI/Project 3 - Behavioural studies/Project 3 - Behavioural studies/5 - metadata/frequency_length_position.csv')

# %% load saved data produced with extract_RTs_stimuluswise_errorcorrection
#RTs = pd.read_csv('K:/Project/Project 3 - Behavioural studies/3 - SPR main/analysis/readingtimes_fillers_88pp.csv')
RTs = pd.read_csv('//nasac-faculty.isis.unige.ch/MEDECINE_HOME_PAPAT/Neufo/slaats/MPI/Project 3 - Behavioural studies/Project 3 - Behavioural studies/3 - SPR main/analysis/readingtimes_88pp.csv')

# %% preprocessing
# store original length
data_length = len(RTs)

RTs = RTs[~RTs['RT'].isna()] # remove NaN values - always check if this happened
RTs['RT'] = pd.to_numeric(RTs['RT'])

print(f'Removed: {(data_length-len(RTs))/data_length}')

RTs = RTs[~(RTs['RT'] > 2500.0)]
RTs = RTs[~(RTs['RT'] < 100.0)]

print(f'Removed: {(data_length-len(RTs))/data_length}')

# log transform (natural log)
RTs['logRT'] = np.log(RTs['RT'])

plt.hist(RTs['logRT'])

#%% outlier rejection on the basis of the general mean and std per participant
# SKIP THIS FOR REVIEWS

RTs['max_values'] = np.zeros(len(RTs))
RTs['min_values'] = np.zeros(len(RTs))

# take mean per participant, remove all that are 2.5 sds away from that
for pp in RTs['UserId'].unique():
    
    pp_dat = RTs.loc[RTs['UserId'] == pp]
    
    # sd & mean
    sd = np.std(pp_dat['logRT'])
    mean = np.mean(pp_dat['logRT'])
    
    # get maximum value
    max_value = mean + 2.5 * sd
    min_value = mean - 2.5 * sd
    
    # add the limits to the dataframe
    RTs.loc[RTs['UserId'] == pp, 'max_values'] = [max_value] * len(RTs.loc[RTs['UserId'] == pp])
    RTs.loc[RTs['UserId'] == pp, 'min_values'] = [min_value] * len(RTs.loc[RTs['UserId'] == pp])
    
    
# %% here we remove all the values that are above or below the outlier values
RTs['outlier'] = (RTs['logRT'] > RTs['max_values']) | (RTs['logRT'] < RTs['min_values'])
RTs = RTs.loc[~RTs['outlier']]
plt.hist(RTs['logRT'])

# %%
# %%
print(f'Removed: {(data_length-len(RTs))/data_length}')
# in total, 2.8 % of the data was removed - still holds

#%%
RTs.to_csv('//nasac-faculty.isis.unige.ch/MEDECINE_HOME_PAPAT/Neufo/slaats/MPI/Project 3 - Behavioural studies/Project 3 - Behavioural studies/3 - SPR main/analysis/readingtimes_88pp_preprocessed-outliers-REVIEWS.csv')

# %% residual log reading times in LMM
base = smf.ols('logRT ~ word_frequency + word_length + sentence_order', RTs) #  + '
basef = base.fit()

RTs['residuals'] = basef.resid

# Covariates: word length, word frequency, sentence number (position in stimulus list)
# Random slope per participant
# Model 1: continuous surprisal values + agreement condition (to check for main effects - pilot)
# Model 2: surprisal category * agreement
# Model 3: continuous surprisal values * agreement


# %% inspect results per subject
mpw_pp = RTs.groupby(by=['agreement', 'surprisal', 'word_index', 'UserId']).aggregate(mean=pd.NamedAgg('RT', np.mean), 
                     std=pd.NamedAgg('RT', np.std), logmean=pd.NamedAgg('logRT', np.mean), logstd=pd.NamedAgg('logRT', np.std),
                     residmean=pd.NamedAgg('residuals', np.mean), residstd=pd.NamedAgg('residuals', np.std))

mpw_pp.reset_index(inplace=True)
mpw_pp['word_index'] = mpw_pp['word_index'].astype(int)

#%% plot
for participant in set(mpw_pp['UserId']):
    print(participant)
    pp_data = mpw_pp.loc[mpw_pp['UserId'] == participant]
    
    fig,ax = plt.subplots(figsize=(9,6))

    corr_high = pp_data.loc[(pp_data['agreement'] == 'correct') & (pp_data['surprisal'] == 'high') & (pp_data['word_index'] > 3)]
    corr_low = pp_data.loc[(pp_data['agreement'] == 'correct') & (pp_data['surprisal'] == 'low') & (pp_data['word_index'] > 3)]
    incorr_high = pp_data.loc[(pp_data['agreement'] == 'incorrect') & (pp_data['surprisal'] == 'high') & (pp_data['word_index'] > 3)]
    incorr_low = pp_data.loc[(pp_data['agreement'] == 'incorrect') & (pp_data['surprisal'] == 'low') & (pp_data['word_index'] > 3)]
    
    ax.plot(range(4,10), corr_high['mean'], color='green', alpha=1)
    ax.plot(range(4,10), corr_low['mean'], color='green', linestyle='dashed', alpha=1)
    ax.plot(range(4,10), incorr_high['mean'], color='red', alpha=1)
    ax.plot(range(4,10), incorr_low['mean'], color='red', linestyle='dashed', alpha=1)
    
    plt.legend(['Correct/High', 'Correct/Low', 'Incorrect/High', 'Incorrect/Low'], frameon=False)
    
    ax.fill_between(range(4,10), corr_high['mean']-corr_high['std'], corr_high['mean']+corr_high['std'], color='green', alpha=0.2)
    ax.fill_between(range(4,10), corr_low['mean']-corr_low['std'], corr_low['mean']+corr_low['std'], color='green', hatch='/', alpha=0.2)
    ax.fill_between(range(4,10), incorr_high['mean']-incorr_high['std'], incorr_high['mean']+incorr_high['std'], color='red', alpha=0.2)
    ax.fill_between(range(4,10), incorr_low['mean']-incorr_low['std'], incorr_low['mean']+incorr_low['std'], color='red', hatch='/', alpha=0.2)
    
    ax.axvline(5, color='grey', zorder=0, linestyle='--')     
    
    ax.set_xlabel('Word position')
    ax.set_ylabel('RT (ms)')
    ax.set_title(participant)
    sns.despine()

# %% plot RTs per word (mpw = means per word)
mpw = mpw_pp.groupby(by=['agreement', 'surprisal', 'word_index']).aggregate(mean=pd.NamedAgg('mean', np.mean), std=pd.NamedAgg('mean', np.std),
                     logmean=pd.NamedAgg('logmean', np.mean), logstd=pd.NamedAgg('logmean', np.std),
                     residmean=pd.NamedAgg('residmean', np.mean), residstd=pd.NamedAgg('residmean', np.std))
mpw.reset_index(inplace=True)
mpw['word_index'] = mpw['word_index'].astype(int)

# %% plot the averages
fig,ax = plt.subplots(figsize=(9,6))

corr_high = mpw.loc[(mpw['agreement'] == 'correct') & (mpw['surprisal'] == 'high') & (mpw['word_index'] > 3)]
corr_low = mpw.loc[(mpw['agreement'] == 'correct') & (mpw['surprisal'] == 'low') & (mpw['word_index'] > 3)]
incorr_high = mpw.loc[(mpw['agreement'] == 'incorrect') & (mpw['surprisal'] == 'high') & (mpw['word_index'] > 3)]
incorr_low = mpw.loc[(mpw['agreement'] == 'incorrect') & (mpw['surprisal'] == 'low') & (mpw['word_index'] > 3)]

ax.plot(range(4,10), corr_high['mean'], color='green', alpha=1)
ax.plot(range(4,10), corr_low['mean'], color='green', linestyle='dashed', alpha=1)
ax.plot(range(4,10), incorr_high['mean'], color='red', alpha=1)
ax.plot(range(4,10), incorr_low['mean'], color='red', linestyle='dashed', alpha=1)

plt.legend(['Correct/High', 'Correct/Low', 'Incorrect/High', 'Incorrect/Low'], frameon=False)

ax.fill_between(range(4,10), corr_high['mean']-corr_high['std'], corr_high['mean']+corr_high['std'], color='green', alpha=0.2)
ax.fill_between(range(4,10), corr_low['mean']-corr_low['std'], corr_low['mean']+corr_low['std'], color='green', hatch='/', alpha=0.2)
ax.fill_between(range(4,10), incorr_high['mean']-incorr_high['std'], incorr_high['mean']+incorr_high['std'], color='red', alpha=0.2)
ax.fill_between(range(4,10), incorr_low['mean']-incorr_low['std'], incorr_low['mean']+incorr_low['std'], color='red', hatch='/', alpha=0.2)

ax.axvline(5, color='grey', zorder=0, linestyle='--')     

ax.set_xlabel('Word position')
ax.set_ylabel('RT (ms)')
sns.despine()


# %% do this for the log-transformed
fig,ax = plt.subplots(figsize=(9,6))

ax.plot(range(4,10), corr_high['logmean'], color='green', alpha=1)
ax.plot(range(4,10), corr_low['logmean'], color='green', linestyle='dashed', alpha=1)
ax.plot(range(4,10), incorr_high['logmean'], color='red', alpha=1)
ax.plot(range(4,10), incorr_low['logmean'], color='red', linestyle='dashed', alpha=1)

plt.legend(['Correct/High', 'Correct/Low', 'Incorrect/High', 'Incorrect/Low'], frameon=False)

ax.fill_between(range(4,10), corr_high['logmean']-corr_high['logstd'], corr_high['logmean']+corr_high['logstd'], color='green', alpha=0.2)
ax.fill_between(range(4,10), corr_low['logmean']-corr_low['logstd'], corr_low['logmean']+corr_low['logstd'], color='green', hatch='/', alpha=0.2)
ax.fill_between(range(4,10), incorr_high['logmean']-incorr_high['logstd'], incorr_high['logmean']+incorr_high['logstd'], color='red', alpha=0.2)
ax.fill_between(range(4,10), incorr_low['logmean']-incorr_low['logstd'], incorr_low['logmean']+incorr_low['logstd'], color='red', hatch='/', alpha=0.2)

ax.axvline(5, color='grey', zorder=0, linestyle='--')     

ax.set_xlabel('Word position')
ax.set_ylabel('logRT')
sns.despine()


# %% and for the residuals
fig,ax = plt.subplots(figsize=(9,6))

ax.plot(range(4,10), corr_high['residmean'], color='green', alpha=1)
ax.plot(range(4,10), corr_low['residmean'], color='green', linestyle='dashed', alpha=1)
ax.plot(range(4,10), incorr_high['residmean'], color='red', alpha=1)
ax.plot(range(4,10), incorr_low['residmean'], color='red', linestyle='dashed', alpha=1)

plt.legend(['Correct/High', 'Correct/Low', 'Incorrect/High', 'Incorrect/Low'], frameon=False)

ax.fill_between(range(4,10), corr_high['residmean']-corr_high['residstd'], corr_high['residmean']+corr_high['residstd'], color='green', alpha=0.2)
ax.fill_between(range(4,10), corr_low['residmean']-corr_low['residstd'], corr_low['residmean']+corr_low['residstd'], color='green', hatch='/', alpha=0.2)
ax.fill_between(range(4,10), incorr_high['residmean']-incorr_high['residstd'], incorr_high['residmean']+incorr_high['residstd'], color='red', alpha=0.2)
ax.fill_between(range(4,10), incorr_low['residmean']-incorr_low['residstd'], incorr_low['residmean']+incorr_low['residstd'], color='red', hatch='/', alpha=0.2)

ax.axvline(5, color='grey', zorder=0, linestyle='--')     

ax.set_xlabel('Word position')
ax.set_ylabel('logRT')
sns.despine()

