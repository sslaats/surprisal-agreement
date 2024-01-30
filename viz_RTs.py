# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 16:38:22 2023

@author: sopsla
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import scipy.stats as stats

# %% load the new data
resultsdir = 'K:/Project/Project 3 - Behavioural studies/3 - SPR main/analysis/'
RTs = pd.read_csv('K:/Project/Project 3 - Behavioural studies/3 - SPR main/analysis/readingtimes_88pp_residuals_manual.csv')

def get_interval(x):
    return stats.norm.interval(alpha=0.95, loc=np.mean(x), scale=stats.sem(x))

# %% create a predictor for number on the noun
RTs['noun_number'] = np.zeros(len(RTs))
RTs['noun_number'][np.where(((RTs['agreement'] == 'correct')&(RTs['correct_number']=='singular'))| 
                                  (RTs['agreement'] == 'incorrect')&(RTs['correct_number']=='plural'))[0]] = 'singular'
RTs['noun_number'][np.where((RTs['noun_number'] == 0))[0]] = 'plural'   

# %% inspect results per participant
mpw_pp = RTs.groupby(by=['agreement', 'surprisal', 'word_index', 'UserId']).aggregate(mean=pd.NamedAgg('RT', np.mean), 
                     std=pd.NamedAgg('RT', np.std), logmean=pd.NamedAgg('logRT', np.mean), logstd=pd.NamedAgg('logRT', np.std),
                     residmean=pd.NamedAgg('logRTresidual', np.mean), residstd=pd.NamedAgg('logRTresidual', np.std))
        
mpw_pp.reset_index(inplace=True)
mpw_pp['word_index'] = mpw_pp['word_index'].astype(int)

#%% plot per participant
"""
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
"""
# %% plot RTs per word (mpw = means per word)
mpw = mpw_pp.groupby(by=['agreement', 'surprisal', 'word_index']).aggregate(mean=pd.NamedAgg('mean', np.mean), std=pd.NamedAgg('mean', np.std),
                     lin_interval=pd.NamedAgg('mean', get_interval),
                     logmean=pd.NamedAgg('logmean', np.mean), logstd=pd.NamedAgg('logmean', np.std),
                     residmean=pd.NamedAgg('residmean', np.mean), residstd=pd.NamedAgg('residmean', np.std),
                     interval=pd.NamedAgg('residmean', get_interval))
mpw.reset_index(inplace=True)
mpw['word_index'] = mpw['word_index'].astype(int)

mpw[['interval_high', 'interval_low']] = mpw['interval'].apply(pd.Series)
mpw[['lin_interval_high', 'lin_interval_low']] = mpw['lin_interval'].apply(pd.Series)

corr_high = mpw.loc[(mpw['agreement'] == 'correct') & (mpw['surprisal'] == 'high') & (mpw['word_index'] > 3)]
corr_low = mpw.loc[(mpw['agreement'] == 'correct') & (mpw['surprisal'] == 'low') & (mpw['word_index'] > 3)]
incorr_high = mpw.loc[(mpw['agreement'] == 'incorrect') & (mpw['surprisal'] == 'high') & (mpw['word_index'] > 3)]
incorr_low = mpw.loc[(mpw['agreement'] == 'incorrect') & (mpw['surprisal'] == 'low') & (mpw['word_index'] > 3)]


# %% plot the averages
fig,ax = plt.subplots(figsize=(7,4))

ax.plot(range(4,10), corr_high['mean'], color='green', alpha=1)
ax.plot(range(4,10), corr_low['mean'], color='green', linestyle='dashed', alpha=1)
ax.plot(range(4,10), incorr_high['mean'], color='red', alpha=1)
ax.plot(range(4,10), incorr_low['mean'], color='red', linestyle='dashed', alpha=1)

plt.legend(['Correct/High', 'Correct/Low', 'Incorrect/High', 'Incorrect/Low'], frameon=False, bbox_to_anchor=[1,1])

ax.fill_between(range(4,10), corr_high['lin_interval_low'], corr_high['lin_interval_high'], color='green', alpha=0.2)
ax.fill_between(range(4,10), corr_low['lin_interval_low'], corr_low['lin_interval_high'], color='green', hatch='//', alpha=0.2)
ax.fill_between(range(4,10), incorr_high['lin_interval_low'], incorr_high['lin_interval_high'], color='red', alpha=0.2)
ax.fill_between(range(4,10), incorr_low['lin_interval_low'], incorr_low['lin_interval_high'], color='red', hatch='//', alpha=0.2)

ax.axvline(5, color='grey', zorder=0, linestyle='--')      
ax.margins(x=0)  

ax.set_xlabel('Word position')
ax.set_ylabel('RT (ms)')
sns.despine()
plt.tight_layout()
fig.savefig(f'{resultsdir}/88_pps_linRTs_confint_manual.png', dpi=300)
fig.savefig(f'{resultsdir}/88_pps_linRTs_confint_manual.svg')


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
ax.margins(x=0)

ax.set_xlabel('Word position')
ax.set_ylabel('logRT')
sns.despine()

fig.savefig(f'{resultsdir}/88_pps_logRTs.svg')
#fig.savefig(f'{resultsdir}/12_first_pps_log.jpg', dpi=300)
#fig.savefig(f'{resultsdir}/16_pps_logRTs.svg')

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
ax.margins(x=0)

ax.set_xlabel('Word position')
ax.set_ylabel('Residual logRT')
sns.despine()

fig.savefig(f'{resultsdir}/88_pps_resid_logRTs.svg')

# %% confidence interval plot
fig,ax = plt.subplots(figsize=(7,4))

ax.plot(range(4,10), corr_high['residmean'], color='green', alpha=1)
ax.plot(range(4,10), corr_low['residmean'], color='green', linestyle='dashed', alpha=1)
ax.plot(range(4,10), incorr_high['residmean'], color='red', alpha=1)
ax.plot(range(4,10), incorr_low['residmean'], color='red', linestyle='dashed', alpha=1)

plt.legend(['Correct/High', 'Correct/Low', 'Incorrect/High', 'Incorrect/Low'], frameon=False, bbox_to_anchor=[1,1])

ax.fill_between(range(4,10), corr_high['interval_low'], corr_high['interval_high'], color='green', alpha=0.2)
ax.fill_between(range(4,10), corr_low['interval_low'], corr_low['interval_high'], color='green', hatch='//', alpha=0.2)
ax.fill_between(range(4,10), incorr_high['interval_low'], incorr_high['interval_high'], color='red', alpha=0.2)
ax.fill_between(range(4,10), incorr_low['interval_low'], incorr_low['interval_high'], color='red', hatch='//', alpha=0.2)

#ax.errorbar(x=range(4,10), y=corr_high['residmean'], yerr=np.asarray([np.abs(corr_high['interval_low']), np.abs(corr_high['interval_high'])]), color='green')
#ax.errorbar(x=range(4,10), y=corr_low['residmean'], yerr=np.asarray([np.abs(corr_low['interval_low']), np.abs(corr_low['interval_high'])]), color='green', linestyle='dashed')
#ax.errorbar(x=range(4,10), y=incorr_high['residmean'], yerr=np.asarray([np.abs(incorr_high['interval_low']), np.abs(incorr_high['interval_high'])]), color='red')
#ax.errorbar(x=range(4,10), y=incorr_low['residmean'], yerr=np.asarray([np.abs(incorr_low['interval_low']), np.abs(incorr_low['interval_high'])]), color='red', linestyle='dashed')

ax.axvline(5, color='grey', zorder=0, linestyle='--')     
ax.margins(x=0)

ax.set_xlabel('Word position')
ax.set_ylabel('Residual logRT')
sns.despine()
plt.tight_layout()
fig.savefig(f'{resultsdir}/88_pps_resid_logRTs_confint_manual.png', dpi=300)
fig.savefig(f'{resultsdir}/88_pps_resid_logRTs_confint_manual.svg')

# %% Include the effect of plurality
plural_pp = RTs.groupby(by=['agreement', 'correct_number', 'word_index', 'UserId']).aggregate(mean=pd.NamedAgg('RT', np.mean), 
                     std=pd.NamedAgg('RT', np.std), logmean=pd.NamedAgg('logRT', np.mean), logstd=pd.NamedAgg('logRT', np.std),
                     residmean=pd.NamedAgg('logRTresidual', np.mean), residstd=pd.NamedAgg('logRTresidual', np.std))
        
plural_pp.reset_index(inplace=True)
plural_pp['word_index'] = plural_pp['word_index'].astype(int)


# %% group by plural
plural = plural_pp.groupby(by=['agreement', 'correct_number', 'word_index']).aggregate(mean=pd.NamedAgg('mean', np.mean), std=pd.NamedAgg('mean', np.std),
                     lin_interval=pd.NamedAgg('mean', get_interval),
                     logmean=pd.NamedAgg('logmean', np.mean), logstd=pd.NamedAgg('logmean', np.std),
                     residmean=pd.NamedAgg('residmean', np.mean), residstd=pd.NamedAgg('residmean', np.std),
                     interval=pd.NamedAgg('residmean', get_interval))
plural.reset_index(inplace=True)
plural['word_index'] = plural['word_index'].astype(int)

plural[['interval_high', 'interval_low']] = plural['interval'].apply(pd.Series)
plural[['lin_interval_high', 'lin_interval_low']] = plural['lin_interval'].apply(pd.Series)

corr_sin = plural.loc[(plural['agreement'] == 'correct') & (plural['correct_number'] == 'singular') & (plural['word_index'] > 3)]
corr_plu = plural.loc[(plural['agreement'] == 'correct') & (plural['correct_number'] == 'plural') & (plural['word_index'] > 3)]
incorr_sin = mpw.loc[(plural['agreement'] == 'incorrect') & (plural['correct_number'] == 'singular') & (plural['word_index'] > 3)]
incorr_plu = mpw.loc[(plural['agreement'] == 'incorrect') & (plural['correct_number'] == 'plural') & (plural['word_index'] > 3)]

# %% plurality plot
fig,ax = plt.subplots(figsize=(7,4))

ax.plot(range(4,10), corr_sin['residmean'], color='green', alpha=1)
ax.plot(range(4,10), corr_plu['residmean'], color='green', linestyle='--', alpha=1)
ax.plot(range(4,10), incorr_sin['residmean'], color='red', alpha=1)
ax.plot(range(4,10), incorr_plu['residmean'], color='red', linestyle='--', alpha=1)

plt.legend(['Correct/Singular', 'Correct/Plural', 'Incorrect/Singular', 'Incorrect/Plural'], frameon=False, bbox_to_anchor=[1,1])

ax.fill_between(range(4,10), corr_sin['interval_low'], corr_sin['interval_high'], color='green', alpha=0.2)
ax.fill_between(range(4,10), corr_plu['interval_low'], corr_plu['interval_high'], color='green', hatch='|', alpha=0.2)
ax.fill_between(range(4,10), incorr_sin['interval_low'], incorr_sin['interval_high'], color='red', alpha=0.2)
ax.fill_between(range(4,10), incorr_plu['interval_low'], incorr_plu['interval_high'], color='red', hatch='|', alpha=0.2)

ax.axvline(5, color='grey', zorder=0, linestyle='--')     
ax.margins(x=0)

ax.set_xlabel('Word position')
ax.set_ylabel('Residual logRT')
sns.despine()
plt.tight_layout()

fig.savefig(f'{resultsdir}/88_pps_resid_logRTs_confint_manual_PLURALITY.png', dpi=300)
fig.savefig(f'{resultsdir}/88_pps_resid_logRTs_confint_manual_PLURALITY.svg')


# %% ungrammaticality effect per plurality
incorrect = plural_pp.loc[plural_pp['agreement'] == 'incorrect']
correct = plural_pp.loc[plural_pp['agreement'] == 'correct']

ungram = correct.copy()
ungram['difference'] = incorrect['residmean'].values- ungram['residmean'].values

# %% now we group
ungram_plural = ungram.groupby(by=['correct_number', 'word_index']).aggregate(mean=pd.NamedAgg('difference', np.mean), std=pd.NamedAgg('difference', np.std),
                     interval=pd.NamedAgg('difference', get_interval))
ungram_plural.reset_index(inplace=True)

sin = ungram_plural.loc[(ungram_plural['correct_number'] == 'singular') & (ungram_plural['word_index'] > 3)]
plu = ungram_plural.loc[(ungram_plural['correct_number'] == 'plural') & (ungram_plural['word_index'] > 3)]

# %%
fig,ax=plt.subplots(figsize=(7,4))
ax.plot(range(4,10), sin['mean'], color='black', alpha=1)
ax.plot(range(4,10), plu['mean'], color='black', linestyle='--', alpha=1)

plt.legend(['Singular', 'Plural'], frameon=False, bbox_to_anchor=[1.25,1])

ax.fill_between(range(4,10), [s[0] for s in sin['interval']], [s[1] for s in sin['interval']], color='blue', alpha=0.2)
ax.fill_between(range(4,10), [pl[0] for pl in plu['interval']], [pl[1] for pl in plu['interval']], color='blue', hatch='|', alpha=0.2)

ax.axvline(5, color='grey', zorder=0, linestyle='--')
ax.axhline(0, color='grey', zorder=0, linestyle=':')     
ax.margins(x=0)

ax.set_xlabel('Word position')
ax.set_ylabel('$\Delta$ residual logRT')
sns.despine()
plt.tight_layout()

fig.savefig(f'{resultsdir}/88_pps_resid_logRTs_confint_manual_ungram-by-number.png', dpi=300)
fig.savefig(f'{resultsdir}/88_pps_resid_logRTs_confint_manual_ungram-by-number.svg')

# %% ungrammaticality effect per surprisal category
correct = mpw_pp.loc[mpw_pp['agreement'] == 'correct']
incorrect = mpw_pp.loc[mpw_pp['agreement'] == 'incorrect']

ungram_surp = correct.copy()
ungram_surp['correct'] = correct['residmean'].values
ungram_surp['incorrect'] = incorrect['residmean'].values
ungram_surp['difference'] = incorrect['residmean'].values - correct['residmean'].values

ungram_surp = ungram_surp.groupby(by=['surprisal', 'word_index']).aggregate(
                    mean=pd.NamedAgg('difference', np.mean), std=pd.NamedAgg('difference', np.std),
                    interval=pd.NamedAgg('difference', get_interval),
                    cormean=pd.NamedAgg('correct', np.mean), corstd=pd.NamedAgg('correct', np.std),
                    corinterval=pd.NamedAgg('correct', get_interval),
                    incormean=pd.NamedAgg('incorrect', np.mean), incorstd=pd.NamedAgg('incorrect', np.std),
                    incorinterval=pd.NamedAgg('incorrect', get_interval)
                    )
ungram_surp.reset_index(inplace=True)

high_surp = ungram_surp.loc[(ungram_surp['surprisal'] == 'high') & (ungram_surp['word_index'] > 3)]
low_surp = ungram_surp.loc[(ungram_surp['surprisal'] == 'low') & (ungram_surp['word_index'] > 3)]

# %% plot this
fig,ax=plt.subplots(ncols=2, figsize=(7,3), sharey=True)

ax[0].plot(range(4,10), high_surp['mean'], color='#253494', alpha=1)
ax[0].plot(range(4,10), low_surp['mean'], color='#2c7fb8', linestyle='--', alpha=1)

ax[0].legend(['High surprisal', 'Low surprisal'], frameon=False, bbox_to_anchor=[1,1])

ax[0].fill_between(range(4,10), [s[0] for s in high_surp['interval']], [s[1] for s in high_surp['interval']], color='#253494', alpha=0.2)
ax[0].fill_between(range(4,10), [pl[0] for pl in low_surp['interval']], [pl[1] for pl in low_surp['interval']], color='#2c7fb8', hatch='|', alpha=0.2)

ax[0].axvline(5, color='grey', zorder=0, linestyle='--')
ax[0].axhline(0, color='grey', zorder=0, linestyle=':')     
ax[0].margins(x=0)
ax[0].set_xlabel('Word position')
ax[0].set_ylabel('$\Delta$ residual logRT')

ax[1].plot(range(4,10), sin['mean'], color='#d95f02', alpha=1)
ax[1].plot(range(4,10), plu['mean'], color='#7570b3', linestyle='--', alpha=1)

ax[1].legend(['Singular', 'Plural'], frameon=False, bbox_to_anchor=[1,1])

ax[1].fill_between(range(4,10), [s[0] for s in sin['interval']], [s[1] for s in sin['interval']], color='#d95f02', alpha=0.2)
ax[1].fill_between(range(4,10), [pl[0] for pl in plu['interval']], [pl[1] for pl in plu['interval']], color='#7570b3', hatch='|', alpha=0.2)

ax[1].axvline(5, color='grey', zorder=0, linestyle='--')
ax[1].axhline(0, color='grey', zorder=0, linestyle=':')     
ax[1].margins(x=0)

ax[1].set_xlabel('Word position')
ax[1].set_ylabel('$\Delta$ residual logRT')
ax[1].get_yaxis().set_visible(False)

#ax[0].text(3, 0.14, 'A', fontsize=14)
#ax[1].text(3.7, 0.14, 'B', fontsize=14)

sns.despine()
plt.tight_layout()

fig.savefig(f'{resultsdir}/88_pps_resid_logRTs_confint_manual_ungram.png', dpi=300)
fig.savefig(f'{resultsdir}/88_pps_resid_logRTs_confint_manual_ungram.svg')


# %% per surprisal value...
# first, we aggregate over participants
mpw_items = RTs.groupby(by=['agreement', 'surprisal', 'surprisal_value', 'word_index', 'correct_number', 'id', 'list', 'word']).aggregate(
                        residRT=pd.NamedAgg('logRTresidual', np.mean), residRT_int=pd.NamedAgg('logRTresidual', get_interval))
mpw_items[['ci_high', 'ci_low']] = mpw_items['residRT_int'].apply(pd.Series)
mpw_items.reset_index(inplace=True)

# %%
fig,ax=plt.subplots(ncols=4, figsize=(7,2), sharey=True, sharex=True)

titles = {5: 'Target',
          6: 'Spill-over 1',
          7: 'Spill-over 2',
          8: 'Spill-over 3'}

for word_index, axi in zip([5,6,7,8],ax):
    dt_cor = mpw_items.loc[(mpw_items['word_index'] == word_index) & (mpw_items['agreement'] == 'correct')]
    dt_incor = mpw_items.loc[(mpw_items['word_index'] == word_index) & (mpw_items['agreement'] == 'incorrect')]
    
    dif = dt_cor.copy()
    dif['difference'] = dt_incor['residRT'].values - dt_cor['residRT'].values
    
    axi.scatter(dif['surprisal_value'], dif['difference'], s=2)
    
    # let's estimate a line
    #obtain m (slope) and b(intercept) of linear regression line
    m, b = np.polyfit(dif['surprisal_value'], dif['difference'], 1)

    #add linear regression line to scatterplot 
    axi.plot(dif['surprisal_value'], m*dif['surprisal_value']+b, color='orange')
    #print(m)
    
    #axi.scatter(dt_cor['surprisal_value'], dt_cor['residRT'], s=2)
    #axi.scatter(dt_incor['surprisal_value'], dt_incor['residRT'], s=2)
    
    axi.set_title(titles[word_index])
    axi.set_xlabel('Surprisal (bits)')
    axi.set_ylabel('$\Delta$ residual logRT')
    
    if word_index != 5:
        axi.get_yaxis().set_visible(False)
    
sns.despine()
plt.tight_layout()

fig.savefig(f'{resultsdir}/88_pps_resid_logRTs_scatter_surprisal_agreement_interaction.png', dpi=300)
fig.savefig(f'{resultsdir}/88_pps_resid_logRTs_scatter_surprisal_agreement_interaction.svg')

# %%
fig,ax=plt.subplots(ncols=4, figsize=(7,2), sharey=True, sharex=True)

titles = {5: 'Target',
          6: 'Spill-over 1',
          7: 'Spill-over 2',
          8: 'Spill-over 3'}

for word_index, axi in zip([5,6,7,8],ax):
    dt_cor = mpw_items.loc[(mpw_items['word_index'] == word_index) & (mpw_items['agreement'] == 'correct')]
    dt_incor = mpw_items.loc[(mpw_items['word_index'] == word_index) & (mpw_items['agreement'] == 'incorrect')]
    
    #dif = dt_cor.copy()
    #dif['difference'] = dt_incor['residRT'].values - dt_cor['residRT'].values
     # let's estimate a line
    #obtain m (slope) and b(intercept) of linear regression line
    m, b = np.polyfit(dt_incor['surprisal_value'], dt_incor['residRT'], 1)
    mc,bc = np.polyfit(dt_cor['surprisal_value'], dt_cor['residRT'], 1)

    #add linear regression line to scatterplot 
    axi.plot(dt_cor['surprisal_value'], mc*dt_cor['surprisal_value']+bc, color='green')
    axi.plot(dt_incor['surprisal_value'], m*dt_incor['surprisal_value']+b, color='red')
    
    if word_index == 8:
        axi.legend(['Correct', 'Incorrect'], frameon=False, bbox_to_anchor=[1,1])
    
    axi.scatter(dt_cor['surprisal_value'], dt_cor['residRT'], s=2, color='green', alpha=0.5, zorder=0)
    axi.scatter(dt_incor['surprisal_value'], dt_incor['residRT'], s=2, color='red', alpha=0.5, zorder=0)
    
    axi.set_title(titles[word_index])
    axi.set_xlabel('Surprisal (bits)')
    axi.set_ylabel('Residual logRT')
    
    if word_index != 5:
        axi.get_yaxis().set_visible(False)

    
sns.despine()
plt.tight_layout()

fig.savefig(f'{resultsdir}/88_pps_resid_logRTs_scatter_surprisal_agreement_separate.png', dpi=300)
fig.savefig(f'{resultsdir}/88_pps_resid_logRTs_scatter_surprisal_agreement_separate.svg')

# %% the effect of agreement on the effect of verb number
singular = plural_pp.loc[plural_pp['correct_number'] == 'singular']
plural = plural_pp.loc[plural_pp['correct_number'] == 'plural']

tmpnumber = singular.copy()
tmpnumber['difference'] = plural['residmean'].values - singular['residmean'].values

number = tmpnumber.groupby(by=['agreement', 'word_index']).aggregate(mean=pd.NamedAgg('difference', np.mean), std=pd.NamedAgg('difference', np.std),
                     interval=pd.NamedAgg('difference', get_interval))

number.reset_index(inplace=True)

cor = number.loc[(number['agreement'] =='correct') & (number['word_index'] > 3)]
incor = number.loc[(number['agreement'] =='incorrect') & (number['word_index'] > 3)]

# %%# effect of agreement on surprisal
high_surprisal = mpw_pp.loc[mpw_pp['surprisal'] == 'high']
low_surprisal = mpw_pp.loc[mpw_pp['surprisal'] == 'low']

surp_by_agr = high_surprisal.copy()
surp_by_agr['difference'] = high_surprisal['residmean'].values - low_surprisal['residmean'].values

surp_by_agr = surp_by_agr.groupby(by=['agreement', 'word_index']).aggregate(
                mean=pd.NamedAgg('difference', np.mean), interval=pd.NamedAgg('difference', get_interval))
surp_by_agr.reset_index(inplace=True)
corsur = surp_by_agr.loc[(surp_by_agr['agreement'] == 'correct') & (number['word_index'] > 3)]
incorsur = surp_by_agr.loc[(surp_by_agr['agreement'] == 'incorrect') & (number['word_index'] > 3)]

#%% number by agreement
fig,ax=plt.subplots(figsize=(7,3), ncols=2, sharey=True, sharex=True)

# surprisal by agreement
ax[0].plot(corsur['word_index'], corsur['mean'], color='green')
ax[0].plot(incorsur['word_index'], incorsur['mean'], color='red', linestyle='--')

ax[0].fill_between(range(4,10), [s[0] for s in corsur['interval']], [s[1] for s in corsur['interval']], color='green', alpha=0.2)
ax[0].fill_between(range(4,10), [pl[0] for pl in incorsur['interval']], [pl[1] for pl in incorsur['interval']], color='red', hatch='|', alpha=0.2)

ax[0].set_xlabel('Word position')
ax[0].set_ylabel('$\Delta$ residual logRT')
ax[0].axvline(5, color='grey', zorder=0, linestyle='--')


# number by agreement
ax[1].plot(cor['word_index'], cor['mean'], color='green')
ax[1].plot(incor['word_index'], incor['mean'], color='red', linestyle='--')

ax[1].legend(['Correct', 'Incorrect'], frameon=False, bbox_to_anchor=[1,1])

ax[1].fill_between(range(4,10), [s[0] for s in cor['interval']], [s[1] for s in cor['interval']], color='green', alpha=0.2)
ax[1].fill_between(range(4,10), [pl[0] for pl in incor['interval']], [pl[1] for pl in incor['interval']], color='red', hatch='|', alpha=0.2)

ax[1].set_xlabel('Word position')
ax[1].get_yaxis().set_visible(False)
ax[1].axvline(5, color='grey', zorder=0, linestyle='--')

sns.despine()
plt.tight_layout()

# %% all in one plot with abcd
fig,ax=plt.subplots(figsize=(7,6), nrows=2, ncols=2, sharey=True, sharex=True)

# agreement by surprisal
ax[0,0].plot(range(4,10), high_surp['mean'], color='#253494', alpha=1)
ax[0,0].plot(range(4,10), low_surp['mean'], color='#2c7fb8', linestyle='--', alpha=1)

ax[0,0].legend(['High surprisal', 'Low surprisal'], frameon=False, bbox_to_anchor=[1,1])

ax[0,0].fill_between(range(4,10), [s[0] for s in high_surp['interval']], [s[1] for s in high_surp['interval']], color='#253494', alpha=0.2)
ax[0,0].fill_between(range(4,10), [pl[0] for pl in low_surp['interval']], [pl[1] for pl in low_surp['interval']], color='#2c7fb8', hatch='|', alpha=0.2)

ax[0,0].axvline(5, color='grey', zorder=0, linestyle='--')
ax[0,0].axhline(0, color='grey', zorder=0, linestyle=':')     
ax[0,0].margins(x=0)
ax[0,0].get_xaxis().set_visible(False)
ax[0,0].set_ylabel('$\Delta$ residual logRT')

ax[0,0].set_title('Agreement by surprisal')

ax[0,1].plot(range(4,10), sin['mean'], color='#d95f02', alpha=1)
ax[0,1].plot(range(4,10), plu['mean'], color='#7570b3', linestyle='--', alpha=1)

ax[0,1].legend(['Singular', 'Plural'], frameon=False, bbox_to_anchor=[1,1])

ax[0,1].fill_between(range(4,10), [s[0] for s in sin['interval']], [s[1] for s in sin['interval']], color='#d95f02', alpha=0.2)
ax[0,1].fill_between(range(4,10), [pl[0] for pl in plu['interval']], [pl[1] for pl in plu['interval']], color='#7570b3', hatch='|', alpha=0.2)

ax[0,1].axvline(5, color='grey', zorder=0, linestyle='--')
ax[0,1].axhline(0, color='grey', zorder=0, linestyle=':')     
ax[0,1].margins(x=0)

ax[0,1].get_xaxis().set_visible(False)
ax[0,1].get_yaxis().set_visible(False)

ax[0,1].set_title('Agreement by verb number')

# surprisal by agreement
ax[1,0].plot(corsur['word_index'], corsur['mean'], color='green')
ax[1,0].plot(incorsur['word_index'], incorsur['mean'], color='red', linestyle='--')

ax[1,0].fill_between(range(4,10), [s[0] for s in corsur['interval']], [s[1] for s in corsur['interval']], color='green', alpha=0.2)
ax[1,0].fill_between(range(4,10), [pl[0] for pl in incorsur['interval']], [pl[1] for pl in incorsur['interval']], color='red', hatch='|', alpha=0.2)

ax[1,0].set_xlabel('Word position')
ax[1,0].set_ylabel('$\Delta$ residual logRT')
ax[1,0].axvline(5, color='grey', zorder=0, linestyle='--')
ax[1,0].axhline(0, color='grey', zorder=0, linestyle=':')     

ax[1,0].legend(['Correct', 'Incorrect'], frameon=False, bbox_to_anchor=[1,1])
ax[1,0].set_title('Surprisal by agreement')

# number by agreement
ax[1,1].plot(cor['word_index'], cor['mean'], color='green')
ax[1,1].plot(incor['word_index'], incor['mean'], color='red', linestyle='--')

ax[1,1].legend(['Correct', 'Incorrect'], frameon=False, bbox_to_anchor=[1,1])

ax[1,1].fill_between(range(4,10), [s[0] for s in cor['interval']], [s[1] for s in cor['interval']], color='green', alpha=0.2)
ax[1,1].fill_between(range(4,10), [pl[0] for pl in incor['interval']], [pl[1] for pl in incor['interval']], color='red', hatch='|', alpha=0.2)

ax[1,1].set_xlabel('Word position')
ax[1,1].get_yaxis().set_visible(False)
ax[1,1].axvline(5, color='grey', zorder=0, linestyle='--')
ax[1,1].axhline(0, color='grey', zorder=0, linestyle=':')

ax[1,1].set_title('Verb number by agreement')

ax[0,0].text(3.4, 0.15, 'A', fontsize=14)
ax[0,1].text(3.4, 0.15, 'B', fontsize=14)
ax[1,0].text(3.4, 0.15, 'C', fontsize=14)
ax[1,1].text(3.4, 0.15, 'D', fontsize=14)

ax[0,0].margins(x=0)
ax[0,1].margins(x=0)
ax[1,0].margins(x=0)
ax[1,1].margins(x=0)

sns.despine()
plt.tight_layout()

fig.savefig(f'{resultsdir}/88_pps_resid_logRTs_line_interactions_pairwise.png', dpi=300)
fig.savefig(f'{resultsdir}/88_pps_resid_logRTs_line_interactions_pairwise.svg')

# %% noun number
noun_pp = RTs.groupby(by=['agreement', 'noun_number', 'word_index', 'UserId']).aggregate(mean=pd.NamedAgg('RT', np.mean), 
                     std=pd.NamedAgg('RT', np.std), logmean=pd.NamedAgg('logRT', np.mean), logstd=pd.NamedAgg('logRT', np.std),
                     residmean=pd.NamedAgg('logRTresidual', np.mean), residstd=pd.NamedAgg('logRTresidual', np.std))
        
noun_pp.reset_index(inplace=True)
noun_pp['word_index'] = noun_pp['word_index'].astype(int)

# %% group by plural
noun = noun_pp.groupby(by=['agreement', 'noun_number', 'word_index']).aggregate(mean=pd.NamedAgg('mean', np.mean), std=pd.NamedAgg('mean', np.std),
                     lin_interval=pd.NamedAgg('mean', get_interval),
                     logmean=pd.NamedAgg('logmean', np.mean), logstd=pd.NamedAgg('logmean', np.std),
                     residmean=pd.NamedAgg('residmean', np.mean), residstd=pd.NamedAgg('residmean', np.std),
                     interval=pd.NamedAgg('residmean', get_interval))
noun.reset_index(inplace=True)
noun['word_index'] = noun['word_index'].astype(int)

noun[['interval_high', 'interval_low']] = noun['interval'].apply(pd.Series)
noun[['lin_interval_high', 'lin_interval_low']] = noun['lin_interval'].apply(pd.Series)

corr_sin = noun.loc[(noun['agreement'] == 'correct') & (noun['noun_number'] == 'singular') & (noun['word_index'] > 3)]
corr_plu = noun.loc[(noun['agreement'] == 'correct') & (noun['noun_number'] == 'plural') & (noun['word_index'] > 3)]
incorr_sin = mpw.loc[(noun['agreement'] == 'incorrect') & (noun['noun_number'] == 'singular') & (noun['word_index'] > 3)]
incorr_plu = mpw.loc[(noun['agreement'] == 'incorrect') & (noun['noun_number'] == 'plural') & (noun['word_index'] > 3)]

# %% plurality plot
fig,ax = plt.subplots(figsize=(7,4))

ax.plot(range(4,10), corr_sin['residmean'], color='green', alpha=1)
ax.plot(range(4,10), corr_plu['residmean'], color='green', linestyle='--', alpha=1)
ax.plot(range(4,10), incorr_sin['residmean'], color='red', alpha=1)
ax.plot(range(4,10), incorr_plu['residmean'], color='red', linestyle='--', alpha=1)

plt.legend(['Correct/Singular', 'Correct/Plural', 'Incorrect/Singular', 'Incorrect/Plural'], frameon=False, bbox_to_anchor=[1,1])

ax.fill_between(range(4,10), corr_sin['interval_low'], corr_sin['interval_high'], color='green', alpha=0.2)
ax.fill_between(range(4,10), corr_plu['interval_low'], corr_plu['interval_high'], color='green', hatch='|', alpha=0.2)
ax.fill_between(range(4,10), incorr_sin['interval_low'], incorr_sin['interval_high'], color='red', alpha=0.2)
ax.fill_between(range(4,10), incorr_plu['interval_low'], incorr_plu['interval_high'], color='red', hatch='|', alpha=0.2)

ax.axvline(5, color='grey', zorder=0, linestyle='--')     
ax.margins(x=0)

ax.set_xlabel('Word position')
ax.set_ylabel('Residual logRT')
sns.despine()
plt.tight_layout()

fig.savefig(f'{resultsdir}/88_pps_resid_logRTs_confint_manual_PLURALITY-N.png', dpi=300)
fig.savefig(f'{resultsdir}/88_pps_resid_logRTs_confint_manual_PLURALITY-N.svg')


# %% ungrammaticality effect per plurality
incorrect = noun_pp.loc[noun_pp['agreement'] == 'incorrect']
correct = noun_pp.loc[noun_pp['agreement'] == 'correct']

ungram = correct.copy()
ungram['difference'] = incorrect['residmean'].values- ungram['residmean'].values

# %% now we group
ungram_noun = ungram.groupby(by=['noun_number', 'word_index']).aggregate(mean=pd.NamedAgg('difference', np.mean), std=pd.NamedAgg('difference', np.std),
                     interval=pd.NamedAgg('difference', get_interval))
ungram_noun.reset_index(inplace=True)

sin = ungram_noun.loc[(ungram_noun['noun_number'] == 'singular') & (ungram_noun['word_index'] > 3)]
plu = ungram_noun.loc[(ungram_noun['noun_number'] == 'plural') & (ungram_noun['word_index'] > 3)]

# %%
fig,ax=plt.subplots(figsize=(7,4))
ax.plot(range(4,10), sin['mean'], color='#d95f02', alpha=1)
ax.plot(range(4,10), plu['mean'], color='#7570b3', linestyle='--', alpha=1)

plt.legend(['Singular', 'Plural'], frameon=False, bbox_to_anchor=[1.25,1])

ax.fill_between(range(4,10), [s[0] for s in sin['interval']], [s[1] for s in sin['interval']], color='#d95f02', alpha=0.2)
ax.fill_between(range(4,10), [pl[0] for pl in plu['interval']], [pl[1] for pl in plu['interval']], color='#7570b3', hatch='|', alpha=0.2)

ax.axvline(5, color='grey', zorder=0, linestyle='--')
ax.axhline(0, color='grey', zorder=0, linestyle=':')     
ax.margins(x=0)

ax.set_xlabel('Word position')
ax.set_ylabel('$\Delta$ residual logRT')
sns.despine()
plt.tight_layout()

fig.savefig(f'{resultsdir}/88_pps_resid_logRTs_confint_manual_ungram-by-number-N.png', dpi=300)
fig.savefig(f'{resultsdir}/88_pps_resid_logRTs_confint_manual_ungram-by-number-N.svg')

# %% the opposite effect
singular = noun_pp.loc[noun_pp['noun_number'] == 'singular']
plural = noun_pp.loc[noun_pp['noun_number'] == 'plural']

tmpnumber = singular.copy()
tmpnumber['difference'] = plural['residmean'].values - singular['residmean'].values

number = tmpnumber.groupby(by=['agreement', 'word_index']).aggregate(mean=pd.NamedAgg('difference', np.mean), std=pd.NamedAgg('difference', np.std),
                     interval=pd.NamedAgg('difference', get_interval))

number.reset_index(inplace=True)

cor = number.loc[(number['agreement'] =='correct') & (number['word_index'] > 3)]
incor = number.loc[(number['agreement'] =='incorrect') & (number['word_index'] > 3)]

# number by agreement
fig,ax=plt.subplots(figsize=(7,4))

ax.plot(cor['word_index'], cor['mean'], color='green')
ax.plot(incor['word_index'], incor['mean'], color='red', linestyle='--')

ax.legend(['Correct', 'Incorrect'], frameon=False, bbox_to_anchor=[1,1])

ax.fill_between(range(4,10), [s[0] for s in cor['interval']], [s[1] for s in cor['interval']], color='green', alpha=0.2)
ax.fill_between(range(4,10), [pl[0] for pl in incor['interval']], [pl[1] for pl in incor['interval']], color='red', hatch='|', alpha=0.2)

ax.set_xlabel('Word position')
ax.get_yaxis().set_visible(False)
ax.axvline(5, color='grey', zorder=0, linestyle='--')
ax.axhline(0, color='grey', zorder=0, linestyle=':')
ax.margins(x=0)
ax.set_ylabel('$\Delta$ residual logRT')
sns.despine()
plt.tight_layout()

#ax[1,1].set_title('Verb number by agreement')
