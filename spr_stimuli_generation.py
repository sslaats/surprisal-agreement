#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:54:26 2023

@author: sopsla
"""
from minicons import scorer
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from text import get_word_frequency
import os

# place ideas
specific = ['dierentuin', 'zandbak', 'schoolplein', 'kantoor', 'markt', 'gemeenteraad']
general = ['buurt', 'plaats', 'plein']

# %% load the model
model = scorer.IncrementalLMScorer('GroNLP/gpt2-small-dutch', 'cpu')

# %% new method: context, find probable continuation, try out all words with similar frequency
wf_file='SUBTLEX-NL.cd-above2.txt'
df_wf = pd.read_csv(os.path.join('/project/3027005.01/', 'wordfreq', wf_file), delim_whitespace=True)

def get_words_same_wf(word, number_of_words=50, frequency_df=df_wf):
    freq = get_word_frequency(word)
    
    vals = frequency_df.iloc[(frequency_df['Lg10WF']-freq).abs().argsort()[:number_of_words]]
    return vals['Word'].tolist()

# %% context with match & wrong
context = "op het kantoor legt de"
target = "zakenman"
error = "zakenmannen"

print(get_word_frequency(target))
alternatives = get_words_same_wf(target, number_of_words=100)

stimuli = [" ".join([context, target]), " ".join([context, error])]

stimscores = model.token_score(stimuli, surprisal=True, base_two=True)

wrds = [st[0] for st in stimscores[0]]
i = wrds.index(target)

target_surprisal = stimscores[0][i]
error_surprisal = stimscores[1][i]

print(target_surprisal, error_surprisal)

if target_surprisal[1] < 12.0:
    
    for word in alternatives:
        surprisal = model.token_score([" ".join([context, word])], surprisal=True, base_two=True)[0][i]
        if (surprisal[-1] > 14) & (surprisal[-1] >= error_surprisal[-1]):
            print(" ".join([context, word]), surprisal)

# %% surprisal error chosen alternative
alt = "kapper"
alt_error = "kappers"
print(model.token_score([" ".join([context, alt_error])], surprisal=True, base_two=True)[0][i])

# %% all word frequencies and surprisals
for w in [target, error, alt, alt_error]:
    print(w, model.token_score([" ".join([context, w])], surprisal=True, base_two=True)[0][i],
          get_word_frequency(w))

# %% if word split in two
w = error
print(model.token_score([" ".join([context, w])], surprisal=True, base_two=True)[0][-1][1] +
      model.token_score([" ".join([context, w])], surprisal=True, base_two=True)[0][-2][1])

# %% print current scores
print(model.token_score([" ".join([context, w])], surprisal=True, base_two=True)[0])

# %% places
places = ['in de gang', 'in de duinen', 'in de tuin', 'in de zomer', 'in de supermarkt', 'in de gemeenteraad']
verbs = ['stond de', 'liep de', 'ging de', 'mocht de', 'zocht de']
contexts = [pl + ' ' + v for pl in places for v in verbs]

# %%
contexts = ['op de straat ligt de', 'in de sloot drijft de'] #['na het gevecht vloog de', 'na het feest vloog de']
targets = ['man', 'mannen', 'vis', 'vissen']
end = 'de schreeuw niet'

stimuli = []
for context in contexts:
    for target in targets:
        stim = context + ' ' + target + ' ' + end
        stimuli.append(stim)

# in de gang liep de 9.2 in de zomer liep de 11.2
# in de tuin 

# %% entropy
logprobs = model.next_word_distribution(contexts)
es = []
for ct,context in zip(contexts, logprobs):
    context = np.asarray(context)
    cs = [np.exp(c) for c in context]
    context_prob = np.exp(context)
    print(ct, stats.entropy(context_prob, base=2))
#    plt.hist(context[:-1],alpha=0.5)

# %% surprisal
stimscores = model.token_score(stimuli, surprisal=True, base_two=True)
for score in stimscores:
    print(score[3:7])

