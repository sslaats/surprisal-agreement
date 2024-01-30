# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:31:25 2023

@author: sopsla
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

#%% getting the stimuli

dpath = 'K:/Project/Project 3 - Behavioural studies/stimuli/'
fname = 'stimuli_with_questions_selection.xlsx'

stimuli = pd.read_excel(os.path.join(dpath, fname))

# %% get the word frequency
def get_word_frequency(word, wf_type='Zipf', wf_file='SUBTLEX-NL with pos and Zipf.txt'):
    """
    Get word frequencies as given by the SUBTLEX-NL corpus.

    Parameters
    ----------
    word : str
        Word (lowered characters)
    wf_type : str
        zipf | Lg10CD | FREQcount
    fallback : float
        Value to fall back to if word is not in corpus. Default to the minimum 
        value encountered in the corpus.

    Returns
    -------
    float
        -log(Word frequency) of a word

    """
    if "df_wf" not in globals():
        global df_wf
        df_wf = pd.read_csv(os.path.join('K:/Project/Project 3 - Behavioural studies/5 - metadata', wf_file), delim_whitespace=True)
    if word.lower() not in df_wf.Word.to_list():
        return min(df_wf[wf_type].values) # return lowest value
    else:
        return df_wf.loc[df_wf.Word==word.lower(), wf_type].to_numpy()[0]

# %% get the stimulus order per version
dpath = 'K:/Project/Project 3 - Behavioural studies/stimuli/'

orders = {}

for lst in [1,2,3,4]:
    orders[lst] = {}
    
    with open(os.path.join(dpath, f'frinex_main_stimuli_questions_list{lst}.txt'), 'r') as f:
        stimuli = f.readlines()
    
    orders[lst] = pd.DataFrame(columns=['list_position', 'stimulus_id', 'agreement', 'surprisal'], dtype=object)
   
    position = []
    stimid = []
    agr = []
    sur = []
    
    for stim in stimuli:
        splitstim =  stim.split(" ")
       
        identifier = splitstim[2].split("__")
        
        if identifier[1].startswith('F'):
            continue
        elif splitstim[-1].startswith('question'):
            continue
        
        else:
            stimid.append(identifier[1])
            agr.append(identifier[2])
            sur.append(identifier[3])
            
            pos = int(splitstim[1].split("=")[1][1:-1])
            position.append(pos)
        
    orders[lst]['list_position'] = position
    orders[lst]['stimulus_id'] = stimid
    orders[lst]['agreement'] = agr
    orders[lst]['surprisal'] = sur
         # [stim.split(" ")[1][5:] for stim in stimuli]

# %% create a dataframe with all words underneath each other
df_words = pd.DataFrame(columns=['set_number', 'cat_surprisal', 'agreement', 'correct_number', 'tense', 'plural_type', 'list', 'list_position', 'word', 'word_index', 'word_length', 'word_frequency'],
                        data=np.zeros((1600,12)))

target_stimuli = stimuli.loc[stimuli['cat_surprisal'] != 'filler']

iterator = 0
for i,sentence in target_stimuli.iterrows():
    
    # only take the not-fillers
    words = sentence['sentence'].split(' ')
    listorder = orders[sentence['list']]
    
    list_position = listorder.loc[listorder['stimulus_id'] == str(sentence['set_number']), 'list_position'].values
    if len(set(list_position)) > 1:
        print('More than one stimulus appears to match, please check')
        break
    else:
        list_position = list_position[0]

    for x, word in enumerate(words):
        df_words.iloc[iterator] = sentence[['set_number', 'cat_surprisal', 'agreement', 'correct_number', 'tense', 'plural_type', 'list']]
        df_words.iloc[iterator, 7] = list_position
        df_words.iloc[iterator, 8] = word
        df_words.iloc[iterator, 9] = x
        df_words.iloc[iterator, 10] = len(word)
        df_words.iloc[iterator, 11] = get_word_frequency(word)
        iterator += 1

# %% store
df_words.to_csv('K:/Project/Project 3 - Behavioural studies/5 - metadata/frequency_length_position.csv')

# %% let's add surprisal
df_words = pd.read_csv('K:/Project/Project 3 - Behavioural studies/5 - metadata/frequency_length_position.csv')
surprisal_values = pd.read_excel('K:/Project/Project 3 - Behavioural studies/stimuli/agreement_surprisal.xlsx')

df_words['surprisal_value'] = np.zeros((len(df_words)))
surps = []

for i,row in df_words.iterrows():
    surps.append(surprisal_values.loc[(surprisal_values['set_number'] == row['set_number']) & (surprisal_values['cat_surprisal'] == row['cat_surprisal']) &
                                                                (surprisal_values['agreement'] == row['agreement']), 'surprisal'].values[0])
    
df_words['surprisal_value'] = np.asarray(surps)
    
df_words.to_csv('K:/Project/Project 3 - Behavioural studies/5 - metadata/frequency_length_position.csv')

