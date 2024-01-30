# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:30:45 2023

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

# remove empty rows
#stimuli = stimuli[0:192]

# %% creating a sentence column
#stimuli['sentence'] = stimuli['context'] + ' ' + stimuli['target'] + ' ' + stimuli['spillover'] + '.'

# %% creating a column for every word
word_stimuli = pd.DataFrame(columns=['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10'],index=range(len(stimuli)), dtype='object')


# %% frinex frame generation
"""
<stimulus code="0" identifier="G1_42_correct_high_surprisal__sentence__wordnumber__word.(TARGET-2)" label="---- word -- ------ --" pauseMs="0" tags="main word list1"

tags:
    practice / main
    first_word / word / last_word
    list1/ list2 / list3 / list4
    PRONOUNm2 PRONOUNm1 PRONOUN PRONOUNp1 PRONOUNp2 PRONOUNp3
    TARGETm2 TARGETm1 TARGET TARGETp1 TARGETp2 TARGETp3
"""
#randomize the dataframe
#stimuli = stimuli.sample(frac=1).reset_index(drop=True)

# %%
def generate_random_index(low=int, high=int, space_between=int, previously_generated=list):
    
    random_index = np.random.randint(low=low, high=high)

    if len(previously_generated) == 0:
        return random_index
    
    if random_index in previously_generated:
        return generate_random_index(low, high, space_between, previously_generated)
    
    distance_from_rest = [np.abs(x-random_index) for x in previously_generated]
    
    if min(distance_from_rest) < space_between+1:
        return generate_random_index(low, high, space_between, previously_generated)
    
    else:
        return random_index

# %% randomize the trials and create frinex structure

for lst in [1,2,3,4]:
    frinex = []
    list_stimuli = stimuli.loc[stimuli['list'] == lst]
    
    # define a stimulus order - we have 200 items
    # the test trials need to be spread by at least 2
    idx_spread = []
    for tr_idx in list(range(40)):
        # generate a random number
        random_idx = generate_random_index(1, 161, 2, idx_spread)
        idx_spread.append(random_idx)
    
    #idx_spread = np.asarray(idx_spread, dtype=int)
    
    # now we fill in th rest of the trials
    idx_fillers = []
    while len(idx_fillers) < 160-40:
        random_idx = generate_random_index(1,161,0,idx_spread+idx_fillers)
            
        idx_fillers.append(random_idx)

    # now we add this to the stimulus dataframe
    list_stimuli['order'] = idx_spread + idx_fillers
    list_stimuli = list_stimuli.sort_values(by='order')
    
    for i,sentence in enumerate(list_stimuli['sentence']):
        
        sentence = sentence[0].upper() + sentence[1:] + '.'
        words = sentence.split(' ')
        frame = [len(w)*'-' for w in words]
        
        # create the frinex style notation
        # for in identifier
        code = list_stimuli.iloc[i]['order'] # this specifies the pseudorandom order of the stimuli
        number = list_stimuli.iloc[i]['set_number']
        surprisal = list_stimuli.iloc[i]['cat_surprisal']
        agreement = list_stimuli.iloc[i]['agreement']

        for k,word in enumerate(words):
                
            stim = frame.copy()
            stim[k] = word
            stim = " ".join(stim)
            
            # word order
            if k == 0:
                w_order = 'first_word'
            elif k == len(words)-1 and list_stimuli.iloc[i]['has_question'] == 0: # use last_word only if the stimulus does not have a question
                w_order = 'last_word'
            else:
                w_order = 'word'
            
            # word type
            if type(number) != int:
                tgt = 'FILLER'
            elif k == 5:
                tgt = 'TARGET'
            elif k < 5:
                tgt = f'CONTEXT{k+1}'
            elif k > 5:
                tgt = f'SPILL{k-5}'
            
            wrds = [w if words.index(w) != len(words)-1 else w[:-1] for w in words]        
            frinex_frame = f'<stimulus code="{code}" identifier="L{lst}__{number}__{agreement}__{surprisal}__{k}__{word}" label="{stim}" pauseMs="0" tags="main {w_order} {tgt}"/>'
            
            frinex.append(frinex_frame)
            
        if list_stimuli.iloc[i]['has_question'] == 1:
            k += 1 # add one to the maximum k
            question = list_stimuli.iloc[i]['question']
            a1 = list_stimuli.iloc[i]['answer1']
            a2 = list_stimuli.iloc[i]['answer2']
            correct = list_stimuli.iloc[i]['correct']
            
            qframe = f'<stimulus code="{code}" identifier="L{lst}__{number}__{agreement}__{surprisal}__{k}__q" label="{question}" ratingLabels="{a1},{a2}" correctResponses="{correct}" pauseMs="0" tags="main question"/>'
    
            frinex.append(qframe)
            
    frinex_text = "\n".join(frinex)
    
    with open(os.path.join(dpath, f'frinex_main_stimuli_questions_list{lst}.txt'), 'w') as f:
        f.write(frinex_text)
        
# %% 
frinex_text1 = "\n".join(frinex1)

# %%
with open(os.path.join(dpath, 'frinex_main_stimuli_questions_list1.txt'), 'w') as f:
    f.write(frinex_text1)
    
# %% number of words
no_words = []
for lst in [1,2,3,4]:
    no_words_list = []
    list_stimuli = stimuli.loc[stimuli['list'] == lst]
    
    for i,sentence in enumerate(list_stimuli['sentence']):
        words = sentence.split(' ')
        no_words_list.append(len(words))
        
    no_words.append(no_words_list)
    
# duration of experiments
no_words_total = [np.sum(x) for x in no_words] # same for every list
duration_seconds = (no_words_total[0]*600 + 160*500 + 40*1000) / 1000 # 600ms per word, 500ms before every sentence, and 1000 ms to answer question
duration_minutes = duration_seconds/60

print(f'Without the intro/outtro, experiment will take approx {duration_minutes} minutes.')

# %% for the questions
 # 		<stimulus code="1" identifier="L4_1_incorrect_high__op het kantoor legt de kappers meteen zijn notitieboekje neer.__10__q" label="Legt de kapper zijn notitieboekje neer?" ratingLabels="Ja,Nee" correctResponses="Ja" pauseMs="0" tags="main question list4"/>