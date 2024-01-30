# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 11:17:09 2023

@author: sopsla
"""
import os
import numpy as np
import pandas as pd
from utils import extract_RTs, extract_RTs_stimuluswise

# %% get the stimulus information
stim_info = pd.read_csv('K:/Project/Project 3 - Behavioural studies/5 - metadata/frequency_length_position.csv')

# %% the pilot subjects - need to remove them
pilot_info = pd.read_excel('I:/shared/Sophie/Self-paced reading/2 - SPR - pilot/pp_log_SPR_pilot.xlsx')
pilot_ids = pilot_info['pp_ID_Frinex'].values

# %% loop over the files
# load csv file
resultsdir = "K:/Project/Project 3 - Behavioural studies/3 - SPR main/data-88pp"
subjects_versions = ['v1', 'v2', 'v3', 'v4']

# gather the results in some lists
vs = []
all_subjects = []
accuracy = []

for version in subjects_versions:
    lst = version[1]
    listinfo = stim_info.loc[stim_info['list'] == int(lst)]
    listinfo = listinfo.sort_values(by=['list_position', 'word_index'])
    listinfo = listinfo.drop(['Unnamed: 0', 'list', 'agreement', 'word', 'word_index'], axis=1)
    listinfo.reset_index(inplace=True)
    
    fpath = f'{resultsdir}/{version}'
    
    results = pd.read_csv(os.path.join(fpath, 'stimulusresponse.csv'))

    print('Found the files')
    
    if len(set(results['UserId'])) != 1:
        print(f'{len(set(results["UserId"]))} participants in dataframe from folder {version}, please check')
    
        for sub_number,subject in enumerate(set(results['UserId'])):
            if subject in pilot_ids:
                continue
            
             # subset data for subject
            sub_results = results.loc[results['UserId'] == subject]
            sub_results.reset_index(inplace=True)
            sub_results = sub_results.loc[sub_results['ScreenName'] == 'stimuliScreen_Test']
            sub_results.reset_index(inplace=True)
            
            sub_results[['list', 'id', 'agreement', 'surprisal', 'word_index', 'word']] = sub_results['StimulusId'].str.split('__', expand=True)
            extracted = sub_results[(sub_results['word'] == 'q')&(sub_results['ResponseGroup'] == 'defaultGroup')]
            correct = extracted['IsCorrect']
              
            vs.append(lst)
            all_subjects.append(subject)
            accuracy.append(sum(correct)/len(correct)) # accuracy calculation
            
# %% create one dataframe with all accuracy values
comprehension_accuracy = pd.DataFrame(data=np.asarray([vs,all_subjects,accuracy],dtype=object).T, columns=['list', 'participant', 'accuracy'])
comprehension_accuracy['above_75'] = comprehension_accuracy['accuracy'] >= 0.75

print(sum(comprehension_accuracy['above_75'])/len(comprehension_accuracy)*100)
# all subjects above 75%

# means and sd
print(np.mean(comprehension_accuracy['accuracy']))
print(np.std(comprehension_accuracy['accuracy']))