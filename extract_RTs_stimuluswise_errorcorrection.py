# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:22:42 2023

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

all_subjects = []
all_responses = []
missing_subjects = []

for version in subjects_versions:
    lst = version[1]
    listinfo = stim_info.loc[stim_info['list'] == int(lst)]
    listinfo = listinfo.sort_values(by=['list_position', 'word_index'])
    listinfo = listinfo.drop(['Unnamed: 0', 'list', 'agreement', 'word', 'word_index'], axis=1)
    listinfo.reset_index(inplace=True)
    
    fpath = f'{resultsdir}/{version}'
    
    results = pd.read_csv(os.path.join(fpath, 'stimulusresponse.csv'))
    timestamps = pd.read_csv(os.path.join(fpath, 'timestampdata.csv'))
    
    print('Found the files')
    
    if len(set(results['UserId'])) != 1:
        print(f'{len(set(results["UserId"]))} participants in dataframe from folder {version}, please check')
    
        for sub_number,subject in enumerate(set(results['UserId'])):
            if subject in pilot_ids:
                continue
            
             # subset data for subject
            sub_results = results.loc[results['UserId'] == subject]
            sub_results.reset_index(inplace=True)
            sub_timestamps = timestamps.loc[timestamps['UserId'] == subject]
            sub_timestamps.reset_index(inplace=True)
            
            # get RTs for all words
            RTs = extract_RTs_stimuluswise(sub_timestamps, sub_results)
            RTs['UserId'] = [subject] * len(RTs)
                    
            if len(RTs) != 1793:
                print("dataframe not correct length")
                print(len(RTs))
            
            all_responses.append(RTs)

            # extract only the experimental trials
            RTs = RTs[(RTs['surprisal'] == 'low')|(RTs['surprisal'] == 'high')]
            RTs.reset_index(inplace=True)

            # COMBINE the two dataframes
            RTs = pd.concat([RTs, listinfo], axis=1)         
            RTs['UserId'] = [subject] * len(RTs)
            RTs = RTs.T.drop_duplicates().T
            all_subjects.append(RTs.copy())
        
            
# %%
for sub in all_subjects:
    print(sub.columns)
        
# %%
RTs = pd.concat(all_subjects)
RTs_with_fillers = pd.concat(all_responses)

# here we can check how many nan values we have
#sum(RTs_with_fillers['RT'].isna()) # with 15 participants: 9 words (= 1 sentence)
sum(RTs['RT'].isna()) # no issues

# %% save the data
RTs.to_csv('K:/Project/Project 3 - Behavioural studies/3 - SPR main/analysis/readingtimes_88pp.csv')
RTs_with_fillers.to_csv('K:/Project/Project 3 - Behavioural studies/3 - SPR main/analysis/readingtimes_fillers_88pp.csv')

#RTs.reset_index(inplace=True)

        