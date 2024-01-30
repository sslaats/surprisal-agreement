# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:00:02 2023

@author: sopsla
"""
import pandas as pd
import numpy as np

def extract_RTs(timestamps, results):
    # we use 'results' to get the value of the stimulus and 'timestamps' to get the RTs
    # remove practice trials
    if len(np.where(timestamps['EventTag'] == 'main_stimuli')[0]) > 1:
        raise ValueError("More than one tag of main_stimuli found, check this participant")
        
    results = results.loc[results['ScreenName'] == 'stimuliScreen_Test']
    results.reset_index(inplace=True)
    try:
        timestamps = timestamps.iloc[np.where(timestamps['EventTag'] == 'main_stimuli')[0][0]:]
    except KeyError:
        timestamps = timestamps.loc[timestamps['ScreenName'] == 'stimuliScreen_Test']
    timestamps.reset_index(inplace=True)     
    
    sentencecounter = 0
    
    responses = []
    
    for ix, row in timestamps.iterrows():
        if ix == 0:
            continue
        
        if row['EventTag'] == 'QUESTION':
            # skip the questions
            continue
        
        elif row['EventTag'] == 'FIRST_WORD':
            # initiate new sentence
            sentencecounter += 1
            wordcounter = 0
            
            # it has happened that after FIRST_WORD there is no log of StimulusLabel_out
            # in that case we use FIRST_WORD as the start of the word
            if timestamps.iloc[ix+1]['EventTag'] == 'stimulusButton':

                # we have a word
                wordcounter += 1
                
                # compute the RTs and save them to a list with the stimulus and word counter
                RT_word = timestamps.iloc[ix+1]['EventMs'] - row['EventMs']
                responses.append([sentencecounter, wordcounter, RT_word])

        elif row['EventTag'] == 'stimulusLabel_out':
            if timestamps.iloc[ix+1]['EventTag'] == 'stimulusButton':

                # we have a word
                wordcounter += 1

                # compute the RTs and save them to a list with the stimulus and word counter
                RT_word = timestamps.iloc[ix+1]['EventMs'] - row['EventMs']
                responses.append([sentencecounter, wordcounter, RT_word])
                          
            # sometimes somehow the label appears to be presented twice (?)
            elif timestamps.iloc[ix+1]['EventTag'] == 'stimulusLabel_out' and timestamps.iloc[ix+2]['EventTag'] == 'stimulusButton':

                    # treat this as a word
                    wordcounter += 1
                    
                    # compute the RTs and save them to a list with the stimulus and word counter
                    RT_word = timestamps.iloc[ix+1]['EventMs'] - row['EventMs']
                    responses.append([sentencecounter, wordcounter, RT_word])
                    
            else:
                continue
    
    # convert it to a dataframe
    RTs = pd.DataFrame(columns = ['sentence_order', 'word_index', 'RT'],
                       data = responses)
    
    # % now we combine it with the information from the other dataframe
    results[['list', 'id', 'agreement', 'surprisal', 'word_index', 'word']] = results['StimulusId'].str.split('__', expand=True)
    results = results.loc[results['word'] != 'q'] # remove the questions
    results = results.loc[results['ResponseGroup'] != 'defaultGroup'] # remove the answers to the questions

    # get the stimulus information back into the dataframe
    stim_ids = []
    
    results = results.drop('level_0', axis=1)
    results.reset_index(inplace=True)
    for ir, row in results.iterrows():
        # skip double occurrences, caused by the questions
        if row['StimulusId'] == results.iloc[ir-1]['StimulusId']:
            # this works
            continue
        else:
            stim_ids.append(row['StimulusId'])
            
    stim_ids = pd.DataFrame(data=stim_ids, columns=['StimulusId'])
    stim_ids[['list', 'id', 'agreement', 'surprisal', 'word_index', 'word']] = stim_ids['StimulusId'].str.split('__', expand=True)
    stim_ids = stim_ids.drop(['word_index'],axis=1)            
    
    # add this to RTs
    if len(RTs) != len(stim_ids):
        raise ValueError("The dataframes are not the same length. Check this subject")
        
    RTs = pd.concat([RTs, stim_ids], axis=1)
    RTs = RTs.T.drop_duplicates().T
    
    return RTs

def extract_RTs_stimuluswise(timestamps, results):
    # we use 'results' to get the value of the stimulus and 'timestamps' to get the RTs
    # remove practice trials
    if len(np.where(timestamps['EventTag'] == 'main_stimuli')[0]) > 1:
        raise ValueError("More than one tag of main_stimuli found, check this participant")
        
    try:
        timestamps = timestamps.iloc[np.where(timestamps['EventTag'] == 'main_stimuli')[0][0]:]
    except KeyError:
        timestamps = timestamps.loc[timestamps['ScreenName'] == 'stimuliScreen_Test']
    timestamps.reset_index(inplace=True)     
    
    # we crop the timestamps between the occurrences of 'FIRST_WORD'
    start_of_sentence = np.where(timestamps['EventTag'] == 'FIRST_WORD')[0]
    stimulus_timestamps = [timestamps[start:start_of_sentence[i+1]] for i,start in enumerate(start_of_sentence[:-1])]
    stimulus_timestamps.append(timestamps[start_of_sentence[-1]:]) # last one is not included
    
    # preprocess the results dataframe (stimulusresponse)
    results = results.loc[results['ScreenName'] == 'stimuliScreen_Test']  
    results[['list', 'id', 'agreement', 'surprisal', 'word_index', 'word']] = results['StimulusId'].str.split('__', expand=True)
    results = results.loc[results['word'] != 'q'] # remove the questions
    results = results.loc[results['ResponseGroup'] != 'defaultGroup'] # remove the answers to the questions
    results.reset_index(inplace=True)

    stim_ids = []
    for ir, row in results.iterrows():
        # skip double occurrences, caused by the questions
        if row['StimulusId'] == results.iloc[ir-1]['StimulusId']:
            # this works
            continue
        else:
            stim_ids.append(row['StimulusId'])
    
    stim_ids = pd.DataFrame(data=stim_ids, columns=['StimulusId'])
    stim_ids[['list', 'id', 'agreement', 'surprisal', 'word_index', 'word']] = stim_ids['StimulusId'].str.split('__', expand=True)
    stim_ids = stim_ids.drop(['word_index'],axis=1)   
    stim_ids['all_word_index'] = np.asarray(range(0,len(stim_ids))) # to maintain order
    
    # now we loop over the stimuli
    sentencecounter = 0
    
    responses = []
    for stim_number,stimulus in enumerate(stimulus_timestamps):
        # log new sentence
        sentencecounter += 1
        
        # remove questions and answers - these also contain 'stimulusButton' and 'stimulusLabel_out', 
        # so must be removed before we select on the basis of tag
        if 'QUESTION' in set(stimulus['EventTag']):
            stimulus = stimulus[:np.where(stimulus['EventTag'] == 'QUESTION')[0][0]] # the zeros take the value out of the tuple
        
        stimulus = stimulus.loc[(stimulus['EventTag'] == 'stimulusButton')|(stimulus['EventTag'] == 'stimulusLabel_out')]
        
        # extract word responses
        # there are two 'stimulusButton' logs at the end of every stimulus
        # and sometimes three - here we remove them
        number_of_stimbutton = 0
        for reverse_row in range(stimulus.shape[0] -1, -1, -1):
            row = stimulus.iloc[reverse_row]
            if row['EventTag'] == 'stimulusButton':
                number_of_stimbutton += 1
            elif row['EventTag'] == 'stimulusLabel_out':
                break
        
        end_index = len(stimulus) - (number_of_stimbutton - 1)
        """
        if stimulus.iloc[-1]['EventTag'] == 'stimulusButton':
            
            if stimulus.iloc[-2]['EventTag'] == 'stimulusButton':
                if stimulus.iloc[-3]['EventTag'] == 'stimulusButton':
                    end_index = -2
                elif stimulus.iloc[-3]['EventTag'] == 'stimulusLabel_out':
                    end_index = -1
            elif stimulus.iloc[-2]['EventTag'] == 'stimulusLabel_out':
                end_index = None
        """
        
        stimulus = stimulus[0:end_index]
        
        if len(stimulus) % 2 != 0:
            # missing value for this sentence
            # could be because of extra button press in middle of stimulus
            # find out how many words it is supposed to have
            stim_ids_separated = {number:stim_ids.loc[stim_ids['id'] == stimd] for number,stimd in enumerate(pd.unique(stim_ids['id']))}
            # get stimulus information
            indiv_stim_info = stim_ids_separated[stim_number]
            sentence_RTs = np.empty(len(indiv_stim_info))
            sentence_RTs[:] = np.nan # we insert nan values for the missing sentence
            
            #raise ValueError(f'{str(set(stimulus["UserId"]))} has an error in stimulus number {sentencecounter}')
                
        else:
            # we put the 'StimulusButton' and 'stimulusLabel_out' tags next to each other
            # and subtract
            grouped = stimulus.groupby(by='EventTag') 
            sentence_RTs = grouped.get_group('stimulusButton')['EventMs'].values - grouped.get_group('stimulusLabel_out')['EventMs'].values
        
        sentence_RTs = pd.DataFrame(data=np.vstack([np.asarray([sentencecounter]*len(sentence_RTs)), np.asarray(range(0,len(sentence_RTs))), sentence_RTs]).T,
                                          columns=['sentence_order', 'word_index', 'RT'])
        responses.append(sentence_RTs)
            
    RTs = pd.concat(responses)
    RTs.reset_index(inplace=True)
    RTs = RTs.drop(['index'], axis=1)
    stim_ids.reset_index(inplace=True)
    stim_ids = stim_ids.drop(['index'], axis=1)
    RTs = pd.concat([RTs, stim_ids], axis=1)
    #RTs = RTs.T.drop_duplicates().T
    
    return RTs