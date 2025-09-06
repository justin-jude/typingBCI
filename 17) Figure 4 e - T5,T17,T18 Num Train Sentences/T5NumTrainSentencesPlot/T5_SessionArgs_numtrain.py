import numpy as np

def get_session_info(session):

    if session == 't5.2019.12.09':
        block_nums = [1,2,3,5,8,13,14,15,16]
        num_test_trials = 1
        trials_to_remove = {}
    

    elif session == 't5.2019.12.11':
        block_nums = [2,4,5,8,9,10,11,13,14]
        num_test_trials = 1
        trials_to_remove = {}

    elif session == 't5.2019.12.18':
        block_nums = [3,6,7,9,10,12,14,15,17]
        num_test_trials = 1
        trials_to_remove = {}

    elif session == 't5.2019.12.20':
        block_nums = [3,4,6,9,10,11,12,13]
        num_test_trials = 1
        trials_to_remove = {7:[0,1,2]}

    elif session == 't5.2020.01.06':
        block_nums = [3,5,6,8,9,13,15,17,19]
        num_test_trials = 1
        trials_to_remove = {}

    elif session == 't5.2020.01.08':
        block_nums = [5,6,8,12,13,16,17,18,19]
        num_test_trials = 1
        trials_to_remove = {}


    elif session == 't5.2020.01.13':
        block_nums = [3,4,5,6,7,11,12,13,16,18,19]
        num_test_trials = 1
        trials_to_remove = {}

    elif session == 't5.2020.01.15':
        block_nums = [2,4,5,6,7,9,10,12,15,17,18]
        num_test_trials = 1
        trials_to_remove = {}

    

    else:
        print('[get_bad_trials.py] session name not recognized')
        trials_to_remove = {}
        block_nums = []
        num_test_trials = []

    return trials_to_remove, block_nums, num_test_trials
