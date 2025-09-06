import numpy as np

def get_session_info(session):

    if session == 't18.2024.12.04':
        block_nums = [5,6,7]
        num_test_trials = 1
        trials_to_remove = {5:[9,12,16]}

    elif session == 't18.2024.12.05':
        block_nums = [1,2,4,6]
        num_test_trials = 1
        trials_to_remove = {}
    
    elif session == 't18.2025.01.14':
        block_nums = [1,2,3,4,5,6,7,9] #[1,2,3,4,5,6,7,9,10,11]
        num_test_trials = 1
        trials_to_remove = {1:[0], 2:[2],3:[4],4:[1,8],10:[0,3]}

    elif session == 't18.2025.01.15':
        block_nums = [5,6,7,8,9,10,11,12,13,14] #[1,2,3,5,6,7,8,9,10,11,12,13,14] 
        num_test_trials = 88
        trials_to_remove = {5:[8], 7:[9]}

    elif session == 't18.2025.01.21':
        block_nums = [1,2,3,4,5,6,7,8,9,10,11]
        num_test_trials = 1
        trials_to_remove = {}
        
    elif session == 't18.2025.01.22':
        block_nums = [1,2,3,4,5,6,7,8,9,10,11,12,13]
        num_test_trials = 1
        trials_to_remove = {3:[1,8]}

    elif session == 't18.2025.02.04':
        block_nums = [4,6,7,8,9,10,11,12,13,14]#[1,2,3,4,6,7,8,9,10,11,12,13,14]
        num_test_trials = 1
        trials_to_remove = {1:[0,1,2,3]}

    elif session == 't18.2025.02.05':
        block_nums = [1,2,3,4,5,6,7,8,9]
        num_test_trials = 1
        trials_to_remove = {1:[0,1]}

    else:
        print('[get_bad_trials.py] session name not recognized')
        trials_to_remove = {}
        block_nums = []
        num_test_trials = []

    return trials_to_remove, block_nums, num_test_trials