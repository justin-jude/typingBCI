import numpy as np

def get_session_info(session):


    if session == 't18.2025.02.05':
        block_nums = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        num_test_trials = 100
        trials_to_remove = {1:[0,1]}

    else:
        print('[get_bad_trials.py] session name not recognized')
        trials_to_remove = {}
        block_nums = []
        num_test_trials = []

    return trials_to_remove, block_nums, num_test_trials