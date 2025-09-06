import numpy as np

def get_session_info(session):

    if session == 't18.2025.04.01_CERWER':
        block_nums = [3,4,5,6,7,8,9,10] #not 11,12
        num_test_trials = 40
        trials_to_remove = {3:[20,21,22]}


    else:
        print('[get_bad_trials.py] session name not recognized')
        trials_to_remove = {}
        block_nums = []
        num_test_trials = []

    return trials_to_remove, block_nums, num_test_trials
