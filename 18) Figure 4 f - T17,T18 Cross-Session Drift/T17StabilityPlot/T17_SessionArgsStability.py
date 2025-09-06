import numpy as np

def get_session_info(session):

    if session == 't17.2024.04.25':
        block_nums = [1,3,4,5,6]
        num_test_trials = 1
        trials_to_remove = {}
    

    elif session == 't17.2024.06.03':
        block_nums = [4,5,7,8,9,11,12]
        num_test_trials = 10
        trials_to_remove = {}

    elif session == 't17.2024.06.04':
        block_nums = [1,2,3,5,6]
        num_test_trials = 100
        trials_to_remove = {}

    elif session == 't17.2024.06.05':
        block_nums = [1,2,6]
        num_test_trials = 60
        trials_to_remove = {}

    elif session == 't17.2024.06.13':
        block_nums = [7,8]
        num_test_trials = 40
        trials_to_remove = {7:[0,1,2]}

    elif session == 't17.2024.07.09':
        block_nums = [1,2,3]
        num_test_trials = 60
        trials_to_remove = {}

    elif session == 't17.2024.07.10':
        block_nums = [2]
        num_test_trials = 20
        trials_to_remove = {}


    elif session == 't17.2024.07.22':
        block_nums = [1,2,3]
        num_test_trials = 60
        trials_to_remove = {1:[18]}
        




    # elif session == 't15.2023.08.13':
    #     if not use_synthesis_blocks:
    #         block_nums = [1,2,3,4,5,7,8,9]
    #         num_test_trials = 40
    #     else:
    #         block_nums = [1,2,3,4,5,7,8,9,11,12]
    #         num_test_trials = 40

    #     trials_to_remove = {}

    

    else:
        print('[get_bad_trials.py] session name not recognized')
        trials_to_remove = {}
        block_nums = []
        num_test_trials = []

    return trials_to_remove, block_nums, num_test_trials
