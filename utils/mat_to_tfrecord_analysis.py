# mat_to_tfrecord.py
# Written by Nick Card, July 2023

# This function loads multiple blocks of brain-to-text training data (.mat format, from rdb_to_mat.py) from a single session
# and then formats and converts that data to .tfrecord files (one for training, one for testing) which can be used to train
# the brain-to-text decoder.


# Adapted for Typing (T5 data) by Justin Jude 2024

import os
import re
from pathlib import Path
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
import json



# function that trializes and formats the .mat data
def formatSessionData(blocks,
                      trialsToRemove,
                      dataDir,
                      start_offset=0,
                      end_offset=0,
                      channels_to_exclude=[],
                      channels_to_zero=[],
                      includeThreshCrossings=True,
                      includeSpikePower=True,
                      spikePowerMax=10000,
                      globalStd=True,
                      zscoreData=True,
                      bin_compression_factor=1):

    inputFeatures = []
    rawInputFeatures = []
    microphoneData = []
    transcriptions = []
    frameLens = []
    trialTimes = []
    blockMeans = []
    blockStds = []
    blockList = []

    # loop through blocks
    for b in blocks:

        # load .mat file for this block
        redisFile = sorted([str(x) for x in Path(dataDir, 'RedisMat').glob('*('+str(b)+').mat')])
        redisFile = redisFile[-1]
        print(f'RedisMat file for block {b}: {redisFile}')
        redisDat = scipy.io.loadmat(redisFile)

        # loop through trials
        blockStartIdx = len(inputFeatures)
        for x in range(len(redisDat['cue'])):
            if start_offset < -150 and x == 0:
                continue
            # if this trial was specified as bad, continue
            if b in trialsToRemove and x in trialsToRemove[b]:
                print(f"Removed block {b}'s trial {x} because it was manually specified as bad.")
                continue
            elif redisDat['trial_paused_by_CNRA'][0][x]==1:
                print(f"Skipping block {b}'s trial {x} because it ended with a pause.")
                continue
            elif redisDat['trial_timed_out'][0][x]==1:
                print(f"Skipping block {b}'s trial {x} because it timed out.")
                continue

            # get start and end NSP timestamp for this trial
            startTime = redisDat['go_cue_nsp_neural_time'][0][x]
            #goCueTime = redisDat['go_cue_nsp_neural_time'][0][x]
            endTime = redisDat['trial_end_nsp_neural_time'][0][x]

            # startTime_analog = redisDat['go_cue_nsp_analog_time'][0][x]
            # endTime_analog = redisDat['trial_end_nsp_analog_time'][0][x]

            # get start and end time step for this trial
            startTimeStep = np.argmin(np.abs(redisDat['binned_neural_nsp_timestamp']-startTime)) +start_offset # T17: -450, T18:-250
            endTimeStep = np.argmin(np.abs(redisDat['binned_neural_nsp_timestamp']-endTime)) +end_offset

            # startTimeStep_analog = np.argmin(np.abs(redisDat['microphone_nsp_time']-startTime_analog))
            # endTimeStep_analog = np.argmin(np.abs(redisDat['microphone_nsp_time']-endTime_analog))

            if endTimeStep <= startTimeStep:
                print(f'Trial #{x}: endTimeStep ({endTimeStep}) is less than or equal to startTimeStep ({startTimeStep}). Skipping this trial.')
                continue

            # calulate trial duration in seconds
            trialTimes.append((endTime - startTime))

            # get neural data for this trial: Threshold crossings and/or optionally spike power.
            if includeThreshCrossings:
                thresholdCrossings = redisDat['binned_neural_threshold_crossings'][startTimeStep:endTimeStep,:].astype(np.float32)
                if channels_to_zero != []:
                    for c in channels_to_zero:
                        thresholdCrossings[:, c] = 0*thresholdCrossings[:, c]
                if channels_to_exclude != []:
                    thresholdCrossings = np.delete(thresholdCrossings, channels_to_exclude, axis=1)

            if includeSpikePower:
                spikePower = redisDat['binned_neural_spike_band_power'][startTimeStep:endTimeStep,:].copy()
                spikePower[spikePower>spikePowerMax]=spikePowerMax
                if channels_to_zero != []:
                    for c in channels_to_zero:
                        spikePower[:, c] = 0*spikePower[:, c]
                if channels_to_exclude != []:
                    spikePower = np.delete(spikePower, channels_to_exclude, axis=1)


            # optionally compress bins
            if bin_compression_factor > 1:

                if includeThreshCrossings:
                    if np.shape(thresholdCrossings)[0] % bin_compression_factor != 0:
                        # remove extra bins to make binning nice
                        bins_to_remove = np.shape(thresholdCrossings)[0] % bin_compression_factor
                        thresholdCrossings = thresholdCrossings[0:-bins_to_remove, :]

                    # sum sequential threshold crossing bins together to compress
                    thresholdCrossings = np.sum(thresholdCrossings.transpose().reshape(np.shape(thresholdCrossings)[1], -1, bin_compression_factor).transpose(1,0,2), axis=-1)

                if includeSpikePower:
                    if np.shape(spikePower)[0] % bin_compression_factor != 0:
                        # remove extra bins to make binning nice
                        bins_to_remove = np.shape(spikePower)[0] % bin_compression_factor
                        spikePower = spikePower[0:-bins_to_remove, :]
                    
                    # average sequential spikepow bins together to compress
                    spikePower = np.mean(spikePower.transpose().reshape(np.shape(spikePower)[1], -1, bin_compression_factor).transpose(1,0,2), axis=-1)


            # concatenate threshold crossings and spike power
            if includeThreshCrossings and includeSpikePower:
                newInputFeatures = np.concatenate([thresholdCrossings, spikePower], axis=1)
            elif includeThreshCrossings:
                newInputFeatures = thresholdCrossings
            elif includeSpikePower:
                newInputFeatures = spikePower

            # get microphone data for this trial
            # microphoneData.append(redisDat['microphone_data'][startTimeStep_analog:endTimeStep_analog,:].astype(np.int16))

            # get cue for this trial
            newTranscription = redisDat['cue'][x]

            # append data to lists
            rawInputFeatures.append(newInputFeatures)
            inputFeatures.append(newInputFeatures)
            transcriptions.append(newTranscription)
            frameLens.append(newInputFeatures.shape[0])
            blockList.append(b)

        # calculate block means and standard deviations
        blockEndIdx = len(inputFeatures)
        block = np.concatenate(inputFeatures[blockStartIdx:blockEndIdx], 0)
        blockMean = np.mean(block, axis=0, keepdims=True).astype(np.float32)
        blockMeans.append(blockMean)
        blockStd = np.std(block, axis=0, keepdims=True).astype(np.float32)
        blockStds.append(blockStd)

        # z-score data. If globalStd is True, use global standard deviation instead of block standard deviation.
        if zscoreData:
            for i in range(blockStartIdx, blockEndIdx):
                if globalStd:
                    inputFeatures[i] = (inputFeatures[i].astype(np.float32) - blockMean)
                else:
                    inputFeatures[i] = (inputFeatures[i].astype(np.float32) - blockMean)# / (blockStd + 1e-8)

    # calculate and optionally finish z-scoring with global standard deviation
    gStd = np.std(block, axis=0, keepdims=True).astype(np.float32)
    if globalStd and zscoreData:
        for i in range(len(inputFeatures)):
            inputFeatures[i] = (inputFeatures[i].astype(np.float32)) / (gStd + 1e-8)

    # return data
    return {
        'inputFeatures': inputFeatures,
        'rawInputFeatures': rawInputFeatures,
        # 'microphoneData': microphoneData,
        'transcriptions': transcriptions,
        'trialTimes': trialTimes,
        'frameLens': frameLens,
        'blockList': blockList,
        'blockMeans': blockMeans,
        'blockStds': blockStds,
        'globalStd': gStd,
    }
