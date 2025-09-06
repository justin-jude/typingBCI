# mat_to_tfrecord.py
# Nick Card, July 2023

# This function loads multiple blocks of brain-to-text training data (.mat format, from rdb_to_mat.py) from a single session
# and then formats and converts that data to .tfrecord files (one for training, one for testing) which can be used to train
# the brain-to-text decoder.

## TODO: add trials_to_remove parsing

import os
import re
from pathlib import Path
import numpy as np
import scipy.io
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json


CHAR_DEF = [
    '>', ',', '?', '.',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 
    'H', 'I', 'J', 'K', 'L', 'M', 'N', 
    'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
    'V', 'W', 'X', 'Y', 'Z']


# function that trializes and formats the .mat data
def formatSessionData(blocks,
                      trialsToRemove,
                      dataDir,
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
            #print(len(redisDat['cue'][x]))
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
            elif len(redisDat['cue'][x].split()) == 0 or 'DO_NOTHING' in redisDat['cue'][x]:
                print(f"Skipping block {b}'s trial {x} because the cue was empty or too short.")
                continue

            # get start and end NSP timestamp for this trial
            startTime = redisDat['go_cue_nsp_neural_time'][0][x]
            endTime = redisDat['trial_end_nsp_neural_time'][0][x]

            # startTime_analog = redisDat['go_cue_nsp_analog_time'][0][x]
            # endTime_analog = redisDat['trial_end_nsp_analog_time'][0][x]

            # get start and end time step for this trial
            startTimeStep = np.argmin(np.abs(redisDat['binned_neural_nsp_timestamp']-startTime))
            endTimeStep = np.argmin(np.abs(redisDat['binned_neural_nsp_timestamp']-endTime))

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
                    inputFeatures[i] = (inputFeatures[i].astype(np.float32) - blockMean) / (blockStd + 1e-8)

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

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _ints_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _convert_to_ascii(text):
    return [ord(char) for char in text]

# function that converts loaded data to .tfrecord files
def convertToTFRecord(sessionData, outputDir, trainTrials, testTrials):

    # initialize
    partNames = ['train', 'test']
    partSets = [trainTrials, testTrials]
    maxSeqLen = 200

    # loop through train and test sets
    for pIter in range(len(partNames)):

        partIdx = partSets[pIter]
        saveDir = Path(outputDir, partNames[pIter])
        saveDir.mkdir(parents=True, exist_ok=True)

        with tf.io.TFRecordWriter(str(saveDir.joinpath('chunk_0.tfrecord'))) as writer:
            # loop through trials in this set
            for trialIdx in partIdx:
                inputFeats = sessionData['inputFeatures'][trialIdx]

                seqClassIDs = np.zeros([maxSeqLen]).astype(np.int32)

                thisTranscription = sessionData['transcriptions'][trialIdx]

                # Remove punctuation
                thisTranscription = re.sub(r'[^a-zA-Z?>~.,\- \']', '', thisTranscription)
                thisTranscription = thisTranscription.replace('--', '').lower()
                thisTranscription = thisTranscription.replace(" '", "'").lower()
                thisTranscription = thisTranscription.strip()
                # Convert to characters
                characters = []
                if len(thisTranscription) == 0:
                    characters = '>'
                else:
                    for c_iter, c in enumerate(list(thisTranscription.upper())):
                        if c == '~':
                            characters.append('.')

                        c = re.sub(r'[0-9]', '', c)  # Remove stress
                        if c in CHAR_DEF:  # Only keep letters and punctuation
                            characters.append(c)

                    seqLen = len(characters)
                    seqClassIDs[0:seqLen] = [CHAR_DEF.index(c) + 1 for c in characters]
                    print(seqClassIDs)
                # print a warning if there are more characters than timesteps
                print(f'Transcription: {thisTranscription}')
                print(f'Characters: {characters}')
                print(f'Character seq length: {len(characters)}')
                print(f'Data size: {inputFeats.shape}')
                if inputFeats.shape[0] < len(characters):
                    print(f'WARNING! (TRIAL #{trialIdx}): DATA LENGTH SHORTER THAN CHARACTERS')

                ceMask = np.zeros([inputFeats.shape[0]]).astype(np.float32)
                ceMask[0:sessionData['frameLens'][trialIdx]] = 1

                paddedTranscription = np.zeros([maxSeqLen]).astype(np.int32)
                paddedTranscription[0:len(thisTranscription)] = np.array(_convert_to_ascii(thisTranscription))

                # create feature dict to write to .tfrecord file
                feature = {
                    'inputFeatures': _floats_feature(np.ravel(inputFeats).tolist()),
                    'seqClassIDs': _ints_feature(seqClassIDs),
                    'nTimeSteps': _ints_feature([sessionData['frameLens'][trialIdx]]),
                    'nSeqElements': _ints_feature([seqLen]),
                    'ceMask': _floats_feature(np.ravel(ceMask).tolist()),
                    'transcription': _ints_feature(paddedTranscription)
                    }

                if pIter==0 and trialIdx==partIdx[0]:
                    print(f'inputFeatures shape: {np.shape(np.ravel(inputFeats).tolist())}')
                    # print(np.ravel(inputFeats).tolist()[0])
                    print(f'seqClassIDs shape: {np.shape(seqClassIDs)}')
                    print(f'nTimeSteps: {sessionData["frameLens"][trialIdx]}')
                    print(f'seqLen: {seqLen}')
                    print(f'ceMask shape: {np.shape(np.ravel(ceMask).tolist())}')
                    print(f'transcription shape: {np.shape(paddedTranscription)}')

                if inputFeats.shape[0] > (len(characters)*4 + 100): # this is hardcoded for the normal Stanford parameters
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                else:
                    # print(f'WARNING: Not including trial #{trialIdx} in the {partNames[partIdx]} dataset because it is too short.')
                    print(f'WARNING: Not including trial #{trialIdx} because it is too short.')

                print('\n')



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    

def main(args):

    # Mandatory input parameters
    session_mat_path = args['session_mat_path']
    block_nums = args['block_nums']
    num_test_trials = args['num_test_trials']

    # Optional input parameters
    trials_to_remove = args.get('trials_to_remove', {})
    channels_to_zero = args.get('channels_to_zero', [])
    channels_to_exclude = args.get('channels_to_exclude', [])
    include_thresh_crossings = args.get('include_thresh_crossings', True)
    include_spike_power = args.get('include_spike_power', True)
    spike_power_max = args.get('spike_pow_max', 50000)
    z_score_data = args.get('z_score_data', True)
    global_std = args.get('global_std', True)
    bin_compression_factor = args.get('bin_compression_factor', 1)
    save_path = args.get('save_path', str(Path(session_mat_path, 'tfdata')))

    assert len(block_nums) > 0, "Must specify at least one block number"
    assert num_test_trials > 0, "Must specify at least one test trial"

    # print input parameters
    print('mat_to_tfrecord_V3.py input parameters:')
    print(f'\tSession mat path: {session_mat_path}')
    print(f'\tBlock numbers: {block_nums}')
    print(f'\tNumber of test trials: {num_test_trials}')
    print(f'\tTrials to remove: {trials_to_remove}')
    print(f'\tChannels to exclude: {channels_to_exclude}')
    print(f'\tChannels to zero: {channels_to_zero}')
    print(f'\tInclude threshold crossings: {include_thresh_crossings}')
    print(f'\tInclude spike power: {include_spike_power}')
    print(f'\tSpike power max: {spike_power_max}')
    print(f'\tZ-score data: {z_score_data}')
    print(f'\tGlobal std: {global_std}')
    print(f'\tBin compression factor: {bin_compression_factor}')
    print(f'\tSave path: {save_path}')
    print('\n')

    mat_to_tfrecord_params = {
        'session_mat_path': session_mat_path,
        'block_nums': block_nums,
        'trials_to_remove': trials_to_remove,
        'channels_to_exclude': channels_to_exclude,
        'channels_to_zero': channels_to_zero,
        'num_test_trials': int(num_test_trials),
        'include_thresh_crossings': include_thresh_crossings,
        'include_spike_power': include_spike_power,
        'spike_power_max': int(spike_power_max),
        'z_score_data': z_score_data,
        'global_std': global_std,
        'bin_compression_factor': bin_compression_factor,
        'save_path': save_path,
        }
    
    os.makedirs(save_path, exist_ok=True)
    with open(str(Path(save_path,'mat_to_tfrecord_params.json')), 'w') as f:
        json.dump(mat_to_tfrecord_params, f, indent=4, cls=NpEncoder)

    sessionData = formatSessionData(
        blocks = block_nums,
        trialsToRemove = trials_to_remove,
        dataDir = session_mat_path,
        channels_to_exclude = channels_to_exclude, 
        channels_to_zero = channels_to_zero,
        includeThreshCrossings = include_thresh_crossings,
        includeSpikePower = include_spike_power,
        zscoreData = z_score_data,
        globalStd = global_std,
        spikePowerMax = spike_power_max,
        bin_compression_factor = bin_compression_factor,
        )

    # plot some data for trial #idx
    idx = 1

    plt.figure(figsize=(16,8))
    plt.subplot2grid((4,1),(0,0),rowspan=2)
    plt.imshow(sessionData['inputFeatures'][idx].T, aspect='auto', clim=(-1.5,1.5), interpolation='none')
    # plt.colorbar()
    plt.xticks(np.arange(0, sessionData['inputFeatures'][idx].shape[0], 200), np.arange(0, sessionData['inputFeatures'][idx].shape[0], 200)*0.01)
    plt.xlabel("time (s)")
    plt.ylabel("Neural features")
    plt.title("Neural features for trial " + str(idx) + ": ''" + sessionData['transcriptions'][idx].strip()+ "''")

    # plt.subplot2grid((4,1),(2,0))
    # mic_time = np.arange(0, sessionData['microphoneData'][idx].shape[0]) / 30000
    # plt.plot(mic_time, sessionData['microphoneData'][idx])
    # plt.xlim([0, mic_time[-1]])
    # plt.xlabel("time (s)")

    plt.subplot2grid((4,1),(3,0),4)
    plt.plot(np.squeeze(sessionData['blockMeans'][0]) + 1E-9)
    plt.xlim([0, sessionData['blockMeans'][0].shape[1]])
    plt.yscale('log')
    plt.ylabel('Block means')
    plt.xlabel("Neural features")

    plt.tight_layout()
    plt.savefig(f'{save_path}/trial_{idx}_neural_features.png')
    print(f'Saved neural features plot to: {save_path}/trial_{idx}_neural_features.png')
    # plt.show()


    # print sentence cues and trial duration for each trial
    for i in range(len(sessionData['transcriptions'])):
        print(f'Trial {i} ({sessionData["trialTimes"][i]} seconds): {sessionData["transcriptions"][i]}')


    # decide which trials are for testing or training
    nTrials = len(sessionData['inputFeatures'])
    np.random.seed(9)
    trials = np.arange(0, nTrials)
    np.random.shuffle(trials)
    trainTrials = trials[:-num_test_trials].astype(np.int32)
    testTrials = trials[-num_test_trials:].astype(np.int32)

    print(f'train trials: {trainTrials}')
    print(f'test trials: {testTrials}')
    print('\n')

    # convert data to .tfrecord files
    convertToTFRecord(
        sessionData = sessionData, 
        outputDir   = save_path, 
        trainTrials = trainTrials, 
        testTrials  = testTrials,
        )

    # save block means
    np.save(os.path.join(save_path, 'blockMean'), np.squeeze(sessionData['blockMeans'][-1], 0))
    np.save(os.path.join(save_path, 'blockStd'), np.squeeze(sessionData['globalStd'], 0)+1e-8)

    print("Saved .tfrecord data and normalization stats to: " + save_path)
