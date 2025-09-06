# Sourced from https://github.com/fwillett/speechBCI/blob/main/NeuralDecoder/neuralDecoder/utils/rnnEval.py

import numpy as np

def calculate_wer(r, h):
    """
    Calculation of WER or PER with Levenshtein distance.
    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.
    ----------
    Parameters:
    r : list of true words or phonemes
    h : list of predicted words or phonemes
    ----------
    Returns:
    Word error rate (WER) or phoneme error rate (PER) [int]
    ----------
    Examples:
    >>> calculate_wer("who is there".split(), "is there".split())
    1
    >>> calculate_wer("who is there".split(), "".split())
    3
    >>> calculate_wer("".split(), "who is there".split())
    3
    """
    # initialization
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]


def _per_and_wer(decodedSentences, trueSentences, returnCI=False):
    allCharErr = []
    allChar = []
    allWordErr = []
    allWord = []
    for x in range(len(decodedSentences)):
        decSent = decodedSentences[x]
        trueSent = trueSentences[x]

        nCharErr = calculate_wer([c for c in trueSent], [c for c in decSent])

        trueWords = trueSent.split(" ")
        decWords = decSent.split(" ")
        nWordErr = calculate_wer(trueWords, decWords)

        allCharErr.append(nCharErr)
        allWordErr.append(nWordErr)
        allChar.append(len(trueSent))
        allWord.append(len(trueWords))

    per = np.sum(allCharErr) / np.sum(allChar)
    wer = np.sum(allWordErr) / np.sum(allWord)

    if not returnCI:
        return per, wer
    else:
        allChar = np.array(allChar)
        allCharErr = np.array(allCharErr)
        allWord = np.array(allWord)
        allWordErr = np.array(allWordErr)

        nResamples = 10000
        resampledPER = np.zeros([nResamples,])
        resampledWER = np.zeros([nResamples,])
        for n in range(nResamples):
            resampleIdx = np.random.randint(0, allChar.shape[0], [allChar.shape[0]])
            resampledPER[n] = np.sum(allCharErr[resampleIdx]) / np.sum(allChar[resampleIdx])
            resampledWER[n] = np.sum(allWordErr[resampleIdx]) / np.sum(allWord[resampleIdx])
        perCI = np.percentile(resampledPER, [2.5, 97.5])
        werCI = np.percentile(resampledWER, [2.5, 97.5])

        return (per, perCI[0], perCI[1]), (wer, werCI[0], werCI[1])