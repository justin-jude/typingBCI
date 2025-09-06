# Sourced from https://github.com/fwillett/speechBCI/blob/main/NeuralDecoder/neuralDecoder/utils/lmDecoderUtils.py

import os
import lm_decoder
import numpy as np
import time

def lm_decode(decoder, logits, returnNBest=False, rescore=False,
              blankPenalty=0.0,
              logPriors=None):
    assert len(logits.shape) == 2

    if logPriors is None:
        logPriors = np.zeros([1, logits.shape[1]])
    lm_decoder.DecodeNumpy(decoder, logits, logPriors, blankPenalty)
    decoder.FinishDecoding()
    if rescore:
        decoder.Rescore()


    if not returnNBest:
        if len(decoder.result()) == 0:
            decoded = ''
        else:
            decoded = decoder.result()[0].sentence
    else:
        decoded = []
        for r in decoder.result():
            decoded.append((r.sentence, r.ac_score, r.lm_score))

    decoder.Reset()

    return decoded


def _cer_and_wer(decodedSentences, trueSentences, outputType='handwriting',
                 returnCI=False):
    allCharErr = []
    allChar = []
    allWordErr = []
    allWord = []
    for x in range(len(decodedSentences)):
        decSent = decodedSentences[x]
        trueSent = trueSentences[x]
        
        nCharErr = calc_wer([c for c in trueSent], [c for c in decSent])
        trueWords = trueSent.split(" ")
        decWords = decSent.split(" ")
        nWordErr = calc_wer(trueWords, decWords)

        allCharErr.append(nCharErr)
        allWordErr.append(nWordErr)
        allChar.append(len(trueSent))
        allWord.append(len(trueWords))

    cer = np.sum(allCharErr) / np.sum(allChar)
    wer = np.sum(allWordErr) / np.sum(allWord)

    if not returnCI:
        return cer, wer, np.array(allCharErr)/np.array(allChar), np.array(allWordErr)/np.array(allWord)
    else:
        allChar = np.array(allChar)
        allCharErr = np.array(allCharErr)
        allWord = np.array(allWord)
        allWordErr = np.array(allWordErr)

        nResamples = 10000
        resampledCER = np.zeros([nResamples,])
        resampledWER = np.zeros([nResamples,])
        for n in range(nResamples):
            resampleIdx = np.random.randint(0, allChar.shape[0], [allChar.shape[0]])
            resampledCER[n] = np.sum(allCharErr[resampleIdx]) / np.sum(allChar[resampleIdx])
            resampledWER[n] = np.sum(allWordErr[resampleIdx]) / np.sum(allWord[resampleIdx])
        cerCI = np.percentile(resampledCER, [2.5, 97.5])
        werCI = np.percentile(resampledWER, [2.5, 97.5])

        return (cer, cerCI[0], cerCI[1]), (wer, werCI[0], werCI[1])


def cer_with_lm_decoder(decoder, inferenceOut, includeSpaceSymbol=True,
                        outputType='speech',
                        returnCI=False,
                        rescore=False,
                        blankPenalty=0.0,
                        logPriors=None):
    # Reshape logits to kaldi order
    logits = inferenceOut['logits']
    print(np.argmax(logits[0], -1))
    logits = rearrange_speech_logits(logits, False)
    trueSentences = _extract_transcriptions(inferenceOut)
    print(trueSentences)
    #trueSentences = inferenceOut['transcriptions']
    # Decode with language model
    decodedSentences = []
    decodeTime = []
    for l in range(len(inferenceOut['logits'])):
        decoder.Reset()

        logitLen = inferenceOut['logitLengths'][l]
        start = time.time()
        decoded = lm_decode(decoder,
                            logits[l],
                            rescore=rescore,
                            blankPenalty=blankPenalty,
                            logPriors=logPriors)

        # Post-process
        decoded = decoded.strip()

        decoded = decoded.replace('.', '')
        decoded = decoded.replace('?', '')
        decoded = decoded.replace(',', '')
        decoded = decoded.replace(' , ', '')
        decoded = decoded.replace('~', '')
        decoded = decoded.strip()
        decoded = decoded.strip()
        decoded = decoded.strip()
        
        if decoded[-1] == 'm':
            decoded = decoded[:-1]
        decoded = decoded.strip()
        decodeTime.append((time.time() - start) * 1000)
        decodedSentences.append(decoded)

    assert len(trueSentences) == len(decodedSentences)

    cer, wer, all_cer, all_wer = _cer_and_wer(decodedSentences, trueSentences, outputType, returnCI)

    return {
        'cer': cer,
        'wer': wer,
        'all_cer' : all_cer,
        'all_wer' : all_wer, 
        'decoded_transcripts': decodedSentences,
        'true_transcripts': trueSentences,
        'decode_time': decodeTime
    }

def build_lm_decoder(model_path,
                        max_active=7000,
                        min_active=200,
                        beam=17.,
                        lattice_beam=8.,
                        acoustic_scale=1.5,
                        ctc_blank_skip_threshold=1.0,
                        length_penalty=0.0,
                        nbest=1):

        decode_opts = lm_decoder.DecodeOptions(
            max_active,
            min_active,
            beam,
            lattice_beam,
            acoustic_scale,
            ctc_blank_skip_threshold,
            length_penalty,
            nbest
        )

        TLG_path = os.path.join(model_path, 'TLG.fst')
        words_path = os.path.join(model_path, 'words.txt')
        G_path = os.path.join(model_path, 'G.fst')
        rescore_G_path = os.path.join(model_path, 'G_no_prune.fst')
        if not os.path.exists(rescore_G_path):
            rescore_G_path = ""
            G_path = ""
        if not os.path.exists(TLG_path):
            raise ValueError('TLG file not found at {}'.format(TLG_path))
        if not os.path.exists(words_path):
            raise ValueError('words file not found at {}'.format(words_path))

        decode_resource = lm_decoder.DecodeResource(
            TLG_path,
            G_path,
            rescore_G_path,
            words_path,
            ""
        )
        decoder = lm_decoder.BrainSpeechDecoder(decode_resource, decode_opts)

        return decoder

def rearrange_speech_logits(logits, has_sil=False):
    if not has_sil:
        logits = np.concatenate([logits[:, :, -1:], logits[:, :, :-1]], axis=-1)
    else:
        logits = np.concatenate([logits[:, :, -1:], logits[:, :, -2:-1], logits[:, :, :-2]], axis=-1)
    return logits

def calc_wer(r, h):
    # initialisation
    import numpy
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
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

def _extract_transcriptions(inferenceOut):
    transcriptions = []
    for i in range(len(inferenceOut['transcriptions'])):
        endIdx = np.argwhere(inferenceOut['transcriptions'][i] == 0)[0, 0]
        trans = ''
        for c in range(endIdx):
            trans += chr(inferenceOut['transcriptions'][i][c])
        trans = trans.strip()
        trans = trans.replace('.', '')
        trans = trans.replace('?', '')
        trans = trans.replace(',', '')
        trans = trans.replace(' > ', ' ')
        trans = trans.replace('~', '')
        trans = trans.strip()
        trans = trans.strip()
        transcriptions.append(trans)

    return transcriptions

def _extract_true_sentences(inferenceOut):
    CHAR_DEF = [
        '>', ',', '?', '.',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 
        'H', 'I', 'J', 'K', 'L', 'M', 'N', 
        'O', 'P ', 'Q', 'R', 'S', 'T', 'U', 
        'V', 'W', 'X', 'Y', 'Z']

    trueSentences = []
    for i in range(len(inferenceOut['trueSeqs'])):
        trueSent = ''
        endIdx = np.argwhere(inferenceOut['trueSeqs'][i] == -1)
        endIdx = endIdx[0,0]
        for c in range(endIdx):
            trueSent += CHAR_DEF[inferenceOut['trueSeqs'][i][c]]
        trueSentences.append(trueSent)

    return trueSentences
