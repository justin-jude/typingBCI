# All functions adapted from https://github.com/fwillett/speechBCI/blob/main/AnalysisExamples/analysis.py

import numpy as np
import scipy.stats
from scipy.ndimage import gaussian_filter1d
from numba import njit
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os

@njit
def meanResamples(trlConcat, nResamples):
    resampleMeans = np.zeros((nResamples, trlConcat.shape[1], trlConcat.shape[2]))
    for rIdx in range(nResamples):
        resampleIdx = np.random.randint(0,trlConcat.shape[0],trlConcat.shape[0])
        resampleTrl = trlConcat[resampleIdx,:,:]
        resampleMeans[rIdx,:,:] = np.sum(resampleTrl, axis=0)/trlConcat.shape[0]

    return resampleMeans
    
def plotPreamble():
    import matplotlib.pyplot as plt

    SMALL_SIZE=5
    MEDIUM_SIZE=6
    BIGGER_SIZE=7

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rcParams['svg.fonttype'] = 'none'
    
def makeTuningHeatmapT17(mapping, dat, window, num_conditions = 15):
    nFeat = dat[0].shape[2]
   # print(nFeat)
    nClasses = len(mapping)
    sets = [list(range(num_conditions))]
    nTrials = 0
    for cue in dat:
        nTrials +=  cue.shape[0]
        
    trialVectors = np.zeros([nTrials, nFeat])
    predVectors = np.zeros([nTrials, nFeat])
    trial_cues = []
    tuningR2 = np.zeros([nFeat, 1])
    tuningPVal = np.zeros([nFeat, 1])
    
    t = 0
    for cue_ind, cue in enumerate(dat):
        for trial in cue:
            trialVectors[t,:] = np.mean(trial[window[0]:window[1],:], axis=0)
            t+=1
            trial_cues.append(cue_ind)
    trial_cues = np.array(trial_cues)

    random_order = np.random.choice(np.arange(trial_cues.shape[0]), trial_cues.shape[0], replace=False)
    trial_cues = trial_cues[random_order]
    trialVectors = trialVectors[random_order, :]
   # print(trial_cues.shape)
    #split observations into folds
    nFolds = 5
    heldOutIdx = []
    minPerFold = np.floor(nTrials/nFolds).astype(np.int32)
    remainder = nTrials-minPerFold*nFolds
    if remainder>0:
        currIdx = np.arange(0,(minPerFold)).astype(np.int32)
    else:
        currIdx = np.arange(0,minPerFold).astype(np.int32)

    for x in range(nFolds):
        heldOutIdx.append(currIdx.copy())
        currIdx += len(currIdx)
        if remainder!=0 and x==remainder:
            currIdx = currIdx[0:-1]
    
    for foldIdx in range(nFolds):
        meanVectors = np.zeros([nClasses, nFeat])
        for m in range(nClasses):
            trlIdx = np.squeeze(np.argwhere(trial_cues==m))
            
            trlIdx = np.setdiff1d(trlIdx, heldOutIdx[foldIdx])
            #print(trlIdx)
            meanVectors[m,:] = np.mean(trialVectors[trlIdx,:], axis=0)

        for t in heldOutIdx[foldIdx]:
            predVectors[t,:] = meanVectors[trial_cues[t],:]

   # print(trialVectors)
   # print(predVectors)
    for setIdx in range(len(sets)):
        mSet = sets[setIdx]
        trlIdx = np.argwhere(np.in1d(trial_cues, mSet))
        SSTOT = np.sum(np.square(trialVectors[trlIdx,:]-np.mean(trialVectors[trlIdx,:],axis=0,keepdims=True)), axis=0)
        SSERR = np.sum(np.square(trialVectors[trlIdx,:]-predVectors[trlIdx,:]), axis=0)
        
        tuningR2[:,setIdx] = 1-SSERR/SSTOT
        groupVectors = []
        for m in mSet:
            trlIdx = np.argwhere(trial_cues==m)
            groupVectors.append(trialVectors[trlIdx,:])
            
        fResults = scipy.stats.f_oneway(*groupVectors,axis=0)
        tuningPVal[:,setIdx] = fResults[1]

    return tuningR2, tuningPVal

def heatmapPlotCirclesT17(tuning, isSig, clim, titles, layout):
    circle_cmap = cm.Blues(np.linspace(0,1,256))
    nPlots = tuning.shape[1]
    arrRows = [ np.arange(0,64).astype(np.int32), 
                np.arange(64,128).astype(np.int32),
                np.arange(128,192).astype(np.int32),
                np.arange(192,256).astype(np.int32),
               np.arange(256,320).astype(np.int32),
               np.arange(320,384).astype(np.int32),]
        
    plt.figure(figsize=(len(arrRows)*0.8,nPlots*0.8))
    for arrIdx in range(len(arrRows)):
        for plotIdx in range(nPlots):
            plt.subplot(nPlots,len(arrRows),1+plotIdx+arrIdx*nPlots)
            
            matVals = tuning[arrRows[arrIdx], plotIdx]
            mat = np.reshape(matVals, [8,8], 'F')
            
            matVals_sig = isSig[arrRows[arrIdx], plotIdx]
            mat_sig = np.reshape(matVals_sig, [8,8], 'F')
            for x in range(8):
                for y in range(8):
                    if mat_sig[y,x]:
                        thisColor = np.round(255*mat[y,x]/clim[1]).astype(np.int32)
                        if thisColor>255:
                            thisColor = 255
                        if thisColor<0:
                            thisColor = 0
                            
                        plt.plot(x,-y,'o',color=circle_cmap[thisColor,:],markersize=4,markeredgecolor ='k',markeredgewidth=0.1)
                    else:
                        plt.plot(x,-y,'x',color='k',markersize=1,alpha=0.1)
            
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            
            ax = plt.gca()
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(0.75)
            
            if arrIdx==0:
                plt.title(titles[plotIdx],fontsize=6)
                
import matplotlib.pyplot as plt
def heatmapPlot(tuning, clim, titles, layout):
    nPlots = tuning.shape[1]
    if layout=='6v':
        arrRows = [np.arange(64,128).astype(np.int32), np.arange(0,64).astype(np.int32)]
    elif layout=='ifg':
        arrRows = [np.flip(np.arange(64,128).astype(np.int32)), np.flip(np.arange(0,64).astype(np.int32))]
        
    plt.figure(figsize=(nPlots,2), dpi=300)
    for plotIdx in range(nPlots):
        for arrIdx in range(len(arrRows)):
            plt.subplot(2,nPlots,1+plotIdx+arrIdx*nPlots)
            
            matVals = tuning[arrRows[arrIdx], plotIdx]
            mat = np.reshape(matVals, [8,8], 'F')
            
            plt.imshow(mat, aspect='auto', clim=clim, cmap='RdBu')
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            if arrIdx==0:
                plt.title(titles[plotIdx],fontsize=6)
                
#gaussian naive bayes classifier with variable time window and channel set
def gnb_loo(trials_input, timeWindow, chanIdx):
   #chanIdx = np.concatenate(( chanIdx, 384 + chanIdx), axis=-1)
    unroll_Feat = []
    for t in range(len(trials_input)):
        for x in range(trials_input[t].shape[0]):
            unroll_Feat.append(trials_input[t][x,:,:])

    unroll_Feat = np.concatenate(unroll_Feat, axis=0)
    mn = np.mean(unroll_Feat, axis=0)
    sd = np.std(unroll_Feat, axis=0)
    
    unroll_X = []
    unroll_y = []

    for t in range(len(trials_input)):
        for x in range(trials_input[t].shape[0]):
            tmp = (trials_input[t][x,:,:] - mn[np.newaxis,:])/sd[np.newaxis,:]
            b1 = np.mean(tmp[timeWindow[0]:timeWindow[1],chanIdx], axis=0)
            
            unroll_X.append(np.concatenate([b1]))
            unroll_y.append(t)

    unroll_X = np.stack(unroll_X, axis=0)
    unroll_y = np.array(unroll_y).astype(np.int32)
    
    from sklearn.naive_bayes import GaussianNB

    y_pred = np.zeros([unroll_X.shape[0]])
    for t in range(unroll_X.shape[0]):
        X_train = np.concatenate([unroll_X[0:t,:], unroll_X[(t+1):,:]], axis=0)
        y_train = np.concatenate([unroll_y[0:t], unroll_y[(t+1):]])

        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        gnb.var_ = np.ones(gnb.var_.shape)*np.mean(gnb.var_)

        pred_val = gnb.predict(unroll_X[np.newaxis,t,:])
        y_pred[t] = pred_val
        
    return y_pred, unroll_y

def plot_tsne(trials_input, timeWindow, chanIdx, mapping, random_seed):
    chanIdx = np.concatenate(( chanIdx, 384 + chanIdx), axis=-1)
    #chanIdx = chanIdx + 384
    #print(chanIdx)
    unroll_Feat = []
   # print(len(trials_input))
    for t in range(len(trials_input)):
        for x in range(trials_input[t].shape[0]):
            unroll_Feat.append(trials_input[t][x,:,:])

    unroll_Feat = np.concatenate(unroll_Feat, axis=0)
    mn = np.mean(unroll_Feat, axis=0)
    sd = np.std(unroll_Feat, axis=0)
    
    unroll_X = []
    unroll_y = []

    for t in range(len(trials_input)):
        for x in range(trials_input[t].shape[0]):
            tmp = (trials_input[t][x,:,:] - mn[np.newaxis,:])/(sd[np.newaxis,:] + 0.00000001)
            b1 = np.reshape(tmp[timeWindow[0]:timeWindow[1],chanIdx], ((timeWindow[1]-timeWindow[0]) * chanIdx.shape[-1]))
           # print(b1.shape)
            unroll_X.append(np.concatenate([b1]))
            unroll_y.append(t)

    unroll_X = np.stack(unroll_X, axis=0)
    unroll_y = np.array(unroll_y).astype(np.int32)
    
    from sklearn.decomposition import PCA
    import matplotlib.colors as mcolors
    from sklearn.manifold import TSNE
    #from umap import UMAP

    pca = TSNE(n_components=2, perplexity=10.0, init="pca", method="exact", random_state=random_seed)
    print(unroll_X.shape)
    print(unroll_y.shape)
    components = pca.fit_transform(unroll_X)
    outliers = []
    for i in range(components.shape[0]):
        
        if abs(components[i,0]) > 150 or abs(components[i,1]) > 150:
            outliers.append(i)
    if outliers != []:
        components = np.delete(components, outliers, axis=0)
        unroll_y = np.delete(unroll_y, outliers, axis=0)

    print(components.shape)
    #colors1 =  plt.cm.tab20b()
    #colors2 =  plt.cm.tab20c()
    #colors = np.vstack((colors1, colors2))
    #mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

   # for i,_ in enumerate(components) :
    print(unroll_y)
    from matplotlib import cm as cm
    import matplotlib
    colors = []
    for tabcol in range(10):
        for pos in range(3):
            colors.append((cm.tab10(tabcol))) 
    colors.append(('black'))
    new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=31)
    #colors = new_map(np.linspace(0,1,len(mapping)))
    fig = plt.figure(figsize=(5,5), dpi=300)
    fig.set_size_inches(16, 10)
    ax = fig.add_subplot(111)
    markers = ['o', '^', '+','s']
    #ax = fig.add_subplot(projection='3d')
    all_scatters = []
    all_lables = []
    for comp in range(components.shape[0]):
        current_marker = unroll_y[comp] % 3
        label = ""
        if mapping[unroll_y[comp]] not in all_lables:
            all_lables.append(mapping[unroll_y[comp]])
            label = mapping[unroll_y[comp]]
        if unroll_y[comp] == 30:
            current_marker = 3
        if markers[current_marker] in ['o', 's', '^']:
            all_scatters.append(ax.scatter(components[comp,0], components[comp,1],color=new_map(unroll_y[comp]), s=100, marker=markers[current_marker], label=label, facecolors='none'))
        else:
            all_scatters.append(ax.scatter(components[comp,0], components[comp,1],color=new_map(unroll_y[comp]), s=100, marker=markers[current_marker], label=label))
    #print(mapping)
    x = np.arange(10)


   # ax = plt.subplot(111)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    lgd = ax.legend(handles=all_scatters, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #plt.xlabel('PC1')
   # plt.ylabel('PC2')
    
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    #plt.savefig('pca_scatter_6v6d_fingersweep.png')
    fig.savefig('T17ptsneHollow-seed-' + str(random_seed) + '.pdf', bbox_inches='tight')
    plt.show()
    return

def plot_pca_trajectories(trials_input, timeWindow, chanIdx, mapping):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from scipy.ndimage.filters import gaussian_filter1d
    import matplotlib.cm as cm
    chanIdx = np.concatenate(( chanIdx, 384 + chanIdx), axis=-1)
    #chanIdx = chanIdx + 384
    #print(chanIdx)
    unroll_Feat = []
    #print(len(trials_input))
    for t in range(len(trials_input)):
        #for x in range(trials_input[t].shape[0]):
        #print(trials_input[t].shape)
        #for x in range(trials_input[t].shape[0]):
          #  if t == 6:
            #    trials_input[t] = np.delete(trials_input[t], 5, axis=0)
            #    break
                #print(t, x, np.sum(trials_input[t][x,timeWindow[0]:timeWindow[1],chanIdx],axis=(0,1)))
                
        unroll_Feat.append(np.median(trials_input[t][:,timeWindow[0]:timeWindow[1],chanIdx],axis=0))

    unroll_Feat = np.vstack(unroll_Feat)
    print(unroll_Feat.shape, 'hstack shape')
    mn = np.mean(unroll_Feat, axis=0)
    sd = np.std(unroll_Feat, axis=0) + 0.0000001
    unroll_Feat = (unroll_Feat - mn)/sd
    trial_size = timeWindow[1] - timeWindow[0]
    print(unroll_Feat.shape)
    combine_pca = PCA(n_components=5).fit_transform(unroll_Feat).T
    component_variance = PCA().fit(unroll_Feat).explained_variance_ratio_
    #combine_pca = unroll_Feat[timeWindow[0]:timeWindow[1],chanIdx]
    combine_pca = combine_pca.reshape((5, len(trials_input), trial_size))

    from matplotlib import cm as cm
    import matplotlib
    print(combine_pca.shape)
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0

    whole_color_list = [(cm.gist_rainbow(i)) for i in range(0,256)][::-1] 
    colors = ['#99004C', '#FF007F', '#FF66B2', '#990099', '#FF00FF', '#FF66FF', '#4C0099', '#7F00FF', '#CC99FF', '#000099', '#0000FF', '#99CCFF', '#009999', '#00FFFF', '#99FFFF', '#00CC00', '#33FF33', '#B2FF66', '#CCCC00', '#FFFF33', '#FFFF99', '#CC6E00', '#FF9933', '#FFCC99', '#CC0000', '#FF0000', '#FF9999', 'sienna', 'chocolate', 'peru', 'black']
    linestyles = ['solid', 'solid', 'solid'] * 10
    linestyles.append('solid')
   # print(colors)
    #colors[-1] = [0,0,0,1]
    
    for i in range(5):
        fig = plt.figure(figsize=(10, 5), dpi=300)
        ax = fig.add_subplot(111)
        for kk in range(combine_pca.shape[1]):
            x = combine_pca[i, kk, :]
            x = gaussian_filter1d(x, sigma=20)
            ax.plot((np.arange(timeWindow[0],timeWindow[1]))/100, x, color=colors[kk], linestyle=linestyles[kk])
            
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        lgd = ax.legend(labels=mapping, loc='center left', bbox_to_anchor=(1, 0.5))
        for line in lgd.get_lines():
            line.set_linewidth(4)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        #def add_stim_to_plot(ax):
        shade_alpha = 0.2
        lines_alpha = 0.3
        #"delay_min_ms": 3500,
        #"delay_max_ms": 4500,
        start_stim = 0 #T17: 0 T18:0
        end_stim = 100  #T17: 100  T18:100
        go_cue = 450   #T17: 450 T18:250
        trial_end = 700  #T17: 700 T18:500
        iti_end = 950  #T17: 950 T18:750
        text_height = np.percentile(combine_pca[i, :, :], 99) + 6.8
        print(text_height)
        ax.axvspan((start_stim)/100, (end_stim)/100, alpha=shade_alpha,color='gray')
        ax.axvline((start_stim)/100, alpha=lines_alpha, color='gray', ls='--')
        plt.text(((end_stim)/100) - 0.87,text_height,'Delay', fontsize=16)
        ax.axvline((end_stim)/100, alpha=lines_alpha, color='gray', ls='--')
        ax.axvline((go_cue)/100, alpha=lines_alpha, color='gray', ls='--')
        plt.text(((go_cue)/100) - 0.45,text_height,'Go', fontsize=16)
        ax.axvline((trial_end)/100, alpha=lines_alpha, color='gray', ls='--')
        plt.text(((trial_end)/100) - 1.3,text_height,'Trial End', fontsize=16)
        ax.axvline((iti_end)/100, alpha=lines_alpha, color='gray', ls='--')
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=14)
        plt.text(((iti_end)/100) -1.2,text_height,'ITI End', fontsize=16)
        plt.xticks(np.arange(start_stim, iti_end+100, 100)/100)
        plt.xlabel('Time (s)', fontsize=16)
        plt.ylabel('PC' + str(i+1), fontsize=18, rotation=0)
        #plt.xticks((np.arange(timeWindow[0],timeWindow[1])*2)/1000)
        plt.title('Explained Variance: ' + str(round(component_variance[i]*100,2)) + '%', fontsize=14)
        plt.savefig('PC' + str(i+1) + 'T17PCATrajectoriesNewColors.pdf', bbox_inches="tight")
        plt.show()
    return
    

def bootCI(x,y):
    nReps = 10000
    bootAcc = np.zeros([nReps])
    for n in range(nReps):
        shuffIdx = np.random.randint(len(x),size=len(x))
        bootAcc[n] = np.mean(x[shuffIdx]==y[shuffIdx])
        
    return np.percentile(bootAcc,[10, 90])
            

def plot_psth_jjj(trials_input, timeWindow, chanIdx, mapping, subtractMeansWithinBlock =False):
    #plot the PSTH for the channel specified by channelIdx
    plotPreamble()
    avg_tx = []
    ci_tx = []
    for t in range(len(trials_input)):
        trials_input[t] = gaussian_filter1d(trials_input[t], 6, axis=1)
        avg_tx.append(np.mean(trials_input[t], axis=0))
        ci_tx.append(np.percentile(meanResamples(trials_input[t], 1000), [2.5, 97.5], axis=0, method='linear'))   
    avg_tx = np.stack(avg_tx)
    ci_tx = np.stack(ci_tx)

    print(avg_tx.shape)
    print(ci_tx.shape)

    timeAxis = np.arange(timeWindow[0]-100,timeWindow[1]-100)*0.02

    
    legends = [['Upwards','Downwards','In to Palm','Do Nothing']]
    three_positions_each_finger = [[0,1,2,30],[3,4,5,30],[6,7,8,30],[9,10,11,30],[12,13,14,30],[15,16,17,30], [18,19,20,30],[21,22,23,30],[24,25,26,30], [27,28,29,30]]
    channelIdx_finger = [352,357,367,301,382,370,374,382,302,382]
    setTitles = ['Right Pinky','Right Ring','Right Middle','Right Index','Right Thumb','Left Pinky','Left Ring','Left Middle', 'Left Index', 'Left Thumb']
    plt.figure(figsize=(len(setTitles)*(4/5),0.7), dpi=500)
    for channelIdx, setIdx in zip(channelIdx_finger, range(len(three_positions_each_finger))):
        conIdx = three_positions_each_finger[setIdx]
        plt.subplot(1,len(three_positions_each_finger)+2,setIdx+1)
        lines = []
        for c in range(len(conIdx)):
            tmp = plt.plot(timeAxis, 50*avg_tx[conIdx[c], :, channelIdx],linewidth=1)
            lines.append(tmp[0])
            plt.fill_between(timeAxis, 
                             50*ci_tx[conIdx[c], 0, :, channelIdx], 
                             50*ci_tx[conIdx[c], 1, :, channelIdx],alpha=0.3)
        plt.ylim([0,170])
        plt.plot([0,0],plt.gca().get_ylim(),'--k',linewidth=0.75)
        if setIdx>0:
            plt.gca().set_yticklabels([])
        else:
            plt.ylabel('TX Rate (Hz)', fontsize=9)
        ax = plt.gca()
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.75)
        ax.tick_params(length=2)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=8)
        plt.xlabel('Time (s)', fontsize=8)
        plt.title(setTitles[setIdx])
        plt.figtext(ax.get_position().x0 +0.028,-0.37, str(channelIdx), ha="center", va="top", fontsize=7, color="k")
        if setIdx ==0:
            plt.figtext(ax.get_position().x0 -0.03,-0.36, 'Channel ', ha="center", va="top", fontsize=7, color="k")
    plt.legend(lines,legends[0],loc='upper left', bbox_to_anchor=(1.25, 1.0),fontsize=9,frameon=False)
        #plt.xlim([-0.5,1.0])
    plt.savefig('PSTH_movements_per_finger150lim.pdf',bbox_inches='tight')
    plt.show()

def plot_psth_jjj_loop_all(trials_input, timeWindow, chanIdx, mapping, subtractMeansWithinBlock =False):
    #plot the PSTH for the channel specified by channelIdx
    plotPreamble()
    avg_tx = []
    ci_tx = []
    for t in range(len(trials_input)):
        trials_input[t] = gaussian_filter1d(trials_input[t], 8, axis=1)
        avg_tx.append(np.mean(trials_input[t], axis=0))
        ci_tx.append(np.percentile(meanResamples(trials_input[t], 1000), [2.5, 97.5], axis=0, method='linear'))   
    avg_tx = np.stack(avg_tx)
    ci_tx = np.stack(ci_tx)
    print(avg_tx.shape)
    print(ci_tx.shape)

    timeAxis = np.arange(timeWindow[0]-100,timeWindow[1]-100)*0.02

    for chann in chanIdx:
        channelIdx_finger = [chann] * 10
        legends = [['In','Down','Up','None']]
        three_positions_each_finger = [[15,16,17,30], [18,19,20,30],[21,22,23,30],[24,25,26,30], [27,28,29,30] ,[12,13,14,30],[9,10,11,30], [6,7,8,30],[3,4,5,30], [0,1,2,30]]
        #channelIdx_finger = [352,357,367,300,382,370,374,382,302,382]
        setTitles = ['Left Pinky','Left Ring','Left Middle', 'Left Index', 'Left Thumb','Right Thumb','Right Index','Right Middle','Right Ring', 'Right Pinky']
        plt.figure(figsize=(len(setTitles)*(4/5),0.7), dpi=500)
        for channelIdx, setIdx in zip(channelIdx_finger, range(len(three_positions_each_finger))):
            conIdx = three_positions_each_finger[setIdx]
            plt.subplot(1,len(three_positions_each_finger)+2,setIdx+1)
            lines = []
            for c in range(len(conIdx)):
                tmp = plt.plot(timeAxis, 50*avg_tx[conIdx[c], :, channelIdx],linewidth=1)
                lines.append(tmp[0])
                plt.fill_between(timeAxis, 
                                 50*ci_tx[conIdx[c], 0, :, channelIdx], 
                                 50*ci_tx[conIdx[c], 1, :, channelIdx],alpha=0.3)
            plt.ylim([-30, 50])
            plt.plot([0,0],plt.gca().get_ylim(),'--k',linewidth=0.75)
            if setIdx>0:
                plt.gca().set_yticklabels([])
            else:
                plt.ylabel('TX Rate (ΔHz)', fontsize=9)
            ax = plt.gca()
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(0.75)

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(length=2)
            plt.xticks(fontsize=7)
            plt.yticks([-20, 0, 20, 40],fontsize=8)
            
            plt.xlabel('Time (s)', fontsize=8)
            plt.title(setTitles[setIdx], fontsize=6)
            plt.figtext(ax.get_position().x0 +0.028,-0.37, str(channelIdx), ha="center", va="top", fontsize=5, color="k")
            if setIdx ==0:
                plt.figtext(ax.get_position().x0 -0.03,-0.36, 'Channel ', ha="center", va="top", fontsize=6, color="k")

        lgd = plt.legend(lines,legends[0],loc='upper left', bbox_to_anchor=(1.25, 1.0),fontsize=9,frameon=False)
        for line in lgd.get_lines():
            line.set_linewidth(3)
        plt.show()
