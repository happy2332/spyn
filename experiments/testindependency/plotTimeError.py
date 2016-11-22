'''
Created on 05.07.2016

@author: alejomc
'''

import json
from math import floor
from matplotlib import pyplot as plt    
from natsort import natsorted, ns
import numpy
from pylab import plot, show, savefig, xlim, figure, hold, ylim, legend, boxplot, setp, axes
from sortedcontainers import SortedSet
from matplotlib.ticker import NullFormatter

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


def plotBoxes(data, col, ylabel, minticks, fname, plotXLabels=False, plotLegend=False, figsize=(6,3.6)):    
    fig = figure(figsize=figsize)
    ax = axes()
    hold(True)

    maxval = 0
    
    batch = 0
    alldims = natsorted(data.keys(), alg=ns.IGNORECASE)
    labels = SortedSet()
    for dims in alldims:
        print(dims)
        expsdata = data[dims]
        bpdata = numpy.zeros((50, 0))
        for method in natsorted(expsdata.keys(), alg=ns.IGNORECASE):
            labels.add(method)
            mdata = expsdata[method]
            print(method)
            networkerror = numpy.asarray(mdata)[:, col].reshape(50, 1)
            if numpy.max(networkerror) > maxval:
                maxval = numpy.max(networkerror)
            bpdata = numpy.append(bpdata, networkerror, axis=1)
        #bpdata = numpy.log(bpdata)    
        # print(bpdata)
        positions = numpy.asarray([1, 2, 3, 4]) + (5 * batch)
        box = boxplot(bpdata+1, positions=positions, widths=0.85)
        batch += 1
        
        colors = ['b', 'g', 'k', 'r']
        for elem in ['boxes', 'caps', 'whiskers', 'medians']:
            elems = len(box[elem])
            i = 1
            if elems > 4:
                i = 2
            for e in range(elems):
                setp(box[elem][e], linewidth=3)
                setp(box[elem][e], color=colors[floor(e / i)])
    
    
    plt.ylabel(ylabel, fontsize=18)
    if plotXLabels :
        ax.set_xticklabels(map(lambda dim: "n=" + dim.split("_")[0] + "\nd=" + dim.split("_")[1], alldims), multialignment='left', fontsize=16)
        ax.set_xticks(numpy.asarray([2.5, 7.5, 12.5, 17.5, 22.5]))
    else:
        ax.xaxis.set_major_formatter(NullFormatter())
    
    xlim(0, batch * 5)
    #ylim(-maxval * 0.1, maxval * 1.1)
    
    for b in range(batch):
        ax.axvline(x=5*b,c="k",linewidth=0.5,zorder=0)
    
    ax.set_yscale("log")
    #ylim(-10000.0,10.0)   
    
    #ax.set_yticks(numpy.arange(maxval, maxval*0.1)*2, minor=True)
    #ax.set_yticks(numpy.arange(0, maxval*1.1, minticks), minor=True)
    ax.yaxis.grid(True, which='major', linestyle='--', color='k')
    ax.yaxis.grid(True, which='minor', linestyle='--', color='#DBDBDB')
    [line.set_zorder(3) for line in ax.lines]
    
    if plotLegend:
        leg = legend(list(map(lambda l: l.replace("_", " ").upper().replace("XMRF", "LPGM").replace("GLMPTEST ", "PSPN p="), labels)), bbox_to_anchor=(0.54, 1.02))
        ax.add_artist(leg)
        ltext = leg.get_texts()
        setp(ltext[0], fontsize=14, color='b')
        setp(ltext[1], fontsize=14, color='g')
        setp(ltext[2], fontsize=14, color='k')
        setp(ltext[3], fontsize=14, color='r')
        for line, txt in zip(leg.get_lines(), leg.get_texts()):
            line.set_linewidth(10)
            line.set_color(txt.get_color())
        for lh in leg.legendHandles:
            lh.set_dashes((None, None))
        
    [tick.label.set_fontsize(16) for tick in ax.yaxis.get_major_ticks()]
    
    
    
    
         

        

    savefig(fname, bbox_inches='tight', dpi=600)
    
with open('gnspnoutfile3_withtime.txt') as data_file:    
    data = json.load(data_file)


plotBoxes(data, 0, "Time (in Seconds)", 100, "plot_time.pdf", plotXLabels=True, figsize=(6,3.3))
plotBoxes(data, 1, "Difference in Edges", 2, "plot_error.pdf", plotLegend=True, figsize=(6,4.2))
