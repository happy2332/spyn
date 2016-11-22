'''
Created on 05.07.2016

@author: alejomc
'''

import json
from math import floor
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerLine2D
from natsort import natsorted, ns
import numpy
from pylab import plot, show, savefig, xlim, figure, hold, ylim, legend, boxplot, setp, axes
from sortedcontainers import SortedSet
from matplotlib import pyplot as plt    
import re
from glob import glob
from mlutils.benchmarks import Stats

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def plotBoxes(data, ylabel, minticks, fname, figsize=(6,3.6)):    
    fig = figure(figsize=figsize)
    ax = axes()
    hold(True)

    #plt.title(key)

    maxyval = 0
    maxxval = 0
    xtickspos = []
    batch = 0
    alldims = natsorted(data.keys(), alg=ns.IGNORECASE)
    labels = SortedSet()
    for dims in alldims:
        expsdata = data[dims]
                
        bpdata = numpy.zeros((5, 0))
        for method in natsorted(expsdata.keys(), alg=ns.IGNORECASE):
            
            labels.add(method)
            mdata = expsdata[method]
            print(method)
            networkerror = numpy.asarray(mdata).reshape(5, 1)
            if numpy.max(networkerror) > maxyval:
                maxyval = numpy.max(networkerror)
            bpdata = numpy.append(bpdata, networkerror, axis=1)
            
        # print(bpdata)
        positions = numpy.arange(1, bpdata.shape[1]+1) + ((bpdata.shape[1]+1) * batch )
        maxxval = positions[-1]
        xtickspos.append(numpy.mean(positions))
        box = boxplot(bpdata, positions=positions, widths=0.85, boxprops={"linewidth":3})
        batch += 1
        
        colors = ['b', 'g', 'r', 'k']
        colors = ['b', 'g', 'r', 'k'] 
        for elem in ['boxes', 'caps', 'whiskers', 'medians']:
            elems = len(box[elem])
            i = 1
            if elems > bpdata.shape[1]:
                i = 2
            for e in range(elems):
                setp(box[elem][e], color=colors[floor(e / i)])
                
    plt.ylabel(ylabel, fontsize=18)
    print(alldims)
    alldims = list(map(lambda l: l.strip().replace("C&C","C\&C"),alldims))

    ax.set_xticklabels(alldims, multialignment='left')
    ax.set_xticks(xtickspos)
    #ylim(-maxyval * 0.1, maxyval * 1.1)
    xlim(0, maxxval+1)
     
    #ax.set_yticks(numpy.arange(maxyval, maxyval*0.1)*2, minor=True)
    #ax.set_yticks(numpy.arange(0, maxyval*1.1, minticks), minor=True)
    ax.yaxis.grid(True, which='major', linestyle='--', color='k')
    ax.yaxis.grid(True, which='minor', linestyle='--', color='#DBDBDB')
    [line.set_zorder(3) for line in ax.lines]
    
    ax.set_yscale("log")
    
    for b in range(batch):
        ax.axvline(x=5*b,c="k",linewidth=0.5,zorder=0)
    
    btoa = (0.95, 1)
    leg = legend(list(map(lambda l: l.replace("PSPN", "MaxP PSPN").replace("MaxP PSPN_MJ", "MaxM PSPN").strip(), labels)), bbox_to_anchor=btoa)
    ax.add_artist(leg)
    
    for tick in sum([ax.yaxis.get_major_ticks(),ax.xaxis.get_major_ticks()],[]):
        tick.label.set_fontsize(16) 
    
    for lh in leg.legendHandles:
        lh.set_dashes((None, None))
    
    ltext = leg.get_texts()
    setp(ltext[0], fontsize=14, color='b')
    setp(ltext[1], fontsize=14, color='g')
    setp(ltext[2], fontsize=14, color='r')
    setp(ltext[3], fontsize=14, color='k')
         
    for line, txt in zip(leg.get_lines(), leg.get_texts()):
        line.set_linewidth(10)
        line.set_color(txt.get_color())
        

    savefig(fname, bbox_inches='tight', dpi=600)



data = {}    
for fname in glob("*.json"):

    stats = Stats(fname=fname)
    data[stats.name] = {}
    
    for method in stats.getMethods(Stats.SQUARED_ERROR):
        data[stats.name][method] = stats.getValues(method, Stats.SQUARED_ERROR)

print(data)                
plotBoxes(data, "Prediction Error", 0.1, "ploterr.pdf", figsize=(6,4))
