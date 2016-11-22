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

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)



def plotBoxes(data, key, ylabel, minticks, fname):    
    fig = figure(figsize=(6,3.4))
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
        
        keys = [k for k in natsorted(expsdata.keys(), alg=ns.IGNORECASE) if key in k]
        print(len(keys), keys)
        bpdata = numpy.zeros((5, 0))
        for method in keys:
            
            if key not in method:
                continue
            print(key, method)
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
        box = boxplot(bpdata, positions=positions, widths=1.0, boxprops={"linewidth":2.5})
        batch += 1
        
        colors = ['b', 'r', 'r', 'k']
        colors = ['b', 'r', 'r', 'k'] * len(keys)
        for elem in ['boxes', 'caps', 'whiskers', 'medians']:
            elems = len(box[elem])
            i = 1
            if elems > bpdata.shape[1]:
                i = 2
            for e in range(elems):
                setp(box[elem][e], color=colors[floor(e / i)])
                
    plt.ylabel(ylabel, fontsize=18)
    print(alldims)
    ax.set_xticklabels(alldims, multialignment='left')
    ax.set_xticks(xtickspos)
    #ylim(-maxyval * 0.1, maxyval * 1.1)
    xlim(0, maxxval+1)
     
    #ax.set_yticks(numpy.arange(maxyval, maxyval*0.1)*2, minor=True)
    #ax.set_yticks(numpy.arange(0, maxyval*1.1, minticks), minor=True)
    ax.yaxis.grid(True, which='major', linestyle='--', color='#DBDBDB')
    #ax.yaxis.grid(True, which='minor', linestyle='--', color='#DBDBDB')
    [line.set_zorder(3) for line in ax.lines]
    
    
    for b in range(batch):
        ax.axvline(x=3*b,c="k",linewidth=0.5,zorder=0)
    
    btoa = (0.30, 1)
    leg = legend(list(map(lambda l: l.replace(key, "").replace(" -> ", " ").replace("SPN per class ", "").replace("activations", r"$\alpha$PSPN").replace("gradients", r"$\Delta$PSPN").replace("ll", r"PSPN").replace("LR", "").replace("raw","").strip(), labels)), bbox_to_anchor=btoa)
    ax.add_artist(leg)
    
    for tick in sum([ax.yaxis.get_major_ticks(),ax.xaxis.get_major_ticks()],[]):
        tick.label.set_fontsize(16) 
    
    for lh in leg.legendHandles:
        lh.set_dashes((None, None))
    
    ltext = leg.get_texts()
    setp(ltext[0], fontsize=14, color='b')
    setp(ltext[1], fontsize=14, color='r')
    #setp(ltext[2], fontsize=14, color='r')
    #setp(ltext[3], fontsize=14, color='k')
         
    for line, txt in zip(leg.get_lines(), leg.get_texts()):
        line.set_linewidth(10)
        line.set_color(txt.get_color())
        

    savefig(fname, bbox_inches='tight', dpi=600)

data = {}    
with open('backupoutput1.txt') as data_file:    
    lines = data_file.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if len(line) > 0 and line[0] == "{":
            dsname = re.findall(r'/\d(\w+)', lines[i-2])[0].title()
            data[dsname] = {}
            for k, v in json.loads(line).items():
                if not ("per class ll" in k or "SVM raw Accuracy raw" in k ):
                    continue
                
                data[dsname][k] = v



                
plotBoxes(data, "Accuracy raw", "Accuracy", 0.1, "plots/accuracy_raw.pdf")
0/0
plotBoxes(data, "Accuracy std", "Accuracy std", 0.1, "plots/accuracy_std.pdf")
plotBoxes(data, "Accuracy nml", "Accuracy nml", 0.1, "plots/accuracy_nml.pdf")
plotBoxes(data, "AUC raw", "Accuracy", 0.1, "plots/auc_raw.pdf")

print(data)
