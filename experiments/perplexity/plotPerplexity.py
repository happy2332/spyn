'''
Created on 05.07.2016

@author: alejomc
'''

from glob import glob
from math import floor
from matplotlib import pyplot as plt    , cm
from natsort import natsorted, ns
import numpy
from pylab import plot, show, savefig, xlim, figure, hold, ylim, legend, boxplot, setp, axes
import re

from mlutils.benchmarks import Stats


from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def getLabelText(txt):
    
    if "HLDA" in txt:
        return "HLDA"
    
    if "LDA" in txt:
        return txt.replace("LDA", "LDA ")
    
    if "PDN" in txt:
        return txt.replace("8_", " ")
    
    if "SAMPLED" in txt:
        return txt.replace("PSPN_", "RPSPN ").replace("SAMPLED", "")
    
    if "PSPN" in txt:
        return txt.replace("PSPN", "PSPN ")
    
    
    
    return txt


def plotBoxes(data, colors, ylabel, fname, plotLegend=False, figsize=(6,3)):    
    fig = figure(figsize=figsize)
    ax = fig.add_subplot(111)
    hold(True)

    data, methods = data

#     colorCounts = {}
#     
#     for method in methods:
#         key = re.sub(r'[0-9]+', '', method).replace("_", " ").strip()
#         if key not in colorCounts:
#             colorCounts[key] = []
#         colorCounts[key].append(method)
#         colorCounts[key] = natsorted(colorCounts[key], alg=ns.IGNORECASE)
#     
#     colors = {}
#         
#     for k, v in colorCounts.items():
#         for i, color in enumerate(map(lambda x: colorMaps[k](x), numpy.linspace(0.4, 0.6, len(v)))):
#             colors[v[i]] = color
#                     
# 
#     print(colors)
    
    num_datacases = len(data[list(data.keys())[0]].keys())
    print(num_datacases)
    
    
    NUM_COLORS = num_datacases
    
    # cm = plt.get_cmap('gist_rainbow')
    # colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

    maxyval = 0
    maxxval = 0
    xtickspos = []
    batch = 0
    alldims = natsorted(data.keys(), alg=ns.IGNORECASE)
    print(alldims)
    labels = []
    for dims in alldims:
        expsdata = data[dims]
        print(dims)
        bpdata = numpy.zeros((5, 0))
        for method in natsorted(expsdata.keys(), alg=ns.IGNORECASE):
            if method not in labels:
                labels.append(method)
            mdata = expsdata[method]
            print(method)
            print(mdata)
            networkerror = numpy.asarray(mdata).reshape(5, 1)
            if numpy.max(networkerror) > maxyval:
                maxyval = numpy.max(networkerror)
            bpdata = numpy.append(bpdata, networkerror, axis=1)
            
        # print(bpdata)
        positions = numpy.arange(1, bpdata.shape[1] + 1) + ((bpdata.shape[1] + 1) * batch)
        maxxval = positions[-1]
        xtickspos.append(numpy.mean(positions))
        box = boxplot(bpdata, positions=positions, widths=0.9, boxprops={"linewidth":3.5}, showfliers=True)
        batch += 1
        

        for elem in ['boxes', 'caps', 'whiskers', 'medians']:
            elems = len(box[elem])
            i = 1
            if elems > bpdata.shape[1]:
                i = 2
            for e in range(elems):
                setp(box[elem][e], color=colors[labels[floor(e / i)]])
                
    plt.ylabel(ylabel, fontsize=18)
    print(alldims)
    alldims = list(map(lambda l: l.strip().replace("C&C","C\&C"),alldims))
    print(alldims)

    ax.set_xticklabels(alldims, multialignment='left')
    ax.set_xticks(xtickspos)
    ylim(1, maxyval * 1.2)
    xlim(0, maxxval + 1)
     
    # ax.set_yticks(numpy.arange(maxyval, maxyval*0.1)*2, minor=True)
    # ax.set_yticks(numpy.arange(0, maxyval*1.1, minticks), minor=True)
    ax.yaxis.grid(True, which='major', linestyle='--', color='k')
    ax.yaxis.grid(True, which='minor', linestyle='--', color='#DBDBDB')
    # plt.tick_params(axis='y', which='minor')
    # ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3f"))
    [line.set_zorder(3) for line in ax.lines]
    
    ax.set_yscale("log")
    
    # Vertical lines
    for b in range(batch):
        ax.axvline(x=(num_datacases + 1) * b, c="k", linewidth=0.5, zorder=0)
    
    # legend on the left
    # btoa = (0.2250, 1)
    # leg = legend(list(map(lambda l: l.strip().replace("_", " "), labels)), bbox_to_anchor=btoa, prop={'size':4}, markerscale=0)
    
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    
    
    
    print(labels)
    
    if plotLegend:
        leg = ax.legend(labels, bbox_to_anchor=(0.0, 1.01, 1.0, 0.0), prop={'size':5}, markerscale=0, loc=3, ncol=4, mode="expand", borderaxespad=0.0)
        i = 0
        for line, txt in zip(leg.get_lines(), leg.get_texts()):
            color = colors[txt.get_text()]
            setp(txt, fontsize=12, color=color)
            txt.set_text(getLabelText(txt.get_text()))
            line.set_linewidth(6)
            line.set_color(color)
            line.set_dashes((None, None))
              
            i += 1

    
    for tick in sum([ax.yaxis.get_major_ticks(), ax.xaxis.get_major_ticks()], []):
        tick.label.set_fontsize(16) 
    
    print(labels)
    
        
    # plt.tight_layout()
    savefig(fname, bbox_inches='tight', dpi=600)


colorMaps = {}
# colorMaps["LDA"] = cm.get_cmap('Purples')
# colorMaps["HLDA"] = cm.get_cmap('Reds')
# colorMaps["PSPN"] = cm.get_cmap('Blues')
# colorMaps["PSPN SAMPLED"] = cm.get_cmap('Greens')
# colorMaps["PDN"] = cm.get_cmap('Oranges')

colorMaps["LDA"] = cm.get_cmap('Greys')
colorMaps["HLDA"] = cm.get_cmap('Greens')
colorMaps["PSPN"] = cm.get_cmap('Reds')
colorMaps["PSPN SAMPLED"] = cm.get_cmap('Oranges')
colorMaps["PDN"] = cm.get_cmap('Blues')

    

def getData(measure):
    data = {}
    methods = set()
    for fname in glob("stats8*.json"):
    
        stats = Stats(fname=fname)
        data[stats.name] = {}
        
        for method in stats.getMethods(measure):
            data[stats.name][method] = stats.getValues(method, measure)
            methods.add(method)

    return data, methods

colors = {


'HLDA100': (0.59607844948768618, 0.834509813785553, 0.5788235485553741, 1.0), 

'LDA5': (0.71058825254440305, 0.71058825254440305, 0.71058825254440305, 1.0), 
'LDA10': (0.64821224703508262, 0.64821224703508262, 0.64821224703508262, 1.0), 
'LDA20': (0.58608230025160546, 0.58608230025160546, 0.58608230025160546, 1.0), 
'LDA50': (0.53440985843247057, 0.53440985843247057, 0.53440985843247057, 1.0), 
'LDA100': (0.47843137979507444, 0.47843137979507444, 0.47843137979507444, 1.0), 

'PDN8_1': (0.57960786223411564, 0.77019609212875362, 0.87372549772262575, 1.0), 
'PDN8_5': (0.51686275858505104, 0.73574780506246229, 0.8601922420894399, 1.0), 
'PDN8_10': (0.45411765493598649, 0.70129951799617096, 0.84665898645625393, 1.0), 
'PDN8_25': (0.39186467388096979, 0.66340640222325042, 0.828389091351453, 1.0), 
'PDN8_50': (0.341422539248186, 0.6289581151569591, 0.80870435588500078, 1.0), 
'PDN8_100': (0.2909804046154022, 0.59450982809066777, 0.78901962041854856, 1.0), 

'PSPN10': (0.98745098114013674, 0.54117649197578432, 0.41568627953529358, 1.0),
'PSPN25': (0.98622068517348349, 0.49196464395990558, 0.36647444086916309, 1.0),
'PSPN50': (0.98499038920683024, 0.44275279594402694, 0.3172626022030326, 1.0), 
'PSPN75': (0.97619377304525934, 0.38388312622612597, 0.26989620491570115, 1.0), 
'PSPN90': (0.96143022144542023, 0.32605921111854852, 0.23298732124122917, 1.0), 
'PSPN99': (0.94666666984558101, 0.26823529601097107, 0.19607843756675719, 1.0), 
 
#'PSPN_SAMPLED10': (0.99215686321258545, 0.65647060871124263, 0.38274510204792023, 1.0), 
#'PSPN_SAMPLED25': (0.99215686321258545, 0.61587084181168505, 0.32492118694034278, 1.0), 
#'PSPN_SAMPLED50': (0.99215686321258545, 0.57527107491212737, 0.26709727183276544, 1.0), 
#'PSPN_SAMPLED75': (0.98403690983267389, 0.5285813378352745, 0.20755094318997627, 1.0), 
#'PSPN_SAMPLED90': (0.96927335823283478, 0.48429067368600881, 0.15710880621975543, 1.0), 
#'PSPN_SAMPLED99': (0.95450980663299556, 0.44000000953674318, 0.10666666924953461, 1.0), 

'PSPN_SAMPLED10': (0.52, 0.76, 1.0, 1.0),
'PSPN_SAMPLED25': (0.42, 0.65, 1.0, 1.0),
'PSPN_SAMPLED50': (0.32, 0.53, 1.0, 1.0), 
'PSPN_SAMPLED75': (0.22, 0.42, 1.0, 1.0), 
'PSPN_SAMPLED90': (0.11, 0.31, 1.0, 1.0), 
'PSPN_SAMPLED99': (0.02, 0.20, 1.0, 1.0), 

}


plotBoxes(getData(Stats.PERPLEXITY), colors, "Perplexity", "perplexity.pdf", plotLegend=True, figsize=(6,3.3))

plotBoxes(getData(Stats.TIME), colors, "Time", "time.pdf", figsize=(6,4))

