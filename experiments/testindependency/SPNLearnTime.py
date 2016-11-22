'''
Created on 21.07.2016

@author: alejomc
'''
# we want a shallow spn here
from glob import glob
import json
import time

from natsort.natsort import natsorted
from natsort.ns_enum import ns
import numpy

from algo.learnspn import LearnSPN



result = json.load(open('gnspnoutfile4.json'))
oldres = json.dumps(result)
for fname in natsorted(glob("datasets/simdata*.csv"), alg=ns.IGNORECASE):
    print(fname)
    name = "%s_%s" % (fname.split("_")[1], fname.split("_")[2])
    idx = int(fname.split("_")[3])-1
    
    data = numpy.loadtxt(fname, dtype=float, delimiter=",", skiprows=1)
    for alpha in ["0.001", "0.0001", "0.00001"]:
        t0 = time.time()
        spn = LearnSPN(alpha=float(alpha), min_instances_slice=data.shape[0] - 1, cluster_first=False).fit_structure(data)

        ptime = (time.time() - t0)
        
        result[name]["glmptest_%s"%(alpha)][idx][0] = ptime
        # print(spn.to_text(list(map(lambda x: "V"+str(x),range(2,200000)))))
print(oldres)
print(json.dumps(result))

with open('gnspnoutfile4_withtime.txt', 'w') as outfile:
    json.dump(result, outfile)
