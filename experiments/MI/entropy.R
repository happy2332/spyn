require(entropy)
require(infotheo)

n = 5000
p = 20
sim <- XMRF.Sim(n=n, p=p, model="LPGM", graph.type="scale-free")
data = t(sim$X)
#write.csv2(data, file="/Users/alejomc/Dropbox/pspn/spyn/bin/experiments/MI/synth100x20.csv", row.names = FALSE)

#data <- read.csv("~/Dropbox/pspn/spyn/bin/experiments/MI/synthetic2.csv")
data <- read.csv("~/Dropbox/pspn/spyn/bin/experiments/MI/synth100x20.csv", sep=";")
j <- read.csv("~/Dropbox/pspn/spyn/bin/experiments/MI/synth100x20XY.csv", sep=" ", header = FALSE)

numbins = max(data[,1],data[,2])
maxnum = numbins
#numbins = maxnum
disc = discretize2d(data[,1],data[,2], numBins1=numbins, numBins2=numbins, r1=c(0,numbins), r2=c(0,numbins))
MI = mi.empirical(disc, unit = "log2")

d = as.data.frame.matrix(disc) 
colnames(d)<-colnames(j)
rownames(d)<-NULL

jj = floor(as.matrix(j)*194)
rownames(jj)<-NULL

