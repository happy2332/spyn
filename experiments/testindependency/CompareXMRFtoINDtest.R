library("XMRF")
library(jsonlite)
library(iterators)
library(grid)
library(partykit)
library(Formula)
library(foreach)
library(doMC)
library(parallel)
library(dplyr)
library(gtools)
library(igraph)
registerDoMC(detectCores()-1)

options(warn = -1)

setwd("/home/molina/Dropbox/Papers/pspn/spyn/experiments/testindependency")

#data<-read.csv2("/Users/alejomc/Dropbox/pspn/spyn/bin/data/graphlets/out/wl/1mutag.build_wl_corpus.csv", header = FALSE, sep = ",",  quote="\"", skip=1)

#data<-read.csv2("/home/molina/Dropbox/Papers/pspn/spyn/bin/data/graphlets/out/wl/1mutag.build_wl_corpus.csv", header = FALSE, sep = ",",  quote="\"", skip=1)

#data<-read.csv2("/home/molina/Dropbox/Papers/pspn/spyn/bin/data/graphlets/out/wl/2ptc.build_wl_corpus.csv" , header = FALSE,  sep = ",",  quote="\"", skip=1)


ptestpdnglm <- function(data) {
  
  data = as.data.frame(data)
  cols = ncol(data)
  
  vard = apply(data, 2, var)
  n = mixedsort(names(data))
  adjc <- foreach(i = 1:cols, .combine=rbind) %dopar% {
    
    fmla = as.formula(paste(n[i], " ~ . | .", sep = ''))
    
    tree = glmtree(fmla, data, family="poisson", verbose = FALSE, maxdepth=2)
    #tree = glmtree(fmla, data, family="poisson", verbose = FALSE, maxit = 25, maxdepth=2)
    #tree = glmtree(fmla, data, family="poisson", verbose = FALSE, minsize=5)
    
    treeresults = sctest.modelparty(tree,node=1)
    ptest = as.data.frame(matrix(1, ncol=cols,nrow=1))
    colnames(ptest)<-n
    
    if(!is.null(ptest)){
      ptest[1,colnames(treeresults)] = treeresults[2,colnames(treeresults)]
      return(ptest)
    }
    
    return(matrix(1,ncol=cols))
    
  }
  
  rownames(adjc)<-n
  
  pvals = pmin(adjc[upper.tri(adjc)],t(adjc)[upper.tri(adjc)])
  
  adjc[upper.tri(adjc)] = pvals
  adjc = adjc + t(adjc)
  diag(adjc)<-1
  return(adjc) 
}



findpval <- function(data){
  
  ptests = ptestglmblock(data)
  
  return(median(ptests[ptests<0.05]))
}

ptestglmblock <- function(data, family) {
  
  
  if(dim(data)[2] < 5){
    return(ptestpdnglm(data))  
  }
  
  start.time <- Sys.time()
  
  n = mixedsort(names(data))
  
  nblocks = ceiling(dim(data)[2] / 100)
  if(nblocks < 1){
    nblocks = 1
  }
  
  blocks = split(n, rank(n) %% nblocks)
  
  
  ptests = foreach(ni = n, .combine=rbind) %dopar% {
    #for(ni in n){
    
    ptestscols = foreach(bk = blocks, .combine=c) %do% {
      #for(bk in blocks){
      othervars = bk[bk != ni]
      fmla = as.formula(paste(ni, " ~ ", paste(othervars, collapse = ' + '), sep=' '))
      
      tree = glmtree(fmla, data, family=family, verbose = FALSE, maxdepth=2)
      
      ptest = sctest.modelparty(tree,node=1)[2,]
      
      if(is.null(ptest)){
        ptest = array(1, dim=length(othervars))
        #ptest = matrix(1,ncol=length(othervars))
        names(ptest) <- othervars
      }
      return(ptest)
    }
    ptestscols[ni] = 1
    ptestscols = ptestscols[mixedsort(names(ptestscols))]
    ptestscols[]
    return(ptestscols)
  }
  rownames(ptests) = n
  #if there was a problem computing the pvalues, default to 1
  ptests[is.na(ptests)] <- 1
  
  
  print(paste("ptest time: ", format(difftime(Sys.time(), start.time)),
              "ptest min: ", min(ptests),
              "ptest max: ", max(ptests),
              "ptest mean: ", mean(ptests)
  ))
  
  
  return(ptests)
}




subconnected2 <- function(ptests) {
  library(igraph)
  
  start.time <- Sys.time()
  
  
  wptests = 1-ptests
  vas = matrix(0, nrow = dim(ptests)[1], ncol=2)
  rownames(vas)<-rownames(ptests)
  maxval = max(wptests)
  repeat{
    #get clusters
    g = graph.adjacency(wptests, mode="max", weighted=TRUE, diag=FALSE)
    c = clusters(g)
    
    #variable assignment to subsets
    vas[,]=0
    
    for(w in order(c$csize, decreasing = TRUE)){
      j = which.min(apply(vas, 2, sum))
      vas[c$membership == w,j] = 1      
    }
    binsizes = apply(vas, 2, sum)
    r = min(binsizes)/sum(vas)
    
    ptestcut = median(wptests[wptests>0])
    
    if(ptestcut == maxval){
      #we couldn't really partition this
      vas[,1]=1
      vas[,2]=0
      break
    }
    
    if(r >= 0.3){
      break
    }
    
    wptests[wptests <= ptestcut] = 0
  }
  
  print("independent components")
  print(Sys.time() - start.time)
  
  return(vas)
  
  #plot.igraph(g,vertex.label=V(g)$name,layout=layout.fruchterman.reingold, edge.color="black",edge.width=E(g)$weight)
  
  #min_cut(g, capacity = E(g)$weight, value.only = FALSE)
  #cluster_spinglass(g, spins = 2)
}

subconnected <- function(ptests, alpha) {
  
  #alpha = 1-alpha
  #wptests = 1-ptests
  #wptests[wptests <= alpha] = 0
  
  ptests[ptests>alpha] = 0
  
  vas = matrix(0, nrow = dim(ptests)[1], ncol=2)
  rownames(vas)<-rownames(ptests)
  
  g = graph.adjacency(ptests, mode="min", weighted=TRUE, diag=FALSE)
  c = clusters(g)
  print(paste("Number of connected components", c$no,":",toString(table(c$membership))))
  
  for(w in order(c$csize, decreasing = TRUE)){
    j = which.min(apply(vas, 2, sum))
    vas[c$membership == w,j] = 1      
  }
  
  return(vas)
}

subclusters <- function(ptests, alpha) {
  ptests[ptests>alpha] = 0
  ptests[ptests > 0] = 1
  
  g = graph.adjacency(as.matrix(ptests), mode="min", weighted=TRUE, diag=FALSE)
  c = clusters(g)
  res = c$membership
  names(res)<-NULL
  return(res) 
}

getIndependentGroupsAlpha <- function(data, alpha, family) {
  return(subclusters(ptestglmblock(data, family), alpha))
}

getIndependentGroupsAlpha3 <- function(data, alpha) {
  return(subconnected(ptestglmblock(data), alpha))
}

getIndependentGroupsAlpha2 <- function(data, alpha) {
  return(subconnectedCombined(ptestglmblock(data)))
}


subconnectedCombined <- function(ptests){
  library(igraph)
  
  start.time <- Sys.time()
  
  wptests = 1-ptests
  vas = matrix(0, nrow = dim(ptests)[1], ncol=2)
  rownames(vas)<-rownames(ptests)
  g = graph.adjacency(wptests, mode="max", weighted=TRUE, diag=FALSE)
  c = clusters(g)
  
  if(c$no ==  1){
    s = cluster_spinglass(g, spins = 2)
    
    vas[s[[1]],1] = 1
    
    if(length(s) > 1){
      vas[s[[2]],2] = 1
    }
    
  }else{
    for(w in order(c$csize, decreasing = TRUE)){
      j = which.min(apply(vas, 2, sum))
      vas[c$membership == w,j] = 1      
    }
  }
  
  
  print("independent components")
  print(Sys.time() - start.time)
  
  return(vas)
  
}


subconnectedCD <- function(ptests) {
  library(igraph)
  
  start.time <- Sys.time()
  
  wptests = 1-ptests
  vas = matrix(0, nrow = dim(ptests)[1], ncol=2)
  rownames(vas)<-rownames(ptests)
  g = graph.adjacency(wptests, mode="max", weighted=TRUE, diag=FALSE)
  
  s = cluster_spinglass(g, spins = 2)
  
  vas[s[[1]],1] = 1
  
  if(length(s) > 1){
    vas[s[[2]],2] = 1
  }
  
  print("independent components")
  print(Sys.time() - start.time)
  
  return(vas)
}

#dim = 8
#nval = 100*dim
#data = as.data.frame(ceiling(matrix(runif(nval,min=1,max=10),ncol=dim)))


#start.time <- Sys.time()
#pt = ptestpdnglm(data[,c(117,15,114,590,668,sample(1:dim(data)[2],10))])
#ptests = getIndependentGroups(data)
#end.time <- Sys.time()
#time.taken <- end.time - start.time
#print(time.taken)
#print(round(pt,2))
#diag(pt) = 1
#print(any(round(pt,2) < 0.05))


set.seed(123)

simData <- function(n, p) {
  
  sim1 <- XMRF.Sim(n=n, p=p/2, model="LPGM", graph.type="scale-free")
  #hist(sim1$X)
  #plotNet(sim1$B)
  
  sim2 <- XMRF.Sim(n=n, p=p/2, model="LPGM", graph.type="hub")
  #hist(sim2$X)
  #plotNet(sim2$B)
  
  data = cbind(t(sim1$X), t(sim2$X))
  
  z = matrix(0, p/2, p/2)
  network = rbind(cbind(sim1$B, z),cbind(z, sim2$B))
  return(list("network" = network, "data" = data))
  
  #write.table(t(sim1$X), file = "sim1X.csv", sep = ",", row.names=FALSE)
  #write.table(t(sim2$X), file = "sim2X.csv", sep = ",", row.names=FALSE)
  #write.table(t(sim1$B), file = "sim1B.csv", sep = ",", row.names=FALSE)
  #write.table(t(sim2$B), file = "sim2B.csv", sep = ",", row.names=FALSE)
  #write.table(data, file = "synthetic.csv", sep = ",", row.names=FALSE)
  #write.table(network, file = "network.csv", sep = ",", row.names=FALSE)
  #pos = plotNet(network)
  #write.table(pos, file = "pos.csv", sep = ",", row.names=FALSE)
}

runExp <- function(n, p, r, alphas) {
  
  xmrfre = matrix(0, nrow=r, ncol=2)
  result = list()
  for(alpha in alphas){
    glmpt = matrix(1, nrow=r, ncol=2)
    result[[paste("glmptest",format(alpha, scientific = FALSE),sep="_")]] <- glmpt
  }
  for(i in 1:r) {
    net = simData(n,p)
    write.table(net$data, file = paste("datasets/simdata",n,p,i,".csv", sep="_"), sep = ",", row.names=FALSE)
    write.table(net$network, file = paste("datasets/simnet",n,p,i,".csv", sep="_"), sep = ",", row.names=FALSE)
    
    for(alpha in alphas){
      start.time <- Sys.time()
      ptestvals = ptestpdnglm(net$data)
      result[[paste("glmptest",format(alpha, scientific = FALSE),sep="_")]][i,1] = as.numeric(difftime(Sys.time(), start.time, units="secs"))
      result[[paste("glmptest",format(alpha, scientific = FALSE),sep="_")]][i,2] = sum((ptestvals < alpha) != (net$network==TRUE))/2
    }
    
    start.time <- Sys.time()
    lpgm.fit <- XMRF(t(net$data), method="LPGM", lambda.path=0.01* sqrt(log(p)/n) * lambdaMax(net$data))
    xmrfre[i,1] = as.numeric(difftime(Sys.time(), start.time, units="secs"))
    xmrfre[i,2] = sum((as.data.frame(lpgm.fit$network) == 1) != (net$network==TRUE))/2
    result[["xmrf"]] = xmrfre
    cat(i, n,p,"\n")
  }
  return(result)
}

nrsamples = 50
results = list()
results[["100_5"]] = runExp(100,5,nrsamples,c(0.001,0.0001,0.00001))
cat(toJSON(results),file="gnspnoutfile4.json",sep="\n")

results[["500_10"]] = runExp(500,10,nrsamples,c(0.001,0.0001,0.00001))
cat(toJSON(results),file="gnspnoutfile4.json",sep="\n")

results[["1000_20"]] = runExp(1000,20,nrsamples,c(0.001,0.0001,0.00001))
cat(toJSON(results),file="gnspnoutfile4.json",sep="\n")

results[["2000_50"]] = runExp(2000,50,nrsamples,c(0.001,0.0001,0.00001))
cat(toJSON(results),file="gnspnoutfile4.json",sep="\n")

results[["4000_100"]] = runExp(4000,100,nrsamples,c(0.001,0.0001,0.00001))
cat(toJSON(results),file="gnspnoutfile4.json",sep="\n")
