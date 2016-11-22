'''
Created on Nov 12, 2015

@author: molina
'''
import numpy
import os
from rpy2 import robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr, SignatureTranslatedAnonymousPackage
from scipy.stats._continuous_distns import chi2
from scipy.stats._discrete_distns import poisson
from scipy.stats.contingency import chi2_contingency
from statistics import mean


# kmeans log transformation log(ex+1)
# kmeans hellinger distance
# LDA
# pmrf / admixture
# mixtures of pdns
# A Kernel Statistical Test of Independence 
# TODO: add histograms
def getContingency(A, B):
    minim = min(min(A), min(B))
    maxim = max(max(A), max(B))
    
    contingency = numpy.zeros((maxim - minim + 1, 2))
        
    for i in range(0, len(A)):
        contingency[A[i] - minim, 0] += 1
        
    for i in range(0, len(B)):
        contingency[B[i] - minim, 1] += 1
        
    contingency = contingency[~numpy.all(contingency == 0, axis=1)]
    
    return contingency

def independentChi(A, B, alpha=0.05):
    
    contingency = getContingency(A, B)
    
    res = chi2_contingency(contingency, correction=False)
    print(res)
    # A and B are independant if true
    return res[1] < alpha

def independentG(A, B, alpha=0.05):
    
    contingency = getContingency(A, B)
    
    res = chi2_contingency(contingency, correction=False, lambda_="log-likelihood")
    print(res)

    # A and B are independant if true
    return res[1] < alpha
    
def independentBivPois(A, B, alpha=0.05):
    robjects.r["set.seed"](1337)
    
    m1 = numpy.array(A)
    m2 = numpy.array(B)
    diff = m1 - m2
    skellam = importr('skellam')
    simplebp = importr('bivpois')

    mu1 = mean(m1)
    mu2 = mean(m2)
    
    print(mu1)
    print(mu2)
    numpy2ri.activate()
    ll1 = numpy.sum(numpy.log(skellam.dskellam(diff, lambda1=mu1, lambda2=mu2)))
    
    bp = simplebp.simple_bp(robjects.r["as.vector"](m1), robjects.r["as.vector"](m2), maxit=4)
    

    lambdas = list(map(lambda x: round(float(x), 3), bp[0]))
    print(lambdas)
    
    mu1 = lambdas[0]
    mu2 = lambdas[1]
    corr = lambdas[2]

    ll2 = numpy.sum(numpy.log(skellam.dskellam(diff, lambda1=mu1, lambda2=mu2)))

    print(ll1)
    print(ll2)

    val = -2.0 * ll1 + 2.0 * ll2
    print(val)

    pcso = robjects.r["pchisq"](val, df=1, lower_tail=False)
    
    print(pcso)

    numpy2ri.deactivate()
    
    return val > float(pcso[0])
    
    # pchisq(2 * (loglik.bp - loglik.ip),df = 1, lower.tail = FALSE)
      
def independentBivPois2(A, B, alpha=0.01):
    robjects.r["set.seed"](1337)
    
    m1 = numpy.array(A)
    m2 = numpy.array(B)
    bivpois = importr('bivpois')

    mu1 = mean(m1)
    mu2 = mean(m2)
    

    numpy2ri.activate()
    # print(bivpois.pbivpois(robjects.r["as.vector"](m1), robjects.r["as.vector"](m2), robjects.r["c"](mu1,mu2,0.00000001)))
    
    bp = bivpois.simple_bp(robjects.r["as.vector"](m1), robjects.r["as.vector"](m2), maxit=35)
    dpll = max(bp[1])
    
    idll = numpy.sum(numpy.log(poisson.pmf(m1, mu=mu1))) + numpy.sum(numpy.log(poisson.pmf(m2, mu=mu2)))
    
    
    lrt = -2.0 * (idll - dpll)
    
    chisq = chi2.ppf(1.0 - alpha, 1)
    print(bp[0])
    print(dpll)
    print(idll)
    
    print(lrt)
    print(chisq)
    # is independent
    return lrt <= chisq
    
 
    
if __name__ == '__main__':
    A = [10] * 200 + [25] * 250
    B = [10] * 200 + [25] * 250
    B = [20] * 200 + [40] * 250
    print(independentBivPois2(A, B))
    print(independentChi(A, B))
    print(independentG(A, B))
