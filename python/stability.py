import numpy as np
from scipy.stats import norm
import math

def getStability(X):
    M,d=X.shape
    hatPF=np.mean(X,axis=0)
    kbar=np.sum(hatPF)
    denom=(kbar/d)*(1-kbar/d)
    return 1-(M/(M-1))*np.mean(np.multiply(hatPF,1-hatPF))/denom

def getVarianceofStability(X):
    stab=getStability(X)
    M,d=X.shape
    hatPF=np.mean(X,axis=0)
    kbar=np.sum(hatPF)
    k=np.sum(X,axis=1)
    denom=(kbar/d)*(1-kbar/d)
    stab=1-(M/(M-1))*np.mean(np.multiply(hatPF,1-hatPF))/denom
    phi=np.zeros(M)
    for i in range(M):
        phi[i]=(1/denom)*(np.mean(np.multiply(X[i,],hatPF))-(k[i]*kbar)/d**2+(stab/2)*((2*k[i]*kbar)/d**2-k[i]/d-kbar/d+1))
    phiAv=np.mean(phi)
    variance=(4/M**2)*np.sum(np.power(phi-phiAv,2))
    return {'stability':stab,'variance':variance}

def confidenceIntervals(X,alpha=0.05):
    res=getVarianceofStability(X)
    lower=res['stability']-norm.ppf(1-alpha/2)*math.sqrt(res['variance'])
    upper=res['stability']+norm.ppf(1-alpha/2)*math.sqrt(res['variance'])
    return {'stability':res['stability'],'lower':lower,'upper':upper}

## this tests whether the true stability is equal to a given value stab0
def hypothesisTestV(X,stab0,alpha):
    res=getVarianceofStability(X)
    V=(res['stability']-stab0)/math.sqrt(res['variance'])
    zCrit=norm.ppf(1-alpha)
    if V>=zCrit: reject=True
    else: reject=False
    pValue=1-norm.cdf(V)
    return {'reject':reject,'V':V,'p-value':pValue}

# this tests the equality of the stability of two algorithms
def hypothesisTestT(X1,X2,alpha):
    res1=getVarianceofStability(X1)
    res2=getVarianceofStability(X2)
    stab1=res1['stability']
    stab2=res2['stability']
    var1=res1['variance']
    var2=res2['variance']
    T=(stab2-stab1)/math.sqrt(var1+var2)
    zCrit=norm.ppf(1-alpha/2) 
    ## the cumulative inverse of the gaussian at 1-alpha/2
    if(abs(T)>=zCrit):
        reject=True
        #print('Reject H0: the two algorithms have different population stabilities')
    else:
        reject=False
        #print('Do not reject H0')
    pValue=2*(1-norm.cdf(abs(T)))
    return {'reject':reject,'T':T,'p-value':pValue}