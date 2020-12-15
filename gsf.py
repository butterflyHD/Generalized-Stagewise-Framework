import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import example1 as ex1
from sklearn.linear_model import LinearRegression

def logloptimizor(args):
    # TweedieRegression
    # min_w 1/2n \sum_i d(yi yi^hat) + \alpha/2||w||_2
    # normal distribution case
    # simple array
    


    model = args.modelPara.model
    X     = pd.DataFrame(args.data.X.ix[:,7].reshape(-1,1))
    y     = pd.DataFrame(args.data.y.reshape(-1,1))
    reg   = LinearRegression(fit_intercept=False) 
    reg.fit(X, y)
    return reg.coef_


def GSF(args):
    # Generalized Stagewise Framework main code

    # parsing parameters 
    epsilon  = args.modelPara.epsilon
    maxiter  = args.modelPara.maxiter
    normType = args.modelPara.normType
    Q        = args.modelPara.Q
    tol      = args.modelPara.tol
    A        = args.data.X
    y        = args.data.y
    w        = args.w
    G        = args.G
    omega    = args.omega
    subsett  = args.subsett
    M        = args.M
    sizeP    = args.sizeP
    nOri     = sum(sizeP)

    #print(sizeP)

    if(subsett is None):
        M     = None
        betaM = None
    else:
        groupsize = 1
        indexpointer = 0
        tempsubsett = subsett
        for i in range(0, len(sizeP)): #0 to len(sizep) - 1
            tempsubsett = subsett[indexpointer:len(subsett)]
            if len(tempsubsett) == 0: 
                break
            groupsize += sizeP[i]
            tempsubsize = [x % groupsize for x in tempsubsett]
            #print(type(tempsubsize))
            findex = np.array([x == y for x in tempsubsize for y in tempsubsett].index(True)) + 1
            #print(type(findex))
            #print(sizeP[i]-findex.size)
            sizeP[i] = sizeP[i] - findex.size
            indexpointer = indexpointer + findex.size
        
        M = A.ix[:,subsett]
        csubsett = set(range(nOri)) - set(subsett)
        A = A.ix[:, csubsett] # A = A[, -subsett]
        betaM = np.zeros((len(sizeP), 1))
        sizeP = [x for x in sizeP if x != 0]
        G = len(sizeP)
        omega = np.sqrt(sizeP)
    
    ll = np.zeros((maxiter, 1))
    PenValue  = np.zeros((maxiter, 1))
    solution  = np.zeros((sum(sizeP)+len(subsett), maxiter + 1))
    Ucount    = np.zeros((G, 1))
    iCand     = np.zeros((G, 1))
    nablaF    = np.zeros((sum(sizeP), 1))
    beta      = np.zeros((sum(sizeP), 1))
    nablaFseq = np.zeros((sum(sizeP), maxiter + 1))

   # for iter in range(0,maxiter):
   #     a = 0

    return 1


if __name__ == "__main__":
    ex1.example1()



