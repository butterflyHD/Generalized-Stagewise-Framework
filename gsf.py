import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import example1 as ex1
import utility as util
from numpy import linalg as LA
from sklearn.linear_model import LinearRegression

def gradient(model, beta0, beta, betaM, jstart, pj, y, A, M):
    n,_ = A.shape
    jend = jstart + pj - 1 # watch out index starts from zero
    # .values convert dataframe to array
    if (M is None):
        MbetaM = 0
    else:
        MnoIntercept = M.drop('intercept', axis = 1)
        MbetaM = np.matmul(MnoIntercept.values, betaM)
    
    #print(MbetaM)

    mu = beta0 + np.matmul(A.values, beta) + MbetaM
    #print(mu)

    if (model == "gaussian"):
        if(jstart == jend):
            Xj  = A.values[:,jstart].reshape(-1, 1)
        else:
            Xj  = A.values[:,jstart:jend]
        #print(Xj)
        r   = y.values.reshape(-1, 1) - mu
        XTj = Xj.transpose()
        
        #print(mu)
        #print(r)
        #print(mu.shape)
        #print(XTj)
        #print(mu)
        v   = - np.matmul(XTj, r)
        #print(v)

    if (model == "binomial"):
        temp = 0
        v = 0
        #binomial not supported for now
        

    if (model == "poisson"):
        v = 0
        #poisson not supported for now

    return(v)

def ChooseI(beta0, beta, betaM, model, normType, nablaF, iCand, omega, sizeP, G, y, A, M):
    # normType not supported for now
    print(beta0)
    print(beta)
    print(betaM)
    pj = sizeP[0]
    jstart = 0
    jend = sizeP[0] - 1
    if(jstart == jend):
        nablaF[jstart] = gradient(model, beta0, beta, betaM, jstart, pj, y, A, M)
    else:
        nablaF[jstart:jend] = gradient(model, beta0, beta, betaM, jstart, pj, y, A, M)
    
    print(nablaF)
    if(jstart == jend):
        iCand[0] = LA.norm(nablaF[jstart]) # for now 2 norm only
    else:
        iCand[0] = LA.norm(nablaF[jstart:jend])
    print(iCand)

    if(G>1):
        for i in range(1, G): # 2:G in R
            jstart = int(np.sum(sizeP[0:(i-1)]) + 1)
            jend   = int(jstart + sizeP[i] - 1)
            if(jstart == jend):
                nablaF[jstart] = gradient(model, beta0, beta, betaM, jstart, pj, y, A, M)
                iCand[i] = LA.norm(nablaF[jstart])
            else:
                nablaF[jstart:jend] = gradient(model, beta0, beta, betaM, jstart, pj, y, A, M)
                iCand[i] = LA.norm(nablaF[jstart:jend])
            
            print(iCand)

    ii = np.argmax(iCand, axis = 0)
    print(ii)
    return(ii)

def nablafGaussian(y, X, beta, beta0):
    XT = X.transpose()
    v  = - np.matmul(XT.values,y.values.reshape(-1,1) - np.matmul(X.values, beta) - beta0)
    return(v)

def nablafBinomial(y, X, beta, beta0):
    n,_  = X.shape # total number of rows 
    mu   = beta0 + np.matmul(X.values, beta)
    temp = 0
    for i in range(0, n):
        temp += np.dot((np.exp(mu[i])/(1+ np.exp([i])) - y[i]), X.ix[i,:].values.transpose())
    return(temp)

def nablafPoisson(y, X, beta, beta0):
    n,_  = X.shape
    mu   = beta0 + np.matmul(X.values, beta)
    temp = 0
    for i in range(0, n):
        temp += np.dot((-y[i] + np.exp((mu[i]))), X.ix[i,:].values.transpose()) 
    return(temp)

def logloptimizor(args):
    # TweedieRegression
    # min_w 1/2n \sum_i d(yi yi^hat) + \alpha/2||w||_2
    # normal distribution case
    # simple array
    ## simple example fit linear regression ##
    # model = args.modelPara.model
    # X     = pd.DataFrame(args.data.X.ix[:,7].reshape(-1,1))
    # y     = pd.DataFrame(args.data.y.reshape(-1,1))
    # reg   = LinearRegression(fit_intercept=False) 
    # reg.fit(X, y)
    # return reg.coef_

    nablalogl = args.nablaf
    tol       = args.tol
    print(tol)
    A         = args.data.X
    y         = args.data.y
    beta0     = args.offset

    maxiter   = 100000
    _,p       = A.shape
    beta      = np.zeros((p, 1))
    gamma     = 1
    print(A)
    print(beta0)
    print(y)

    # gradient descent 
    for i in range(0, maxiter):
        delta    = gamma * nablalogl(y, A, beta, beta0)
        betaCand = beta - delta
        nablaFcand = nablalogl(y, A, betaCand, beta0)
        nablaFprev = nablalogl(y, A, beta, beta0)
        
        # Barzilai-Borwein method
        temp = abs(nablaFcand - nablaFprev) 
        print(i)
        print(gamma)
        gamma  = abs(np.matmul(delta.transpose(), temp)/ LA.norm(temp)**2)
        if ((LA.norm(temp) <= tol) or (LA.norm(nablaFcand) <= tol) or (LA.norm(beta- betaCand) <= tol)):
            break 
        beta = betaCand
    return(betaCand) 
    

def GSF(args):
    # Generalized Stagewise Framework main code

    # parsing parameters 
    epsilon  = args.modelPara.epsilon
    maxiter  = args.modelPara.maxiter
    normType = args.modelPara.normType
    model    = args.modelPara.model
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
    n,p      = A.shape

    #print(sizeP)
    print(A)
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
            findex = np.array([xx == yy for xx in tempsubsize for yy in tempsubsett].index(True)) + 1
            #print(type(findex))
            #print(sizeP[i]-findex.size)
            print(y)
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
    OptimizeParser = util.UpdateCovariateParser()
    datap = util.data() 
    print(y)
    datap.add('y', y)
    print(datap.y)
    # add intercept
    intercept = 1 
    M.insert(0, "intercept", intercept)
    print(M)
    datap.add('X', M)
    OptimizeParser.add('epsilon', 0.01)
    OptimizeParser.add('data', datap)
    OptimizeParser.add('tol', 1e-14)
    OptimizeParser.add('beta', beta)
    OptimizeParser.add('offset', np.matmul(A.values, beta)) # offset will be an array

    if(subsett != None):
        if(model == 'gaussian'):
            OptimizeParser.add('nablaf', nablafGaussian)
        if(model == 'binomial'):
            OptimizeParser.add('nablaf', nablafBinomial)
        if(model == 'poisson'):
            OptimizeParser.add('nablaf', nablafPoisson)
    
    temp = OptimizeParser.nablaf(y, A, beta, OptimizeParser.offset)
    tempIni = logloptimizor(OptimizeParser)
    print(tempIni)
    beta0 = tempIni[0]
    print(beta0)
    betaM = tempIni[1:]  ## get all except first element
    print(betaM)
    pj = sizeP[0]
    jstart = 0
    jend = sizeP[0]


    ii  = ChooseI(beta0, beta, betaM, model, normType, nablaF, iCand, omega, sizeP, G, y, A, M)
    ##vv = gradient(model, beta0, beta, betaM, jstart, pj, y, A, M)
   # for iter in range(0,maxiter):
   #     a = 0

    return 1


if __name__ == "__main__":
    ex1.example1()



