import utility as util
import csv
import pandas as pd
import gsf as gsf

def example1():
    # read data y and X
    path = "/Users/mengyang/Documents/python/Generalized-Stagewise-Framework/data/prostate.txt"
    data = pd.read_csv(path, sep = "\t", header = 0)    
    y = data.lpsa
    X = data.ix[:, 1:9]
    print(X)
    ind = data.train
    # parameter set up 
    modelp = util.modelPara()
    modelp.add('epsilon', 0.01)
    modelp.add('x0', 0)
    modelp.add('model', 'gaussian')
    modelp.add('normType', "2")
    modelp.add('Q', 0)
    modelp.add('maxiter', 1000)
    modelp.add('tol', 1e-7)

    data = util.data()
    X    = util.scale(X)
    print(X)
    index = X[ind == "F"].index
    X = X.drop(index)
    y = y.drop(index)
    data.add('y', y)
    data.add('X', X)
    print(X)
    #print(X)

    infoP = util.infoParser()
    infoP.add('modelPara', modelp)
    infoP.add('data', data)
    infoP.add('sizeP', [1,1,1,1,1,1,1,1])
    infoP.add('G', 8)
    infoP.add('subsett', [0,1])
    #print(infoP.modelPara.epsilon)
    #fit = gsf.logloptimizor(infoP)
    #print(fit)
 
    fit = gsf.GSF(infoP)
if __name__ == "__main__":
    example1()

