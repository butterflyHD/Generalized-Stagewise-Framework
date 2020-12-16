import numpy as np

# variable definition
class modelPara:
    def __init__(self):
        self.epsilon  = []
        self.x0       = []
        self.model    = []
        self.normType = []
        self.Q        = []
        self.maxiter  = []
        self.tol      = []
    def add(self, name, parm):
        setattr(self, name, parm) # self.name = par doesnt work. related to string, attribute and variable issues
    def showAll(self):
        for attribute, value in self.__dict__.items():
            print(attribute, '=', value)

class data:
    def __init__(self):
        self.y = []
        self.X = []
    def add(self, name, parm):
        setattr(self, name, parm)
    def scaleX(self):
        self.X = scale(X)
    def showAll(self):
        for attribute, value in self.__dict__.items():
            print(attribute, '=', value)

class infoParser:
    def __init__(self):
        self.modelPara = []
        self.data      = []
        self.M         = []
        self.w         = []
        self.G         = []
        self.sizeP     = []
        self.omega     = []
        self.subsett   = []
    def add(self, name, parm):
        setattr(self, name, parm)
    def showAll(self):
        for attribute, value in self.__dict__.items():
            print(attribute, '=', value) 

class UpdateCovariateParser:
    def __int__(self):
        self.data   = []
        self.tol    = []
        self.offset = []
        self.beta   = []
        self.nablaf = []
    def add(self, name, parm):
        setattr(self, name, parm)
    def showAll(self):
        for attribute, value in self.__dict__.items():
            print(attribute, "=", value)

def scale(X):
    # rescale a matrix to [0, 1]
    colMeanX = X.mean(0)
    colVarX = X.ix[:,0].var(0, ddof = 1) # same resuls from R but it is based on 6 digit precision when reading data.
    #d = X - colMeanX
    #d = d**2
    #print(d.lcavol.sum()/96)
    #print(colVarX)
    scaledX = (X- colMeanX)/np.sqrt(colVarX)
    return scaledX

if __name__ == '__main__':
    points = np.matrix([
        [1.,2.,3.],   # 1st point
        [4.,5.,6.]]   # 2nd point
    )
    X = scale(points)
    print X
    d = modelPara()
    d.add("x0", 0.1)
    d.add("x0", 1)