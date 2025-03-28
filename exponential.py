import numpy as np
from tqdm import tqdm

class ExponentialNN():
    def __init__(self, numberOfInputs, sizeOfHiddenLayer, numberOfOutputs, epochs):
        self.numberOfInputs = numberOfInputs
        self.sizeOfHiddenLayer = sizeOfHiddenLayer
        self.numberOfOutputs = numberOfOutputs
        self.epochs = epochs

        np.random.seed(1)

        stdDevHl = np.sqrt(1 / (self.sizeOfHiddenLayer + self.numberOfInputs))
        stdDevOl = np.sqrt(1 / (sizeOfHiddenLayer + numberOfOutputs))

        initialWeightsHl = np.random.normal(loc = 0, scale = stdDevHl, size = (self.sizeOfHiddenLayer, self.numberOfInputs) ) #* Initial Weights Hidden Layer
        initialWeightsOl = np.random.normal(loc = 0, scale = stdDevOl, size = (self.numberOfOutputs, self.sizeOfHiddenLayer)) #* Initial Weights Output Layer

        self.hlWeights = initialWeightsHl.copy() #* ------------------- Exponential
        self.hlBiases = np.zeros((self.sizeOfHiddenLayer, 1)) #* HL Biases Exponential

        self.olWeights = initialWeightsOl.copy() #* Output Layer Exponential
        self.olBias = 0

        self.costEpochArr = []

    def gradient(self, x):
        w1 = self.hlWeights @ x + self.hlBiases
        z1 = np.maximum(0, w1) #? RELU ACTIVATION
        w2 = self.olWeights @ z1 + self.olBias

        y = w2 #? No Nonlinearity Needed Here

        v2 = np.array([1]).reshape(1,1) #? Derivative 

        u1 = self.olWeights.T @ v2

        v1 = u1 * np.where(w1 > 0, 1, 0)

        dA2 = v2 @ z1.T
        dB2 = v2

        dA1 = v1 @ x.T
        dB1 = v1

        fin = np.array([dA1, dB1, dA2, dB2], dtype =object), y.item()
        return fin
    
    def train(self, x_H0, x_H1, m, l):
        print("Started Training With Method Exponential, Step Size:", m, "Normalization:", l)

        lenX = len(x_H0)
        costEpoch = 0
        iterations = self.  epochs * lenX

        index = 0
        epoch = 0

        gradsX1, x1 = self.gradient(x_H0[0])
        gradsX2, x2 = self.gradient(x_H1[0])

        phiDev = 0.5 * np.exp(0.5*x1)
        psiDev = -0.5 * np.exp(-0.5*x2)

        px = ((phiDev*gradsX1 + psiDev*gradsX2) ** 2)

        c = 10**(-8)

        pbar = tqdm(range(iterations), colour='blue', position=1, desc=f"Exponential, Epoch {epoch}, Cost {costEpoch}")
        for _ in pbar:
            index +=1

            if index == lenX - 1:
                index = 0
                c = costEpoch / lenX
                self.costEpochArr.append(c)
                epoch +=1
                pbar.set_description(f"Exponential, Epoch: {epoch}, Cost: {c:.9f}")
                costEpoch = 0

            self.hlWeights -=  m * (phiDev * gradsX1[0] + psiDev * gradsX2[0]) / np.sqrt(c + px[0])
            self.hlBiases  -=  m * (phiDev * gradsX1[1] + psiDev * gradsX2[1]) / np.sqrt(c + px[1])
            self.olWeights -=  m * (phiDev * gradsX1[2] + psiDev * gradsX2[2]) / np.sqrt(c + px[2])
            self.olBias    -=  m * (phiDev * gradsX1[3] + psiDev * gradsX2[3]) / np.sqrt(c + px[3])
            
            cost = (2 * phiDev - 2 * psiDev)
            costEpoch += cost

            gradsX1, x1 = self.gradient(x_H0[index])
            gradsX2, x2 = self.gradient(x_H1[index])

            phiDev = 0.5 * np.exp(0.5*x1)
            psiDev = -0.5 * np.exp(-0.5*x2)

            px = (1 - l) * px + l * ((phiDev * gradsX1 + psiDev * gradsX2) ** 2)


    def storeParameters(self): 
        costEpochArr = np.array(self.costEpochArr)
        np.savez("nn_exponential_parameters_"+str(self.epochs)+".npz", hlw = self.hlWeights, hlb = self.hlBiases, olw = self.olWeights, olb = self.olBias, costEpoch = costEpochArr)

    def loadParameters(self):
        data = np.load("nn_exponential_parameters_"+str(self.epochs)+".npz")

        self.hlWeights = data["hlw"]
        self.hlBiases  = data["hlb"]
        self.olWeights = data["olw"]
        self.olBias    = data["olb"]
        self.costEpochArr = data["costEpoch"]

    def u(self, x):
        w1 = self.hlWeights @ x + self.hlBiases
        z1 = np.maximum(0, w1) #? RELU ACTIVATION
        w2 = self.olWeights @ z1 + self.olBias

        #? y = w2 No Nonlinearity Needed Here

        return w2.item()
