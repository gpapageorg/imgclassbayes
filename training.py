from crossEntropy import CrossEntropyNN
from exponential import ExponentialNN
from PIL import Image
import numpy as np
import subprocess
import sys
import os


def loadImages():
    zeros = []
    eights = []
    zerostest = []
    eightstest = []

    zerosTrainingDirectory = "MNIST/training/0"
    eightsTrainingDirectory = "MNIST/training/8"
    zerosTestingDirectory = "MNIST/testing/0"
    eightsTestingDirectory = "MNIST/testing/8"

    for filename in os.listdir(zerosTrainingDirectory):
        imgPath = os.path.join(zerosTrainingDirectory, filename)
        with Image.open(imgPath) as img:
            imgA  = np.array(img).astype('float32') / 255.0
            imgA = imgA.flatten().reshape(784,1)
            zeros.append(imgA)

    for filename in os.listdir(eightsTrainingDirectory):
        imgPath = os.path.join(eightsTrainingDirectory, filename)
        with Image.open(imgPath) as img:
            imgA  = np.array(img).astype('float32') / 255.0
            imgA = imgA.flatten().reshape(784,1)
            eights.append(imgA)
    for filename in os.listdir(zerosTestingDirectory):
        imgPath = os.path.join(zerosTestingDirectory, filename)
        with Image.open(imgPath) as img:
            imgA  = np.array(img).astype('float32')/ 255.0
            imgA = imgA.flatten().reshape(784,1)
            zerostest.append(imgA)

    for filename in os.listdir(eightsTestingDirectory):
        imgPath = os.path.join(eightsTestingDirectory, filename)
        with Image.open(imgPath) as img:
            imgA  = np.array(img).astype('float32') / 255.0
            imgA = imgA.flatten().reshape(784,1)
            eightstest.append(imgA)

    zerosTrain = np.array(zeros[:5000])
    eightsTrain = np.array(eights[:5000])
    zerosTest = np.array(zerostest[:970])
    eightsTest = np.array(eightstest[:970])

    return [zerosTrain, eightsTrain, zerosTest, eightsTest]

def main():
    data = loadImages()
    ceNN = CrossEntropyNN(784, 300, 1, 50)
    expNN = ExponentialNN(784, 300, 1, 50)
    
    subprocess.run("clear") 

    # * Run This Only For Training
    if (len(sys.argv) > 1):
        if(sys.argv[1] == '1'):
            ceNN.train(data[0], data[1], 0.001, 0.01)
            ceNN.storeParameters()

        elif(sys.argv[1] == '2'):
            expNN.train(data[0], data[1], 0.001, 0.01)
            expNN.storeParameters()
    else:
        print("set argv 1 for cross_entropy 2 for exponential")
        return

if __name__ == "__main__":
    main()