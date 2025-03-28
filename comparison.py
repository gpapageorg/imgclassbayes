from PIL import Image
import os
import numpy as np
from crossEntropy import CrossEntropyNN
from exponential import ExponentialNN
import subprocess
import matplotlib.pyplot as plt

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

def plot(x, y, xlabel, ylabel, title):
    
    plt.figure(figsize=(8,6))
    plt.style.use('seaborn-v0_8-deep')
    plt.plot(x, y, linewidth = 2)
    plt.xlabel(xlabel, fontsize=14, fontweight='bold', color='#555')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold', color='#555')
    plt.xticks(fontsize=12, color='#444')
    plt.yticks(fontsize=12, color='#444')
    plt.show()

def testing(zeros, eights, nn1, nn2):
    failsCE = 0
    failsExp= 0
    wrongCEH0 = []
    wrongExpH0 = []
    for x in eights:
        f1 = nn1.u(x)
        f2 = nn2.u(x)
        if f1 <= 1/2:
            failsCE += 1
            wrongCEH0.append(x)
        if f2 <= 0:
            failsExp +=1
            wrongExpH0.append(x)
    print("\nChose H0 Instead of H1 CE:", 100 * failsCE / 997, "LLR:", 100 * failsExp / 997)  
    failsCE = 0
    failsExp= 0
    wrongCEH1 = []
    wrongExpH1 = []

    for x in zeros:
        f1 = nn1.u(x)
        f2 = nn2.u(x)

        if f1 > 1/2:
            failsCE += 1
            wrongCEH1.append(x)
        if f2 > 0:
            failsExp +=1
            wrongExpH1.append(x)           

    fig, axs = plt.subplots(2, 2, figsize=(15,5))  
    axs = axs.flatten()

    for i, image in enumerate(wrongCEH0[:2]):
        axs[i].imshow(image.reshape(28, 28), cmap='gray')   
        axs[i].axis('off')  # Turn off axis

    for i, image in enumerate(wrongCEH1[:2], start=2):
        axs[i].imshow(image.reshape(28, 28), cmap='gray')   
        axs[i].axis('off')  # Turn off axis

    fig, axs = plt.subplots(2, 2, figsize=(15,5))  # Number of columns based on length of images list
    axs = axs.flatten()
    
    for i, image in enumerate(wrongExpH0[:2]):
        axs[i].imshow(image.reshape(28, 28), cmap='gray')   
        axs[i].axis('off')  # Turn off axis

    for i, image in enumerate(wrongExpH1[:2], start=2):
        axs[i].imshow(image.reshape(28, 28), cmap='gray')   
        axs[i].axis('off')  # Turn off axis

    plt.tight_layout()
    plt.show()

    print("\nChose H1 Instead of H0 CE:", 100 * failsCE / 997, "LLR:", 100 * failsExp / 997)  


def main():
    data = loadImages() 

    ceNN = CrossEntropyNN(784, 300, 1, 50)
    expNN = ExponentialNN(784, 300, 1, 50)
    
    subprocess.run("clear") 

    # * Run This Only For Training
    #ceNN.trainCE(data[0], data[1], 0.001, 0.01)
    #ceNN.storeParameters()
    #expNN.trainCE(data[0], data[1], 0.001, 0.01)
    #expNN.storeParameters()

    # * Assume training has already been done
    ceNN.loadParameters()
    expNN.loadParameters()
    # print(ceNN.costEpochArr)
    # plot(np.arange(50), ceNN.costEpochArr, "Epochs", "Cost Value", "Cross Entropy")
    # plot(np.arange(50), expNN.costEpochArr, "Epochs", "Cost Value", "Exponential")
    
    testing(data[2], data[3], ceNN, expNN)
if __name__ == "__main__":
    main()