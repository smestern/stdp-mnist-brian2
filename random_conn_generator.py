'''
Created on 15.12.2014

@author: Peter U. Diehl
'''

import scipy.ndimage as sp
import numpy as np
import pylab


def randomDelay(minDelay, maxDelay):
    return np.random.rand()*(maxDelay-minDelay) + minDelay
        
        
def computePopVector(popArray):
    size = len(popArray)
    complex_unit_roots = np.array([np.exp(1j*(2*np.pi/size)*cur_pos) for cur_pos in range(size)])
    cur_pos = (np.angle(np.sum(popArray * complex_unit_roots)) % (2*np.pi)) / (2*np.pi)
    return cur_pos


def sparsenMatrix(baseMatrix, pConn):
    weightMatrix = np.zeros(baseMatrix.shape)
    numWeights = 0
    numTargetWeights = baseMatrix.shape[0] * baseMatrix.shape[1] * pConn
    weightList = [0]*int(numTargetWeights)
    while numWeights < numTargetWeights:
        idx = (np.int32(np.random.rand()*baseMatrix.shape[0]), np.int32(np.random.rand()*baseMatrix.shape[1]))
        if not (weightMatrix[idx]):
            weightMatrix[idx] = baseMatrix[idx]
            weightList[numWeights] = (idx[0], idx[1], baseMatrix[idx])
            numWeights += 1
    return weightMatrix, weightList


def create_weights():

    DEFAULT_NE = 400

    weightNormSF = 0.09948979591836734693877551020408 #a magic number from the original pub, they normalize the 784->400 weights to 78. 
    #which is 78.0/784.0 = 0.09948979591836734693877551020408

    nInput = 784
    nE = 400
    nI = nE
    SF = (1/(nE/DEFAULT_NE))
    SF_in = 1 
    SF_ee = weightNormSF * nE

    dataPath = './random/'
    weight = {}
    weight['ee_input'] = 0.3 * SF_in
    weight['ei_input'] = 0.2 * SF_in
    weight['ee'] = 0.3
    weight['ei'] = 10.4 * SF
    weight['ie'] = 17.0 * SF
    weight['ii'] = 0.4  * SF
    pConn = {}
    pConn['ee_input'] = 1.0
    pConn['ei_input'] = 0.1
    pConn['ee'] = 1.0
    pConn['ei'] = 0.0025
    pConn['ie'] = 0.9
    pConn['ii'] = 0.1

    #MUTE_NEURONS
    muteNeurons = np.arange(10).tolist()

    #E->E?
    e_to_e = True


    print( 'create random connection matrices')
    connNameList = ['XeAe']
    for name in connNameList:
        weightMatrix = np.random.random((nInput, nE)) + 0.01
        weightMatrix *= weight['ee_input']
        if len(muteNeurons) > 0:
            for i in muteNeurons:
                weightMatrix[:,i] = 0
        if pConn['ee_input'] < 1.0:
            weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ee_input'])
        else:
            weightList = [(i, j, weightMatrix[i,j]) for j in range(nE) for i in range(nInput)]
        np.save(dataPath+name, weightList)



    print( 'create connection matrices from X->I which are purely random')
    connNameList = ['XeAi']
    for name in connNameList:
        weightMatrix = np.random.random((nInput, nI))
        weightMatrix *= weight['ei_input']
        weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ei_input'])
        print( 'save connection matrix', name)
        np.save(dataPath+name, weightList)



    print( 'create connection matrices from E->I which are purely random')
    connNameList = ['AeAi']
    for name in connNameList:
        if nE == nI:
            weightList = [(i, i, weight['ei']) for i in range(nE)]
        else:
            weightMatrix = np.random.random((nE, nI))
            weightMatrix *= weight['ei']
            weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ei'])
        print( 'save connection matrix', name)
        np.save(dataPath+name, weightList)



    print( 'create connection matrices from I->E which are purely random')
    connNameList = ['AiAe']
    for name in connNameList:
        if nE == nI:
            weightMatrix = np.ones((nI, nE))
            weightMatrix *= weight['ie']
            for i in range(nI):
                weightMatrix[i,i] = 0

            if len(muteNeurons) > 0:
                #for the muted neurons, dial the inhibition down
                for i in muteNeurons:
                    weightMatrix[:, i] *= (len(muteNeurons)/nInput) * 0.0
            weightList = [(i, j, weightMatrix[i,j]) for i in range(nI) for j in range(nE)]
        else:
            weightMatrix = np.random.random((nI, nE))
            weightMatrix *= weight['ie']
            
            weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ie'])
        print( 'save connection matrix', name)
        np.save(dataPath+name, weightList)

    if e_to_e:
        connNameList = ['AeAe']
        for name in connNameList:
            if nE == nE:
                weightMatrix = np.random.random((nE, nE))
                weightMatrix *= weight['ee'] 
                for i in range(nE):
                    weightMatrix[i,i] = 0
                if muteNeurons is not None: 
                    for i in muteNeurons:
                        weightMatrix[:, i] += weight['ee']
                weightList = [(i, j, weightMatrix[i,j]) for i in range(nE) for j in range(nE)]
            else:
                weightMatrix = np.random.random((nI, nE))
                weightMatrix *= weight['ee']
                weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ee'])

            #normalize weights such that the maximum weight is equal to SF_ee
            weightList = [(i, j, weightMatrix[i,j]) for j in range(nE) for i in range(nE)]
            print( 'save connection matrix', name)
            np.save(dataPath+name, weightList)


if __name__ == "__main__":
    create_weights()










