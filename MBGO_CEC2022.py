# coding:UTF-8
'''
Created by Yuefeng XU (xyf20070623@gmail.com) on October 1, 2023
benchmark function: 10 functions of the CEC2022 test suite (https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2017/CEC2017.htm)
'''

# import packages
import os
import math
from opfunu.cec_based.cec2022 import *
import numpy as np
from copy import deepcopy


PopSize = 100  # the number of Pop
DimSize = 10  # the number of variables
LB = [-100] * DimSize  # the maximum value of the variable range
UB = [100] * DimSize  # the minimum value of the variable range
TrialRuns = 30  # the number of independent runs
MaxFEs = 1000 * DimSize  # the maximum number of fitness evaluations

Pop = np.zeros((PopSize, DimSize))  # the coordinates of the individual (candidate solutions)
FitPop = np.zeros(PopSize)  # the fitness value of all Pop
curFEs = 0  # the current number of fitness evaluations
FuncNum = 1  # the serial number of benchmark function
curIter = 0  # the current number of generations
MaxIter = int(MaxFEs / PopSize / 2)
curBest = np.zeros(DimSize)  # the best individual in the current generation
FitBest = 0  # the fitness of the best individual in the current generation
curWorst = np.zeros(DimSize)  # the worst individual in the current generation


# initialize the M randomly
def Initialization(func):
    global Pop, FitPop, curBest, FitBest
    # randomly generate Pop
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
            # calculate the fitness of the i-th individual
        FitPop[i] = func.evaluate(Pop[i])
    bestIdx = np.argmin(FitPop)
    curBest = Pop[bestIdx].copy()
    FitBest = FitPop[bestIdx]


def Check(indi):
    global LB, UB
    for i in range(DimSize):
        range_width = UB[i] - LB[i]
        if indi[i] > UB[i]:
            n = int((indi[i] - UB[i]) / range_width)
            mirrorRange = (indi[i] - UB[i]) - (n * range_width)
            indi[i] = UB[i] - mirrorRange
        elif indi[i] < LB[i]:
            n = int((LB[i] - indi[i]) / range_width)
            mirrorRange = (LB[i] - indi[i]) - (n * range_width)
            indi[i] = LB[i] + mirrorRange
        else:
            pass
    return indi


def MBGO(func):
    global Pop, FitPop, curBest, FitBest, curWorst
    # update the current worst Pop
    worstIdx = np.argmax(FitPop)
    curWorst = Pop[worstIdx].copy()
    # calculate the distance between the best individual and the worst individual
    maxDist = np.linalg.norm(curBest - curWorst + np.finfo(float).eps, ord=2)
    safeRadius = np.random.uniform(0.8, 1.2) * maxDist
    # record the generated Off individual
    Off = np.zeros(DimSize)
    # movement of Pop in the first round
    for i in range(PopSize):
        # calculate the distance between the current individual and the best Pop
        distance = np.linalg.norm(curBest - Pop[i] + np.finfo(float).eps, ord=2)
        # if the individual falls within the range of the first round, it indicates that the individual has potential.
        if distance < safeRadius:
            Off = Pop[i] + curBest * np.sin(np.random.rand() * 2 * np.pi)
        else:
            for j in range(DimSize):
                if np.random.rand() < 0.5:
                    Off[j] = Pop[i][j] + np.random.normal()
                else:
                    Off[j] = Pop[i][j] + (curBest[j] - Pop[i][j]) * np.random.rand()

        Off = Check(Off)
        FitOff = func.evaluate(Off)
        # If the Off individual is better, replace its parent.
        if FitOff < FitPop[i]:
            Pop[i] = Off.copy()
            FitPop[i] = FitOff
    # end of first round
    # battle Phase
    for i in range(PopSize):
        # randomly select an individual different from the i-th individual
        selectedIdx = np.random.randint(0, PopSize)
        while selectedIdx == i:
            selectedIdx = np.random.randint(0, PopSize)
        # compare the fitness of two Pop, if the randomly selected individual is better
        if FitPop[i] > FitPop[selectedIdx]:
            # compute the vector between two Pop
            space = Pop[selectedIdx] - Pop[i]
            for j in range(DimSize):
                if np.random.uniform() < 0.5:
                    Off[j] = Pop[i][j] + space[j] * 0.5 * np.random.rand()
                else:
                    Off[j] = Pop[selectedIdx][j] + space[j] * 0.5 * np.random.rand()
        else:
            # compute the vector between two Pop
            space = Pop[i] - Pop[selectedIdx]
            Off = Pop[i] + space * np.cos(np.random.rand() * 2 * np.pi)
        Off = Check(Off)
        FitOff = func.evaluate(Off)
        # If the Off individual is better, replace its parent.
        if FitOff < FitPop[i]:
            Pop[i] = Off.copy()
            FitPop[i] = FitOff
    # update the current best
    bestIdx = np.argmin(FitPop)
    curBest = deepcopy(Pop[bestIdx])
    FitBest = FitPop[bestIdx]
    # end of battle Phase


def RunMBGO(func):
    global curFEs, curIter, MaxFEs, TrialRuns, DimSize
    All_Trial_Best = []
    for i in range(TrialRuns):
        BestList = []
        curFEs = 0
        curIter = 0
        Initialization(func)
        BestList.append(FitBest)
        np.random.seed(2024 + 88 * i)
        while curIter <= MaxIter:
            MBGO(func)
            curIter += 1
            BestList.append(FitBest)
        All_Trial_Best.append(BestList)
    np.savetxt("./MBGO_Data/CEC2022/F" + str(FuncNum) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")


def main(dim):
    global FuncNum, DimSize, MaxFEs, MaxIter, Pop, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    MaxIter = int(MaxFEs / PopSize / 2)
    LB = [-100] * dim
    UB = [100] * dim

    CEC2022 = [F12022(DimSize), F22022(DimSize), F32022(DimSize), F42022(DimSize), F52022(DimSize), F62022(DimSize),
               F72022(DimSize), F82022(DimSize), F92022(DimSize), F102022(DimSize), F112022(DimSize), F122022(DimSize)]
    FuncNum = 0
    for i in range(len(CEC2022)):
        FuncNum = i + 1
        RunMBGO(CEC2022[i])


if __name__ == "__main__":
    if os.path.exists('./MBGO_Data/CEC2022') == False:
        os.makedirs('./MBGO_Data/CEC2022')
    Dims = [10, 20]
    for Dim in Dims:
        main(Dim)


