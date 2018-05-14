"""""
    # Python 3.0
    Author: Arkaditya
    CS429 Information Retrieval
    Assignment 4
"""
import re
import os
import collections
import time
import numpy as np
import operator
import sys
from itertools import islice


class PageRank:

    def __init__(self, path):
        self.path = path

    # import other modules as needed
    def pagerank(self, input_file, iter):
        # function to implement pagerank algorithm
        # input_file - input file that follows the format provided in the assignment description
        alpha = 0.15
        pageDic = {}

        with open(input_file) as file:
            numpages = int(file.readline())
            numlinks = int(file.readline())

            P = np.zeros(shape=(numpages, numpages))  # Initialize Transition Matrix
            tmatrix = [[0 for x in range(int(numpages))] for y in range(int(numpages))]
            pStateMatrix = [1 / numpages for x in range(int(numpages))]  # initial Page Rank matrix

            S = np.matrix(pStateMatrix, dtype=float)

            for line in file:
                line = re.split(r'\D', line)
                tmatrix[int(line[0])][int(line[1])] = 1
        file.close()

        A = np.matrix(tmatrix, dtype=float)
        sumrow = np.array(A.sum(1))[:, 0]

        for rows in range(A.__len__()):
            if sumrow[rows] == 0:
                A[rows] = [1 / numpages for x in range(int(numpages))]
                A[rows] = (1 - alpha) * A[rows]
            else:
                A[rows] = (1 - alpha) * (A[rows] / sumrow[rows])
            P[rows] = A[rows] + (alpha / numpages)

        if iter == "":
            iter = 15

        for i in range(int(iter)):
            X = np.matmul(S, P)
            S = X
        S = list(np.array(S))



        for i in range(numpages):
            pageDic[i] = np.take(S, i)

        sortedranks = sorted(pageDic.items(), key=operator.itemgetter(1))

        # Store the sorted top 10 PageRank values
        toptenranks = sortedranks[:10]

        print("\n========= THE Top 10 DocID, PAGE RANK pairs =========")
        for docId, ranks in toptenranks:
            print(docId, ":  ", ranks)

        # Redirect standard output to a file
        sys.stdout = open('out.txt', 'w')
        print("========= DocID, PAGE RANK values for %s =========" % input_file)
        for docId, ranks in sortedranks:
            print(docId, ":  ", ranks)



if __name__ == '__main__':
    pr = PageRank('.')
    inputFile = input("Enter the input file:\t")
    iterations = input("Enter the no of iterations to be run for PageRank computation:\t")
    pr.pagerank(inputFile, iterations)
