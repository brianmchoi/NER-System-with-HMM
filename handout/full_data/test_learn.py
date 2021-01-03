#dylan-wyl10
# -*- coding: utf-8 -*-

"""
@author: Dylan Wang
"""

import numpy as np
import sys


def splitWords(file):
    f = open(file)
    word_matrix, tag_matrix = [], []
    for i in f:
        temp = i.strip('\n').split(' ')
        wli = [] #word_list for lines
        tli = [] #tag_list for lines
        for ii in temp:
            wli.append(ii.split('_')[0])
            tli.append(ii.split('_')[1])
        word_matrix.append(wli)
        tag_matrix.append(tli)
    return word_matrix, tag_matrix #output two split 2darray


def splitIndex(file):
    f = open(file)
    idxli = []
    for i in f:
        idxli.append(i.strip('\n'))
    return idxli


def calPi(tgli, tgidx):
    pili = np.zeros((len(tgidx), 1))
    for i in tgli:
        fwd = i[0]
        pili[tgidx.index(fwd)] += 1
    pili_final = (pili + 1)/(sum(pili) + len(pili))
    return pili_final


def calA(tgli, tgidx):
    ali = np.zeros((len(tgidx), len(tgidx)))
    for li in tgli:
        for id in range(len(li) - 1):
            next = id + 1
            tagCurr = li[id]
            tagNext = li[next]
            ali[tgidx.index(tagCurr), tgidx.index(tagNext)] += 1
    ali_final = (ali + 1) / (np.sum(ali, axis=1).reshape((-1, 1)) + ali.shape[1])
    return ali_final


def calB(wdli, wdidx, tgli, tgidx):
    bli = np.zeros((len(tgidx), len(wdidx)))
    for line in range(len(tgli)):
        for item in range(len(tgli[line])):
            word = wdli[line][item]
            tag = tgli[line][item]
            bli[tgidx.index(tag), wdidx.index(word)] += 1
    bli_final = (bli + 1) / (np.sum(bli, axis=1).reshape((-1, 1)) + bli.shape[1])
    return bli_final


if __name__ == '__main__':
    Train_Input = sys.argv[1]
    Index_2_Word = sys.argv[2]
    Index_2_Tag = sys.argv[3]
    Hmmprior = sys.argv[4]
    Hmmemit = sys.argv[5]
    Hmmtrans = sys.argv[6]
    #
    # Train_Input = 'toy_data/toytrain.txt'
    # Index_2_Word = 'toy_data/toy_index_to_word.txt'
    # Index_2_Tag = 'toy_data/toy_index_to_tag.txt'

    wdli, tgli = splitWords(Train_Input)
    wdidx = splitIndex(Index_2_Word)
    tgidx = splitIndex(Index_2_Tag)
    pi = calPi(tgli, tgidx)
    a = calA(tgli, tgidx)
    b = calB(wdli, wdidx, tgli, tgidx)

    # Hmmprior = '1.txt'
    # Hmmemit = '2.txt'
    # Hmmtrans = '3.txt'

    np.savetxt(Hmmprior, pi, delimiter=' ')
    np.savetxt(Hmmtrans, a, delimiter=' ')
    np.savetxt(Hmmemit, b, delimiter=' ')