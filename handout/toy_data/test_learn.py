# Hayden Moore
# 10-301 / 10-601
# HW7 HMM

import sys
import numpy


def load_file(file):
    with open(file, 'r') as f:
        data = f.readlines()
    i = 0
    for line in data:
        #print(line)
        data[i] = line.rstrip()
        #print(data[i])
        i += 1

    return data


def compute_priors(data, tag_index):
    priors = [0.0] * len(tag_index)

    # count number of observations that start with each tag
    for line in data:
        pretag = line[0].split('_')
        tag = pretag[1]

        index = tag_index.index(tag)
        priors[index] += 1.0

    # Pseudo
    i = 0
    sum = 0.0
    for prior in priors:
        priors[i] += 1
        sum += priors[i]
        i += 1

    # Norm
    i = 0
    for prior in priors:
        priors[i] = "{:e}".format(priors[i] / sum)
        i += 1

    return priors


def compute_trans(data, tag_index):
    size = len(tag_index)
    trans = numpy.zeros((size, size))
    print(trans)

    for line in data:
        i = 0
        for word in line:
            if i == len(line) - 1:
                break
            tag = word.split('_')
            w_index = tag_index.index(tag[1])
            next_tag = line[i + 1].split('_')
            n_index = tag_index.index(next_tag[1])

            trans[w_index][n_index] += 1.0

            i += 1

    # Pseudo
    trans = numpy.add(trans, 1)
    print(trans)
    # Norm
    i = 0
    for tran in trans:
        sum = numpy.sum(tran)
        trans[i] = numpy.divide(trans[i], sum)

        i += 1
    print(trans)
    return trans

def compute_emit(data, word_index, tag_index):
    t_size = len(tag_index)
    w_size = len(word_index)
    emit = numpy.zeros((t_size, w_size))

    for line in data:
        for word in line:
            word = word.split('_')

            t = word[1]
            w = word[0]

            t = tag_index.index(t)
            w = word_index.index(w)

            emit[t][w] += 1.0

    # Pseudo
    emit = numpy.add(emit, 1)

    # Norm
    i = 0
    for e in emit:
        sum = numpy.sum(e)
        emit[i] = numpy.divide(emit[i], sum)

        i += 1

    return emit


def main():
    train_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]

    # load in training data
    train_data = load_file(train_input)
    i = 0
    for line in train_data:
        train_data[i] = line.split()
        #print(train_data[i])
        i += 1

    # load in word index
    word_index = load_file(index_to_word)

    # load in tag idex
    tag_index = load_file(index_to_tag)

    # compute priors
    print(train_data)
    print(tag_index)
    print(word_index)
    print("\n")
    priors = compute_priors(train_data, tag_index)
    print(priors)

    # compute trans
    trans = compute_trans(train_data, tag_index)

    # compute emit
    emit = compute_emit(train_data, word_index, tag_index)
    print(emit)

    # write outputs
    with open(hmmprior, 'w') as f:
        for prior in priors:
            f.write("{:.18e}".format(float(prior)) + '\n')

    with open(hmmtrans, 'w') as f:
        for tran in trans:
            for word in tran:
                f.write("{:.18e}".format(word) + ' ')
            f.write('\n')

    with open(hmmemit, 'w') as f:
        for e in emit:
            for word in e:
                f.write("{:.18e}".format(word) + ' ')
            f.write('\n')


if __name__ == "__main__":
    main()