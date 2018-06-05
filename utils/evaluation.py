#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import math


def get_percentage_records(w1, w2, score, _percentage):
    len_file = len(w1)
    population = [x for x in range(len_file)]
    n_samples = len(population) * (_percentage / 100)
    n_samples = math.ceil(n_samples)
    #print (n_samples)
    random.seed()
    s = random.randint(1, 1000000)
    random_indexes = random.sample(population, n_samples)

    w1 = [w1[i] for i in random_indexes]
    w2 = [w2[i] for i in random_indexes]
    score = [score[i] for i in random_indexes]
    return w1, w2, score


def get_words_for_men_dataset(line):
    
    words = line.split(' ')
    w1 = words[0].split('-')[0]
    w2 = words[1].split('-')[0]
    score = words[2].replace('\n', '')
    return [w1, w2, score]


def get_words_for_truk_dataset(line):
    
    words = line.split(' ')
    w1 = words[0]
    w2 = words[1]
    score = words[2].replace('\n', '')
    return [w1, w2, score]


def get_words_for_rg65_dataset(line):
    words = line.split('\t')
    w1 = words[0]
    w2 = words[1]
    score = float(words[2].replace('\n', ''))
    return [w1, w2, score]


def get_words_for_ws353_dataset(line):
    words = line.split('\t')
    w1 = words[0]
    w2 = words[1]
    score = float(words[2].replace('\n', ''))
    return [w1, w2, score]