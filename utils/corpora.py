#!/usr/bin/env python
# -*- coding: utf-8 -*-

import psycopg2
from psycopg2 import IntegrityError
import os, sys, re, json
import collections
import numpy as np
import re
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
from .database import insert_cleaned_text

def freq_toquens(tokens):
    #tokens=nltk.word_tokenize(sentence)
    return nltk.FreqDist(tokens)

def idf(df, ndocs):
    return np.log(ndocs/df)

def get_tokens_spacy(texto, **kwargs):
    #nlp = spacy.load('en')
    nlp = spacy.load('en', disable=kwargs['spacy_disabled_components'])
    tokens = nlp(texto)
    #print (tokens)

    if kwargs['remove_spaces']:
        tokens = [x for x in tokens if x.is_space == False]
 
    
    #parse NER tokens and update Doc
    entity = ''
    for idx, val in enumerate(tokens):
        #print (val.lemma_ + ' ' + val.ent_iob_ + ' ' + str(val.is_stop))
        if val.ent_iob_ == 'B':
            entity = val.text
            tokens[idx].lemma_ = '_'
            
        elif val.ent_iob_ == 'I':
            if entity != '':
                entity += '_'+val.text
            else:
                entity = val.text
            tokens[idx].lemma_ = '_'
            
        elif val.ent_iob_ == 'O':
            #tokens[idx].lemma_ = tokens[idx].lemma_.lower()
            if entity != '':
                tokens[idx - 1].lemma_ = entity
            entity = ''
    if entity != '':
        tokens[idx - 1].lemma_ = entity
    tokens = [x for x in tokens if x.lemma_ != '_']

    #for ent in tokens.ents:
        #print(ent.text, ent.start_char, ent.end_char, ent.label_)
        #print(texto[ent.start_char:ent.end_char].replace(' ', '_')) 


    if kwargs['remove_stop_words']:
        tokens = [x for x in tokens if x.is_stop == False]

    if kwargs['remove_punct']:
        tokens = [x for x in tokens if x.is_punct == False]

    if kwargs['remove_numbers']:
        tokens = [x for x in tokens if x.like_num == False]

    

    if kwargs['remove_apostrophe']:
        tokens = [x for x in tokens if x.shape_ != "'x"]

    if kwargs['remove_2letters_words']:
        tokens = [x for x in tokens if len(x) > 2]
    
    if kwargs['lemmas']:
        tokens = [token.lemma_ for token in tokens]
    else:
        tokens = [token.text for token in tokens]
    return tokens

def get_tokens_default(texto, **kwargs):
    tokens = texto.lower().strip().split()
    tokens = remove_stop_words_from_tokens(tokens)
    return tokens

def get_tokens(texto, **kwargs):
    tokens = []
    if kwargs['method'] == 'default':
        tokens = get_tokens_default(texto, **kwargs)
    elif kwargs['method'] == 'spacy':
        tokens = get_tokens_spacy(texto, **kwargs)
    return tokens

def remove_stop_words_from_tokens(data):
    return [x for x in data if x not in STOP_WORDS]

def remove_square_brackets(_text):
    return re.sub("(\[([^\]])*\])", "", _text)

def clean_text(_text, **kwargs):
    
    tokens = []
    if type(_text) is str:
        snippet = remove_square_brackets(_text)
        tokens.append(get_tokens(snippet, **kwargs))

    elif type(_text) is dict:
        
        data = _text
        snippet = remove_square_brackets(data['text'])
        tokens.append({'id': data['id'], 'snippet': get_tokens(snippet, **kwargs)})

    elif type(_text) is list:
        
        data = _text
        for row in data:
            snippet = remove_square_brackets(data['text'])
            tokens.append({'id': data['id'], 'snippet': get_tokens(snippet, **kwargs)})

        
    elif type(_text) is pd.core.frame.DataFrame:
        dataframe = _text
        for index, row in dataframe.iterrows():
            
            snippet = remove_square_brackets(row['text'])
            tokens.append({'id': row['id'], 'snippet': get_tokens(snippet, **kwargs)})
    return tokens

def get_word_counts_per_snippet(dataframe, **kwargs):

    docs = []
    for index, row in dataframe.iterrows():
        doc = collections.Counter()
        doc['id'] = row['id']
        if kwargs['clean_text']:
            tokens = get_tokens(row['text'], **kwargs)
        else:
            tokens = row['text'].split()

        for w in tokens:
            doc[w] += 1
        docs.append( doc )
    return docs

def get_vocabulary(word_docs):
    vocabulary = set()
    for d in word_docs:
        for w in d:
            vocabulary.add(w)
    return vocabulary

def get_snippets_by_word2(dataframe):
   
    snippets_by_word = {}
    for index, row in dataframe.iterrows():
        tokens = row['text'].split()
        for w in tokens:
            if w not in snippets_by_word:
                snippets_by_word[w] = [row['id']]
            else:
                snippets_by_word[w].append(row['id'])
        
    return snippets_by_word

def get_snippets_by_word(word_counts_per_snippet):
    
    snippets_by_word = {}
    for d in word_counts_per_snippet:
        for w in d:
            snippet_word_counts = {'snippet': d['id'], 'counts': d[w]}
            if w not in snippets_by_word:
                #snippets_by_word[w] = [d['id']]
                snippets_by_word[w] = [snippet_word_counts]
            else:
                snippets_by_word[w].append(snippet_word_counts)
    return snippets_by_word

def get_frequencies(docs):
    tf = collections.Counter()
    df = collections.Counter()
    for d in docs:
        for w in d:
            tf[w] += d[w]
            df[w] += 1
    return tf 

def get_idfs(docs):
    tf = collections.Counter()
    df = collections.Counter()
    for d in docs:
        for w in d:
            tf[w] += d[w]
            df[w] += 1

    idfs = {}
    for w in tf:
        if tf[w] > 1:
            idfs[w] = idf(df[w], len(docs))
    voc=sorted(idfs, key=idfs.get, reverse=True)[:200]
    return idfs, voc


def test_sentent_spacy(_sentence):
    nlp = spacy.load('en')
    Doc = nlp(_sentence)
    print (Doc.ents)
    for ent in Doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
    print (Doc.cats)
    for token in Doc:
        #if token.lemma_ == "'s":
        print ('%s - %s - %s - %s - %s - %s - %s' % (token.lemma_, token.like_num, token.like_num, token.shape_, token.is_punct, token.is_quote, token.is_stop)) 
        print (token.ent_type)
        print (token.ent_iob)
        

            
    