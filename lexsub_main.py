#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import numpy as np
import tensorflow

import gensim
import transformers 

from typing import List

from collections import defaultdict
import string


def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    candidates = set()
    for syn in wn.synsets(lemma, pos):
        for lem in syn.lemmas():
            if lem.name() != lemma:
                candidates.add(lem.name().replace('_', ' '))

    return list(candidates) 

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    candidates = defaultdict(int)
    for syn in wn.synsets(context.lemma, context.pos):
        for lem in syn.lemmas():
            if lem.name() != context.lemma:            
                candidates[lem] += lem.count()
    
    return max(candidates, key=candidates.get).name().replace('_', ' ')

def wn_simple_lesk_predictor(context : Context) -> str:
    target = None
    synset = None
    freqsyn = None
    highfreq = 0
    maxOverlap = 0
    overLap = defaultdict(int)
    mfs = defaultdict(int)
    count = defaultdict(int)
    for syn in wn.synsets(context.lemma, context.pos):
        if len(syn.lemmas()) == 1 and syn.lemmas()[0].name() == context.lemma:
            continue

        definition = getDef(syn)
        fullContext = getCon(context)
        
        overlap = getOverlap(definition, fullContext)
        if overlap >= maxOverlap:
            overLap[syn] = overlap
            maxOverlap = overlap

        for lem in syn.lemmas():
            if lem.count() > highfreq and lem.name() != context.lemma:
                freqsyn = syn
                highfreq = lem.count()

    
    if maxOverlap == 0: #if there are no overlapping words
        synset = freqsyn
        #return wn_frequency_predictor(context)
    else:
        ovlist = [k for k,v in overLap.items() if v == maxOverlap]
        
        if len(ovlist) == 1: #if there is one highest overlapping word
            synset = ovlist[0]
        else:                   #if there is a tie
            for syn in ovlist:
                for lem in syn.lemmas():
                    if lem.name() != context.lemma:
                        count[syn] += lem.count()

            if count:
                synset = max(count, key=count.get)

    freq = {}
    if synset:
        for lex in synset.lemmas():
            if lex.name() != context.lemma:
                freq[lex] = lex.count()

    if freq:
        target = max(freq, key=freq.get).name().replace('_', ' ')

    return target
   
def lemmatize(lemmas, pos):
    lemmatizer = WordNetLemmatizer()
    lemmatized = set()
    if pos is None:
        for lemma in lemmas:
            lemmatized.add(lemmatizer.lemmatize(lemma))
    else:
        for lemma in lemmas:
            lemmatized.add(lemmatizer.lemmatize(lemma, pos))

    return lemmatized

def getOverlap(defi, cont):
    stop = set(stopwords.words('english'))
    punc = set(string.punctuation)
    stoppunc = stop.union(punc)
    
    overlapping = defi.intersection(cont)
    overlapping = overlapping.difference(stoppunc)
    return len(overlapping)

def getCon(context):
    fullContext = ' '.join(context.left_context) + ' ' + ' '.join(context.right_context)
    fullContext = tokenize(fullContext)
    fullContext = lemmatize(fullContext, None)

    return set(fullContext)

def getDef(syn):
    definition = tokenize(syn.definition())
    for ex in syn.examples():
        definition.extend(tokenize(ex))
    definition.extend(examples(syn))
    definition = lemmatize(definition, None)
    return set(definition)

def examples(syn):
    examples = []
    for hyp in syn.hypernyms():
        examples.extend(tokenize(hyp.definition()))
        examples.extend(hyp.lemma_names())
        for hyp in hyp.examples():
            examples.extend(tokenize(hyp))

    return examples

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        simil = {}
        cans = get_candidates(context.lemma, context.pos)
        for word in cans:
            if word in self.model.wv.vocab:
                simil[word] = self.model.similarity(context.lemma, word)

        return max(simil, key=simil.get)
    
    def predictFromGroup(self, context, group): #part 6
        simil = {}
        candidates = get_candidates(context.lemma, context.pos)
        for syn in group:
            candidates += syn.lemma_names()

        for word in candidates:
            if word in self.model.wv.vocab and word != context.lemma:
                simil[word] = self.model.similarity(context.lemma, word)

        return max(simil, key=simil.get)

    def predictForBert(self, context, group):
        simil = {}
        candidates = get_candidates(context.lemma, context.pos)
        candidates += group

        for word in candidates:
            if word in self.model.wv.vocab and word != context.lemma:
                simil[word] = self.model.similarity(context.lemma, word)

        return max(simil, key=simil.get)

    def lesk2Vec(self, context): # this method is for part 6
        
        overLap = []
        
        for syn in wn.synsets(context.lemma, context.pos):
            if len(syn.lemmas()) == 1 and syn.lemmas()[0].name() == context.lemma:
                continue

            definition = getDef(syn)
            fullContext = getCon(context)
        
            overlap = getOverlap(definition, fullContext)
            if overlap > 0:
                overLap.append(syn)

        if len(overLap) == 0: #if there are no overlapping words
            return self.predict_nearest(context)
        else:

            return self.predictFromGroup(context, overLap)

        return None


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)

        left = self.tokenizer.tokenize(' '.join(context.left_context))
        right = self.tokenizer.tokenize(' '.join(context.right_context))
        input = left + ["[MASK]"] + right
        idx = len(left) + 1

        input_toks = self.tokenizer.encode(input)
        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.model.predict(input_mat)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][idx])[::-1] 
        bestTokens = self.tokenizer.convert_ids_to_tokens(best_words)

        for token in bestTokens:
            if token in candidates:
                return token

        return None # replace for part 5

    

if __name__=="__main__":

    # For part 6 I wrote the lesk2Vec method in the Word2VecSubst class. In it I tried adding context to the Word2Vec model by combining it 
    # with the Lesk algorithm to generate a larger bag of words for Word2Vec to select from. Also I added Hypernyms themselves and their 
    # lexemes to the definition. However none of it added or took away from my precision. All three part4, part5, and part6 got the same results.


    #W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)
    predictor = BertPredictor()

    for context in read_lexsub_xml(sys.argv[1]):
    #    print(context)  # useful for debugging
    #    prediction = smurf_predictor(context)
        #prediction = get_candidates(context.lemma, context.pos)
        #prediction = wn_frequency_predictor(context)
        #prediction = wn_simple_lesk_predictor(context)
        #prediction = predictor.predict_nearest(context)
        prediction = predictor.predict(context)
        #prediction = predictor.lesk2Vec(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
    



    #print(get_candidates('slow', 'a'))
    #left = 'During the siege , George Robertson had appointed Shuja-ul-Mulk , who was a'.split()
    #right = 'boy only 12 years old and the youngest surviving son of Aman-ul-Mulk , as the ruler of Chitral .'.split()

    #input = ' '.join(left) + " [MASK] " + ' '.join(right)
    #print(input)
    #context = Context(cid=1, word_form='bright', lemma='bright', pos='a', left_context=left, right_context=right)
    #prediction = wn_simple_lesk_predictor(context)
    #print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
