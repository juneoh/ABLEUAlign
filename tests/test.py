from unittest import TestCase

import numpy as np
from torchtext.vocab import pretrained_aliases

from ableualign import ableu
from ableualign import align


class TestABLEU(TestCase):
    def setUp(self):
        self.similarity = ableu.Similarity()

    def test_precision_identical(self):
        sentence = 'hello world'.split()

        self.assertEquals(
            ableu.modified_precision([sentence], sentence,
                                     similarity=self.similarity, n=2),
            1.0,
            'The precision of two identical sentences should be 1.0')

    def test_ableu_identical(self):
        sentence = 'hello world'.split()

        self.assertEquals(
            ableu.sentence_ableu([sentence], sentence,
                                 similarity=self.similarity),
            1.0,
            'The ABLEU of two identical sentences should be 1.0')

    def test_precision_monogram(self):
        reference = ['hello']
        hypothesis = ['world']

        similarity = ableu.Similarity()
        self.assertEquals(
            ableu.modified_precision([reference], hypothesis, 
                                     similarity=self.similarity, n=1),

            similarity(reference[0], hypothesis[0]),
            'The precision of two monograms should be their distance')

    def test_precision_bigram(self):
        reference = 'hello world'.split()
        hypothesis = 'hi universe'.split()

        precision = np.mean([
            self.similarity(reference[0], hypothesis[0]),
            self.similarity(reference[1], hypothesis[1])])

        self.assertEquals(
            float(ableu.modified_precision([reference], hypothesis, n=2,
                                           similarity=self.similarity)),
            precision,
            'The precision of two bigrams should be the average of their '
            'distance')

    def test_unknown(self):
        reference = ['hello']
        hypothesis = ['%%']

        ableu.modified_precision([reference], hypothesis, n=1,
                                 similarity=self.similarity)


class TestAlign(TestCase):
    def test_align(self):
        answer = ['Beautiful is better than ugly',
                  'Explicit is better than implicit',
                  'Simple is better than complex']
        query = [answer[2], answer[0], answer[1]]
        translation = ['Beauty is better than ugliness',
                       'Specification is better than implied',
                       'Simplicity is better than complexity']

        aligned = align(query, translation)
        self.assertEquals(
            aligned,
            answer)
