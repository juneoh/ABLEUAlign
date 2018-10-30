from unittest import TestCase

import torch
from torchtext.vocab import pretrained_aliases

from ableualign import ableu_score, align


class TestABLEU(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.similarity = ableu_score.Similarity()

    def test_precision_identical(self):
        sentence = 'a b c d'.split()

        precision = ableu_score.modified_precision([sentence], sentence,
                                             similarity=self.similarity, n=2)

        self.assertEquals(
            float(precision),
            1.0,
            'The precision of two identical sentences should be 1.0')

    def test_ableu_identical(self):
        sentence = 'a b c d'.split()

        self.assertEquals(
            ableu_score.sentence_ableu([sentence], sentence,
                                 similarity=self.similarity),
            1.0,
            'The ABLEU of two identical sentences should be 1.0')

    def test_precision_monogram(self):
        reference = ['hello']
        hypothesis = ['world']

        precision = ableu_score.modified_precision([reference], hypothesis, 
                                             similarity=self.similarity, n=1)
        self.assertEquals(
            float(precision),
            self.similarity(reference[0], hypothesis[0]),
            'The precision of two monograms should be their distance')

    def test_precision_bigram(self):
        reference = 'hello world'.split()
        hypothesis = 'hi universe'.split()

        precision = torch.Tensor([
            self.similarity(reference[0], hypothesis[0]),
            self.similarity(reference[1], hypothesis[1])]).mean()

        self.assertEquals(
            float(ableu_score.modified_precision([reference], hypothesis, n=2,
                                           similarity=self.similarity)),
            precision,
            'The precision of two bigrams should be the average of their '
            'distance')

    def test_unknown(self):
        reference = ['hello']
        hypothesis = ['%%']

        ableu_score.modified_precision([reference], hypothesis, n=1,
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

        aligned = [sentence for sentence in align(query, translation)]
        self.assertEquals(
            aligned,
            answer)
