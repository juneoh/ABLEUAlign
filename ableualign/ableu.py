#!/usr/bin/env python3
from fractions import Fraction
from functools import partial
from unittest.mock import patch

from nltk.translate.bleu_score import Counter, ngrams, sentence_bleu
import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from torchtext.vocab import pretrained_aliases

from .args import VOCAB


class Similarity:
    def __init__(self, vocab=VOCAB, cache_dir=None):
        if cache_dir:
            self._vocab = pretrained_aliases[vocab](cache=cache_dir)
        else:
            self._vocab = pretrained_aliases[vocab]()
        self._cache = {}

    def _pair(self, word1, word2):
        return tuple(sorted([word1, word2]))

    def _vector(self, word):
        try:
            return self._vocab.vectors[self._vocab.stoi[word]]
        except KeyError:
            return torch.zeros(self._vocab.dim)

    def __call__(self, word1, word2):
        if word1 == word2:
            return 1.0

        pair = self._pair(word1, word2)

        try:
            return self._cache[pair]

        except KeyError:
            similarity = cosine_similarity(
                self._vector(word1), self._vector(word2), dim=0).item()
            self._cache[pair] = similarity

            return similarity


def modified_precision(references, hypothesis, n, similarity):
    hypothesis_counts = Counter(ngrams(hypothesis, n))

    ngrams_similarity = {}
    for reference in references:
        reference_counts = Counter(ngrams(reference, n))

        for hypothesis_ngram in hypothesis_counts:
            for reference_ngram in reference_counts:
                ngram_similarity = np.mean([
                    similarity(hypothesis_ngram[i], reference_ngram[i])
                    for i in range(n)
                ])
                # Clip
                ngram_similarity *= min(hypothesis_counts[hypothesis_ngram],
                                        reference_counts[reference_ngram])

                ngrams_similarity[hypothesis_ngram] = max(
                    ngrams_similarity.get(hypothesis_ngram, 0),
                    ngram_similarity)

    numerator = sum(ngrams_similarity.values())
    denominator = max(1, sum(hypothesis_counts.values()))

    return Fraction.from_float(numerator / denominator)


def sentence_ableu(references, hypothesis, similarity=None, *args, **kwargs):
    if similarity is None:
        similarity = Similarity()

    with patch('nltk.translate.bleu_score.modified_precision',
               partial(modified_precision, similarity=similarity)):
        return sentence_bleu(references, hypothesis, *args, **kwargs)
