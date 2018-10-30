#!/usr/bin/env python3
from functools import partial
from unittest.mock import patch

from nltk.translate.bleu_score import Counter, ngrams, sentence_bleu
import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from torchtext.vocab import pretrained_aliases

from .args import DEVICE, VOCAB


class IrrationalFraction:
    def __init__(self, numerator, denominator, *args, **kwargs):
        self.numerator = numerator
        self.denominator = denominator

    def __float__(self):
        return self.numerator / self.denominator


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
            similarity = (similarity + 1) / 2
            self._cache[pair] = similarity

            return similarity


def modified_precision(references, hypothesis, n, similarity,
                       device=DEVICE):
    if len(hypothesis) < n:
        return IrrationalFraction(0, 1)

    hypothesis_counts = Counter(ngrams(hypothesis, n))

    numerator_matrix = []
    for reference in references:
        if len(reference) < n:
            numerator_matrix.append(torch.zeros([len(hypothesis_counts)]))
            continue

        reference_counts = Counter(ngrams(reference, n))
        similarity_tensor = []
        clip_vector = []

        for hypothesis_ngram in hypothesis_counts:

            similarity_vector = []
            for w in range(n):
                similarity_vector.append([
                    similarity(hypothesis_ngram[w], reference_ngram[w])
                    for reference_ngram in reference_counts])
            similarity_tensor.append(similarity_vector)

            clip_vector.append(min(hypothesis_counts[hypothesis_ngram],
                                   sum(reference_counts.values())))

        similarity_tensor = torch.Tensor(similarity_tensor, device=device)
        clip_vector = torch.Tensor(clip_vector, device=device)
        numerator_matrix.append(
            similarity_tensor.mean(dim=1).max(dim=1)[0] * clip_vector)

    numerator = torch.stack(
        numerator_matrix).max(dim=0)[0].sum().item()

    fraction = IrrationalFraction(
        numerator=numerator,
        denominator=max(1, sum(hypothesis_counts.values())))

    return fraction


def sentence_ableu(references, hypothesis, similarity=None, device=DEVICE,
                   *args, **kwargs):
    if similarity is None:
        similarity = Similarity()

    modified_precision_ = partial(modified_precision, similarity=similarity,
                                  device=device)
    with patch('nltk.translate.bleu_score.modified_precision',
               modified_precision_):
        with patch('nltk.translate.bleu_score.Fraction', IrrationalFraction):
            return sentence_bleu(references, hypothesis, *args, **kwargs)
