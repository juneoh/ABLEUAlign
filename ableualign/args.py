"""Align target text to reference translation.
"""
import argparse

WINDOW_SIZE = 30
MAX_THRESHOLD = 0.9
MIN_THRESHOLD = 0.4
VOCAB = 'glove.840B.300d'
PROGRESS = False
DEVICE = 'cpu'


def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--target', '-t', required=True,
        help='The target text file to align.')
    parser.add_argument(
        '--reference', '-r', required=True,
        help='The reference translation to align to.')
    parser.add_argument(
        '--output', '-o', required=True,
        help='The output file to write the aligned target text.')

    parser.add_argument(
        '--window_size', '-w', type=int, default=WINDOW_SIZE,
        help='The number of reference sentences to compare per target.')
    parser.add_argument(
        '--max_threshold', type=float, default=MAX_THRESHOLD,
        help='The ABLEU threshold to assume best matching sentences.')
    parser.add_argument(
        '--min_threshold', type=float, default=MIN_THRESHOLD,
        help='The minimum ABLEU score for valid alignment.')
    parser.add_argument(
        '--vocab', '-v', default=VOCAB,
        help='The pretrained alias from `torchtext.vocab` to use.')
    parser.add_argument(
        '--cache_dir',
        help='The directory to save vocabulary cache.')
    parser.add_argument(
        '--progress', '-p', action='store_true', default=PROGRESS,
        help='Show progress bar.')
    parser.add_argument(
        '--device', '-d', default=DEVICE,
        help='The `torch.device` value to use in calculations.')

    return parser
