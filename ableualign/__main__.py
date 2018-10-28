from tqdm import tqdm

from .align import align
from .args import get_parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    with open(args.target) as f:
        target = f.readlines()

    with open(args.reference) as f:
        reference = f.readlines()

    aligned = align(target, reference,
                    max_threshold=args.max_threshold,
                    min_threshold=args.min_threshold,
                    window_size=args.window_size,
                    vocab=args.vocab,
                    cache_dir=args.cache_dir)

    if args.progress:
        aligned = tqdm(aligned, total=len(reference))

    with open(args.output, 'w') as f:
        for line in aligned:
            print(line, file=f, flush=True)


if __name__ == '__main__':
    main()
