from .ableu_score import sentence_ableu, Similarity
from .args import MAX_THRESHOLD, MIN_THRESHOLD, WINDOW_SIZE, VOCAB


def align(target, reference, max_threshold=MAX_THRESHOLD,
          min_threshold=MIN_THRESHOLD, window_size=WINDOW_SIZE,
          vocab=VOCAB, cache_dir=None):
    similarity = Similarity(vocab, cache_dir)

    for r in range(len(reference)):
        alignment = None

        if reference[r]:

            highscore = min_threshold

            for t in range(r - window_size // 2, r + window_size // 2):
                try:
                    if not target[t]:
                        continue
                except IndexError:
                    continue

                score = sentence_ableu([reference[r].strip().split()],
                                       target[t].strip().split(),
                                       similarity=similarity)

                if score > max_threshold:
                    alignment = t
                    break

                elif score > highscore:
                    alignment = t
                    highscore = score

        yield target[alignment].strip() if alignment is not None else ''
