from .ableu_score import sentence_ableu, Similarity
from .args import DEVICE, MAX_THRESHOLD, MIN_THRESHOLD, WINDOW_SIZE, VOCAB


def _prepare_sentence(sentence):
    if sentence[-1] == '.':
        sentence = sentence[:-1]

    return sentence.strip().split()

def _drop(p_n, references, hyp_len, *args, **kwargs):
    new_p_n = p_n[:min(hyp_len, *[len(reference) for reference in references])]
    return new_p_n

def align(target, reference, max_threshold=MAX_THRESHOLD,
          min_threshold=MIN_THRESHOLD, window_size=WINDOW_SIZE,
          device=DEVICE, vocab=VOCAB, cache_dir=None):
    similarity = Similarity(vocab, cache_dir)
    offset = 0

    for r in range(len(reference)):
        alignment = None

        if reference[r]:

            highscore = min_threshold

            start = round(r + offset - window_size / 2)

            for t in range(start, start + window_size):
                try:
                    if not target[t]:
                        continue
                except IndexError:
                    continue

                reference_sentence = _prepare_sentence(reference[r])
                target_sentence = _prepare_sentence(target[t])

                score = sentence_ableu([reference_sentence],
                                       target_sentence,
                                       similarity=similarity,
                                       device=device,
                                       auto_reweigh=True,
                                       smoothing_function=_drop)

                if score > max_threshold:
                    alignment = t
                    break

                elif score > highscore:
                    alignment = t
                    highscore = score

        offset = (offset + (alignment - r)) / 2

        yield target[alignment].strip() if alignment is not None else ''
