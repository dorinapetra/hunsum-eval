import collections


def get_ngrams(words, n_):
    queue = collections.deque(maxlen=n_)
    for w in words:
        queue.append(w)
        if len(queue) == n_:
            yield tuple(queue)
