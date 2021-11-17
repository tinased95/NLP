# Copyright 2020 University of Toronto, all rights reserved

'''Calculate BLEU score for one reference and one hypothesis

You do not need to import anything more than what is here
'''

from math import exp  # exp(x) gives e^x


def grouper(seq, n):
    '''Extract all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of words or token ids representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    '''

    # ngrams = zip(*[seq[i:] for i in range(n)])
    # return [" ".join(ngram) for ngram in ngrams]
    return [seq[i:i + n] for i in range(len(seq) - (n - 1))]


def n_gram_precision(reference, candidate, n):
    '''Calculate the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to calculate

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    '''
    if len(candidate) < n:
        return 0.0

    n_gram_cand = grouper(candidate, n)

    n_gram_ref = grouper(reference, n)

    c = sum(el in n_gram_ref for el in n_gram_cand)
    # print(c, len(n_gram_cand))
    # if len(n_gram_cand)!= 0:
    p_n = c/len(n_gram_cand)
    return p_n
    # else:
    #     return 0


def brevity_penalty(reference, candidate):
    '''Calculate the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)

    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    '''

    c = len(candidate)
    r = len(reference)

    if c != 0:
        beverity = r / c
    else:
        beverity = 0

    if beverity < 1:
        return 1
    else:
        return exp(1 - beverity)


def BLEU_score(reference, hypothesis, n):
    '''Calculate the BLEU score

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score
    '''

    bp = brevity_penalty(reference, hypothesis)
    p = 1
    for i in range(1, n + 1):
        p = p * n_gram_precision(reference, hypothesis, i)

    return bp * (p ** (1 / n))
