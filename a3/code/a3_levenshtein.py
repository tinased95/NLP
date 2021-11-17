import os
import numpy as np
import string

dataDir = '/u/cs401/A3/data/'

def Levenshtein(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list of strings
    h : list of strings

    Returns
    -------
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    0.333 0 0 1
    >>> wer("who is there".split(), "".split())
    1.0 0 0 3
    >>> wer("".split(), "who is there".split())
    Inf 0 3 0
    """
    n, m = len(r), len(h)

    # distance matrix initialization
    R = np.zeros((n + 1, m + 1))
    R[0, :] = np.array(list(range(m + 1)))
    R[:, 0] = np.array(list(range(n + 1)))

    # backtrace matrix initialization
    B = np.zeros((n + 1, m + 1), dtype=object)
    B[0, :] = 'left'
    B[:, 0] = 'up'
    B[0, 0] = ''

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            del_error = R[i - 1, j] + 1  # deletion

            sub_error = R[i - 1, j - 1]  # if words match
            if r[i - 1] != h[j - 1]:  # if words differ
                sub_error += 1

            ins_error = R[i, j - 1] + 1  # insertion

            R[i, j] = min([del_error, sub_error, ins_error])

            # update B matrix
            if R[i, j] == del_error:
                B[i, j] = 'up'
            elif R[i, j] == ins_error:
                B[i, j] = 'left'
            else:
                B[i, j] = 'up-left'

    # backtrace
    nI, nD = 0, 0
    t, s = n, m
    while True:
        if t <= 0 and s <= 0:
            break
        step = B[t, s]
        if step == 'up':
            nD += 1
            t = t - 1
        elif step == 'left':
            nI += 1
            s = s - 1
        elif step == 'up-left':
            t = t - 1
            s = s - 1

    wer = 100 * R[n, m] / float(n)
    nS = np.rint(R[n, m] - nI - nD)

    return wer, nS, nI, nD


def preprocess_line(line):
    punc_to_remove = set(string.punctuation) - set("[]")
    return "".join([
        char.lower() if char not in punc_to_remove else " "
        for char in line
    ]).split()


if __name__ == "__main__":
    output = []
    google_errors, kaldi_errors = [], []
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)
            base_path = os.path.join(dataDir, speaker)
            ref_path = os.path.join(base_path, 'transcripts.txt')
            google_path = os.path.join(base_path, 'transcripts.Google.txt')
            kaldi_path = os.path.join(base_path, 'transcripts.Kaldi.txt')

            ref_lines = open(ref_path, 'r').read().splitlines()
            google_lines = open(google_path, 'r').read().splitlines()
            kaldi_lines = open(kaldi_path, 'r').read().splitlines()

            if len(ref_lines) * len(google_lines) * len(kaldi_lines) > 0:
                for i in range(min(len(ref_lines), len(google_lines), len(kaldi_lines))):
                    ref = preprocess_line(ref_lines[i])
                    google = preprocess_line(google_lines[i])
                    kaldi = preprocess_line(kaldi_lines[i])

                    google_result = Levenshtein(ref, google)
                    google_errors.append(google_result[0])

                    kaldi_result = Levenshtein(ref, kaldi)
                    kaldi_errors.append(kaldi_result[0])

                    output.append("%s %s %d %f S:%d I:%d D:%d\n" %
                                  (speaker, "Kaldi", i, kaldi_result[0], kaldi_result[1], kaldi_result[2], kaldi_result[3]))
                    output.append("%s %s %d %f S:%d I:%d D:%d\n" %
                                  (speaker, "Google", i, google_result[0], google_result[1], google_result[2], google_result[3]))

    fout = open("asrDiscussion.txt", 'w')
    for line in output:
        fout.write(line)

    google_mean = np.mean(google_errors)
    google_std = np.std(google_result)
    kaldi_mean = np.mean(kaldi_result)
    kaldi_std = np.std(kaldi_result)
    fout.write("Google mean %f \nGoogle std %f \nKaldi mean %f \nKaldi std %f \n" % (np.mean(google_errors),
                                                                                         np.std(google_errors),
                                                                                         np.mean(kaldi_errors),
                                                                                         np.std(kaldi_errors)))
    fout.write("\n\n")
    fout.write("Discussion: As we can see from the results, Kaldi has better accuracy than Google. The Google "
               "translation will make more substitution deletion errors. By examining the translations from both "
               "models, we can see that the Google translation makes a sentence more readable, whereas Kaldi tries"
               " to keep the origin. One mistake google made is to ignore sounds like um, hm, etc.")
    fout.close()
