import numpy as np
import argparse
import json
import re
import os
import csv
import statistics
import string

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}


WORD_LIST_DIR = '/u/cs401/Wordlists'
FEATS_DIR = '/u/cs401/A1/feats'

BRISTOL_GILHOOLY_LOGIE_DIR = os.path.join(WORD_LIST_DIR, 'BristolNorms+GilhoolyLogie.csv')
WARRINGER_DIR = os.path.join(WORD_LIST_DIR, 'Ratings_Warriner_et_al.csv')

BGL_DICT = {
        row["WORD"]: {
            "AoA": float(row["AoA (100-700)"]),
            "IMG": float(row["IMG"]),
            "FAM": float(row["FAM"])
        }
        for row in csv.DictReader(open(os.path.join(WORD_LIST_DIR, "BristolNorms+GilhoolyLogie.csv")))
        if ((row["AoA (100-700)"] != "") or (row["IMG"] != "") or (row["FAM"] != ""))
    }

WAR_DICT = {
    row["Word"]: {
        "V.Mean.Sum": float(row["V.Mean.Sum"]),
        "A.Mean.Sum": float(row["A.Mean.Sum"]),
        "D.Mean.Sum": float(row["D.Mean.Sum"])
    }
    for row in csv.DictReader(open(os.path.join(WORD_LIST_DIR, "Ratings_Warriner_et_al.csv")))
    if ((row["V.Mean.Sum"] != "") or (row["A.Mean.Sum"] != "") or (row["D.Mean.Sum"] != ""))
}

LIWC_DICT = {
        cat: {
            id.strip(): row for id, row in
            zip(open(os.path.join(FEATS_DIR, cat + "_IDs.txt")), np.load(os.path.join(FEATS_DIR, cat + "_feats.dat.npy")))
        }
        for cat in ["Alt", "Center", "Right", "Left"]
    }

CAT_TO_INT = {"Left": 0, "Center": 1, "Right": 2, "Alt": 3}

def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''

    feat = np.zeros(29)

    splited_comment = comment.split()
    uppercase_count = 0
    uppercase_words = {}

    # Extract features that rely on capitalization.
    for word_with_tag in splited_comment:
        word_tag = word_with_tag.split("/")
        word = word_tag[0]
        # tag = word_tag[1]
        if len(word) >= 3 and word.isupper():
            uppercase_count += 1
            uppercase_words[word] = word.lower()

    # Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    for k, v in uppercase_words.items():
        comment = comment.replace(k, v)

    feat[0] = uppercase_count

    # Number of first-person pronouns
    words_re = re.compile(r'\b%s\b' % '\\b|\\b'.join(FIRST_PERSON_PRONOUNS), flags=re.IGNORECASE)
    first_person = words_re.findall(comment)
    # first_person_count = re.findall(r"\b(I|me|my|mine|we|us|our|ours)\b", comment)
    feat[1] = len(first_person)

    # Number of second-person pronouns
    words_re = re.compile(r'\b%s\b' % '\\b|\\b'.join(SECOND_PERSON_PRONOUNS), flags=re.IGNORECASE)
    second_person = words_re.findall(comment)
    feat[2] = len(second_person)

    # Number of third-person pronouns
    words_re = re.compile(r'\b%s\b' % '\\b|\\b'.join(THIRD_PERSON_PRONOUNS), flags=re.IGNORECASE)
    third_person = words_re.findall(comment)
    feat[3] = len(third_person)

    # Number of coordinating conjunctions
    cc = re.findall(r"/CC\b", comment)
    feat[4] = len(cc)

    # Number of past-tense verbs
    vbd = re.findall(r"/VBD\b", comment)
    feat[5] = len(vbd)

    # Number of future-tense verbs
    future_tense_1 = re.findall(r"('ll|will|gonna)/", comment)
    future_tense_2 = re.compile(r"(?<=\s)\b(go\/VBG)(\s+to\/TO\s+)([\w]+\/VB)\b(?=\s+)").findall(comment)
    feat[6] = len(future_tense_1) + len(future_tense_2)

    # Number of commas
    comma = re.findall(r",/,", comment)
    feat[7] = len(comma)

    # Number of multi-character punctuation tokens
    multi = re.compile(r"(?<=\s)[\\!\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]{2,}(?=\/)").findall(comment)
    feat[8] = len(multi)

    # Number of common nouns
    nn = re.findall(r"/(NN|NNS)\b", comment)
    feat[9] = len(nn)

    # Number of proper nouns
    nnp = re.findall(r"/(NNP|NNPS)\b", comment)
    feat[10] = len(nnp)

    # Number of adverbs
    rb = re.findall(r"/(RB|RBR|RBS)\b", comment)
    feat[11] = len(rb)

    # Number of wh-words
    wh = re.findall(r"/(WDT|WP|WP$|WRB)\b", comment)
    feat[12] = len(wh)

    # Number of slang acronyms
    slang = re.compile(r"(?<=\s)\b(" + r'|'.join(SLANG) + r")\b(?=\/)").findall(comment)
    feat[13] = len(slang)

    # Average length of sentences, in tokens
    sentences = comment.split("\n")
    count_sentences = 0
    for sent in sentences:
        if sent != '':
            count_sentences += 1

    if count_sentences == 0:
        count_sentences = 1

    words = re.findall(r"[\S]+", comment)
    feat[14] = len(words) / count_sentences

    # Average length of tokens, excluding punctuation-only tokens, in characters
    token_len = 0
    words = re.findall(r"[!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~']*\w+[!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~']*(?=/)", comment)
    if len(words) != 0:
        for word in words:
            token_len += len(word)
        feat[15] = token_len / len(words)

    # Number of sentences.

    feat[16] = count_sentences

    words = re.compile(r"(?<=\s)([\w]+)(?=\/[A-Z]+[$]*\s+)").findall(comment)
    aoa = []
    img = []
    fam = []
    v = []
    a = []
    d = []

    for word in words:
        if word in BGL_DICT.keys():
            aoa.append(BGL_DICT[word]["AoA"])
            img.append(BGL_DICT[word]["IMG"])
            fam.append(BGL_DICT[word]["FAM"])
        if word in WAR_DICT.keys():
            v.append((WAR_DICT[word]["V.Mean.Sum"]))
            a.append((WAR_DICT[word]["A.Mean.Sum"]))
            d.append((WAR_DICT[word]["D.Mean.Sum"]))

    if len(aoa) != 0:
        aoa = np.array(aoa)
        # Average of AoA (100-700) from Bristol, Gilhooly, and Logie norms
        feat[17] = np.mean(aoa)
        # Standard deviation of AoA (100-700) from Bristol, Gilhooly
        feat[20] = np.std(aoa)

    if len(img) != 0:
        img = np.array(img)
        # Average of IMG from Bristol, Gilhooly, and Logie norms
        feat[18] = np.mean(img)
        # Standard deviation of IMG from Bristol, Gilhooly, and Logie norms
        feat[21] = np.std(img)

    if len(fam) != 0:
        fam = np.array(fam)
        # Average of FAM from Bristol, Gilhooly, and Logie norms
        feat[19] = np.mean(fam)
        # Standard deviation of FAM from Bristol, Gilhooly, and Logie norms
        feat[22] = np.std(fam)

    if len(v) != 0:
        v = np.array(v)
        # average of V.Mean.Sum from Warringer norms
        feat[23] = np.mean(v)
        # standard deviation of V.Mean.Sum from Warringer norms
        feat[26] = np.std(v)

    if len(a) != 0:
        a = np.array(a)
        # average of A.Mean.Sum from Warringer norms
        feat[24] = np.mean(a)
        # standard deviation of A.Mean.Sum from Warringer norms
        feat[27] = np.std(a)
    if len(d) != 0:
        d = np.array(d)
        # average of D.Mean.Sum from Warringer norms
        feat[25] = np.mean(d)
        # standard deviation of D.Mean.Sum from Warringer norms
        feat[28] = np.std(d)

    return feat


def extract2(feats, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feats: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (this
        function adds feature 30-173). This should be a modified version of
        the parameter feats.
    '''
    return LIWC_DICT[comment_class][comment_id]


def main(args):
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173 + 1))

    for i in range(len(data)):
        # Use extract1 to find the first 29 features for each data point. Add these to feats.
        feats[i][0:29] = extract1(data[i]["body"])

        # Use extract2 to copy LIWC features (features 30-173) into feats. (Note that these rely on each
        # data point's class, which is why we can't add them in extract1).
        feats[i][29:173] = extract2(feats[i], data[i]["cat"], data[i]["id"])
        feats[i][173] = CAT_TO_INT[data[i]["cat"]]

    np.savez_compressed(args.output, feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()

    main(args)

