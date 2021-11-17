import sys
import argparse
import os
import json
import re
import spacy
import html

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)


def preproc1(comment, steps=range(1, 5)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    modComm = comment
    if 1 in steps:  # replace newlines with spaces
        modComm = re.sub(r"\n{1,}", " ", modComm)
    if 2 in steps:  # unescape html
        modComm = html.unescape(modComm)
    if 3 in steps:  # remove URLs
        modComm = re.sub(r"(http|www)\S+", "", modComm)
    if 4 in steps:  # remove duplicate spaces
        modComm = re.sub(' +', ' ', modComm)

    # get Spacy document for modComm
    utt = nlp(modComm)
    # use Spacy document for modComm to create a string.
    modComm = ""

    for sent in utt.sents:
        for token in sent:
            # Split tokens with spaces.
            # Write "/POS" after each token.
            if token.text[0].isnumeric():
                modComm += token.text + " "
            elif token.lemma_[0] == "-" and token.text[0] != "-":
                modComm += token.text + "/" + token.tag_ + " "
            else:
                modComm += token.lemma_ + "/" + token.tag_ + " "
        # Insert "\n" between sentences.
        modComm += "\n"
        
    return modComm


def main(args):
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)

            data = json.load(open(fullFile))
            # select appropriate args.max lines
            student_id = args.ID[0]
            out_file = args.output
            start = student_id % len(data)
            end = start + args.max

            i = start
            count = 0
            while count < args.max:
                # read those lines with something like `j = json.loads(line)`
                j = json.loads(data[i])
                # choose to retain fields from those lines that are relevant to you
                # add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)
                out_line = {"id": j["id"], "body": j["body"], "cat": file}
                # process the body field (j['body']) with preproc1(...) using default for `steps` argument
                # replace the 'body' field with the processed text
                out_line["body"] = preproc1(out_line["body"])
                # append the result to 'allOutput'
                allOutput.append(out_line)
                count += 1
                i += 1
                if i >= end:  # circular loop
                    i = 0
            
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument("--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory "
                                         "for A1 on cdf.", default='/u/cs401/A1')
    
    args = parser.parse_args()
    if args.max > 200272:
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)
    
    indir = os.path.join(args.a1_dir, 'data')
    print(indir)
    main(args)
