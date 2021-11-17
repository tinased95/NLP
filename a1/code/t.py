import numpy as np
import re

comment = "hehe/UH ,/, I/PRP second/VBP this/DT ./.\nI/PRP adore/VBP Clueless/NNP ./.\n"

words = re.compile(r"(?<=\b)\w+(?=/)").findall(comment)

