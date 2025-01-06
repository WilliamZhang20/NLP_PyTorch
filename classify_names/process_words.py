import string
import torch
import unicodedata

allowed_characters = string.ascii_letters + " .,;'"
n_letters = len(allowed_characters)

# Turn unicode to plain ascii
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in allowed_characters
    )

# Find letter index from all letters, e.g. "a" = 0
def letterToIndex(letter):
    return allowed_characters.find(letter)

# turn a line into <line_length x 1 x n_letters> i.e. an array of one-hot
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

