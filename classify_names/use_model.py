import torch
import torch.nn.functional as F

from rnn_chars import CharRNN
import process_words as words
from process_words import lineToTensor
from process_dataset import NamesDataset
# can't import train_nn or it'll compile/run

model_path = 'rnn_classif.pth'

alldata = NamesDataset("data/names")
n_hidden = 128
print('started load')

rnn = CharRNN(words.n_letters, n_hidden, len(alldata.labels_uniq))
rnn.load_state_dict(torch.load(model_path, weights_only=True))
rnn.eval()  # pass data through it

print('ready to roll')

while True:
    request = input()
    if(request == "end"):
        break
    embed = lineToTensor(request)
    output = rnn.forward(embed)
    # Get the index of the maximum probability (highest probability)
    predicted_idx = torch.argmax(output, dim=1).item()
    # Map the predicted index to the language label
    predicted_language = alldata.labels_uniq[predicted_idx]
    print(predicted_language)