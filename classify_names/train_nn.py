from process_dataset import NamesDataset
from rnn_chars import CharRNN
import process_words as words
from process_words import lineToTensor
import torch
import time
import torch.nn as nn
import random
import numpy as np

device = torch.device('cpu')

alldata = NamesDataset("data/names")
print(f"loaded {len(alldata)} items of data")
print(f"example = {alldata[0]}")

train_set, test_set = torch.utils.data.random_split(alldata, [.85, .15], generator=torch.Generator(device=device).manual_seed(2024))

print(f"train examples = {len(train_set)}, validation examples = {len(test_set)}")
n_hidden = 128

rnn = CharRNN(words.n_letters, n_hidden, len(alldata.labels_uniq))
print(rnn)

def label_from_output(output, output_labels):
    top_n, top_i = output.topk(1)
    label_i = top_i[0].item()
    return output_labels[label_i], label_i

input = lineToTensor('Albert')
output = rnn(input) #this is equivalent to ``output = rnn.forward(input)``
print(output)
print(label_from_output(output, alldata.labels_uniq))

def train(rnn, training_data, n_epoch = 10, n_batch_size = 64, report_every = 50, learning_rate = 0.2, criterion = nn.NLLLoss()):
    # learn on batches of data
    # track losses
    current_loss = 0
    all_losses = []
    rnn.train()

    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

    start = time.time()
    print(f"training on data set with n = {len(training_data)}")

    for iter in range(1, n_epoch + 1):
        rnn.zero_grad() # cear gradients

        # create minibatches
        batches = list(range(len(training_data)))
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches) // n_batch_size)

        for idx, batch in enumerate(batches):
            batch_loss = 0
            for i in batch:
                (label_tensor, text_tensor, label, text) = training_data[i]
                output = rnn.forward(text_tensor)
                loss = criterion(output, label_tensor)
                batch_loss += loss

            # optimize parameters
            batch_loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            current_loss += batch_loss.item() / len(batches)

        all_losses.append(current_loss / len(batches))
        if iter % report_every == 0:
            print(f"{iter} ({iter / n_epoch:.0%}): \t average batch loss = {all_losses[-1]}")
        current_loss = 0

    return all_losses

start = time.time()
all_losses = train(rnn, train_set, n_epoch=27, learning_rate=0.15, report_every=5)
end = time.time()

print(f"training took {end-start}s")

PATH = './rnn_classif.pth'
torch.save(rnn.state_dict(), PATH)
print("Model saved")

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)
plt.show()

def evaluate(rnn, testing_data, classes): # create a confusion matrix
    confusion = torch.zeros(len(classes), len(classes))

    rnn.eval()
    with torch.no_grad(): 
        for i in range(len(testing_data)):
            (label_tensor, text_tensor, label, text) = testing_data[i]
            output = rnn(text_tensor)
            guess, guess_i = label_from_output(output, classes)
            label_i = classes.index(label)
            confusion[label_i][guess_i] += 1
        
        # normalize the matrix
        for i in range(len(classes)):
            denom = confusion[i].sum()
            if denom > 0:
                confusion[i] = confusion[i] / denom

        # Set up plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(confusion.cpu().numpy()) #numpy uses cpu here so we need to use a cpu version
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
        ax.set_yticks(np.arange(len(classes)), labels=classes)

        # Force label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        # sphinx_gallery_thumbnail_number = 2
        plt.show()

evaluate(rnn, test_set, classes=alldata.labels_uniq)
