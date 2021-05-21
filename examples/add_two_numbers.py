    """Trying to replicate examples, I copied the function task_add_two_numbers_after_delimiter from 
    https://github.com/philipperemy/keras-attention-mechanism/blob/master/examples/add_two_numbers.py

    The rest of the code is modified to use PyTorch instead of Tensorflow Keras.
    """


import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from captum.attr import LayerActivation
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from attention import Attention


def task_add_two_numbers_after_delimiter(n: int, seq_length: int, delimiter: float = 0.0,
                                         index_1: int = None, index_2: int = None) -> (np.array, np.array):
    """
    Task: Add the two numbers that come right after the delimiter.
    x = [1, 2, 3, 0, 4, 5, 6, 0, 7, 8]. Result is y = 4 + 7 = 11.
    @param n: number of samples in (x, y).
    @param seq_length: length of the sequence of x.
    @param delimiter: value of the delimiter. Default is 0.0
    @param index_1: index of the number that comes after the first 0.
    @param index_2: index of the number that comes after the second 0.
    @return: returns two numpy.array x and y of shape (n, seq_length, 1) and (n, 1).
    """
    x = np.random.uniform(0, 1, (n, seq_length))
    y = np.zeros(shape=(n, 1))
    for i in range(len(x)):
        if index_1 is None and index_2 is None:
            a, b = np.random.choice(range(1, len(x[i])), size=2, replace=False)
        else:
            a, b = index_1, index_2
        y[i] = 0.5 * x[i, a:a + 1] + 0.5 * x[i, b:b + 1]
        x[i, a - 1:a] = delimiter
        x[i, b - 1:b] = delimiter
    x = np.expand_dims(x, axis=-1)
    return x, y


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 100, batch_first=True)
        self.attn = Attention(100)
        self.drop = nn.Dropout(0.2)
        self.lin = nn.Linear(128, 1)

    def forward(self, x):
        y, _ = self.lstm(x)
        y = self.attn(y)
        y = self.drop(y)
        y = self.lin(y)
        return y


def main():
    numpy.random.seed(7)

    # data. definition of the problem.
    seq_length = 20
    x_train, y_train = task_add_two_numbers_after_delimiter(20_000, seq_length)
    x_val, y_val = task_add_two_numbers_after_delimiter(4_000, seq_length)
    train_dataset = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_dataset = TensorDataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val))
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    # just arbitrary values. it's for visual purposes. easy to see than random values.
    test_index_1 = 4
    test_index_2 = 9
    x_test, _ = task_add_two_numbers_after_delimiter(10, seq_length, 0, test_index_1, test_index_2)
    # x_test_mask is just a mask that, if applied to x_test, would still contain the information to solve the problem.
    # we expect the attention map to look like this mask.
    x_test_mask = np.zeros_like(x_test[..., 0])
    x_test_mask[:, test_index_1:test_index_1 + 1] = 1
    x_test_mask[:, test_index_2:test_index_2 + 1] = 1
    model = Model()
    criteria = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    attn_act = LayerActivation(model, model.attn.softmax_layer)
    # attn_act = LayerConductance(model, model.attn)

    output_dir = 'task_add_two_numbers'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 200

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        running_loss = 0.0
        counter = 0
        model.train()
        tk0 = tqdm(train_dataloader, total=int(len(train_dataloader)))
        for i, data in enumerate(tk0):
            x, y = data
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criteria(y_hat, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            counter += 1
            tk0.set_postfix(loss=(running_loss / (counter * train_dataloader.batch_size)))
            # if i > 20:
            #     break
        epoch_loss = running_loss / len(train_dataloader)
        print('Training Loss: {:.4f}'.format(epoch_loss))
        running_loss = 0.0
        model.eval()
        tk1 = tqdm(val_dataloader, total=int(len(val_dataloader)))
        counter = 0
        with torch.no_grad():
            for i, data in enumerate(tk1):
                x, y = data
                y_hat = model(x)
                loss = criteria(y_hat, y)
                running_loss += loss.item() * x.size(0)
                counter += 1
                tk1.set_postfix(loss=(running_loss / (counter * val_dataloader.batch_size)))
                # if i > 20:
                #     break
            epoch_loss = running_loss / len(val_dataloader)
            print('Val Loss: {:.4f}'.format(epoch_loss))

            attention_map = attn_act.attribute(torch.FloatTensor(x_test))
            plt.imshow(np.concatenate([attention_map, x_test_mask]), cmap='hot')
            iteration_no = str(epoch).zfill(3)
            plt.axis('off')
            plt.title(f'Iteration {iteration_no} / {num_epochs}')
            plt.savefig(f'{output_dir}/epoch_{iteration_no}.png')
            plt.close()
            plt.clf()


if __name__ == '__main__':
    main()
