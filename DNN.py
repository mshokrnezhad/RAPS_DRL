import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torch as T
import os
import numpy as np


class DNN(nn.Module):
    def __init__(self, LR, NUM_ACTIONS, INPUT_SHAPE, NAME, CHECKPOINT_DIR, H1=128, H2=32):
        super().__init__()
        self.CHECKPOINT_DIR = CHECKPOINT_DIR
        self.CHECKPOINT_FILE = os.path.join(self.CHECKPOINT_DIR, NAME)
        self.fc1 = nn.Linear(INPUT_SHAPE, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, NUM_ACTIONS)
        self.optimizer = opt.Adam(self.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

    def forward(self, state):  # forward propagation includes defining layers
        out1 = F.relu(self.fc1(state))
        out2 = F.relu(self.fc2(out1))
        out3 = self.fc3(out2)

        return out3

    def save_checkpoint(self):
        print(f'Saving {self.CHECKPOINT_FILE}...')
        T.save(self.state_dict(), self.CHECKPOINT_FILE)

    def load_checkpoint(self):
        print('Loading checkpoint ...')
        self.load_state_dict(T.load(self.CHECKPOINT_FILE))
