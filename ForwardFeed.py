from torch import nn
from torch.nn import functional as F

class ForwardFeed(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout=0.1):
        super(ForwardFeed, self).__init__()
        self.fc1 = nn.Linear(embed_size, hidden_size) # The first linear layer
        self.fc2 = nn.Linear(hidden_size, embed_size) # The seconde linear layer
        self.dropout = nn.Dropout(dropout) # the dropout layer
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x)))) # The output of the forward (a simple classical feed forward network)