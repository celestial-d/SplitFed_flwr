import torch.nn as nn
import torch.nn.functional as F

# model for client and server
'''
The entire model is separated into A and B
Client has A, trains data and sends to server
The server has A and B. When it receives data from the client, it trains B. When backpropagating, both A and B participate.
Server distributes updated A to clients
'''

# server model
'''
The forward function trains data received from the client from the middle (B).
Forward propagation is performed only in the forward function, and backpropagation is equally applied as a loss function.
'''
class Server_Net(nn.Module):
    def __init__(self): # layer 정의
        super(Server_Net, self).__init__()

        # input size = 28x28 
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        
    def forward(self, x):
        #x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        #x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def test(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# client model
'''
The client function is responsible for transmitting learning results to the server.
'''
class Client_Net(nn.Module):
    def __init__(self): # layer 정의
        super(Client_Net, self).__init__()

        # input size = 28x28 
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return x
