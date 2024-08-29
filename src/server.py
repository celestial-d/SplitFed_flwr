import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable,Dict,Optional,Tuple
import torchvision.transforms as transforms
import torchvision.datasets
from torchvision.datasets import CIFAR10
from torch.utils.data import TensorDataset,DataLoader
import numpy as np
import os
import sys
from collections import OrderedDict
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('model'))))
from model import *
import pickle
import json

#configuration
with open("./config_server.json",'rb')as f:
  conf = json.load(f)

#parameter setting
learning_rate = conf['learning_rate']
batch_size = conf['batch_size']
num_classes = conf['num_classes']
epochs = conf['epochs']
num_clients = conf['num_clients']
round = conf['round'] # Round is calculated as epoch * round and passes the global model for each round.
total_itter = epochs*round
num_client_layer = conf['num_client_layer']

# ######################################################
# transfer data loader
# ######################################################
def loadData(transformed_inputs,transformed_labels):
  #data loader
  '''load data from server'''
  #load code, set loader by transformed data
  transformed_input_arr = np.concatenate(transformed_inputs)
  transformed_label_arr = np.concatenate(transformed_labels)
  
  transformed_input_tensor = torch.Tensor(transformed_input_arr)
  transformed_label_tensor = torch.Tensor(transformed_label_arr).type(torch.LongTensor)
  
  transformed_data = TensorDataset(transformed_input_tensor,transformed_label_tensor)
  transformed_loader = DataLoader(transformed_data,batch_size=32,shuffle=False)
  return transformed_loader


# ######################################################
# train
# ######################################################

def train(net, trainloader):
  """Train the frozen network on the training set, only one time"""
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(net.parameters(),lr = learning_rate,momentum=0.9)
  net.train() # frozen part이기 때문에 eval mode에서 진행
  num_examples = 0
  for images, labels in trainloader:
      optimizer.zero_grad()
      loss = criterion(net(images), labels)
      loss.backward()
      optimizer.step()
      num_examples += labels.size(0)

# client Model parameters
def clientNet(net):
  net = splitParameter(net)
  return [val.cpu().numpy() for _, val in net.items()]

# ######################################################
# weight separation
# ######################################################

def splitParameter(net):
  client_layer = {}
  for index,(name, val) in enumerate(net.state_dict().items()):
    if index>= num_client_layer*2:
      break
    client_layer[name] = val
  return client_layer

# Data loading loads and deletes data in the folder (sequential)
# Modify to deploy model only in epoch*rnd cycle


def main():
    # load model to gpu
    server_net = Server_Net()
    
    
    # configuration
    def get_on_fit_config_fn() -> Callable[[int],Dict[str,str]]:
      def fit_config(rnd:int) -> Dict[str,str]:
        config = {
          'rnd':rnd,
          'epoch':epochs
        }
        
        # Send random init weight in the first round
        # Transmit trained weight from the second round
        if rnd ==1:
          print('@round 1 : config for initialization')
          return config, None
        else:
          # load data
          print('@round '+str(rnd)+' : train data')
          return config, clientNet(server_net) # strategy로 전달
      return fit_config
    
   #Learning progress
    def get_eval_fn():
      print('@get_eval')
      def evaluate(t_data,t_label):
          print('@train_data')
          loaders = loadData(t_data,t_label)
          train(server_net,loaders) # config 전달전 학습
          return None
      return evaluate

    # Define strategy
    strategy = fl.server.strategy.FedDist(
        fraction_fit=1,
        fraction_eval=1,
        min_available_clients=num_clients,
        min_fit_clients =num_clients,
        min_eval_clients = 1,
        on_fit_config_fn = get_on_fit_config_fn(),
        eval_fn = get_eval_fn()
    )
    # The model must not be transmitted from the client (solved by not handing it over to the client)
    # The server must sequentially learn the data sent from the client.
    # Requires distribution of updated model from server to client
    
    
    # Start server
    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": total_itter},
        strategy=strategy,
    )
 #The final learned parameters end without being distributed (can be processed by adding one round)
    with open('parameter.pkl','wb')as f:
      pickle.dump(server_net,f)
    
if __name__ == "__main__":
  main()
