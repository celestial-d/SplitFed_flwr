# SplitFed
Hierarchical Federated Learning with model split

### environment
based on Flower, Pytorch

### abstract
The structure of the system consists of cloud server, edge server, and edge device. The edge server and the edge device share a model and update it. The model of the edge device is processed in parallel by several edge devices. The feature generated from the edge device is collected in the edge server and further learned. The data collected in the edge server is sequentially learned. The edge server has a model of the edge device, making it easy to back propagation. The edge server performs federated learning with the cloud server.

### src

source code for SplitFed

### library

customed from flwr 0.17.0

### installation

1. Download and install flwr (replace by the customized flwr)
2. install torch: conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
3. install numpy: pip install numpy==1.22.0
4. run server: python server.py
5. run client: python client.py 0
