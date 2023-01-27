import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


BATCH_SIZE=200


def getActivation(name,activations_dict):
  # the hook signature
  def hook(model, input, output):
    n_samples = output.shape[0]
    activations_dict[name].append(output.detach().reshape(n_samples,-1))
  return hook


def load_mnist_test_data(batch_size=BATCH_SIZE):
    """Get MNIST test dataset"""
    test_dataset = datasets.MNIST('mnist-data', 
        download=True,
        train=False, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]))

    test_loader = DataLoader(test_dataset,batch_size=batch_size, shuffle=False)

    return test_loader




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.act_fc1 = nn.ReLU()

        self.fc2 = nn.Linear(100, 50)
        self.act_fc2 = nn.ReLU()

        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.act_fc1(self.fc1(x))
        x = self.act_fc2(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x,dim=1) 


def get_MNIST_activations(model_weights_path):
    """Run the model on the MNIST test dataset and return activations in each layer """

    net = Net()
    net.load_state_dict(torch.load(model_weights_path))
    net.eval()

    # Create dictionary containing empty lists to store activation from each layer
    layers = ['inputs','act_fc1','act_fc2','fc3','targets']
    activations = { layer : [] for layer in layers}

    

    # register forward hooks on the layers to store the activation values
    net.act_fc1.register_forward_hook(getActivation('act_fc1',activations))
    net.act_fc2.register_forward_hook(getActivation('act_fc2',activations))
    net.fc2.register_forward_hook(getActivation('fc3',activations))



    test_loader = load_mnist_test_data()
    for data, target in test_loader:
        
        data = data.view(-1, 28 * 28)
        
        net_out = net(data)
        # Store input and output to activation dictionary
        n_samples = net_out.shape[0]
        activations['inputs'].append(data.reshape(n_samples,-1))
        activations['targets'].append(target.reshape(n_samples,-1))


    # For each layer, convert list to numpy array
    activations = {layer : torch.cat(act,dim=0).numpy() for layer, act in activations.items()}
    return activations