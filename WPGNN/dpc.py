from imp import load_compiled
from json import load
from wpgnn import WPGNN
import numpy as np
import utils
import torch
from torch_geometric.loader import DataLoader



#edges, nodes, globals input size
eN=2
nN=2
gN=3
#graph_size =    [[32, 32, 32],
#                [16, 16, 16],
#                [16, 16, 16],
#                [ 8,  8,  8],
#                [ 8,  8,  8],
#                [ 4,  1,  2]]

graph_size =    [[ 5,  1,  2]]

scale_factors = {'x_globals': torch.as_tensor([[0., 25.], [0., 25.], [0.09, 0.03]], dtype=torch.float32),
                   'x_nodes': torch.as_tensor([[0., 75000.], [0., 85000.]], dtype=torch.float32),
                   'x_edges': torch.as_tensor([[-100000., 100000.], [0., 75000.]], dtype=torch.float32),
                 'f_globals': torch.as_tensor([[0., 500000000.], [0., 100000.]], dtype=torch.float32),
                   'f_nodes': torch.as_tensor([[0., 5000000.], [0.,25.]], dtype=torch.float32),
                   'f_edges': torch.as_tensor([[0., 0.]], dtype=torch.float32)}


save_model_path = 'example_training_batch_norm'
load_path = save_model_path+'/{0:05d}'.format(500)+'/wpgnn.h5'
model = WPGNN(eN=eN, nN=nN, gN=gN,graph_size=graph_size, model_path=load_path)

dataset, u ,wss, wds = utils.create_PyG_dataset(1)
normed_data, normed_u = utils.norm_data_pyg(xx=dataset, uu=u, scale_factors=scale_factors)
dataset = [normed_data, normed_u, wss, wds]
model.fitDPC(dataset, learning_rate=0.00001  , batch_size=40  ,
          epochs=500, decay_rate=1.0,print_every=20, save_every=100,
          save_model_path=save_model_path)
