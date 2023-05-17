from imp import load_compiled
from json import load
from wpgnn_batched_ws_wd import WPGNN
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
#                [ 5,  1,  2]]

graph_size =    [[ 5,  1,  2]]

scale_factors = {'x_globals': torch.as_tensor([[0., 25.], [0., 25.], [0.09, 0.03]], dtype=torch.float32),
                   'x_nodes': torch.as_tensor([[0., 75000.], [0., 85000.]], dtype=torch.float32),
                   'x_edges': torch.as_tensor([[-100000., 100000.], [0., 75000.]], dtype=torch.float32),
                   'f_globals': torch.as_tensor([[0., 500000000.], [0., 100000.]], dtype=torch.float32),
                   'f_nodes': torch.as_tensor([[0., 5000000.], [0.,25.]], dtype=torch.float32),
                   'f_edges': torch.as_tensor([[0., 0.]], dtype=torch.float32)}


save_model_path = 'FlorisDPC/floris_torch/example_training_batch_norm'
load_path = save_model_path+'/{0:05d}'.format(100)+'/wpgnn.h5'
model = WPGNN(eN=eN, nN=nN, gN=gN,graph_size=graph_size, model_path=load_path)

numWs, numWd, numLayout = 2,20,1
wsBatchSize, wdBatchSize, layoutBatchSize = 2, 20, 1

[[dataset_train, u_train, wss_train, wds_train],[dataset_test, u_test, wss_test, wds_test]] = utils.create_PyG_dataset(numLayout, numWs, numWd)
normed_data_train, normed_u_train = utils.norm_data_pyg(xx=dataset_train, uu=u_train, scale_factors=scale_factors)
normed_data_test, normed_u_test = utils.norm_data_pyg(xx=dataset_test, uu=u_test, scale_factors=scale_factors)
#dataset_train = [dataset_train, u_train, wss_train, wds_train]
#dataset_test = [dataset_test, u_test, wss_test, wds_test]
dataset_train = [normed_data_train, u_train, wss_train, wds_train]
dataset_test = [normed_data_test, u_test, wss_test, wds_test]
model.fitDPC(dataset_train, numWs, numWd, numLayout, 
          wsBatchSize, wdBatchSize, layoutBatchSize, test_data=dataset_test, learning_rate=0.001,
          epochs=300, decay_rate=1.0,print_every=10, save_every=100,
          save_model_path=save_model_path)
