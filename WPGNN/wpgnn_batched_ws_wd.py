from re import X
import torch
from meta import MetaLayer
import modules as mod
import numpy as np 
from torch_geometric.loader import DataLoader
from torch_geometric.nn.norm.batch_norm import BatchNorm
from time import time
import utils
from torch_geometric.nn import SumAggregation
import h5py
import os
import floris_torch_batched
import matplotlib.pyplot as plt
import random
from torch.nn import BatchNorm1d
from utils import *

import floris

from floris.tools import FlorisInterface
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR

umin = torch.tensor([-25.]) # min allowed yaw angle (degrees)
umax = torch.tensor([25.]) # max allowed yaw angle (degrees)
u_penalty = 100.
#u_penalty = 0.001


class WPGNN(torch.nn.Module):
    '''
        Parameters:
            eN_in, eN_out   - number of input/output edge features
            nN_in, nN_out   - number of input/output node features
            gN_in, gN_out   - number of input/output graph features
            n_layers        - number of graph layers in the network
            graph_layers    - list of graph layers
            model_path      - location of a saved model, if None then use randomly initialized weights
            scale_factors   - list of scaling factors used to normalize data
            optmizer        - Sonnet optimizer object that will be used for training
    '''
    def __init__(self, eN=2, nN=3, gN=3, graph_size=None,
                       scale_factors=None, model_path=None, name=None, h5=True):
        super(WPGNN, self).__init__()

        # Set model architecture
        self.eN_in,  self.nN_in,  self.gN_in  = eN, nN, gN
        if graph_size is None:
            graph_size = [[32, 32, 32],
                          [16, 16, 16],
                          [16, 16, 16],
                          [ 8,  8,  8],
                          [ 8,  8,  8],
                          [ 4,  2,  2]]
        self.n_layers = len(graph_size)
        self.eN_out, self.nN_out, self.gN_out = graph_size[-1][0], graph_size[-1][1], graph_size[-1][2]

        # Construct WPGNN model
        self.graph_layers = []
        for i in range(self.n_layers - 1):
            dim_in = [self.eN_in, self.nN_in, self.gN_in] if i == 0 else graph_size[i-1]
            newMetaLayer = self.graph_layer(dim_in, graph_size[i],
                                                      n_layers=10,
                                                      hidden_dim=360,
                                                      output_activation='leaky_relu',
                                                      layer_index=i)
            #add Module to children list such that it will be recursivly found in self.apply
            self.add_module(name='meta{0:03d}'.format(i) ,module=newMetaLayer)
            self.graph_layers.append(newMetaLayer)
        
        dim_in = [self.eN_in, self.nN_in, self.gN_in] if self.n_layers == 1 else graph_size[-2]
        newMetaLayer = self.graph_layer(dim_in, graph_size[-1],
                                                  n_layers=10,
                                                  hidden_dim=360,
                                                  output_activation='none',
                                                  layer_index=self.n_layers-1)
        self.add_module(name='meta{0:03d}'.format(self.n_layers - 1), module=newMetaLayer)
        self.graph_layers.append(newMetaLayer)

        if scale_factors is None:
            self.scale_factors = {'x_globals': np.array([[0., 25.], [0., 25.], [0.09, 0.03]]),
                                    'x_nodes': np.array([[0., 75000.], [0., 85000.], [15., 15.]]),
                                    'x_edges': np.array([[-100000., 100000.], [0., 75000.]]),
                                  'f_globals': np.array([[0., 500000000.], [0., 100000.]]),
                                    'f_nodes': np.array([[0., 5000000.], [0.,25.]]),
                                    'f_edges': np.array([[0., 0.]])}
        else:
            self.scale_factors = scale_factors

        #init weights
        #self.apply(init_weights) 

        if model_path is not None:
            if h5:
                load_weights_h5(self,model_path)
            else:
                self.custom_load_weights(model_path)
        else:
            #pass
            init_weights_zero(self)

        


        self.optimizer = torch.optim.Adam(self.parameters())


    def forward(self, x, edge_index, edge_attr, u, batch=None): #, physical_units=False): what is that doing 
        # Evaluate the WPGNN on a given input graph
        for graph_layer in self.graph_layers:

            x_out, edge_attr_out, u_out = graph_layer.forward(x, edge_index, edge_attr, u, batch)
            

            #skip connections
            tf_edge_dims = (edge_attr.shape[1] == edge_attr_out.shape[1])
            tf_node_dims = (x.shape[1] == x_out.shape[1])
            tf_global_dims = (u.shape[0] == u_out.shape[0])
            if tf_edge_dims & tf_node_dims & tf_global_dims :
                x_out = torch.add(x_out, x)
                edge_attr_out = torch.add(edge_attr_out, edge_attr)
                u_out = torch.add(u_out, u)
            
            x = x_out
            edge_attr = edge_attr_out
            u = u_out
        
        return x, edge_attr, u


    def graph_layer(self, dim_in, dim_out, n_layers=3, hidden_dim=None, output_activation='relu', layer_index=0):
        edge_inputs, edge_outputs = dim_in[0] + 2*dim_in[1] + dim_in[2], dim_out[0]
        hidden_dim = edge_outputs if hidden_dim is None else hidden_dim 
        layer_sizes = [hidden_dim for _ in range(n_layers-1)]
        eModel = mod.EdgeModel(edge_inputs, edge_outputs, layer_sizes=layer_sizes,
                                     output_activation=output_activation,
                                     name='edgeUpdate{0:02d}'.format(layer_index))
        
        #use already processed edge_attr from EdgeModel as input of NodeModel
        node_inputs, node_outputs = 2*dim_out[0] + dim_in[1] + dim_in[2], dim_out[1]
        hidden_dim = node_outputs if hidden_dim is None else hidden_dim 
        layer_sizes = [hidden_dim for _ in range(n_layers-1)]
        nModel = mod.NodeModel(node_inputs, node_outputs, layer_sizes=layer_sizes,
                                     output_activation=output_activation,
                                     name='nodeUpdate{0:02d}'.format(layer_index))

        #use already processed edge_attr, x from EdgeModel as input of GlobalModel
        global_inputs, global_outputs = dim_out[0] + dim_out[1] + dim_in[2], dim_out[2]
        hidden_dim = global_outputs if hidden_dim is None else hidden_dim 
        layer_sizes = [hidden_dim for _ in range(n_layers-1)]
        gModel = mod.GlobalModel(global_inputs, global_outputs, layer_sizes=layer_sizes,
                                         output_activation=output_activation,
                                         name='globalUpdate{0:02d}'.format(layer_index))

        return MetaLayer(eModel, nModel, gModel)
    
    def custom_save_Model(self, filename):
        torch.save(self.state_dict(), filename)



    def custom_load_weights(self, filename):
        #SIMPLIFIED
        self.load_state_dict(torch.load(filename))

    
    def compute_dataset_loss(self, data, batch_size=100, reporting=False):
        # Compute the mean loss across an entire data without updating the model parameters
        dataset, f, u = utils.tfData_to_pygData(data)
        loader = DataLoader(dataset, batch_size=batch_size,follow_batch=['x','edge_attr'], shuffle=False)

        if reporting:
            N, l_tot, l_tp_tot, l_ts_tot, l_pp_tot, l_ps_tot = 0., 0., 0., 0., 0., 0.
        else:
            N, l_tot = 0., 0.
            
        batch_iterator = iter(loader)
        for batch in batch_iterator:
            N_batch = len(batch)

            x_out, edge_attr_out, u_out  = self.forward(batch.x, batch.edge_index, batch.edge_attr, u[batch.y], batch)
            l = self.compute_loss(x_out, u_out, f, batch, reporting=True)

            if reporting:
                l_tot += l[0]*N_batch
                l_tp_tot += l[1]*N_batch
                l_ts_tot += l[2]*N_batch
                l_pp_tot += l[3]*N_batch
                l_ps_tot += l[4]*N_batch
            else:
                l_tot += l*N_batch
            N += N_batch

        if reporting:
            return l_tot/N, l_tp_tot/N, l_ts_tot/N, l_pp_tot/N, l_ps_tot/N
        else:
            return l_tot/N


    def compute_dataset_loss_DPC(self, data, batch_size=100, reporting=False):
        # Compute the mean loss across an entire data without updating the model parameters
        dataset, u, wss, wds = data
        loader = DataLoader(dataset, batch_size=batch_size,follow_batch=['x','edge_attr'], shuffle=False)

        if reporting:
            N, l_tot, l_opt_tot, l_vio_tot = 0., 0., 0., 0.
        else:
            N, l_tot = 0., 0.
            
        batch_iterator = iter(loader)
        for batch in batch_iterator:
            N_batch = len(batch)

            x_out, edge_attr_out, u_out  = self.forward(batch.x, batch.edge_index, batch.edge_attr, u[batch.y], batch)
            l = self.compute_loss_floris(x_out, wss[batch.y], wds[batch.y], batch, reporting)

            if reporting:
                l_tot += l[0]*N_batch
                l_opt_tot += l[1]*N_batch
                l_vio_tot += l[2]*N_batch

            else:
                l_tot += l*N_batch
            N += N_batch

        if reporting:
            return l_tot/N, l_opt_tot/N, l_vio_tot/N
        else:
            return l_tot/N
    
    def compute_loss(self, x_out, u_out, f, batch, reporting=False):
        f_nodes = torch.cat(tuple(np.take(f[0],batch.y)))
        f_globals = torch.as_tensor(np.take(f[1],batch.y,axis=0), dtype=torch.float32)

        # Compute the mean squared error for the target turbine- and plant-level outputs
        turbine_loss = torch.mean((x_out - f_nodes)**2, axis=0)
        plant_loss = torch.mean((u_out - f_globals)**2, axis=0)
        
        loss = torch.sum(plant_loss) + 10.*torch.sum(turbine_loss)

        if reporting:
            return loss, turbine_loss[0], turbine_loss[1], plant_loss[0], plant_loss[1]
        else:
            return loss


    def compute_loss_floris(self, yaws, wss, wds, batch, layoutBatchsize, reporting=False, rand=False, epoch=0):
        if(rand):
            s = yaws.detach().numpy().shape
            if(epoch!=0):
                yaws += torch.normal(0.,1.0*np.exp(-epoch*0.01), s)
                print(1.0*np.exp(-epoch*0.01))
        relu = ReLU_custom.apply
        clipped_yaws = relu(yaws*umax-umin) + umin
        #clipped_yaws = yaws*umax-umin 

        relu2 = ReLU_custom.apply
        clipped_yaws = -relu2(-clipped_yaws + umax) + umax

        '''clipped_yaws = ((yaws+1) % 2) - 1
        clipped_yaws[yaws==1] = 1.
        clipped_yaws[yaws==-1] = -1.
        clipped_yaws*= umax'''
        
        indices = batch.x_ptr.tolist()
        split = []
        for i in range(len(indices)-1):
            split.append(indices[i+1] - indices[i])

        yaws = torch.tensor_split(yaws, indices[1:-1])

        sizeBatch = int(wds.size()[0] * wss.size()[0])
        #sizeBatch = int(len(clipped_yaws)/layoutBatchsize)

        yaws_list = tuple(yaws[e:e + sizeBatch] for e, k in enumerate(yaws) if e % sizeBatch == 0)

        clipped_yaws = torch.tensor_split(clipped_yaws, indices[1:-1])
        
        clipped_yaws_list = tuple(clipped_yaws[e:e + sizeBatch] for e, k in enumerate(yaws) if e % sizeBatch == 0)

        total_cost, power_cost, u_viol_cost = torch.empty((len(clipped_yaws_list), wds.size()[0], wss.size()[0])), torch.empty((len(clipped_yaws_list), wds.size()[0], wss.size()[0])), torch.empty((len(clipped_yaws_list), wds.size()[0], wss.size()[0]))
        opt_loss = 0


        for i_layout in range(len(clipped_yaws_list)):
            clipped_yaws = clipped_yaws_list[i_layout]
            yaws = yaws_list[i_layout]
            clipped_yaws = torch.transpose(torch.concat(clipped_yaws,1),0,1)
            clipped_yaws = clipped_yaws.reshape((wds.size()[0],wss.size()[0],clipped_yaws.size()[1]))
            yaws = torch.transpose(torch.concat(yaws,1),0,1)
            yaws = yaws.reshape((wds.size()[0],wss.size()[0],yaws.size()[1]))


            x_unnormed = utils.unnorm_coordinates(batch.x)
            #x_unnormed = batch.x
            x_coord_all = torch.split(x_unnormed[:,0], split)[i_layout*sizeBatch:(i_layout+1)*sizeBatch]
            y_coord_all = torch.split(x_unnormed[:,1], split)[i_layout*sizeBatch:(i_layout+1)*sizeBatch]
            
            set_yaw = torch.tensor([np.nan] * split[i])
            x_coord = x_coord_all[0].reshape(split[i_layout*sizeBatch])
            y_coord = y_coord_all[0].reshape(split[i_layout*sizeBatch])
            z_coord = torch.tensor([90.0] * split[i_layout*sizeBatch])

            x_coord_rotated, y_coord_rotated, mesh_x_rotated, \
                mesh_y_rotated, mesh_z, inds_sorted = floris_torch_batched.get_turbine_mesh(wds[:,0], wss[:,0], x_coord, y_coord, z_coord)
            
            #clipped_yaws = torch.transpose(clipped_yaws,0,1)
            #order1 = torch.arange(inds_sorted.shape[0]).unsqueeze(1)
            order2 = inds_sorted.squeeze().reshape((wds.size()[0], wss.size()[0],clipped_yaws.size()[2]))

            clipped_yaws = torch.take_along_dim(clipped_yaws, order2, axis=2) # sort

            #clipped_yaws.scatter(2, order2, clipped_yaws)


            flow_field_u, yaw_angle = floris_torch_batched.get_field_rotor(wss[:,0], wds[:,0], clipped_yaws, \
                x_coord_rotated, y_coord_rotated, mesh_x_rotated, \
                mesh_y_rotated, mesh_z, inds_sorted, x_coord, y_coord)

            p = floris_torch_batched.get_power(flow_field_u, x_coord_rotated, yaw_angle)

            #flow_field_u, x_coord_rotated, yaw_angle = floris_torch.get_field_rotor(wss[i], wds[i], clipped_yaws[i].reshape(split[i]), set_yaw, x_coord, y_coord, z_coord)
            #p = floris_torch.get_power(flow_field_u, x_coord_rotated, yaw_angle)


            #power_cost = -torch.log(1.0 + torch.sum(p))
            power_cost_z = -torch.sum(p,2).to(torch.float32)

            # cost from soft constraints (violating bounds on yaw angle)
            u_viol_lower = torch.nn.functional.relu(umin - yaws*umax)
            u_viol_upper = torch.nn.functional.relu(yaws*umax - umax)
            u_viol_cost_z = u_penalty * torch.sum(torch.sqrt(u_viol_lower.pow(2) + u_viol_upper.pow(2) +1e-16),2)

            # total cost
            total_cost_z = power_cost_z + u_viol_cost_z


            total_cost[i_layout,:,:] = total_cost_z
            u_viol_cost[i_layout,:,:] = u_viol_cost_z            
            power_cost[i_layout,:,:] = power_cost_z         


            #changed to divide also by the number of tubines in a data point
            opt_loss += torch.sum(torch.sum(total_cost_z))/(yaws.shape[0]*yaws.shape[1]*yaws.shape[2])


        if reporting:
            return total_cost, power_cost, u_viol_cost 
        else:
            return opt_loss



    def train_step_DPC(self, batch, u, wss, wds, layoutBatchSize, epoch):
        self.optimizer.zero_grad()
        x_out, edge_attr_out, u_out  = self.forward(batch.x, batch.edge_index, batch.edge_attr, u, batch)
        #x_out = torch.tensor([[0.1],[0.2],[0.3],[0.4],[0.9],[ 1.0],[-0.1],[-0.2], [1.188],[-1.583],[0.1844],[0.2107],[0.1850],[0.2104],[1.18],[-1.5]])
        loss = self.compute_loss_floris(x_out, wss, wds, batch, layoutBatchSize,rand=True,epoch=epoch)
        loss.backward()
        self.optimizer.step()
        #print_weights(self)

        return loss


    
    def fitDPC(self, train_data, lengthRow, lengthColumn, lengthLayout, wsBatchSize, wdBatchSize, layoutBatchsize, test_data=None, learning_rate=1e-3, decay_rate=0.99,
                  epochs=100, print_every=10, save_every=100, save_model_path=None, h5=True):
        '''
            Parameters:
                train_data       - training data in (list of input graphs, list of output graphs) format
                test_data        - test data used to monitor training progress, same format as training data
                batch_size       - number of samples to include in each training batch
                learning_rate    - learning rate for the training optimizer
                decay_rate       - rate of decay for the learning rate
                epochs           - the total number of epochs of training to perform
                print_every      - how frequently (in training iterations) to print the batch performance
                save_every       - how frequently (in epochs) to save the model
                save_model_path  - path to directory where to save model during training
        '''
        for g in self.optimizer.param_groups:
            g['lr'] = learning_rate

        # Build data pipelines
        dataset, u, wss, wds = train_data
        dataset_test, u_test, wss_test, wds_test = test_data

        ls_train = []
        ls_test = []
        ls_train_eval = []
        ls_test_eval = []

        # Start training process
        iters = 0

        previous_loss = torch.inf

        
        #test_row = 2*lengthRow-1
        #test_column = 2*lengthColumn-1        
        test_row = 1
        test_column = 719       
        #test_row = lengthRow
        #test_column = lengthColumn






        for epoch in range(1, epochs+1):
            start_time = time()
            print('Beginning epoch {}...'.format(epoch))

            #if epoch <= 10:
            #    for g in self.optimizer.param_groups:
            #        factor = pow(0.1, (10-epoch)/2)
            #        g['lr'] = learning_rate * factor

            rws = random.sample(range(lengthRow),wsBatchSize)
            rwd = random.sample(range(lengthColumn),wdBatchSize)
            rly = random.sample(range(lengthLayout),layoutBatchsize)

            '''rws = range(wsBatchSize)
            rwd = range(wdBatchSize)
            rly = range(layoutBatchsize)'''


            batchList = []# *(lengthColumn*lengthRow)
            uList = [] #*(lengthColumn*lengthRow)

            for layout in rly:
                wsb = wss[torch.tensor(rws)]
                wdb = wds[torch.tensor(rwd)*lengthRow]

                #wsb += torch.normal(0., 0.5, wsb.detach().numpy().shape)
                wdb += torch.rand(wdb.detach().numpy().shape)

                #wsb = torch.tensor(np.random.uniform(8,10,wsBatchSize).reshape((wsBatchSize,1)))
                #wdb = torch.tensor(np.random.uniform(0,359,wdBatchSize).reshape((wdBatchSize,1)))

                #batchList =[]
                #uList = []
                turb_intensity = 0.08
                countup = 0
                for j, wd in enumerate(wdb.flatten()):
                    for i, ws in enumerate(wsb.flatten()):
                        uv = speed_to_velocity([ws, wd])
                        x = unnorm_coordinates(dataset[lengthColumn*lengthRow*layout].x)
                        edge_attr, edge_index = identify_edges_pyg(x.detach().numpy(), wd)
                        #u_train.append([ws,wd,turb_intensity])

                        #i_l = countup%(lengthColumn*lengthRow)
                        index = i + j* lengthColumn
                

                        #batchList[index] = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=countup)
                        batchList.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=countup))

                        #uList[index] = [uv[0],uv[1],turb_intensity]
                        uList.append([uv[0],uv[1],turb_intensity])
                        #u_test.append([(ws-8.0)/(10.0-8.0),torch.sin(torch.tensor(wd)),torch.cos(torch.tensor(wd))])

                        countup+=1


            batchList = norm_data_pyg(batchList)[0]
            uList = torch.as_tensor(uList, dtype=torch.float32) 

            batchList2 = []
            uList2 = []


#            for l in range(layoutBatchsize):
#                for j in range(wdBatchSize):
#                    for i in range(wsBatchSize):
#                        batchList2.append(dataset[rws[i]+(lengthRow)*rwd[j]+(lengthRow*lengthColumn)*rly[l]])
#                         uList2.append(u[rws[i]+(lengthRow)*rwd[j]+(lengthRow*lengthColumn)*rly[l],:].tolist())

            #uList = uList2
            #batchList = batchList2
            #uList = torch.as_tensor(uList, dtype=torch.float32)

            loader = DataLoader(batchList, batch_size=wdBatchSize*wsBatchSize*layoutBatchsize,follow_batch=['x','edge_attr'], shuffle=False) 

            for co, idx_batch in enumerate(loader):
                loss = self.train_step_DPC(idx_batch, uList, wsb, wdb, layoutBatchsize, epoch)
                #t = loss.detach().item()
                #lss[co].append(t)

                '''if (print_every > 0) and ((iters % print_every) == 0):
                    self.eval()
                    x_out, edge_attr_out, u_out = self.forward(idx_batch.x, idx_batch.edge_index, idx_batch.edge_attr, u[idx_batch.y], idx_batch)
                   
                    self.train()
                    l = self.compute_loss_floris(x_out, wsb, wdb, idx_batch, reporting=True)
                    print('yaws ', x_out.tolist())
                    print('Total batch loss = {:.6f}'.format(torch.sum(l[0])))
                    print('Turbine power loss = {:.6f}, '.format(torch.sum(l[1])))
                    print('Yaw violation loss   = {:.6f}, '.format(torch.sum(l[2])))
                    print('')'''

                
            
            # Save current state of the model
            if (save_model_path is not None) and ((epoch % save_every) == 0):
                model_epoch = save_model_path+'/{0:05d}'.format(epoch)
                if not os.path.exists(model_epoch):
                    os.makedirs(model_epoch)
                if h5:
                    save_weights_h5(self,'/'.join([model_epoch, 'wpgnn.h5']))
                else:
                    self.custom_save_Model('/'.join([model_epoch, 'wpgnn.pt']))


            
            # Report current training/testing performance of model
            if (print_every > 0) and ((iters % print_every) == 0):
                '''rws = range(lengthRow)
                rwd = range(int(wds.size()[0]/lengthRow))

                wsb = wss[torch.tensor(rws)*lengthRow]
                wdb = wds[torch.tensor(rwd)]

                batchList = []
                for i in range(wsBatchSize):
                    for j in range(wdBatchSize):
                        batchList.append(dataset[rws[i]*(lengthRow)+rwd[j]])'''

                loader = DataLoader(dataset, batch_size=wss.size()[0]*wds.size()[0],follow_batch=['x','edge_attr'], shuffle=False)
                loader_test = DataLoader(dataset_test, batch_size=wss_test.size()[0]*wds_test.size()[0],follow_batch=['x','edge_attr'], shuffle=False)

                l = self.eveluate(loader, lengthRow, lengthColumn, layoutBatchsize, u, wss, wds, ev=False)      
                l_test = self.eveluate(loader_test, test_row, test_column, layoutBatchsize, u_test, wss_test, wds_test, ev=False)                

                loader_eval = DataLoader(dataset, batch_size=wss.size()[0]*wds.size()[0],follow_batch=['x','edge_attr'], shuffle=False)
                loader_test_eval = DataLoader(dataset_test, batch_size=wss_test.size()[0]*wds_test.size()[0],follow_batch=['x','edge_attr'], shuffle=False)

                l_eval = self.eveluate(loader, lengthRow, lengthColumn, layoutBatchsize, u, wss, wds, ev=True)      
                l_test_eval = self.eveluate(loader_test, test_row, test_column, layoutBatchsize, u_test, wss_test, wds_test, ev=True)                
          
                
                t = torch.sum(torch.sum(l[0])).detach().item() / (lengthColumn*lengthRow*lengthLayout)
                t_test = torch.sum(torch.sum(l_test[0])).detach().item() / ((test_row)*(test_column)*lengthLayout)

                t_eval = torch.sum(torch.sum(l_eval[0])).detach().item() / (lengthColumn*lengthRow*lengthLayout)
                t_test_eval = torch.sum(torch.sum(l_test_eval[0])).detach().item() / ((test_row)*(test_column)*lengthLayout)

                #t2 = torch.sum(torch.sum(l2[0])).detach().item()

                ls_test.append(t_test)
                ls_train.append(t)                
                ls_test_eval.append(t_test_eval)
                ls_train_eval.append(t_eval)

            else:
                ls_test.append(ls_test[-1])
                ls_train.append(ls_train[-1])                
                ls_test_eval.append(ls_test_eval[-1])
                ls_train_eval.append(ls_train_eval[-1])

            
            #if test_data is not None:
            #    l = self.compute_dataset_loss_DPC(test_data, batch_size=batch_size, reporting=True)
            #    print('Total batch loss = {:.6f}'.format(l[0]))
            #    print('Turbine power loss = {:.6f}, '.format(l[1]))
            #    print('Yaw violation loss   = {:.6f}, '.format(l[2]))
            #    print('')
            global u_penalty
            if (epoch+1)%301==0:
                self.optimizer = torch.optim.Adam(self.parameters())
                for g in self.optimizer.param_groups:
                    g['lr'] = learning_rate
                #u_penalty*=1
                #for g in self.optimizer.param_groups:
                #    g['lr'] *= decay_rate
            
            #previous_loss = loss

            iters += 1

            print('Time to complete: {0:02f}\n'.format(time() - start_time), flush=True)
    
        print('###############################')
        
        '''rws = range(int(wss.size()[0]/lengthRow))
        rwd = range(lengthRow)

        wsb = wss[torch.tensor(rws)*lengthRow]
        wdb = wds[torch.tensor(rwd)]

        batchList = []
        for i in range(int(wss.size()[0]/lengthRow)):
            for j in range(lengthRow):
                batchList.append(dataset[rws[i]*(lengthRow)+rwd[j]])

        loader = DataLoader(batchList, batch_size=wdBatchSize*wsBatchSize,follow_batch=['x','edge_attr'], shuffle=False)
        '''

        ##########################from here it is just plotting##################################



        ########changed
        loader = DataLoader(dataset_test, batch_size=wss_test.size()[0]*wds_test.size()[0],follow_batch=['x','edge_attr'], shuffle=False)



        for co, idx_batch in enumerate(loader):
            self.eval()
            x_out, edge_attr_out, u_out = self.forward(idx_batch.x, idx_batch.edge_index, idx_batch.edge_attr, u_test[idx_batch.y], idx_batch)
            self.train()

            wsb = wss_test[torch.tensor(range(test_row))]
            wdb = wds_test[torch.tensor(range(test_column))*(test_row)]

            ##################################calculating the losses for evalutation and trainings mode######################################
            l_all = self.compute_loss_floris(x_out, wsb, wdb, idx_batch, layoutBatchsize, reporting=True)
            l_all_zero = self.compute_loss_floris(x_out*0, wsb, wdb, idx_batch, layoutBatchsize, reporting=True)

            indices = idx_batch.x_ptr.tolist()
            split = []
            for i in range(len(indices)-1):
                split.append(indices[i+1] - indices[i])

            x_out = torch.split(x_out, split)

            #unnormed = utils.unnorm_coordinates(idx_batch.x)
            #x = unnormed[:,0].detach().numpy().tolist()
            #y = unnormed[:,1].detach().numpy().tolist()
            
            #x_out_floris = np.zeros((5,5))

            #test(x,y,wdb.detach().numpy().tolist(), wsb.detach().numpy().tolist())

            #for i in range(int(wss.size()[0]/lengthRow)):
            #    for j in range(lengthRow):
            #        x_out_floris[i][j] = test(x,y,wdb[j].detach().numpy().item(),wsb[i].detach().numpy().item())

        with open('out.txt', 'w') as f:
            print('Results:', file=f)

        loader = DataLoader(dataset_test, batch_size=1,follow_batch=['x','edge_attr'], shuffle=False)
        batch_iterator = iter(loader)

        diffs = []

        energy_zero = []
        energy_pred = []
        energy_opti = []

        diffopt, diffgnn, index = 0,0,0

        opt_yaws = list()

        for i, batch in enumerate(batch_iterator):
            #i_l = i%(lengthColumn*lengthRow)
            #index = (i_l%lengthColumn)*lengthRow + int(i_l/lengthColumn) +int(i/(lengthColumn*lengthRow))*(lengthColumn*lengthRow)
            #index = (i_l%lengthRow)*lengthColumn + int(i_l/lengthRow) +int(i/(lengthColumn*lengthRow))*(lengthColumn*lengthRow)

            unnormed = utils.unnorm_coordinates(batch.x)
            #unnormed = batch.x
            x = unnormed[:,0].detach().numpy().tolist()
            y = unnormed[:,1].detach().numpy().tolist()

            wind_directions = wds_test[batch.y].detach().numpy().flatten().tolist()
            wind_speeds = wss_test[batch.y].detach().numpy().flatten().tolist()

            #self.eval()
            #x_out_s, edge_attr_out, u_out = self.forward(batch.x, batch.edge_index, batch.edge_attr, u[batch.y], batch)            
            #self.train()
            #x_safe = x_out[index]
            #x_out_s = x_out_s.reshape(len(x),1)
            yaws = torch.as_tensor(test(x,y, wind_directions, wind_speeds),dtype=torch.float32).reshape(len(x),1)
            opt_yaws.append(yaws.detach().numpy())
            yaws /= 25.0
            #l2 = self.compute_loss_floris(x_out_s, wss[batch.y], wds[batch.y], batch, reporting=True)
            
            with open('out.txt', 'a') as f:
                print('WPGNN prediction ', i, file=f)
                print('yaws ', x_out[i].tolist(), file=f)
                zw = l_all[0].flatten()
                print('Total batch loss = {:.6f}'.format(l_all[0].flatten()[index]), file=f)
                print('Turbine power loss = {:.6f}, '.format(l_all[1].flatten()[index]), file=f)
                print('Yaw violation loss   = {:.6f}, '.format(l_all[2].flatten()[index]), file=f)
                print('', file=f)

            l2 = l_all[1].flatten()[index].detach().numpy()
            l3 = l_all_zero[1].flatten()[index].detach().numpy()


            '''with open('out.txt', 'a') as f:
                print('WPGNN prediction ', i, file=f)
                print('yaws ', x_out_s.tolist(), file=f)
                print('Total batch loss = {:.6f}'.format(l2[0].flatten()[0]), file=f)
                print('Turbine power loss = {:.6f}, '.format(l2[1].flatten()[0]), file=f)
                print('Yaw violation loss   = {:.6f}, '.format(l2[2].flatten()[0]), file=f)
                print('', file=f)'''


            l = self.compute_loss_floris(yaws, wss_test[batch.y], wds_test[batch.y], batch, 1, reporting=True)
            with open('out.txt', 'a') as f:
                print('Floris Opt, ', i, file=f)
                print('yaws ', yaws.tolist(), file=f)
                print('Total batch loss = {:.6f}'.format(l[0].flatten()[0]), file=f)
                print('Turbine power loss = {:.6f}, '.format(l[1].flatten()[0]), file=f)
                print('Yaw violation loss   = {:.6f}, '.format(l[2].flatten()[0]), file=f)
                print('', file=f)

                print('ws',str(i), wss_test[i], file=f)
                print('wd',str(i), wds_test[i], file=f)

            l1 = l[1].flatten()[0].detach().numpy()
            diffs.append((l2-l1)/l2 * (-100))
            energy_opti.append(-l1)
            energy_pred.append(-l2)
            energy_zero.append(-l3)
            diffopt += (l3-l1)
            diffgnn += (l3-l2)

            index += 1

        
        diffs = np.array(diffs).reshape((test_column*lengthLayout, test_row), order='C')
        

        
        fig, ax = plt.subplots()
        ax.imshow(diffs)

        wsb = wss_test[torch.tensor(range(test_row))].detach().numpy().flatten().tolist()
        wdb = lengthLayout * wds_test[torch.tensor(range(test_column))*(test_row)].detach().numpy().flatten().tolist()
        
        wsb_string = []
        for i in range(test_row):
            wsb_string.append(str(round(wsb[i],1)))

        wdb_string = []
        for i in range(test_column):
            wdb_string.append(str(round(wdb[i],1)))

        ax.set_xticks(np.arange(test_row), labels=wsb_string)
        ax.set_yticks(np.arange(test_column*lengthLayout), labels=lengthLayout*wdb_string)
        # Loop over data dimensions and create text annotations.
        for i in range(lengthLayout*test_column):
            for j in range(test_row):
                text = ax.text(j, i, round(diffs[i,j],1),
                            ha="center", va="center", color="w")
        plt.show()

        
        # power for test set from FLORIS model (baseline and optimized)
        # FLORIS_baseline = (4483. + 2948. + 1771. + 923. + 364.)/5.
        # FLORIS_opt = (4662. + 3094. + 1882. + 1003. + 406.)/5.
        with open('out.txt', 'a') as f:
            print( "wpgnn increase /possible increase", diffgnn/diffopt, file=f)
            for i,date in enumerate(dataset):
                print(i, date.edge_index, file=f)


        
        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(1,2)
        ax[0].plot(-np.array(ls_test), color='tab:green', linestyle='-', linewidth=3, label='test')
        ax[0].plot(-np.array(ls_test_eval), color='tab:red', linestyle='-', linewidth=3, label='test_eval')
        ax[0].plot(-np.array(ls_train), color='tab:blue', linestyle='-', linewidth=3, label='train')
        ax[0].plot(-np.array(ls_train_eval), color='tab:grey', linestyle='-', linewidth=3, label='train_eval')
        ax[0].legend()

        ax[1].plot(-np.array(ls_train), color='#{:0>3x}'.format(0*77), linestyle='-', linewidth=3)
        #cm = plt.get_cmap('gist_rainbow')
        #for i in range(len(dataset)):
        #    ax[1].scatter(dataset[i].x[:,0].detach().numpy(), dataset[i].x[:,1].detach().numpy(), label='{}'.format(i), color=cm(1.*i/len(dataset)))
        #    for j in range(len(dataset[i].edge_index[0])):
        #        ax[1].plot(dataset[i].x[dataset[i].edge_index[0][j]][0], dataset[i].x[dataset[i].edge_index[0][j]][1], dataset[i].x[dataset[i].edge_index[1][j]][0], dataset[i].x[dataset[i].edge_index[1][j]][1], color=cm(1.*i/len(dataset)))
        # plt.axhline(FLORIS_opt, label='lookup table', color='tab:orange', linestyle='--', linewidth=3)
        # plt.axhline(FLORIS_baseline, label='baseline', color='tab:green', linestyle=':', linewidth=3)
        #ax[1].legend()
        fig.set_size_inches(12,6)
        plt.show()

        #opt_yaws = np.array(opt_yaws)
        #opt_yaws = opt_yaws[:,:,0]
        #plt.plot(opt_yaws)
        #plt.title("Optimal yaws over wind directions and wind speed")
        #plt.show()

        angles = np.linspace(0,360,25)
        plt.plot(diffs)
        plt.xlabel('Winddirection')
        plt.ylabel('Energie Opt - Energie Predicted')
        plt.show()

        plt.scatter(range(len(energy_zero)), energy_zero, marker='+', c='red', label='baseline')
        plt.scatter(range(len(energy_zero)), energy_opti, marker='$o$', c='green', label='floris optimum')
        plt.scatter(range(len(energy_zero)), energy_pred, marker='x', c='blue', label = 'wpgnn prediction')
        plt.legend()
        plt.show()





    def eveluate(self, loader, lengthRow, lengthColumn, layoutBatchsize, u, wss, wds, ev=True):
        for co, idx_batch in enumerate(loader):
            #################################here is testing in evaluation mode######################################

                        
            ##################################here is testing in training mode######################################
            with torch.no_grad():
                if ev:
                    self.eval()
                x_out, edge_attr_out, u_out = self.forward(idx_batch.x, idx_batch.edge_index, idx_batch.edge_attr, u[idx_batch.y], idx_batch)
                if ev:
                    self.train()
                #x_out2, edge_attr_out2, u_out2 = self.forward(idx_batch.x, idx_batch.edge_index, idx_batch.edge_attr, u[idx_batch.y], idx_batch)
                        
                        
                wsb = wss[torch.tensor(range(lengthRow))]
                wdb = wds[torch.tensor(range(lengthColumn))*lengthRow]

                ##################################calculating the losses for evalutation and trainings mode######################################
                l = self.compute_loss_floris(x_out, wsb, wdb, idx_batch, layoutBatchsize, reporting=True)
                #l2 = self.compute_loss_floris(x_out2, wsb, wdb, idx_batch, layoutBatchsize, reporting=True)

                print('yaws ', x_out.tolist())
                print('Total batch loss = {:.6f}'.format(torch.sum(l[0])))
                print('Turbine power loss = {:.6f}, '.format(torch.sum(l[1])))
                print('Yaw violation loss   = {:.6f}, '.format(torch.sum(l[2])))
                print('')

        return l


def test(x,y, wind_directions, wind_speeds):
    # 1. Load an input file
    fi = FlorisInterface("C:\\Users\\fabiu\\Documents\\NLRE\\Programming\\FlorisDPC\\floris_torch\\WPGNN\\gch.yaml")
    

    fi.floris.solver

    # 2. Modify the inputs with a more complex wind turbine layout
    D = 126.0  # Design the layout based on turbine diameter
    #x = [0, 0,  6 * D, 6 * D]
    #y = [0, 3 * D, 0, 3 * D]
    #wind_directions = [270.0, 280.0]
    #wind_speeds = [8.0]

    # Pass the new data to FlorisInterface
    fi.reinitialize(
        layout_x = x,
        layout_y = y,
        wind_directions=wind_directions,
        wind_speeds=wind_speeds
    )

    yaw_opt = YawOptimizationSR(
            fi=fi,
            minimum_yaw_angle=umin.tolist()[0],  # Allowable yaw angles lower bound
            maximum_yaw_angle=umax.tolist()[0],  # Allowable yaw angles upper bound
            Ny_passes=[5, 4],
            exclude_downstream_turbines=True,
            exploit_layout_symmetry=True,
    )

    df_opt = yaw_opt.optimize()
    yaw_angles = df_opt["yaw_angles_opt"][0].reshape(1,1,len(x))
    
    # 3. Calculate the velocities at each turbine for all atmospheric conditions
    # All turbines have 0 degrees yaw
    fi.calculate_wake( yaw_angles=yaw_angles )

    # 4. Get the total farm power
    turbine_powers = fi.get_turbine_powers() / 1000.0  # Given in W, so convert to kW
    farm_power_baseline = np.sum(turbine_powers, 2)  # Sum over the third dimension

    # 5. Develop the yaw control settings
    #yaw_angles = np.zeros( (2, 1, 4) )  # Construct the yaw array with dimensions for two wind directions, one wind speed, and four turbines
    #yaw_angles[0, :, 0] = 25            # At 270 degrees, yaw the first turbine 25 degrees
    #yaw_angles[0, :, 1] = 25            # At 270 degrees, yaw the second turbine 25 degrees
    #yaw_angles[1, :, 0] = 10           # At 265 degrees, yaw the first turbine -25 degrees
    #yaw_angles[1, :, 1] = 10           # At 265 degrees, yaw the second turbine -25 degrees

    print(farm_power_baseline)
    return yaw_angles


def init_weights_zero(m):
    for index, childMeta in enumerate(m.children()):
            #childMeta is a MetaLayer
            for childModel in childMeta.children():
                #childeModel is Edge/Node/Global-Model
                for name, childLin in childModel.named_children():
                    if isinstance(childModel, mod.NodeModel) and name=='linear_out' and index==0:
                        torch.nn.init.zeros_(childLin.weight)
                        torch.nn.init.zeros_(childLin.bias)

                        


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight,1.0)
        if m.bias is not None:
            torch.nn.init.uniform_(m.bias,-1.0, 1.0)


def load_weights_h5(m, filename):
    with h5py.File(filename, 'r') as f:
        #m is 
        for index, childMeta in enumerate(m.children()):
            #childMeta is a MetaLayer
            for childModel in childMeta.children():
                #childeModel is Edge/Node/Global-Model
                for name, childLin in childModel.named_children():
                    if (not isinstance(childLin, SumAggregation) and not isinstance(childLin, BatchNorm)):
                        if isinstance(childModel, mod.EdgeModel):
                            weight_name = 'wpgnn/edgeUpdate{0:02d}/'.format(index) + name + '/w:0'
                            bias_name = 'wpgnn/edgeUpdate{0:02d}/'.format(index) + name + '/b:0'
                            if isinstance(childLin, BatchNorm1d):
                                mean_name = 'wpgnn/edgeUpdate{0:02d}/'.format(index) + name + '/m:0'
                                variance_name = 'wpgnn/edgeUpdate{0:02d}/'.format(index) + name + '/v:0'        
                        if isinstance(childModel, mod.NodeModel):
                            weight_name = 'wpgnn/nodeUpdate{0:02d}/'.format(index) + name + '/w:0'
                            bias_name = 'wpgnn/nodeUpdate{0:02d}/'.format(index) + name + '/b:0'
                            if isinstance(childLin, BatchNorm1d):
                                mean_name = 'wpgnn/nodeUpdate{0:02d}/'.format(index) + name + '/m:0'
                                variance_name = 'wpgnn/nodeUpdate{0:02d}/'.format(index) + name + '/v:0'
                        if isinstance(childModel, mod.GlobalModel):
                            weight_name = 'wpgnn/globalUpdate{0:02d}/'.format(index) + name + '/w:0'
                            bias_name = 'wpgnn/globalUpdate{0:02d}/'.format(index) + name + '/b:0'
                            if isinstance(childLin, BatchNorm1d):
                                mean_name = 'wpgnn/globalUpdate{0:02d}/'.format(index) + name + '/m:0'
                                variance_name = 'wpgnn/globalUpdate{0:02d}/'.format(index) + name + '/v:0'
  
                        childLin.weight= torch.nn.Parameter(torch.tensor(f[weight_name].get(weight_name), dtype=torch.float32))
                        childLin.bias= torch.nn.Parameter(torch.tensor(f[bias_name].get(bias_name), dtype=torch.float32))
                        if isinstance(childLin, BatchNorm1d):
                            a = torch.tensor(f[variance_name].get(variance_name))
                            b = torch.tensor(f[mean_name].get(mean_name))
                            test  = childLin.get_buffer('running_mean')
                            childLin.register_buffer('running_var', torch.tensor(f[variance_name].get(variance_name), dtype=torch.float32))
                            childLin.register_buffer('running_mean', torch.tensor(f[mean_name].get(mean_name), dtype=torch.float32))





def save_weights_h5(m, filename):
    with h5py.File(filename, 'w') as f:
        #m is wpgnn
        for index, childMeta in enumerate(m.children()):
            #childMeta is a MetaLayer
            for childModel in childMeta.children():
                #childeModel is Edge/Node/Global-Model
                for name, childLin in childModel.named_children():
                    if not isinstance(childLin, SumAggregation):# and not isinstance(childLin, BatchNorm):
                        if isinstance(childModel, mod.EdgeModel):
                            weight_name = 'wpgnn/edgeUpdate{0:02d}/'.format(index) + name + '/w:0'
                            bias_name = 'wpgnn/edgeUpdate{0:02d}/'.format(index) + name + '/b:0'
                            if isinstance(childLin, BatchNorm1d):
                                mean_name = 'wpgnn/edgeUpdate{0:02d}/'.format(index) + name + '/m:0'
                                variance_name = 'wpgnn/edgeUpdate{0:02d}/'.format(index) + name + '/v:0'
                        if isinstance(childModel, mod.NodeModel):
                            weight_name = 'wpgnn/nodeUpdate{0:02d}/'.format(index) + name + '/w:0'
                            bias_name = 'wpgnn/nodeUpdate{0:02d}/'.format(index) + name + '/b:0'
                            if isinstance(childLin, BatchNorm1d):
                                mean_name = 'wpgnn/nodeUpdate{0:02d}/'.format(index) + name + '/m:0'
                                variance_name = 'wpgnn/nodeUpdate{0:02d}/'.format(index) + name + '/v:0'
 
                        if isinstance(childModel, mod.GlobalModel):
                            weight_name = 'wpgnn/globalUpdate{0:02d}/'.format(index) + name + '/w:0'
                            bias_name = 'wpgnn/globalUpdate{0:02d}/'.format(index) + name + '/b:0'
                            if isinstance(childLin, BatchNorm1d):
                                mean_name = 'wpgnn/globalUpdate{0:02d}/'.format(index) + name + '/m:0'
                                variance_name = 'wpgnn/globalUpdate{0:02d}/'.format(index) + name + '/v:0'
 
                        try:
                            f.require_group(weight_name)
                            f.require_group(bias_name)
                            if isinstance(childLin, BatchNorm1d):
                                f.require_group(mean_name)
                                f.require_group(variance_name)
                                print("done")
                        except:
                            pass
                        f[weight_name].create_dataset(weight_name, data=childLin.weight.detach().numpy())
                        f[bias_name].create_dataset(bias_name, data=childLin.bias.detach().numpy())
                        if isinstance(childLin, BatchNorm1d):
                            f[variance_name].create_dataset(variance_name, data=childLin.running_var.detach().numpy())
                            f[mean_name].create_dataset(mean_name, data=childLin.running_mean.detach().numpy())
                            a = childLin.weight
                            b = childLin.bias
                            c = childLin.running_mean
                            d = childLin.running_var




class ReLU_custom(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.save_for_backward(input)
        return torch.nn.functional.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input, None


def print_weights(m):
        for index, childMeta in enumerate(m.children()):
            #childMeta is a MetaLayer
            for childModel in childMeta.children():
                #childeModel is Edge/Node/Global-Model
                for name, childLin in childModel.named_children():
                    if not isinstance(childLin, SumAggregation):
                        if isinstance(childModel, mod.EdgeModel):
                            weight_name = 'wpgnn/edgeUpdate{0:02d}/'.format(index) + name + '/w:0'
                            bias_name = 'wpgnn/edgeUpdate{0:02d}/'.format(index) + name + '/b:0'
                        if isinstance(childModel, mod.NodeModel):
                            weight_name = 'wpgnn/nodeUpdate{0:02d}/'.format(index) + name + '/w:0'
                            bias_name = 'wpgnn/nodeUpdate{0:02d}/'.format(index) + name + '/b:0'
                        if isinstance(childModel, mod.GlobalModel):
                            weight_name = 'wpgnn/globalUpdate{0:02d}/'.format(index) + name + '/w:0'
                            bias_name = 'wpgnn/globalUpdate{0:02d}/'.format(index) + name + '/b:0'
                        
                        try:
                            print(weight_name)
                            print(childLin.weight.grad.tolist())
                            print(bias_name)
                            print(childLin.bias.grad.tolist())
                            print(' ')
                        except:
                            pass

