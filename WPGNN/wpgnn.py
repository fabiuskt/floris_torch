from re import X
import torch
from meta import MetaLayer
import modules as mod
import numpy as np 
from torch_geometric.loader import DataLoader
from time import time
import utils
from torch_geometric.nn import SumAggregation
import h5py
import os
import floris_torch
import matplotlib.pyplot as plt

import floris

from floris.tools import FlorisInterface
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR

umin = torch.tensor([-25.]) # min allowed yaw angle (degrees)
umax = torch.tensor([25.]) # max allowed yaw angle (degrees)
u_penalty = 0.1
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
                                                      n_layers=5,
                                                      output_activation='leaky_relu',
                                                      layer_index=i)
            #add Module to children list such that it will be recursivly found in self.apply
            self.add_module(name='meta{0:03d}'.format(i) ,module=newMetaLayer)
            self.graph_layers.append(newMetaLayer)
        
        dim_in = [self.eN_in, self.nN_in, self.gN_in] if self.n_layers == 1 else graph_size[-2]
        newMetaLayer = self.graph_layer(dim_in, graph_size[-1],
                                                  n_layers=10,
                                                  hidden_dim=10,
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


    def compute_loss_floris(self, yaws, wss, wds, batch, reporting=False):
        relu = ReLU_custom.apply
        clipped_yaws = relu(yaws*umax-umin) + umin
        #clipped_yaws = yaws*umax-umin 

        relu2 = ReLU_custom.apply
        clipped_yaws = -relu2(-clipped_yaws + umax) + umax
        
        indices = batch.x_ptr.tolist()
        split = []
        for i in range(len(indices)-1):
            split.append(indices[i+1] - indices[i])

        yaws = torch.split(yaws, split)
        clipped_yaws = torch.split(clipped_yaws, split)

        x_unnormed = utils.unnorm_coordinates(batch.x)
        x_coord_all = torch.split(x_unnormed[:,0], split)
        y_coord_all = torch.split(x_unnormed[:,1], split)

        opt_loss = 0.
        u_viol_loss = 0.
        power_loss = 0.
        power_cost = 0.
        for i in range(len(yaws)):
            set_yaw = torch.tensor([np.nan] * split[i], requires_grad=True)
            x_coord = x_coord_all[i].reshape(split[i])
            x_coord.requires_grad = True
            y_coord = y_coord_all[i].reshape(split[i])
            y_coord.requires_grad = True
            z_coord = torch.tensor([90.0] * split[i], requires_grad=True)
            flow_field_u, x_coord_rotated, yaw_angle = floris_torch.get_field_rotor(wss[i], wds[i], clipped_yaws[i].reshape(split[i]), set_yaw, x_coord, y_coord, z_coord)
            p = floris_torch.get_power(flow_field_u, x_coord_rotated, yaw_angle)
            #power_cost = -torch.log(1.0 + torch.sum(p))
            power_cost = -torch.sum(p)

            power_cost = power_cost.to(torch.float32)

            # cost from soft constraints (violating bounds on yaw angle)
            u_viol_lower = torch.nn.functional.relu(umin - yaws[i]*umax)
            u_viol_upper = torch.nn.functional.relu(yaws[i]*umax - umax)
            u_viol_cost = u_penalty * torch.sum(torch.sqrt(u_viol_lower.pow(2) + u_viol_upper.pow(2) +1e-16))

            # total cost
            total_cost = u_viol_cost + power_cost

            # sum over data in batch
            opt_loss += total_cost
            u_viol_loss += u_viol_cost
            power_loss += power_cost

        opt_loss /= len(yaws)
        u_viol_loss /= len(yaws)
        power_loss /= len(yaws)

        if reporting:
            return opt_loss, power_loss, u_viol_loss 
        else:
            return opt_loss



    def train_step(self, batch, f, u):
        self.optimizer.zero_grad()
        x_out, edge_attr_out, u_out  = self.forward(batch.x, batch.edge_index, batch.edge_attr, u[batch.y], batch)
        #loss = self.compute_loss(x_out, u_out, f, batch)
        loss = -torch.sum(x_out) 
        loss.backward()
        self.optimizer.step()
        return loss



    def train_step_DPC(self, batch, u, wss, wds):
        self.optimizer.zero_grad()
        self.train()
        x_out, edge_attr_out, u_out  = self.forward(batch.x, batch.edge_index, batch.edge_attr, u[batch.y], batch)
        loss = self.compute_loss_floris(x_out, wss[batch.y], wds[batch.y], batch)
        #mae_loss = torch.nn.L1Loss()
        #loss = mae_loss(x_out, torch.tensor([[1.0],[0.0],[1.0],[0.0]]))
        #loss = -torch.sum(loss)
        loss.backward()
        #print_weights(self)
        self.optimizer.step()
        return loss
        

    def fit(self, train_data, test_data=None, batch_size=2, learning_rate=1e-3, decay_rate=0.99,
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
        #dataset, f, u = utils.tfData_to_pygData(train_data)
        dataset, u, f, r =  train_data 
        loader = DataLoader(dataset, batch_size=batch_size,follow_batch=['x','edge_attr'], shuffle=False)
        #loader = DataLoader(dataset, batch_size=batch_size,follow_batch=['x','edge_attr'], shuffle=True)

        # Start training process
        iters = 0
        for epoch in range(1, epochs+1):

            start_time = time()
            print('Beginning epoch {}...'.format(epoch))

            batch_loss = 0
            batch_iterator = iter(loader)
            for idx_batch in batch_iterator:


                self.train_step(idx_batch, f, u)

                if (print_every > 0) and ((iters % print_every) == 0):
                    x_out, edge_attr_out, u_out  = self.forward(idx_batch.x, idx_batch.edge_index, idx_batch.edge_attr, u[idx_batch.y], idx_batch)
                    l = self.compute_loss(x_out, u_out, f, idx_batch, reporting=True)
                    print('Total batch loss = {:.6f}'.format(l[0]))
                    print('Turbine power loss = {:.6f}, '.format(l[1]), 'turbine speed loss = {:.6f}'.format(l[2]))
                    print('Plant power loss   = {:.6f}, '.format(l[3]), 'plant cabling loss = {:.6f}'.format(l[4]))
                    print('')

                iters += 1
            
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
            l = self.compute_dataset_loss(train_data, batch_size=batch_size, reporting=True)
            print('Epochs {} Complete'.format(epoch))
            print('Training Loss = {:.6f}, '.format(l[0]))
            print('Turbine power loss = {:.6f}, '.format(l[1]), 'turbine speed loss = {:.6f}'.format(l[2]))
            print('Plant power loss   = {:.6f}, '.format(l[3]), 'plant cabling loss = {:.6f}'.format(l[4]))
            
            if test_data is not None:
                l = self.compute_dataset_loss(test_data, batch_size=batch_size, reporting=True)
                print('Testing Loss = {:.6f}, '.format(l[0]))
                print('Turbine power loss = {:.6f}, '.format(l[1]), 'turbine speed loss = {:.6f}'.format(l[2]))
                print('Plant power loss   = {:.6f}, '.format(l[3]), 'plant cabling loss = {:.6f}'.format(l[4]))
            

            for g in self.optimizer.param_groups:
                g['lr'] *= decay_rate

            print('Time to complete: {0:02f}\n'.format(time() - start_time), flush=True)


    
    def fitDPC(self, train_data, test_data=None, batch_size=2, learning_rate=1e-3, decay_rate=0.99,
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
        #loader = DataLoader(dataset, batch_size=batch_size,follow_batch=['x','edge_attr'], shuffle=False)
        loader = DataLoader(dataset, batch_size=batch_size,follow_batch=['x','edge_attr'], shuffle=False)


        batch_iterator = iter(loader)
        batch = next(batch_iterator)

        self.eval()
        x_out, edge_attr_out, u_out  = self.forward(batch.x, batch.edge_index, batch.edge_attr, u[batch.y], batch)
        
        l = self.compute_loss_floris(x_out, wss[batch.y], wds[batch.y], batch, reporting=True)
        self.train()

        print('WPGNN prediction')
        print('yaws ', x_out.tolist())
        print('Total batch loss = {:.6f}'.format(l[0]))
        print('Turbine power loss = {:.6f}, '.format(l[1]))
        print('Yaw violation loss   = {:.6f}, '.format(l[2]))
        print('')

        lss = [[] for i in range(int(len(dataset)/batch_size))]
        ltots = []

        # Start training process
        iters = 0


        previous_loss = torch.inf


        for epoch in range(1, epochs+1):
            start_time = time()
            l_tot = 0
            print('Beginning epoch {}...'.format(epoch))

            if epoch <= 10:
                for g in self.optimizer.param_groups:
                    factor = pow(0.1, (10-epoch)/2)
                    g['lr'] = learning_rate * factor

            batch_iterator = iter(loader)
            for co, idx_batch in enumerate(batch_iterator):
                loss = self.train_step_DPC(idx_batch, u, wss, wds)
                t = loss.detach().item()
                lss[co].append(t)

                if (print_every > 0) and ((iters % print_every) == 0):
                    self.eval()
                    x_out, edge_attr_out, u_out = self.forward(idx_batch.x, idx_batch.edge_index, idx_batch.edge_attr, u[idx_batch.y], idx_batch)
                    l = self.compute_loss_floris(x_out, wss[idx_batch.y], wds[idx_batch.y], idx_batch, reporting=True)
                    self.train()
                    print('yaws ', x_out.tolist())
                    print('Total batch loss = {:.6f}'.format(l[0]))
                    print('Turbine power loss = {:.6f}, '.format(l[1]))
                    print('Yaw violation loss   = {:.6f}, '.format(l[2]))
                    print('')

                
            
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
                self.eval()
                l = self.compute_dataset_loss_DPC(train_data, batch_size=batch_size, reporting=True)
                self.train()
                print('Total dataset loss = {:.6f}'.format(l[0]))
                print('Turbine power loss = {:.6f}, '.format(l[1]))
                print('Yaw violation loss   = {:.6f}, '.format(l[2]))
                print('')
                t = l[0].detach().item()
                ltots.append(t)
            else:
                ltots.append(ltots[-1])
            
            if test_data is not None:
                self.eval()
                l = self.compute_dataset_loss_DPC(test_data, batch_size=batch_size, reporting=True)
                self.train()
                print('Total batch loss = {:.6f}'.format(l[0]))
                print('Turbine power loss = {:.6f}, '.format(l[1]))
                print('Yaw violation loss   = {:.6f}, '.format(l[2]))
                print('')

            if epoch%1==0:
                for g in self.optimizer.param_groups:
                    g['lr'] *= decay_rate
            
            if (epoch+10)%300<6:
                pass
                #self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
                #for g in self.optimizer.param_groups:
                #    g['lr'] = learning_rate

            
            previous_loss = loss

            iters += 1

            print('Time to complete: {0:02f}\n'.format(time() - start_time), flush=True)
    
        print('###############################')
        
        loader = DataLoader(dataset, batch_size=1,follow_batch=['x','edge_attr'], shuffle=False)
        batch_iterator = iter(loader)

        with open('out.txt', 'w') as f:
            print('Results:', file=f)

        diffs = []
        for i, batch in enumerate(batch_iterator):

            self.eval()
            x_out, edge_attr_out, u_out  = self.forward(batch.x, batch.edge_index, batch.edge_attr, u[batch.y], batch)
            unnormed = utils.unnorm_coordinates(batch.x)
            x = unnormed[:,0].detach().numpy().tolist()
            y = unnormed[:,1].detach().numpy().tolist()

            wind_directions = wds[batch.y].detach().numpy().flatten().tolist()
            wind_speeds = wss[batch.y].detach().numpy().flatten().tolist()

            yaws = torch.as_tensor(test(x,y, wind_directions, wind_speeds),dtype=torch.float32).reshape(len(x),1)
            yaws /= 25.0

            l = self.compute_loss_floris(x_out, wss[batch.y], wds[batch.y], batch, reporting=True)
            
            with open('out.txt', 'a') as f:
                print('WPGNN prediction ', i, file=f)
                print('yaws ', x_out.tolist(), file=f)
                print('Total batch loss = {:.6f}'.format(l[0]), file=f)
                print('Turbine power loss = {:.6f}, '.format(l[1]), file=f)
                print('Yaw violation loss   = {:.6f}, '.format(l[2]), file=f)
                print('', file=f)
            
            l2 = l[1].detach().numpy()

            l = self.compute_loss_floris(yaws, wss[batch.y], wds[batch.y], batch, reporting=True)
            with open('out.txt', 'a') as f:
                print('Floris Opt, ', i, file=f)
                print('yaws ', yaws.tolist(), file=f)
                print('Total batch loss = {:.6f}'.format(l[0]), file=f)
                print('Turbine power loss = {:.6f}, '.format(l[1]), file=f)
                print('Yaw violation loss   = {:.6f}, '.format(l[2]), file=f)
                print('', file=f)

                print('ws', wss[i], file=f)
                print('wd', wds[i], file=f)

            diffs.append((l2-l[1].detach().numpy())/l2 * (-100))

        
        # power for test set from FLORIS model (baseline and optimized)
        # FLORIS_baseline = (4483. + 2948. + 1771. + 923. + 364.)/5.
        # FLORIS_opt = (4662. + 3094. + 1882. + 1003. + 406.)/5.
        with open('out.txt', 'a') as f:
            for i,date in enumerate(dataset):
                print(i, date.edge_index, file=f)


        
        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(1,2)
        for i in range(int(len(dataset)/batch_size)):
            ax[0].plot(-np.array(lss[i]), color='#{:0>3x}'.format(i*77), linestyle='-', linewidth=3, label='turbine {}'.format(i))
        ax[0].plot(-np.array(ltots), color='tab:red', linestyle='-', linewidth=3, label='total')
        #ax[0].legend()

        ax[1].plot(-np.array(lss[0]), color='#{:0>3x}'.format(0*77), linestyle='-', linewidth=3)
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

        angles = np.linspace(0,360,40)
        plt.plot(angles, diffs)
        plt.xlabel('Winddirection')
        plt.ylabel('Energie Opt - Energie Predicted')
        plt.show()





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
        #m is wpgnn
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

                        childLin.weight= torch.nn.Parameter(torch.tensor(f[weight_name].get(weight_name), dtype=torch.float32))
                        childLin.bias= torch.nn.Parameter(torch.tensor(f[bias_name].get(bias_name), dtype=torch.float32))


def save_weights_h5(m, filename):
    with h5py.File(filename, 'w') as f:
        #m is wpgnn
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
                            f.create_group(weight_name)
                            f.create_group(bias_name)
                        except:
                            pass
                        f[weight_name].create_dataset(weight_name, data=childLin.weight.detach().numpy())
                        f[bias_name].create_dataset(bias_name, data=childLin.bias.detach().numpy())


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

