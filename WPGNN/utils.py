import copy
import re
import h5py
import numpy as np
import torch
from torch_geometric.data import Data
from playgen import PLayGen

np.seterr(invalid='ignore')


def load_data(data_path, normalize=False, scale_factors=None):
    x_dict_list, f_dict_list = [], []
    
    with h5py.File(data_path, 'r') as f:
        for idx in [idx for idx in f]:
            f_idx = f[idx]
            
            x_graph = {'globals': f_idx['x/globals'][()],
                         'nodes': f_idx['x/nodes'][()],
                         'edges': f_idx['x/edges'][()],
                       'senders': f_idx['x/senders'][()],
                     'receivers': f_idx['x/receivers'][()]}
            x_dict_list.append(x_graph)

            f_graph = {'globals': f_idx['f/globals'][()],
                         'nodes': f_idx['f/nodes'][()],
                         'edges': f_idx['f/edges'][()],
                       'senders': f_idx['f/senders'][()],
                     'receivers': f_idx['f/receivers'][()]}
            f_dict_list.append(f_graph)

    if normalize:
        x_dict_list, f_dict_list, _ = norm_data(xx=x_dict_list, ff=f_dict_list, scale_factors=scale_factors)
    return (x_dict_list, f_dict_list)



def tfData_to_pygData(data):
    #might need something like torch.from_numpy() for correct conversion
    x_dict_list, f_dict_list = data
    #x=[torch.as_tensor(date['nodes']) for date in x_dict_list]
    #edge_attr = [torch.as_tensor(date['edges']) for date in x_dict_list]
    #edge_index = [torch.as_tensor(np.asarray([date['senders'],date['receivers']])) for date in x_dict_list]
    dataset = [Data(x=torch.as_tensor(x_dict_list[i]['nodes'], dtype=torch.float32),edge_attr=torch.as_tensor(x_dict_list[i]['edges'], dtype=torch.float32), edge_index=torch.as_tensor(np.asarray([x_dict_list[i]['senders'],x_dict_list[i]['receivers']])), y=i) for i in range(len(x_dict_list))]

    #target values:
    #[node_target, globalt_target]
    f_nodes = [torch.as_tensor(date['nodes']) for date in f_dict_list]
    f_globals = [date['globals'] for date in f_dict_list]
    f = [f_nodes, f_globals]

    u = torch.as_tensor(np.asarray([date['globals'] for date in x_dict_list]), dtype=torch.float32)

    return dataset, f, u


def norm_data(xx=None, ff=None, scale_factors=None):
    x, f = copy.deepcopy(xx), copy.deepcopy(ff)

    N_x = len(x) if (x is not None) else 0
    N_f = len(f) if (f is not None) else 0

    assert scale_factors is not None, 'Scale factors must be provided'

    for i in range(N_x):
        x[i]['edges'] = (x[i]['edges'] - scale_factors['x_edges'][:, 0])/scale_factors['x_edges'][:, 1]
        x[i]['nodes'] = (x[i]['nodes'] - scale_factors['x_nodes'][:, 0])/scale_factors['x_nodes'][:, 1]

        x[i]['globals'] = (x[i]['globals'] - scale_factors['x_globals'][:, 0])/scale_factors['x_globals'][:, 1]

    for i in range(N_f):
        f[i]['nodes'] = (f[i]['nodes'] - scale_factors['f_nodes'][:, 0])/scale_factors['f_nodes'][:, 1]
        f[i]['globals'] = (f[i]['globals'] - scale_factors['f_globals'][:, 0])/scale_factors['f_globals'][:, 1]

    if (N_x > 0) and (N_f > 0):
        return x, f, scale_factors
    elif (N_x > 0):
        return x, scale_factors
    elif (N_f > 0):
        return f, scale_factors


def norm_data_pyg(xx=None, uu=None, ff=None, ffu=None, scale_factors=None):
    x, u, f, fu = copy.deepcopy(xx), copy.deepcopy(uu), copy.deepcopy(ff), copy.deepcopy(ffu)

    if(scale_factors==None):
        scale_factors = {'x_globals': torch.as_tensor([[0., 25.], [0., 25.], [0.09, 0.03]], dtype=torch.float32),
                   'x_nodes': torch.as_tensor([[0., 75000.], [0., 85000.]], dtype=torch.float32),
                   'x_edges': torch.as_tensor([[-100000., 100000.], [0., 75000.]], dtype=torch.float32),
                   'f_globals': torch.as_tensor([[0., 500000000.], [0., 100000.]], dtype=torch.float32),
                   'f_nodes': torch.as_tensor([[0., 5000000.], [0.,25.]], dtype=torch.float32),
                   'f_edges': torch.as_tensor([[0., 0.]], dtype=torch.float32)}

    
    if(x is not None):
        for i in range(len(x)):
            x[i].edge_attr = (x[i].edge_attr - scale_factors['x_edges'][:, 0])/scale_factors['x_edges'][:, 1]
            x[i].x = (x[i].x - scale_factors['x_nodes'][:, 0])/scale_factors['x_nodes'][:, 1]

    if(u is not None):
        for i in range(len(x)):
            u[i]= (u[i] - scale_factors['x_globals'][:, 0])/scale_factors['x_globals'][:, 1]

    if(f is not None):
        for i in range(len(f)):
            f[i].x = (x[i].x - scale_factors['f_nodes'][:, 0])/scale_factors['f_nodes'][:, 1]
            fu[i] = (fu[i] - scale_factors['f_globals'][:, 0])/scale_factors['f_globals'][:, 1]

    outputs = []
    if(x is not None):
        outputs.append(x)
    if(u is not None):
        outputs.append(u)
    if(f is not None):
        outputs.append(f)
    if(fu is not None):
        outputs.append(fu)
    return outputs

def unnorm_data(xx=None, ff=None, scale_factors=None):
    x, f = copy.deepcopy(xx), copy.deepcopy(ff)

    N_x = len(x) if (x is not None) else 0
    N_f = len(f) if (f is not None) else 0

    assert scale_factors is not None, 'Scale factors must be provided'

    for i in range(N_x):
        x[i]['edges'] = scale_factors['x_edges'][:, 1]*x[i]['edges'] + scale_factors['x_edges'][:, 0]
        x[i]['nodes'] = scale_factors['x_nodes'][:, 1]*x[i]['nodes'] + scale_factors['x_nodes'][:, 0]

        x[i]['globals'] = scale_factors['x_globals'][:, 1]*x[i]['globals'] + scale_factors['x_globals'][:, 0]

    for i in range(N_f):
        f[i]['nodes'] = scale_factors['f_nodes'][:, 1]*f[i]['nodes'] + scale_factors['f_nodes'][:, 0]
        f[i]['globals'] = scale_factors['f_globals'][:, 1]*f[i]['globals'] + scale_factors['f_globals'][:, 0]
 
    if (N_x > 0) and (N_f > 0):
        return x, f
    elif (N_x > 0):
        return x
    elif (N_f > 0):
        return f


def unnorm_coordinates(xx, scale_factors=None):
    if scale_factors is None:
        scale_factors = {'x_nodes': torch.as_tensor([[0., 75000.], [0., 85000.]], dtype=torch.float32)}

    x = copy.deepcopy(xx)

    x = scale_factors['x_nodes'][:, 1]*x + scale_factors['x_nodes'][:, 0]

    return x


def speed_to_velocity(xx):
    x = np.atleast_2d(copy.deepcopy(xx))

    ws, wd = x[:, 0], -(x[:, 1]+90)*(np.pi/180.)
    u, v = -ws*np.cos(wd), -ws*np.sin(wd)

    if x.shape[0] == 1:
        x = np.concatenate((u, v), axis=0)
    else:
        x = np.concatenate((np.atleast_2d(u), np.atleast_2d(v)), axis=0).T

    return x

def velocity_to_speed(xx):
    x = np.atleast_2d(copy.deepcopy(xx))

    u, v = x[:, 0], x[:, 1]
    ws = np.sqrt(u**2 + v**2)

    wd = 90-np.arctan(v/u)*(180./np.pi)
    wd[u<0] += 180

    # If ws = 0, then no way to recover the direction
    wd[np.isnan(wd)] = 0.

    if x.shape[0] == 1:
        x = np.concatenate((ws, wd), axis=0)
    else:
        x = np.concatenate((np.atleast_2d(ws), np.atleast_2d(wd)), axis=0).T

    return x

def identify_edges(x_loc, wind_dir, cone_deg=15):
    # Identify edges where wake interactions may play a role in power generation
    N_turbs = x_loc.shape[0]

    u, v = speed_to_velocity([10., wind_dir])
    theta = np.arctan(v/u)
    if u < 0:
        theta += np.pi
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    x_loc = x_loc@R

    x_rel = x_loc.reshape((1, N_turbs, 2)) - x_loc.reshape((N_turbs, 1, 2))

    alpha = np.arctan(x_rel[:, :, 1]/x_rel[:, :, 0])*(180./np.pi)
    alpha[np.isnan(alpha)] = 90.

    directed_edge_indices = ((abs(alpha) < cone_deg) & (x_rel[:, :, 0] <= 0)).nonzero()

    senders, receivers = directed_edge_indices[0], directed_edge_indices[1]

    edges = x_rel[senders, receivers, :]

    return edges, senders, receivers



def identify_edges_pyg(x_loc, wind_dir, cone_deg=20):
    edges, senders, receivers = identify_edges(x_loc, wind_dir, cone_deg)
    edge_attr = torch.as_tensor(edges, dtype=torch.float32)
    edge_index = torch.as_tensor(np.asarray([senders,receivers]))
    return edge_attr, edge_index



def create_PyG_dataset(size=10, numWs=10, numWd=10):
    #wind_speed, wind_direction = np.linspace(10, 20, 21), np.linspace(0, 360, 73)
    wind_speed_train, wind_direction_train = np.linspace(8, 10, numWs), np.linspace(181, 359, numWd)
    wind_speed_test, wind_direction_test = np.linspace(8, 10, 2*numWs-1), np.linspace(181, 359,2*numWd-1)
    #wind_speed_test, wind_direction_test = np.linspace(8, 10, numWs), np.linspace(181, 359,numWd)


    turb_intensity = 0.08
    dataset_test = []
    dataset_train = []
    u_test = []
    u_train = []
    count = 0
    wss_test = []
    wss_train = []
    wds_test = []
    wds_train = []
    for i in range(size):
        generator = PLayGen(N_turbs=6)
        xy = generator()


        #if i == 0:
        #   turbine_diameter = 126.0
        #   x_coord = torch.tensor([4*turbine_diameter, 0*turbine_diameter, 0*turbine_diameter] , dtype=torch.float32).reshape(3,1)
        #   y_coord = torch.tensor([0., -0.5*turbine_diameter, 0.5*turbine_diameter] , dtype=torch.float32).reshape(3,1)
        #   xy = torch.cat((x_coord, y_coord), 1).detach().numpy()


        # 2x2 layout
        if i == 0: 
            turbine_diameter = 126.0
            x_coord = torch.tensor([0., 6.*turbine_diameter, 0., 6.*turbine_diameter], dtype=torch.float32).reshape(4,1)
            y_coord = torch.tensor([0., 0., 3.*turbine_diameter, 3.*turbine_diameter], dtype=torch.float32).reshape(4,1)
            xy = torch.cat((x_coord, y_coord), 1).detach().numpy()

        # 3x3 layout
        #if i == 0: 
        #    turbine_diameter = 126.0
        #    x_coord = torch.tensor([0., 6.*turbine_diameter, 12.*turbine_diameter, 0., 6.*turbine_diameter, 12.*turbine_diameter, 0., 6.*turbine_diameter, 12.*turbine_diameter], dtype=torch.float32).reshape(9,1)
        #    y_coord = torch.tensor([0., 0., 0,  3.*turbine_diameter, 3.*turbine_diameter, 3.*turbine_diameter, 6.*turbine_diameter, 6.*turbine_diameter, 6.*turbine_diameter], dtype=torch.float32).reshape(9,1)
        #    xy = torch.cat((x_coord, y_coord), 1).detach().numpy()

         # 4x4 layout
        #if i == 0: 
        #    turbine_diameter = 126.0
        #    x_coord = torch.tensor([0., 6.*turbine_diameter, 12.*turbine_diameter, 18.*turbine_diameter, 0., 6.*turbine_diameter, 12.*turbine_diameter, 18.*turbine_diameter, 0., 6.*turbine_diameter, 12.*turbine_diameter, 18.*turbine_diameter, 0., 6.*turbine_diameter, 12.*turbine_diameter, 18.*turbine_diameter], dtype=torch.float32).reshape(16,1)
        #    y_coord = torch.tensor([0., 0., 0, 0., 3.*turbine_diameter, 3.*turbine_diameter, 3.*turbine_diameter, 3.*turbine_diameter, 6.*turbine_diameter, 6.*turbine_diameter, 6.*turbine_diameter, 6.*turbine_diameter, 9.*turbine_diameter, 9.*turbine_diameter, 9.*turbine_diameter, 9.*turbine_diameter], dtype=torch.float32).reshape(16,1)
        #    xy = torch.cat((x_coord, y_coord), 1).detach().numpy()
        
        for wd in wind_direction_train:
            for ws in wind_speed_train:
                uv = speed_to_velocity([ws, wd])
                edge_attr, edge_index = identify_edges_pyg(xy, wd)
                u_train.append([uv[0],uv[1],turb_intensity])
                #u_train.append([ws,wd,turb_intensity])

                x = torch.as_tensor(xy, dtype=torch.float32)
                dataset_train.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=count))
                wss_train.append([ws])
                wds_train.append([wd])
                count += 1
    
    u_train = torch.tensor(np.asarray(u_train), dtype=torch.float32)
    wss_train = torch.tensor(np.asarray(wss_train), dtype=torch.float32, requires_grad=False)
    wds_train = torch.tensor(np.asarray(wds_train), dtype=torch.float32, requires_grad=False)

    count=0
    for i in range(size):
        generator = PLayGen(N_turbs=16)
        xy = generator()
        if i == 0: 
            turbine_diameter = 126.0
            x_coord = torch.tensor([0., 6.*turbine_diameter, 0., 6.*turbine_diameter], dtype=torch.float32).reshape(4,1)
            y_coord = torch.tensor([0., 0., 3.*turbine_diameter, 3.*turbine_diameter], dtype=torch.float32).reshape(4,1)
            xy = torch.cat((x_coord, y_coord), 1).detach().numpy()
        for wd in wind_direction_test:
            for ws in wind_speed_test:
                    uv = speed_to_velocity([ws, wd])
                    edge_attr, edge_index = identify_edges_pyg(xy, wd)
                    u_test.append([uv[0],uv[1],turb_intensity])
                    #u_test.append([ws,wd,turb_intensity])

                    x = torch.as_tensor(xy, dtype=torch.float32)
                    dataset_test.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=count))
                    wss_test.append([ws])
                    wds_test.append([wd])
                    count += 1
    
    u_test = torch.tensor(np.asarray(u_test), dtype=torch.float32)
    wss_test = torch.tensor(np.asarray(wss_test), dtype=torch.float32, requires_grad=False)
    wds_test = torch.tensor(np.asarray(wds_test), dtype=torch.float32, requires_grad=False)
    return [[dataset_train, u_train, wss_train, wds_train],[dataset_test, u_test, wss_test, wds_test]]
