import numpy as np
import torch
import dgl
from tqdm import tqdm
from samples import SAMPLES, FIGSAMPLES
import os
from dataset_creator import create_dataset

def dataset_load(type,S,dt,alpha,save=False):
    if not os.path.exists(type):
        data = create_dataset(type,alpha_max=alpha,num_samples=S,dt=0.01,save=save)
        return data
    else:
        files = []
        previous = "blabla"
        for f in os.listdir(type):
            if os.path.isfile(f):
                current = f.split("_")[0]
                if current != previous:
                    print(current)
                    previous = current
                    files.append(current)
        collection=[]
        for name in files:
            sample= []
            x,dx,h,t = data_load(name)
            sample.append(x[:,0:S,:,:])
            sample.append(dx[:,0:S,:,:])
            sample.append(h[:,0:S,:,:])
            sample.append(t)
            collection.append(sample)
        return sample
            
                    
                
        
        
        
        
    
    







def data_load(name):
    if name == "fig8":
        x = torch.load("fig8_x.pt") 
        dx = torch.load("fig8_dx.pt")
        h = torch.load("fig8_h.pt")
        t = torch.load("fig8_t.pt") 
    if name == "yarn":
        x = torch.load("yarn_x.pt") 
        dx = torch.load("yarn_dx.pt")
        h = torch.load("yarn_h.pt")
        t = torch.load("yarn_t.pt") 
    if name == "moth":
        x = torch.load("moth_x.pt") 
        dx = torch.load("moth_dx.pt")
        h = torch.load("moth_h.pt")
        t = torch.load("moth_t.pt")
    if name == "googles":
        x = torch.load("googles_x.pt") 
        dx = torch.load("googles_dx.pt")
        h = torch.load("googles_h.pt")
        t = torch.load("googles_t.pt")
    if name == "v810":
        x = torch.load("v810_x.pt") 
        dx = torch.load("v810_dx.pt")
        h = torch.load("v810_h.pt")
        t = torch.load("v810_t.pt") 
    if name == "v1a":
        x = torch.load("v1a_x.pt") 
        dx = torch.load("v1a_dx.pt")
        h = torch.load("v1a_h.pt")
        t = torch.load("v1a_t.pt") 
    if name == "v7d":
        x = torch.load("v7d_x.pt") 
        dx = torch.load("v7d_dx.pt")
        h = torch.load("v7d_h.pt")
        t = torch.load("v7d_t.pt") 
    if name == "v11a":
        x = torch.load("v11a_x.pt") 
        dx = torch.load("v11a_dx.pt")
        h = torch.load("v11a_h.pt")
        t = torch.load("v11a_t.pt")
    if name == "v15a":
        x = torch.load("v15a_x.pt") 
        dx = torch.load("v15a_dx.pt")
        h = torch.load("v15a_h.pt")
        t = torch.load("v15a_t.pt")
    if name == "v17d":
        x = torch.load("v17d_x.pt") 
        dx = torch.load("v17d_dx.pt")
        h = torch.load("v17d_h.pt")
        t = torch.load("v17d_t.pt") 
    if name == "v17f":
        x = torch.load("v17f_x.pt") 
        dx = torch.load("v17f_dx.pt")
        h = torch.load("v17f_h.pt")
        t = torch.load("v17f_t.pt")
    return x, dx, h, t     




def dst_list(nodes,start=0):

    base = list(np.arange(start,start+nodes))
    out=[]
    for i in range(nodes):
        out = out + base
    return out

def src_list(nodes,start=0):
    out=[]
    for i in range(nodes):
        out = out +list(np.zeros((nodes),dtype=int)+start+i)
    return out



def make_graph_no_loops(nodes,start):
    src = src_list(nodes,start)
    #print(src)
    dst = dst_list(nodes,start)
    #print(dst)
    for i, pack in enumerate(zip(src,dst)):
        #print(i)
        #print(pack[0],pack[1])
        if pack[0] == pack[1]:
            src.pop(i)
            dst.pop(i)    
    return src, dst




def make_snapshots(data,H,timesize,stride=1):
    xlist=[]
    Hlist=[]
    
    
    N = data.shape[1]
    print(N)
    T = data.shape[0]
    for i in range(N):
        for j in range(T-timesize-1):
            if j % int(timesize/stride) !=0:
                print("STRIDE")
                continue
            temp = data[j:timesize+j,i,:,:]
           #print("temp {}".format(temp.shape))
            #print(i)
            tempH = H[j:timesize+j,i,:,:]
            xlist.append(temp)
            Hlist.append(tempH)
    return xlist, Hlist

def transform_dgl(src,dst,snaps,hs):
    gs = []
    for snap,h in tqdm(zip(snaps,hs)):
        g = dgl.graph((src,dst))
        g.ndata["x"] = snap[:,:,0:4].transpose(0,1)
        g.ndata["dx"] = snap[:,:,4:].transpose(0,1)
        g.ndata["H"] = torch.cat((h,h,h),dim=1).transpose(0,1)
        gs.append(g)
    return gs
def get_d_dx_H(sample):
    #try:
    gs = dgl.unbatch(sample)
    #except:
    #    gs=[sample]
    H = []
    for g in gs:
        h_raw = g.ndata["H"].transpose(0,1)
        H.append(h_raw[:,0,0:1])
    H_out = torch.cat((H),dim=-1)
    x_out = sample.ndata["x"].transpose(0,1)
    dx_out = sample.ndata["dx"].transpose(0,1)
    
    return x_out, dx_out, H_out

def get_batch_baseline(sample):
    gs = dgl.unbatch(sample)
    x = torch.Tensor(sample.ndata["x"].shape[1],len(gs),1,12)
    dx = torch.Tensor(sample.ndata["dx"].shape[1],len(gs),1,12)
    h = torch.Tensor(sample.ndata["H"].shape[1],len(gs))
    for i,g in enumerate(gs):
        x_g,dx_g,h_g = get_d_dx_H(g)
        
        h[:,i] = h_g.squeeze()
        
        x[:,i,0,0:2] = x_g[:,0,0:2]
        x[:,i,0,2:4] = x_g[:,1,0:2]
        x[:,i,0,4:6] = x_g[:,2,0:2]
        x[:,i,0,6:8] = x_g[:,0,2:4]
        x[:,i,0,8:10] = x_g[:,1,2:4]
        x[:,i,0,10:12] = x_g[:,2,2:4]
        
        dx[:,i,0,0:2] = dx_g[:,0,0:2]
        dx[:,i,0,2:4] = dx_g[:,1,0:2]
        dx[:,i,0,4:6] = dx_g[:,2,0:2]
        dx[:,i,0,6:8] = dx_g[:,0,2:4]
        dx[:,i,0,8:10] = dx_g[:,1,2:4]
        dx[:,i,0,10:12] = dx_g[:,2,2:4]
    
    return x, dx, h 
def minmax(dataset):
    T = dataset.shape[0]
    B = dataset.shape[1]
    maxim=torch.max(dataset.flatten())
    minim=torch.min(dataset.flatten())
    #print(maxim.shape)
    return (dataset - minim)/(maxim-minim), maxim, minim


def inv_minmax(dataset,min_key,max_key):
    return (dataset*(max_key-min_key))+min_key


def minimax_test(dataset):
    d , maxim, minim = minmax(dataset)
    rec = inv_minmax(d,minim,maxim)
    return torch.mean(rec.flatten() - dataset.flatten())

def loss_reader(str):
    if str == "MSE":
        return torch.nn.MSELoss()
    elif str == "HUB":
        return torch.nn.HuberLoss()
    else:
        return torch.nn.MSELoss()
    

def RKroll_for_learning(model,x0,t):
    def evaluate_model(model,x):
        h_pred = model(x)
        H_l = torch.split(h_pred,1,dim=1)
        dHdx = torch.autograd.grad(H_l,x,retain_graph=True, create_graph=True)[0] 
        dqdp_s = torch.split(dHdx,3,dim=-1)
        dx_pred = torch.cat((dqdp_s[1],-dqdp_s[0]),dim=-1)
        return dx_pred
    out_l = []
    out_dl= []
    out_l.append(x0.unsqueeze(0))
    out_dl.append(evaluate_model(model,x0).unsqueeze(0))
    #print(out_l[0].shape)
    for i in range(1,len(t)):
        dt=t[i]-t[i-1]
        K1 = evaluate_model(model,out_l[i-1].squeeze())
        K2 = evaluate_model(model,out_l[i-1].squeeze()+dt*K1/2)
        K3 = evaluate_model(model,out_l[i-1].squeeze()+dt*K2/2)
        K4 = evaluate_model(model,out_l[i-1].squeeze()+dt*K3)
        rk4=out_l[i-1].squeeze()+dt*(K1+2*K2+2*K3+K4)/6
        out_l.append(rk4.unsqueeze(0))
        out_dl.append(K1.unsqueeze(0))
        #print(out_l[i].shape)
    
    return torch.cat((out_l),dim=0), torch.cat((out_dl),dim=0)


def Euler_for_learning(model,x0,t):
    def evaluate_model(model,x):
        h_pred = model(x)
        H_l = torch.split(h_pred,1,dim=1)
        dHdx = torch.autograd.grad(H_l,x,retain_graph=True, create_graph=True)[0] 
        dqdp_s = torch.split(dHdx,3,dim=-1)
        dx_pred = torch.cat((dqdp_s[1],-dqdp_s[0]),dim=-1)
        return dx_pred
    out_l = []
    out_l.append(x0.unsqueeze(0))
    #print(out_l[0].shape)
    for i in range(1,len(t)):
        dt=t[i]-t[i-1]
        K1 = evaluate_model(model,out_l[i-1].squeeze())
        rk4=out_l[i-1].squeeze()+dt*K1
        out_l.append(rk4.unsqueeze(0))
        #print(out_l[i].shape)
    
    return torch.cat((out_l),dim=0)


    


