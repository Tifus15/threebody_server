import torch
from neuralODE import *
import dgl
from data_func import *
import random
from dgl.dataloading import GraphDataLoader
from ghnn_model import *
from hnn_model import *
from tqdm import tqdm
from torchdiffeq import odeint 
import pickle
print("main")
CUDA = True
MONITORING = False
LR=1e-4
BIAS = False
SET_MAX_TRAIN_BATCHES = 50 # set -1 without
SET_MAX_TEST_BATCHES = 5 # set -1 without
src,dst = make_graph_no_loops(3,0)
t_real = torch.linspace(0,1.27,128)
g = dgl.graph((src,dst))
print(src)
print(dst)
print(g)
SOB = [1.0,1.0,1.0]
#TYPES2TRAIN = ["googles","yarn","moth","v810"]
#TYPE2EVAL = "fig8"
TYPES2TRAIN = ["v7d","v11a","v15a","v17d","v17f"]
TYPE2EVAL = "v1a"
BATCHTIME = 32
BATCHSIZE = 32
SAMPLES = 10 # RAM issues for 16GiB - 75 samples * 4 types is optimal
EPOCHS = 60
#DROPSNAPS = 0.4 # whole dataset is only 60% - make it smaller

data=[]
for type in TYPES2TRAIN:
    print("\n"+type)
    x,dx,h,t = data_load(type)
    x = x[:,0:SAMPLES,:]
    dx = dx[:,0:SAMPLES,:]
    print(h.shape)
    h = h[:,0:SAMPLES]
# making data as snapgraphs

    xdx = torch.cat((x,dx),dim=-1)
    xs,hs = make_snapshots(xdx,h[:,0:SAMPLES,:,:],BATCHTIME,4) # 4
    print(len(xs))
    del x,dx,h,t
    temp = transform_dgl(src,dst,xs,hs)
    data = data + temp
del xs, hs, temp, xdx  
  

random.shuffle(data)
ts = t_real[0:BATCHTIME]
#data_s = data[0:int((1-DROPSNAPS)*len(data))]
#del data
border = int(len(data) *0.9)

train = data[0:border]
test = data[border:]
del data
trainset = GraphDataLoader(train,batch_size=BATCHSIZE,drop_last=True,shuffle=True)
testset = GraphDataLoader(test,batch_size=BATCHSIZE,drop_last=True,shuffle=True)
del train, test
rnn_g = g
## DEFINE LOSS and OPTISdict={}
if CUDA:
    
    print(g.device)
    g = g.to('cuda:0')
    GHNN = GNN_maker_HNN(g,4,128,16,["softplus",""],type="GAT",bias=BIAS,dropout = 0.2).to(torch.device("cuda:0"))
    print(GHNN)
    ## DEFINE BASELINE
    HNN_model = HNN(12,128,["softplus"," "],bias =BIAS).to(torch.device("cuda:0"))
    print(HNN_model)
    
    loss_container=torch.zeros(EPOCHS,24).to(torch.device("cuda:0"))

else:
    GHNN = GNN_maker_HNN(g,4,128,16,["softplus",""],type="GAT",bias=BIAS,dropout = 0.2)
    print(GHNN)
    ## DEFINE BASELINE
    HNN_model = HNN(12,128,["softplus"," "],bias=BIAS)
    print(HNN_model)
    loss_container=torch.zeros(EPOCHS,24)
GRUHNN = rollout_GNN_GRU(g,4,256,16,["softplus"],bias=BIAS,type="GAT",dropout = 0.2)
print(GRUHNN)
opti_hnn = torch.optim.AdamW(HNN_model.parameters(),lr=LR)
opti_ghnn = torch.optim.AdamW(GHNN.parameters(),lr=LR)
opti_gruhnn = torch.optim.AdamW(GRUHNN.parameters(),lr=LR)

loss_fn = torch.nn.HuberLoss()

N_train = len(trainset)
N_test = len(testset)
gs=[]
for i in range(BATCHTIME*BATCHSIZE):
    src, dst = make_graph_no_loops(3,0)
    gtemp = dgl.graph((src,dst))
    #print(g.num_nodes())
    gs.append(gtemp)
#print(len(gs))
#print(g.num_nodes())

roll_g = dgl.batch(gs)
if CUDA:
    roll_g = roll_g.to("cuda:0")
#MAIN LOOP
print("max train batches: {}".format(N_train))
print("max test batches: {}".format(N_test))
for epoch in tqdm(range(EPOCHS)):
    HNN_model.train()
    GHNN.train()
    GRUHNN.train()
    for i,batch in tqdm(enumerate(trainset)):
        if i== SET_MAX_TRAIN_BATCHES:
            break
        loss_b = 0
        loss_m = 0
        loss_g = 0
        loss_br = 0
        loss_mr = 0
        loss_gr = 0
        loss_bh =0
        loss_mh =0
        loss_gh = 0
        loss_bv = 0
        loss_mv = 0
        loss_gv = 0
        opti_hnn.zero_grad()
        opti_ghnn.zero_grad()
        opti_gruhnn.zero_grad()
        x_graph,dx_graph,h_graph = get_d_dx_H(batch)
        x,dx,h = get_batch_baseline(batch)
        x0_b = x[0,:,:,:].requires_grad_()
        x0_m = x_graph[0,:,:].requires_grad_()
        x0_g = x_graph[0,:,:].requires_grad_()
        if CUDA:
            x_graph = x_graph.to(torch.device("cuda:0"))
            dx_graph = dx_graph.to(torch.device("cuda:0"))
            h_graph = h_graph.to(torch.device("cuda:0"))
            x = x.to(torch.device("cuda:0"))
            dx = dx.to(torch.device("cuda:0"))
            h = h.to(torch.device("cuda:0"))
            batch = batch.to("cuda:0")
            x0_b = x0_b.to(torch.device("cuda:0"))
            x0_m=x0_m.to(torch.device("cuda:0"))
            
        """
        print("x for g {}".format(x_graph.shape))
        print("dx for g {}".format(dx_graph.shape))
        print("h for g {}".format(h_graph.shape))
        
        print("x for base {}".format(x.shape))
        print("dx for base {}".format(dx.shape))
        print("h for base {}".format(h.shape))
        """
        #print(batch)
        GRUHNN.change_graph(batch.cpu())
        x_g,dx_g,h_g = GRUHNN(ts.cpu(),x0_g.cpu())
       
        loss_gr= loss_fn(x_g[:,:,0:2],x_graph[:,:,0:2].cpu())+loss_fn(x_g[:,:,2:4],x_graph[:,:,2:4].cpu())
        loss_gv = loss_fn(dx_g[:,:,0:2],dx_graph[:,:,0:2].cpu())+loss_fn(dx_g[:,:,2:4],dx_graph[:,:,2:4].cpu())
        loss_gh = loss_fn(h_g.flatten(),h_graph.flatten().cpu())
        
        #graph roll + vec
        GHNN.change_graph(batch)
        
        x_m, dx_m = RKroll_for_learning(GHNN,x0_m,ts)
        #print(dx_m.shape)
        #print(dx_graph.shape)
        loss_mr= loss_fn(x_m[:,:,0:2],x_graph[:,:,0:2])+loss_fn(x_m[:,:,2:4],x_graph[:,:,2:4])
        loss_mv = loss_fn(dx_m[:,:,0:2],dx_graph[:,:,0:2])+loss_fn(dx_m[:,:,2:4],dx_graph[:,:,2:4])
        #baseline roll
        #mod =NeuralODE(HNN_model,t=ts)
        
        x_b = odeint(HNN_model,x0_b,ts,method="rk4")
        #x_b = mod.forward(x0_b)
        #print(x_b.shape)
        loss_br = loss_fn(x_b[:,:,:,0:6],x[:,:,:,0:6])+loss_fn(x_b[:,:,:,6:12],x[:,:,:,6:12])
        #baseline vec
        #x_b_roll = x_b.clone()
        x_b_roll = x_b
        dx_b = rollout_mlp_vec(HNN_model,x_b_roll)
        loss_bv = loss_fn(dx_b[:,:,:,0:6],dx[:,:,:,0:6])+loss_fn(dx_b[:,:,:,6:12],dx[:,:,:,6:12])
       
        
        
        
        GHNN.change_graph(roll_g)
        xm_h = x_m.clone()
        #graph H
        x_graph = x_m
        x_g_flat = x_graph.reshape(-1,4)
        hg_pred = GHNN(x_g_flat)
        loss_mh = loss_fn(hg_pred.flatten(),h_graph.flatten())
        #baseline H
        xb_h = x_b.clone()
        hb_pred = HNN_model.giveH(xb_h)
        #print(hb_pred.shape)
        #print(h.shape)
        loss_bh = loss_fn(hb_pred.squeeze(),h.squeeze())
        
        
        """
        model.change_graph(roll_g)
        #graph H
        x_graph = x_graph.requires_grad_()
        x_g_flat = x_graph.reshape(-1,4)
        hg_pred = model(x_g_flat)
        loss_mh = loss_fn(hg_pred.flatten(),h_graph.flatten())
        #baseline H
        hb_pred = baseline.giveH(x)
        #print(hb_pred.shape)
        #print(h.shape)
        loss_bh = loss_fn(hb_pred.squeeze(),h.squeeze())
        #graph roll
        model.change_graph(batch)
        x_m = RKroll_for_learning(model,x0_m,ts)
        loss_mr= loss_fn(x_m[:,:,0:2],x_graph[:,:,0:2])+loss_fn(x_m[:,:,2:4],x_graph[:,:,2:4])
        #baseline roll
        x_b = odeint(baseline,x0_b,ts,method="rk4")
        #print(x_b.shape)
        loss_br = loss_fn(x_b[:,:,:,0:6],x[:,:,:,0:6])+loss_fn(x_b[:,:,:,6:12],x[:,:,:,6:12])
        """
        
        loss_b = SOB[0]*loss_bh + SOB[1]*loss_br + SOB[2]*loss_bv
        loss_m = SOB[0]*loss_mh + SOB[1]*loss_mr + SOB[2]*loss_mv
        loss_g = SOB[0]*loss_gh + SOB[1]*loss_gr + SOB[2]*loss_gv
        
        
        
        
        loss_b.backward()
        loss_m.backward()
        loss_g.backward()
        if MONITORING:
            print("HNN weights")
            for namew , param in HNN_model.named_parameters():
                print(namew +"_grad",torch.mean(param.grad))
            print(" ")
        
            print("GRUHNN weights")
            for namew , param in GRUHNN.named_parameters():
                print(namew +"_grad",torch.mean(param.grad))
            print(" ")
            
            print("GHNN weights")
            for namew , param in GHNN.named_parameters():
                print(namew +"_grad",torch.mean(param.grad))
            print(" ")
            print("EPOCH {} dt: {}".format(epoch+1, ts[1]-ts[0]))
        opti_hnn.step()
        opti_ghnn.step()
        opti_gruhnn.step()
        
        loss_container[epoch,0] += loss_m.item()
        loss_container[epoch,1] += loss_mr.item()
        loss_container[epoch,2] += loss_mh.item()
        loss_container[epoch,3] += loss_mv.item()
        loss_container[epoch,4] += loss_b.item()
        loss_container[epoch,5] += loss_br.item()
        loss_container[epoch,6] += loss_bh.item()
        loss_container[epoch,7] += loss_bv.item()
        loss_container[epoch,8] += loss_g.item()
        loss_container[epoch,9] += loss_gr.item()
        loss_container[epoch,10] += loss_gh.item()
        loss_container[epoch,11] += loss_gv.item()
    
    HNN_model.eval()
    GHNN.eval()
    GRUHNN.eval()
    for i,batch in tqdm(enumerate(testset)):
        if i== SET_MAX_TEST_BATCHES:
            break
        loss_b = 0
        loss_m = 0
        loss_g = 0
        loss_br = 0
        loss_mr = 0
        loss_gr = 0
        loss_bh =0
        loss_mh =0
        loss_gh = 0
        loss_bv = 0
        loss_mv = 0
        loss_gv = 0
        opti_hnn.zero_grad()
        opti_ghnn.zero_grad()
        opti_gruhnn.zero_grad()
        x_graph,dx_graph,h_graph = get_d_dx_H(batch)
        x,dx,h = get_batch_baseline(batch)
        x0_b = x[0,:,:,:].requires_grad_()
        x0_m = x_graph[0,:,:].requires_grad_()
        x0_g = x_graph[0,:,:].requires_grad_()
        if CUDA:
            x_graph = x_graph.to(torch.device("cuda:0"))
            dx_graph = dx_graph.to(torch.device("cuda:0"))
            h_graph = h_graph.to(torch.device("cuda:0"))
            x = x.to(torch.device("cuda:0"))
            dx = dx.to(torch.device("cuda:0"))
            h = h.to(torch.device("cuda:0"))
            batch = batch.to("cuda:0")
            x0_b = x0_b.to(torch.device("cuda:0"))
            x0_m=x0_m.to(torch.device("cuda:0"))
            
        """
        print("x for g {}".format(x_graph.shape))
        print("dx for g {}".format(dx_graph.shape))
        print("h for g {}".format(h_graph.shape))
        
        print("x for base {}".format(x.shape))
        print("dx for base {}".format(dx.shape))
        print("h for base {}".format(h.shape))
        """
        #print(batch)
        GRUHNN.change_graph(batch.cpu())
        x_g,dx_g,h_g = GRUHNN(ts.cpu(),x0_g)
       
        loss_gr= loss_fn(x_g[:,:,0:2],x_graph[:,:,0:2].cpu())+loss_fn(x_g[:,:,2:4],x_graph[:,:,2:4].cpu())
        loss_gv = loss_fn(dx_g[:,:,0:2],dx_graph[:,:,0:2].cpu())+loss_fn(dx_g[:,:,2:4],dx_graph[:,:,2:4].cpu())
        loss_gh = loss_fn(h_g.flatten(),h_graph.flatten().cpu())
        
        #graph roll + vec
        GHNN.change_graph(batch)
        x_m, dx_m = RKroll_for_learning(GHNN,x0_m,ts)
        #print(dx_m.shape)
        #print(dx_graph.shape)
        loss_mr= loss_fn(x_m[:,:,0:2],x_graph[:,:,0:2])+loss_fn(x_m[:,:,2:4],x_graph[:,:,2:4])
        loss_mv = loss_fn(dx_m[:,:,0:2],dx_graph[:,:,0:2])+loss_fn(dx_m[:,:,2:4],dx_graph[:,:,2:4])
        #baseline roll
        x_b = odeint(HNN_model,x0_b,ts,method="rk4")
        #print(x_b.shape)
        loss_br = loss_fn(x_b[:,:,:,0:6],x[:,:,:,0:6])+loss_fn(x_b[:,:,:,6:12],x[:,:,:,6:12])
        #baseline vec
        dx_b = rollout_mlp_vec(HNN_model,x_b)
        loss_bv = loss_fn(dx_b[:,:,:,0:6],dx[:,:,:,0:6])+loss_fn(dx_b[:,:,:,6:12],dx[:,:,:,6:12])
       
        
        
        
        GHNN.change_graph(roll_g)
        #graph H
        x_graph = x_m
        x_g_flat = x_graph.reshape(-1,4)
        hg_pred = GHNN(x_g_flat)
        loss_mh = loss_fn(hg_pred.flatten(),h_graph.flatten())
        #baseline H
        hb_pred = HNN_model.giveH(x_b)
        #print(hb_pred.shape)
        #print(h.shape)
        loss_bh = loss_fn(hb_pred.squeeze(),h.squeeze())
        
        
        """
        model.change_graph(roll_g)
        #graph H
        x_graph = x_graph.requires_grad_()
        x_g_flat = x_graph.reshape(-1,4)
        hg_pred = model(x_g_flat)
        loss_mh = loss_fn(hg_pred.flatten(),h_graph.flatten())
        #baseline H
        hb_pred = baseline.giveH(x)
        #print(hb_pred.shape)
        #print(h.shape)
        loss_bh = loss_fn(hb_pred.squeeze(),h.squeeze())
        #graph roll
        model.change_graph(batch)
        x_m = RKroll_for_learning(model,x0_m,ts)
        loss_mr= loss_fn(x_m[:,:,0:2],x_graph[:,:,0:2])+loss_fn(x_m[:,:,2:4],x_graph[:,:,2:4])
        #baseline roll
        x_b = odeint(baseline,x0_b,ts,method="rk4")
        #print(x_b.shape)
        loss_br = loss_fn(x_b[:,:,:,0:6],x[:,:,:,0:6])+loss_fn(x_b[:,:,:,6:12],x[:,:,:,6:12])
        """
        
        loss_b = SOB[0]*loss_bh + SOB[1]*loss_br + SOB[2]*loss_bv
        loss_m = SOB[0]*loss_mh + SOB[1]*loss_mr + SOB[2]*loss_mv
        loss_g = SOB[0]*loss_gh + SOB[1]*loss_gr + SOB[2]*loss_gv
        
        
        
        loss_container[epoch,12] += loss_m.item()
        loss_container[epoch,13] += loss_mr.item()
        loss_container[epoch,14] += loss_mh.item()
        loss_container[epoch,15] += loss_mv.item()
        loss_container[epoch,16] += loss_b.item()
        loss_container[epoch,17] += loss_br.item()
        loss_container[epoch,18] += loss_bh.item()
        loss_container[epoch,19] += loss_bv.item()
        loss_container[epoch,20] += loss_g.item()
        loss_container[epoch,21] += loss_gr.item()
        loss_container[epoch,22] += loss_gh.item()
        loss_container[epoch,23] += loss_gv.item()
    if SET_MAX_TRAIN_BATCHES == -1:    
        loss_container[epoch,0:12] /=N_train 
        loss_container[epoch,12:24] /=N_test
    else:
        loss_container[epoch,0:12] /=SET_MAX_TRAIN_BATCHES 
        loss_container[epoch,12:24] /=SET_MAX_TEST_BATCHES
    
    print("####################################################\n"+
        "EPOCH: {}\n".format(epoch+1) +
        "              train                    test\n"+
        "GHNN\n" +
        "   rollout     {}                      {}\n".format(loss_container[epoch,1],loss_container[epoch,13]) +
        "   vector      {}                      {}\n".format(loss_container[epoch,3],loss_container[epoch,15])+
        "   ham         {}                      {}\n".format(loss_container[epoch,2],loss_container[epoch,14])+
        "   summary     {}                      {}\n".format(loss_container[epoch,0],loss_container[epoch,12])+
        "\n"+
        "HNN\n"+
        "   rollout     {}                      {}\n".format(loss_container[epoch,5],loss_container[epoch,17])+
        "   vector      {}                      {}\n".format(loss_container[epoch,7],loss_container[epoch,19])+
        "   ham         {}                      {}\n".format(loss_container[epoch,6],loss_container[epoch,18])+
        "   summary     {}                      {}\n".format(loss_container[epoch,4],loss_container[epoch,16])+
        "GRUGNN\n"+
        "   rollout     {}                      {}\n".format(loss_container[epoch,9],loss_container[epoch,21])+
        "   vector      {}                      {}\n".format(loss_container[epoch,11],loss_container[epoch,23])+
        "   ham         {}                      {}\n".format(loss_container[epoch,10],loss_container[epoch,22])+
        "   summary     {}                      {}\n".format(loss_container[epoch,8],loss_container[epoch,20]))

    
    
    
torch.save(GHNN,"GHNN.pt") 
torch.save(HNN_model,"HNN.pt")
torch.save(GRUHNN,"GRUGHNN.pt")   
torch.save(loss_container,"losses.pt")  
