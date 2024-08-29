import torch
import dgl
from data_func import *
import random
from dgl.dataloading import GraphDataLoader
from ghnn_model import *
from hnn_model import *
from tqdm import tqdm
from torchdiffeq import odeint 
from time import time
import yaml
import wandb



def dataset_train(config):
    WANDB = config["wandb"]
 
    CUDA = config["cuda"]
 
    MONITORING = config["monitoring"]
    LR=config["lr"]
    BIAS = config["bias"]
    SET_MAX_TRAIN_BATCHES = config["train_break"] # set -1 without
    SET_MAX_TEST_BATCHES = config["test_break"] # set -1 without
    SOB = config["sob"]
    BATCHTIME = config["tbatch"]
    BATCHSIZE = config["sbatch"]
    SAMPLES = config["samples"] # RAM issues for 16GiB - 75 samples * 4 types is optimal
    EPOCHS = config["epochs"]
    LAYER_SIZE = config["size"]
    E_SIZE = config["e_size"]
    LAYER = config["layer"]    
    #DROPSNAPS = 0.4 # whole dataset is only 60% - make it smaller
    TIME = config["max_sim_time"]
    SOB = config["sob"]
    
    data = create_dataset(config["sim_type"],alpha_max=config["alpha"],num_samples=SAMPLES,dt=config["dt"],save=config["save"])
    
    
    if config["no_loops"]:
        src,dst = make_graph_no_loops(config["nodes"],0)
    else:
        src = src_list(3)
        dst = dst_list(3)
    
    t_real = torch.linspace(0,1.27,128)
    g = dgl.graph((src,dst))
    
    
    collection=[]     
    for i in range(len(data)):
        xdx = torch.cat((data[i][0],data[i][1]),dim=-1)
        xs,hs = make_snapshots(xdx,data[i][2],BATCHTIME,1) 
        temp = transform_dgl(src,dst,xs,hs)
        collection = collection + temp
    del xs, hs, temp, xdx     
    del data
    
    random.shuffle(collection)
    ts = t_real[0:BATCHTIME]
    
    border = int(len(collection) *0.9)

    train = collection[0:border]
    test = collection[border:]    
    del collection
    
    trainset = GraphDataLoader(train,batch_size=BATCHSIZE,drop_last=True,shuffle=True)
    testset = GraphDataLoader(test,batch_size=BATCHSIZE,drop_last=True,shuffle=True)
    del train, test
    
    rnn_g = g
## DEFINE LOSS and OPTISdict={}
    if CUDA:
        g = g.to('cuda:0')
        GHNN = GNN_maker_HNN(g,4,LAYER_SIZE,E_SIZE,["softplus",""],type=LAYER,bias=BIAS,dropout = 0.3).to(torch.device("cuda:0"))
        print(GHNN)
        ## DEFINE BASELINE
        HNN_model = HNN(12,LAYER_SIZE,["softplus"," "],bias =BIAS).to(torch.device("cuda:0"))
        print(HNN_model)
        
        loss_container=torch.zeros(EPOCHS,24).to(torch.device("cuda:0"))

    else:
        GHNN = GNN_maker_HNN(g,4,LAYER_SIZE,E_SIZE,["softplus",""],type=LAYER,bias=BIAS,dropout = 0.3)
        print(GHNN)
        ## DEFINE BASELINE
        HNN_model = HNN(12,LAYER_SIZE,["softplus"," "],bias =BIAS)
        print(HNN_model)
        loss_container=torch.zeros(EPOCHS,24)
    GRUHNN = rollout_GNN_GRU(g,4,LAYER_SIZE,E_SIZE,["softplus"],bias=BIAS,type=LAYER,dropout = 0.3)
    print(GRUHNN)
    opti_hnn = torch.optim.AdamW(HNN_model.parameters(),lr=LR)
    opti_ghnn = torch.optim.AdamW(GHNN.parameters(),lr=LR)
    opti_gruhnn = torch.optim.AdamW(GRUHNN.parameters(),lr=LR)

    loss_fn = torch.nn.HuberLoss()

    N_train = len(trainset)
    N_test = len(testset)
    gs=[]
    
    if WANDB:
        metrics={"train_roll_HNN":0, "train_roll_GHN":0, "train_roll_GRUGHNN":0, 
                 "train_vec_HNN" :0, "train_vec_GHNN":0,"train_vec_GRUHNN" :0,
                 "train_h_HNN":0, "train_h_GHN":0, "train_h_GRUGHNN":0,
                 "train_summary_HNN":0, "train_summary_GHN":0, "train_summary_GRUGHNN":0,
                 "test_roll_HNN":0, "test_roll_GHN":0, "test_roll_GRUGHNN":0, 
                 "test_vec_HNN" :0, "test_vec_GHNN":0,"test_vec_GRUHNN" :0,
                 "test_h_HNN":0, "test_h_GHN":0, "test_h_GRUGHNN":0,
                 "test_summary_HNN":0, "test_summary_GHN":0, "test_summary_GRUGHNN":0}
        
    
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
    begin = time()
    flag = 0
    if WANDB:
        wandb.watch(GHNN,log='all')
        wandb.watch(HNN_model,log='all')
        wandb.watch(GRUHNN,log='all')
    for epoch in tqdm(range(EPOCHS)):
        if TIME < flag:
            break
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
        if WANDB:
                metrics["train_roll_HNN"] = loss_container[epoch,5]
                metrics["train_roll_GHNN"] = loss_container[epoch,1]
                metrics["train_roll_GRUHNN"] = loss_container[epoch,9]
                metrics["train_vec_HNN"] = loss_container[epoch,7]
                metrics["train_vec_GHNN"] = loss_container[epoch,3]
                metrics["train_vec_GRUHNN"] = loss_container[epoch,11]
                metrics["train_h_HNN"] = loss_container[epoch,6]
                metrics["train_h_GHNN"] = loss_container[epoch,2]
                metrics["train_h_GRUHNN"] = loss_container[epoch,10] 
                metrics["train_summary_HNN"] = loss_container[epoch,4]
                metrics["train_summary_GHNN"] = loss_container[epoch,0]
                metrics["train_summary_GRUHNN"] = loss_container[epoch,8]
                
                metrics["test_roll_HNN"] = loss_container[epoch,17]
                metrics["test_roll_GHNN"] = loss_container[epoch,13]
                metrics["test_roll_GRUHNN"] = loss_container[epoch,21]
                metrics["test_vec_HNN"] = loss_container[epoch,19]
                metrics["test_vec_GHNN"] = loss_container[epoch,15]
                metrics["test_vec_GRUHNN"] = loss_container[epoch,23]
                metrics["test_h_HNN"] = loss_container[epoch,18]
                metrics["test_h_GHNN"] = loss_container[epoch,14]
                metrics["test_h_GRUHNN"] = loss_container[epoch,22] 
                metrics["test_summary_HNN"] = loss_container[epoch,16]
                metrics["test_summary_GHNN"] = loss_container[epoch,12]
                metrics["test_summary_GRUHNN"] = loss_container[epoch,20]
                wandb.log(metrics)
                
        b = time()
        flag = b-begin
        print("####################################################\n"+
            "EPOCH: {} at {}\n".format(epoch+1,flag) +
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

        
    
    if not os.path.exists("dat_fig"):
        os.makedirs(config["sim_type"]+"_res")
    torch.save(GHNN,config["sim_type"]+"_res/"+"GHNN.pt") 
    torch.save(HNN_model,config["sim_type"]+"_res/"+"HNN.pt")
    torch.save(GRUHNN,config["sim_type"]+"_res/"+"GRUGHNN.pt")   
    torch.save(loss_container,config["sim_type"]+"_res/"+"losses.pt")  
    
    
    
    
if __name__ == "__main__":
    print("play")
    with open("configs/diverse.yaml", 'r') as f:
        configs1 = yaml.load(f, yaml.Loader)
        
    with open("configs/fig_cousins.yaml", 'r') as f:
        configs2 = yaml.load(f, yaml.Loader)

    print('Config file content:')
    print(configs1)
    print('Config file content:')
    print(configs2)