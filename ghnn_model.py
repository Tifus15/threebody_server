import torch.nn as nn
import dgl
import torch
import dgl.function as fn
import torch.nn.functional as F

class PulloutLayer(nn.Module):
    def __init__(self,g,in_dim,out_dim,bias=False):
        super(PulloutLayer,self).__init__()
        self.g = g
        self.in_w = nn.Linear(in_dim, out_dim, bias=bias)
        self.out_w = nn.Linear(in_dim, out_dim, bias=bias)
        
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.in_w.weight, gain=gain)
        nn.init.xavier_normal_(self.out_w.weight, gain=gain)
    def message_func(self, edges):
        return {'s': edges.src['z'], 'd': edges.dst['z']}
    
    def reduce_func(self, nodes):
        src = self.in_w(nodes.mailbox['s'])
        dst = self.out_w(nodes.mailbox['d'])
        #print("src: {}".format(src.shape))
        #print("dst: {}".format(dst.shape))
        out=torch.cat((src,dst),dim=1)
        out1 = torch.sum(out,dim=1)
        #print(out.shape)
        return {'h': out1}
        
    def forward(self, h):
            # equation (1)
        #print(h.shape)
        #print(self.g)
        self.g.ndata['z'] = h
            # equation (2)
            
            # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')
    def change_graph(self,g):
        #print("in change {}".format(g))
        self.g = g
        #print("after change {}".format(self.g))

            
        
        
        
class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim,bias=False,dropout = 1.0):
        super(GATLayer, self).__init__()
        self.g = g
        self.drop = dropout
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=bias)
        self.att_drop = nn.Dropout(p=dropout)
        self.reset_parameters()
    
    def change_graph(self,g):
        #print("GAT g exchanged")
        self.g = g
    
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=-1)
        #print(z2.shape)
        a = self.attn_fc(self.att_drop(z2))
    
        
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

def function_act(name):
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "sin":
        return Sin()
    if name == "softplus":
        return nn.Softplus()
    else:
        return nn.Identity()

"""sin activation function as torch module"""
    
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(1.0 * x)


class GNNlayer(nn.Module):
    def __init__(self,g, in_dim, out_dim,bias = True):
        super().__init__()
        self.NN = nn.Linear(in_dim,out_dim,bias)
        self.g = g
        nn.init.normal_(self.NN.weight,mean=0.,std=0.1)
        nn.init.constant_(self.NN.bias,val=0)
    
    def change_graph(self,g):
        #print("GNNlayer g exchanged")
        #print(g)
        self.g = g
    
    def forward(self,feat):
        with self.g.local_scope():
            #print(feat.shape)
            self.g.ndata["h"] = self.NN(feat)
            self.g.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            #print(self.g.ndata["h"].shape)
            h = self.g.ndata["h"]
            return h
        
        
class GNN(nn.Module):
    def __init__(self,g, in_dim,hid_dim, out_dim):
        super().__init__()
        self.g = g
        self.in_layer = GNNlayer(g,in_dim, hid_dim)
        self.tan = nn.Tanh()
        self.out_layer = GNNlayer(g,hid_dim, out_dim)

    def forward(self,feat):
        h = self.in_layer(feat)
  
        h = self.tan(h)
   
        h = self.out_layer(h)

        return h
    
class GNN_maker_HNN(nn.Module):
    def __init__(self,g, in_dim,hid_dim, out_dim,acts,bias=True,type="GCN",dropout =1.0):
        super().__init__()
        self.g = g
        self.g_hnn=g
        modules = []
        if type == "GCN":
            modules.append(GNNlayer(g,in_dim,hid_dim,bias=bias))
            modules.append(function_act(acts[0]))
            for i in range(1,len(acts)-1):
                modules.append(GNNlayer(g,hid_dim,hid_dim,bias=bias))
                if acts[i] != "":
                    modules.append(function_act(acts[i]))
            modules.append(GNNlayer(g,hid_dim,out_dim,bias=bias))
        elif type == "GAT":
            modules.append(GATLayer(g,in_dim,hid_dim,bias=bias,dropout=dropout))
            modules.append(function_act(acts[0]))
            for i in range(1,len(acts)-1):
                modules.append(GATLayer(g,hid_dim,hid_dim,bias=bias))
                if acts[i] != "":
                    modules.append(function_act(acts[i]))
            modules.append(GATLayer(g,hid_dim,out_dim,bias=bias,dropout=dropout))
        elif type == "PULL":
            modules.append(PulloutLayer(g,in_dim,hid_dim,bias=bias))
            modules.append(function_act(acts[0]))
            for i in range(1,len(acts)-1):
                modules.append(PulloutLayer(g,hid_dim,hid_dim,bias=bias))
                if acts[i] != "":
                    modules.append(function_act(acts[i]))
            modules.append(PulloutLayer(g,hid_dim,out_dim,bias=bias))
            
        self.net = nn.Sequential(*modules)
    
    def change_graph(self,g):
        #print("chaning in ghnn")
        for module in self.net.modules():
            if isinstance(module,GATLayer) or isinstance(module,GNNlayer) or isinstance(module,PulloutLayer):
                module.change_graph(g)   
        self.g_hnn = g
    
    def reset_graph(self):
        for module in self.net.modules():
            if isinstance(module,GATLayer) or isinstance(module,GNNlayer) or isinstance(module,PulloutLayer):
                module.change_graph(self.g)
        self.g_hnn = self.g
    
    """   
    def H(self,y):
        return torch.sum(self.net(y.float()),dim=0)
    """
    def forward(self,x,list=False):
        H_feat = self.net(x.float())
        #print(H_feat.shape)
        self.g_hnn.ndata["temp"] = H_feat
        gs = dgl.unbatch(self.g_hnn)
        #print("from model")
        #print("unbatched {}".format(len(gs)))
        h_list = []
        for g in gs:
            h = g.ndata["temp"]
            #print("h {}".format(h.shape))
            h_sc = h.sum()
           # print(h_sc)
            h_list.append(h_sc.unsqueeze(0))
        out=torch.cat((h_list),dim=0).unsqueeze(0)
        #print(out)
        return out

        
            
            
    def H_rollout(self,x):
        h = []
        for i in range(x.shape[0]):
            h.append(self.forward(x[i,:,:]).unsqueeze(0))
        return torch.cat((h),dim=0)
    
    
    def dx_rollout(self,x):
        out_l = []
        N = self.g.num_nodes()
        for i in range(x.shape[0]):
            xi = x[i,:,:].clone().detach()
            #print(xi.shape)
            dx = torch.autograd.functional.jacobian(self.forward,xi)
            h_n = dx.shape[0]
            temp = []
            for j in range(h_n):
                #print("jac:{} {}".format(j,dx[j,N*j:N*(j+1),:].shape))
                temp.append(dx[j,N*j:N*(j+1),:])
            temp1 = torch.cat((temp),dim=0)
            out_l.append(temp1.unsqueeze(0))   
        out = torch.cat((out_l),dim=0)     
        return out
    """ 
    def dx_rollout(self,x):
        dx = []
        for i in range(x.shape[0]):
            xi = x[i,:,:].clone().requires_grad_()
            #print("dx roll x_i: {}".format(x_i.shape))
            h=self.forward(xi,list=True)
            #print("dx roll H: {}".format(h))
            dHdx = torch.autograd.grad(h,xi,retain_graph=True,create_graph=True)[0]
            #print("dx roll dHdx: {}".format(dHdx.shape))
            qp = torch.split(dHdx,int(x.shape[-1]/2),dim=-1)
            dx.append(torch.cat((qp[1].unsqueeze(0),-qp[0].unsqueeze(0)),dim=-1).float())
        return torch.cat((dx),dim=0)
    """
def rk4(x,dt,model):
   # print("x {}".format(x.shape))
    k1 = model(x)
   # print("k1 {}".format(k1.shape))
    k2 = model(x + dt*k1/2)
   # print("k2 {}".format(k2.shape))
    k3 = model(x + dt*k2/2)
   # print("k3 {}".format(k3.shape))
    k4 = model(x + dt*k3)
   # print("k4 {}".format(k4.shape))
    
    return (k1 + 2*k2 + 2*k3 + k4)/6
    

def rollout(x0,t,model, method = "rk4"):
    l = []
    l.append(x0.unsqueeze(0).detach().requires_grad_())
    for i in range(len(t)-1):
       # print("rollout {}".format(i))
        dt = t[i+1]-t[i]
        xi = l[-1].squeeze().detach().requires_grad_()
        if method == "rk4":
            xii = xi + dt * rk4(xi.unsqueeze(0),dt,model.dx_rollout)
        l.append(xii.requires_grad_())
        #l.append(xii.detach().requires_grad_())
        #print("l[{}] is leaf {}".format(i,l[i].is_leaf))
    #print("l[{}] is leaf {}".format(len(t)-1,l[len(t)-1].is_leaf))
    return torch.cat((l),dim=0)
            
    
"""    
def rollout_dx(model_hnn,x):
    dx_l = []
    for i in range(x.shape[0]):
        h = 
    
    def dx_rollout(self,x):
        dx = []
        for i in range(x.shape[0]):
            dx.append(self.forward(0,x[i,:,:]).unsqueeze(0))
        return torch.cat((dx),dim=0)

    def forward(self,t,y):
        sc_val = self.H(y)
        H_l = torch.split(sc_val,1,dim=0)
        dhdx = torch.autograd.grad(H_l,y,retain_graph=True)[0]
        qp = torch.split(dhdx,int(y.shape[-1]/2),dim=-1)
        return torch.cat((qp[1],-qp[0]),dim=-1).float()
"""
#in GRU there is no cuda backward with autodifferentiation implemented. it is error in there
# Im forcing cpu computation
class rollout_GNN_GRU(nn.Module):
    def __init__(self,g,in_dim,hid,gnn_size,acts,bias=True,type="GCN",dropout=1.0):
        super().__init__()
        self.GRU = nn.GRU(in_dim,hid,1).to(torch.device("cpu"))
        self.GNN = GNN_maker_HNN(g,hid,gnn_size,in_dim,acts,bias=bias,type=type,dropout=dropout).cpu()
        self.g = g
        #self.NN = nn.Linear(hid,in_dim)
        nn.init.normal_(self.GRU.weight_ih_l0,mean=0.,std=0.1)
        nn.init.normal_(self.GRU.weight_hh_l0,mean=0.,std=0.1)
        nn.init.constant_(self.GRU.bias_ih_l0,val=0)
        nn.init.constant_(self.GRU.bias_hh_l0,val=0)
    
    def change_graph(self,g):
        #print("changing in GRU")
        self.GNN.change_graph(g)
       
    
    def reset_graph(self):
        self.GNN.change_graph(self.g)
        
        
    def forward(self,t,x0):
        #print(x0.device)
        #print(t.device)
        #print(self.GNN.g_hnn.device)
        batches =len(dgl.unbatch(self.GNN.g_hnn))
        #print("batches {}".format(batches))
        T = len(t)
        output = torch.zeros(T,x0.shape[0],x0.shape[1])
        doutput = torch.zeros(T,x0.shape[0],x0.shape[1])
        houtput = torch.zeros(T,batches)
        #print(output.device)
        #print(doutput.device)
        #print(houtput.device)
        #print(self.GRU.device)
        #print(self.GNN.device)
        output[0,:,:] = x0
        xi, hidden = self.GRU(x0.unsqueeze(dim=0))
        #print("x0: {}".format(x0.shape))
        #print("xi: {}".format(xi.unsqueeze(dim=0).shape))
        hi = self.GNN(xi.squeeze())
        #print("hi {}".format(hi.shape))
        houtput[0,:]=hi
        H_l = torch.split(houtput[0,:],1,dim=0)
        dHdx = torch.autograd.grad(H_l,x0,retain_graph=True, create_graph=True)[0] 
        dqdp_s = torch.split(dHdx,2,dim=-1)
        dx_pred = torch.cat((dqdp_s[1],-dqdp_s[0]),dim=-1)
        doutput[0,:,:] = dx_pred
        #print("xii: {}".format(xii.shape))
        #print("hidden: {}".format(hidden.shape))
        #print(xii.shape)
        
        dt = t[1]-t[0]
        temp = output[0,:,:] + dt*doutput[0,:,:]
        output[1,:,:] = temp
        for i in range(2,T):
            #print("xi: {}".format(xi.unsqueeze(dim=0).shape))
            xi, hidden = self.GRU(temp.unsqueeze(dim=0),hidden)
            #print("hidden: {}".format(hidden.shape))
            hi = self.GNN(xi.squeeze())
            houtput[i-1,:]=hi
            H_l = torch.split(houtput[i-1,:],1,dim=0)
            dHdx = torch.autograd.grad(H_l,temp,retain_graph=True, create_graph=True)[0] 
            dqdp_s = torch.split(dHdx,2,dim=-1)
            dx_pred = torch.cat((dqdp_s[1],-dqdp_s[0]),dim=-1)
            
            
            
            doutput[i-1,:,:] = dx_pred
            #print("xii: {}".format(xii.shape))
            
            dt = t[i]-t[i-1]
            temp = output[i-1,:,:]+ dt*dx_pred
            output[i,:,:] = temp
        xi, hidden = self.GRU(temp.unsqueeze(dim=0),hidden)
        #print("hidden: {}".format(hidden.shape))
        hi = self.GNN(xi.squeeze())
        houtput[-1,:] = hi
        H_l = torch.split(houtput[-1,:],1,dim=0)
        dHdx = torch.autograd.grad(H_l,temp,retain_graph=True, create_graph=True)[0] 
        dqdp_s = torch.split(dHdx,2,dim=-1)
        dx_pred = torch.cat((dqdp_s[1],-dqdp_s[0]),dim=-1)
        doutput[-1,:,:] = dx_pred
        return  output, doutput, houtput
        
