import torch.nn as nn
import torch 

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

"""MLP module"""
class mlp(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,acts,bias=True): # input layer,hidden_layersout,put_layer
        super(mlp,self).__init__()
        modules = []
        modules.append(nn.Linear(input_dim,hidden_dim))
        modules.append(function_act(acts[0]))
        for i in range(1,len(acts)-1):
            modules.append(nn.Linear(hidden_dim,hidden_dim,bias=bias))
            if acts[i] != "":
                modules.append(function_act(acts[i]))
        modules.append(nn.Linear(hidden_dim,output_dim))
        self.net = nn.Sequential(*modules)

        for m in self.net.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,mean=0.,std=0.1)
                nn.init.constant_(m.bias,val=0)
        
    def forward(self,y):
        return self.net(y.float())



class HNN(nn.Module):
    def __init__(self,c_in,E_hid,acts,bias=True):
        super(HNN,self).__init__()
        # it means that we have other coordinates
        self.c_in = c_in  # in dim 
        
        self.NN = mlp(c_in,E_hid,1,acts) 
        
    
        
    def H_layer(self,y):
        #print("in layer H")
        H = self.NN(y) # H output as vector of outputs from K and U
        #print("H_layer {}".format(H.shape))
        return H
    
    def giveH(self,y):
        #print("in giveH")
        H = self.H_layer(y).sum(dim=-1) # H from the layer summing all elements
        #print("giveH {}".format(H.shape))
        return H
    
    def create_J(self):
        M = torch.eye(self.c_in)
        M = torch.cat([M[self.c_in//2:], -M[:self.c_in//2]])  
        return M
      
    def forward(self,t,x):
        #print("in forward")
        #print(t)
        #print(y)
        #x = x.requires_grad_()
        H =self.giveH(x)
        #print("out of giveH")
          # get scalar H or scalars for batches
        #print(H)
        H = H.split(1,dim=0)
        #print(len(H))
        dh = torch.autograd.grad(H,x,retain_graph=True,create_graph=True)[0] # exploit autograd.grad
           # print("dh {}".format(dh.shape))
        
        J = self.create_J()
        if x.is_cuda:
            J = J.cuda()
        return dh @ J.T
    
    
def rollout_mlp_vec(model,x):
    out_list = []
    for i in range(x.shape[0]):
        temp = model(0,x[i,:,:,:]).unsqueeze(0)
        #print(temp.shape)
        out_list.append(temp)
    return torch.cat((out_list),dim=0)