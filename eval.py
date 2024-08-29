import torch
import dgl
from data_func import *
from dgl.dataloading import GraphDataLoader
from ghnn_model import *
from hnn_model import *
from tqdm import tqdm
from torchdiffeq import odeint
import random
import matplotlib.pyplot as plt
ghnn = torch.load("GHNN.pt").cpu()
hnn = torch.load("HNN.pt").cpu()
grughnn = torch.load("GRUGHNN.pt").cpu()


print(ghnn)
print(hnn)
print(grughnn)
x,dx,h,t = data_load("fig8")
#print(dx.shape)
# take one random sample
rand = random.randint(0,99)
x = x[:,rand,:,:].unsqueeze(1)
dx = dx[:,rand,:,:].unsqueeze(1)
h = h[:,rand,:,:].unsqueeze(1)

print(x.shape)
print(dx.shape)
print(h.shape)
print(t.shape)
xs,hs = make_snapshots(torch.cat((x,dx),dim=-1),h,len(t)-2)
print(len(xs))
src,dst=make_graph_no_loops(3,0)
dgl_snap =transform_dgl(src,dst,xs,hs)
print(dgl_snap)

x_g,dx_g,h_g = get_d_dx_H(dgl_snap[0])
x,dx,h = get_batch_baseline(dgl_snap[0])

#print(dx.shape)
t = t[0:630]

#print(dx_g.shape)


x0 = x[0,:,:,:].requires_grad_()
x_baseline = odeint(hnn,x0,t,method="rk4").squeeze()
x=x.squeeze()
print(" ")
print(x.shape)
print(x_baseline.shape)
h_baseline = hnn.giveH(x_baseline).squeeze()
h=h.squeeze()
print(" ")
print(h.shape)
print(h_baseline.shape)
g_list=[]
g = dgl.graph((src,dst))
for i in range(x_baseline.shape[0]):
    g_list.append(g)
g_h = dgl.batch(g_list)
xb0 = x_g[0,:,:].requires_grad_()
ghnn.change_graph(g)
x_model, dx_model = RKroll_for_learning(ghnn,xb0,t)
x_model = x_model.squeeze()
x_g=x_g.squeeze()
print(" ")
print(x_g.shape)
print(x_model.shape)
ghnn.change_graph(g_h)
#graph H
x_g_flat = x_model.reshape(-1,4)
h_model = ghnn(x_g_flat).transpose(0,1).squeeze()
h_g=h_g.squeeze()
print(" ")
print(h_g.shape)
print(h_model.shape)

grughnn.change_graph(g)
xg, dxg, hg = grughnn(t,xb0)
print(" ")
print(x_g.shape)
print(xg.shape)
print(h_g.shape)
print(hg.shape)
xg = xg.squeeze()
hg.squeeze()





fig, ax = plt.subplots(3,4)
ax[0,0].plot(x[:,0],x[:,6])
ax[0,0].scatter(x_baseline[:,0].detach().numpy(),x_baseline[:,6].detach().numpy(),c="g")
ax[0,0].set_title("x and px at body1")
ax[0,0].legend(["ground truth","prediction"])
ax[0,0].set_xlabel("x")
ax[0,0].set_ylabel("px")

ax[0,1].plot(x[:,2],x[:,8])
ax[0,1].scatter(x_baseline[:,0].detach().numpy(),x_baseline[:,6].detach().numpy(),c="g")
ax[0,1].set_title("x and px at body2")
ax[0,1].legend(["ground truth","prediction"])
ax[0,1].set_xlabel("x")
ax[0,1].set_ylabel("px")

ax[0,2].plot(x[:,4],x[:,10])
ax[0,2].scatter(x_baseline[:,0].detach().numpy(),x_baseline[:,6].detach().numpy(),c="g")
ax[0,2].set_title("x and px at body3")
ax[0,2].legend(["ground truth","prediction"])
ax[0,2].set_xlabel("x")
ax[0,2].set_ylabel("px")

ax[1,0].plot(x[:,1],x[:,7])
ax[1,0].scatter(x_baseline[:,1].detach().numpy(),x_baseline[:,7].detach().numpy(),c="g")
ax[1,0].set_title("y and py at body1")
ax[1,0].legend(["ground truth","prediction"])
ax[1,0].set_xlabel("y")
ax[1,0].set_ylabel("py")

ax[1,1].plot(x[:,3],x[:,9])
ax[1,1].scatter(x_baseline[:,3].detach().numpy(),x_baseline[:,9].detach().numpy(),c="g")
ax[1,1].set_title("y and py at body2")
ax[1,1].legend(["ground truth","prediction"])
ax[1,1].set_xlabel("y")
ax[1,1].set_ylabel("py")

ax[1,2].plot(x[:,5],x[:,11])
ax[1,2].scatter(x_baseline[:,5].detach().numpy(),x_baseline[:,11].detach().numpy(),c="g")
ax[1,2].set_title("y and py at body3")
ax[1,2].legend(["ground truth","prediction"])
ax[1,2].set_xlabel("y")
ax[1,2].set_ylabel("py")

ax[2,0].plot(x[:,0],x[:,1])
ax[2,0].scatter(x_baseline[:,0].detach().numpy(),x_baseline[:,1].detach().numpy(),c="g")
ax[2,0].set_title("x and y at body1")
ax[2,0].legend(["ground truth","prediction"])
ax[2,0].set_xlabel("x")
ax[2,0].set_ylabel("y")

ax[2,1].plot(x[:,2],x[:,3])
ax[2,1].scatter(x_baseline[:,2].detach().numpy(),x_baseline[:,3].detach().numpy(),c="g")
ax[2,1].set_title("x and y at body2")
ax[2,1].legend(["ground truth","prediction"])
ax[2,1].set_xlabel("x")
ax[2,1].set_ylabel("y")

ax[2,2].plot(x[:,4],x[:,5])
ax[2,2].scatter(x_baseline[:,5].detach().numpy(),x_baseline[:,5].detach().numpy(),c="g")
ax[2,2].set_title("x and y at body3")
ax[2,2].legend(["ground truth","prediction"])
ax[2,2].set_xlabel("x")
ax[2,2].set_ylabel("y")

ax[2,3].plot(t,h)
ax[2,3].scatter(t,h_baseline.detach().numpy(),c="g")
ax[2,3].set_title("hamiltonian")
ax[2,3].legend(["ground truth","prediction"])
ax[2,3].set_xlabel("t")
ax[2,3].set_ylabel("H")
plt.suptitle('HNN')
plt.show()

fig, ax = plt.subplots(3,4)
ax[0,0].plot(x_g[:,0,0],x_g[:,0,2])
ax[0,0].scatter(x_model[:,0,0].detach().numpy(),x_model[:,0,2].detach().numpy(),c="g")
ax[0,0].set_title("x and px at body1")
ax[0,0].legend(["ground truth","prediction"])
ax[0,0].set_xlabel("x")
ax[0,0].set_ylabel("px")

ax[0,1].plot(x_g[:,1,0],x_g[:,1,2])
ax[0,1].scatter(x_model[:,1,0].detach().numpy(),x_model[:,1,2].detach().numpy(),c="g")
ax[0,1].set_title("x and px at body2")
ax[0,1].legend(["ground truth","prediction"])
ax[0,1].set_xlabel("x")
ax[0,1].set_ylabel("px")

ax[0,2].plot(x_g[:,2,0],x_g[:,2,2])
ax[0,2].scatter(x_model[:,2,0].detach().numpy(),x_model[:,2,2].detach().numpy(),c="g")
ax[0,2].set_title("x and px at body3")
ax[0,2].legend(["ground truth","prediction"])
ax[0,2].set_xlabel("x")
ax[0,2].set_ylabel("px")

ax[1,0].plot(x_g[:,0,1],x_g[:,0,3])
ax[1,0].scatter(x_model[:,0,1].detach().numpy(),x_model[:,0,3].detach().numpy(),c="g")
ax[1,0].set_title("y and py at body1")
ax[1,0].legend(["ground truth","prediction"])
ax[1,0].set_xlabel("y")
ax[1,0].set_ylabel("py")

ax[1,1].plot(x_g[:,1,1],x_g[:,1,3])
ax[1,1].scatter(x_model[:,1,1].detach().numpy(),x_model[:,1,3].detach().numpy(),c="g")
ax[1,1].set_title("y and py at body2")
ax[1,1].legend(["ground truth","prediction"])
ax[1,1].set_xlabel("y")
ax[1,1].set_ylabel("py")

ax[1,2].plot(x_g[:,2,1],x_g[:,2,3])
ax[1,2].scatter(x_model[:,2,1].detach().numpy(),x_model[:,2,3].detach().numpy(),c="g")
ax[1,2].set_title("y and py at body3")
ax[1,2].legend(["ground truth","prediction"])
ax[1,2].set_xlabel("y")
ax[1,2].set_ylabel("py")

ax[2,0].plot(x_g[:,0,0],x_g[:,0,1])
ax[2,0].scatter(x_model[:,0,0].detach().numpy(),x_model[:,0,1].detach().numpy(),c="g")
ax[2,0].set_title("x and y at body1")
ax[2,0].legend(["ground truth","prediction"])
ax[2,0].set_xlabel("x")
ax[2,0].set_ylabel("y")

ax[2,1].plot(x_g[:,1,0],x_g[:,1,1])
ax[2,1].scatter(x_model[:,1,1].detach().numpy(),x_model[:,1,1].detach().numpy(),c="g")
ax[2,1].set_title("x and y at body2")
ax[2,1].legend(["ground truth","prediction"])
ax[2,1].set_xlabel("x")
ax[2,1].set_ylabel("y")

ax[2,2].plot(x_g[:,2,0],x_g[:,2,1])
ax[2,2].scatter(x_model[:,2,1].detach().numpy(),x_model[:,2,1].detach().numpy(),c="g")
ax[2,2].set_title("x and y at body3")
ax[2,2].legend(["ground truth","prediction"])
ax[2,2].set_xlabel("x")
ax[2,2].set_ylabel("y")


ax[2,3].plot(t,h_g)
ax[2,3].scatter(t,h_model.detach().numpy(),c="g")
ax[2,3].set_title("hamiltonian")
ax[2,3].legend(["ground truth","prediction"])
ax[2,3].set_xlabel("t")
ax[2,3].set_ylabel("H")
plt.suptitle('GHNN')
plt.show()

fig, ax = plt.subplots(3,4)
ax[0,0].plot(x_g[:,0,0],x_g[:,0,2])
ax[0,0].scatter(xg[:,0,0].detach().numpy(),xg[:,0,2].detach().numpy(),c="g")
ax[0,0].set_title("x and px at body1")
ax[0,0].legend(["ground truth","prediction"])
ax[0,0].set_xlabel("x")
ax[0,0].set_ylabel("px")

ax[0,1].plot(x_g[:,1,0],x_g[:,1,2])
ax[0,1].scatter(xg[:,1,0].detach().numpy(),xg[:,1,2].detach().numpy(),c="g")
ax[0,1].set_title("x and px at body2")
ax[0,1].legend(["ground truth","prediction"])
ax[0,1].set_xlabel("x")
ax[0,1].set_ylabel("px")

ax[0,2].plot(x_g[:,2,0],x_g[:,2,2])
ax[0,2].scatter(xg[:,2,0].detach().numpy(),xg[:,2,2].detach().numpy(),c="g")
ax[0,2].set_title("x and px at body3")
ax[0,2].legend(["ground truth","prediction"])
ax[0,2].set_xlabel("x")
ax[0,2].set_ylabel("px")

ax[1,0].plot(x_g[:,0,1],x_g[:,0,3])
ax[1,0].scatter(xg[:,0,1].detach().numpy(),xg[:,0,3].detach().numpy(),c="g")
ax[1,0].set_title("y and py at body1")
ax[1,0].legend(["ground truth","prediction"])
ax[1,0].set_xlabel("y")
ax[1,0].set_ylabel("py")

ax[1,1].plot(x_g[:,1,1],x_g[:,1,3])
ax[1,1].scatter(xg[:,1,1].detach().numpy(),xg[:,1,3].detach().numpy(),c="g")
ax[1,1].set_title("y and py at body2")
ax[1,1].legend(["ground truth","prediction"])
ax[1,1].set_xlabel("y")
ax[1,1].set_ylabel("py")

ax[1,2].plot(x_g[:,2,1],x_g[:,2,3])
ax[1,2].scatter(xg[:,2,1].detach().numpy(),xg[:,2,3].detach().numpy(),c="g")
ax[1,2].set_title("y and py at body3")
ax[1,2].legend(["ground truth","prediction"])
ax[1,2].set_xlabel("y")
ax[1,2].set_ylabel("py")

ax[2,0].plot(x_g[:,0,0],x_g[:,0,1])
ax[2,0].scatter(xg[:,0,0].detach().numpy(),xg[:,0,1].detach().numpy(),c="g")
ax[2,0].set_title("x and y at body1")
ax[2,0].legend(["ground truth","prediction"])
ax[2,0].set_xlabel("x")
ax[2,0].set_ylabel("y")

ax[2,1].plot(x_g[:,1,0],x_g[:,1,1])
ax[2,1].scatter(xg[:,1,1].detach().numpy(),xg[:,1,1].detach().numpy(),c="g")
ax[2,1].set_title("x and y at body2")
ax[2,1].legend(["ground truth","prediction"])
ax[2,1].set_xlabel("x")
ax[2,1].set_ylabel("y")

ax[2,2].plot(x_g[:,2,0],x_g[:,2,1])
ax[2,2].scatter(xg[:,2,1].detach().numpy(),xg[:,2,1].detach().numpy(),c="g")
ax[2,2].set_title("x and y at body3")
ax[2,2].legend(["ground truth","prediction"])
ax[2,2].set_xlabel("x")
ax[2,2].set_ylabel("y")


ax[2,3].plot(t,h_g)
ax[2,3].scatter(t,hg.detach().numpy(),c="g")
ax[2,3].set_title("hamiltonian")
ax[2,3].legend(["ground truth","prediction"])
ax[2,3].set_xlabel("t")
ax[2,3].set_ylabel("H")
plt.suptitle('GRUGHNN')
plt.show()

losses = torch.load("losses.pt").cpu()
ep_max =losses.shape[0]

ep = torch.linspace(1,ep_max,ep_max)

fig, ax = plt.subplots(1,4)
ax[0].set_title("rollout")
ax[0].plot(ep,losses[:,1])
ax[0].plot(ep,losses[:,13])
ax[0].set_xlabel("epochs")
ax[0].set_ylabel("loss")
ax[0].legend(["train loss","test loss"])



ax[1].set_title("hamiltonian")
ax[1].plot(ep,losses[:,2])
ax[1].plot(ep,losses[:,14])
ax[1].set_xlabel("epochs")
ax[1].set_ylabel("loss")
ax[1].legend(["train loss","test loss"])


ax[2].set_title("vector")
ax[2].plot(ep,losses[:,3])
ax[2].plot(ep,losses[:,15])
ax[2].set_xlabel("epochs")
ax[2].set_ylabel("loss")
ax[2].legend(["train loss","test loss"])


ax[3].set_title("summary")
ax[3].plot(ep,losses[:,0])
ax[3].plot(ep,losses[:,12])
ax[3].set_xlabel("epochs")
ax[3].set_ylabel("loss")
ax[3].legend(["train loss","test loss"])
plt.suptitle('losses at GHNN')
plt.show()

fig, ax = plt.subplots(1,4)
ax[0].set_title("rollout")
ax[0].plot(ep,losses[:,5])
ax[0].plot(ep,losses[:,17])
ax[0].set_xlabel("epochs")
ax[0].set_ylabel("loss")
ax[0].legend(["train loss","test loss"])



ax[1].set_title("hamiltonian")
ax[1].plot(ep,losses[:,6])
ax[1].plot(ep,losses[:,18])
ax[1].set_xlabel("epochs")
ax[1].set_ylabel("loss")
ax[1].legend(["train loss","test loss"])


ax[2].set_title("vector")
ax[2].plot(ep,losses[:,7])
ax[2].plot(ep,losses[:,19])
ax[2].set_xlabel("epochs")
ax[2].set_ylabel("loss")
ax[2].legend(["train loss","test loss"])


ax[3].set_title("summary")
ax[3].plot(ep,losses[:,4])
ax[3].plot(ep,losses[:,16])
ax[3].set_xlabel("epochs")
ax[3].set_ylabel("loss")
ax[3].legend(["train loss","test loss"])
plt.suptitle('losses at HNN')
plt.show()


fig, ax = plt.subplots(1,4)
ax[0].set_title("rollout")
ax[0].plot(ep,losses[:,9])
ax[0].plot(ep,losses[:,21])
ax[0].set_xlabel("epochs")
ax[0].set_ylabel("loss")
ax[0].legend(["train loss","test loss"])



ax[1].set_title("hamiltonian")
ax[1].plot(ep,losses[:,10])
ax[1].plot(ep,losses[:,22])
ax[1].set_xlabel("epochs")
ax[1].set_ylabel("loss")
ax[1].legend(["train loss","test loss"])


ax[2].set_title("vector")
ax[2].plot(ep,losses[:,11])
ax[2].plot(ep,losses[:,23])
ax[2].set_xlabel("epochs")
ax[2].set_ylabel("loss")
ax[2].legend(["train loss","test loss"])


ax[3].set_title("summary")
ax[3].plot(ep,losses[:,8])
ax[3].plot(ep,losses[:,20])
ax[3].set_xlabel("epochs")
ax[3].set_ylabel("loss")
ax[3].legend(["train loss","test loss"])
plt.suptitle('losses at GRUGHNN')
plt.show()