from threebody import *
from samples import SAMPLES,FIGSAMPLES
from tqdm import tqdm
from time import time
import os


def transform_dataset(dataset):
    new = torch.Tensor(dataset.shape[0],dataset.shape[1],3,4)
    new[:,:,0,0:2] = dataset[:,:,0,0:2]
    new[:,:,1,0:2] = dataset[:,:,0,2:4]
    new[:,:,2,0:2] = dataset[:,:,0,4:6]
    new[:,:,0,2:4] = dataset[:,:,0,6:8]
    new[:,:,1,2:4] = dataset[:,:,0,8:10]
    new[:,:,2,2:4] = dataset[:,:,0,10:12]
    return new


def get_t_array(name,samp_dict,dt):
    d = len(str(dt).split('.')[1]) 
    T = round(samp_dict[name]["T"],d)
    t = torch.linspace(0,T-dt,int(T/dt))
    return t



def create_fig_dataset(alpha_max,num_samples,dt,save=False):
    maker = threebody(FIGSAMPLES)
    list_samples = list(FIGSAMPLES.keys())
    created_samples =[]
    if save:
        if not os.path.exists("dat_fig"):
            os.makedirs("dat_fig")
    for type in tqdm(list_samples):
        print(type)
        sample = []
        time = get_t_array(type,FIGSAMPLES,dt)
        x,dx,t,h = maker.dataset_onekind_t(type,num_samples,time,[0,alpha_max])
        x=transform_dataset(x)
        dx=transform_dataset(dx)
        print(x.shape)
        print(dx.shape)
        print(t.shape)
        print(h.shape)
        if save:
            torch.save(x, "dat_fig/"+type+ "_x.pt")
            torch.save(dx,"dat_fig/"+type+ "_dx.pt")
            torch.save(h, "dat_fig/"+type+ "_h.pt")
            torch.save(t, "dat_fig/"+type+ "_t.pt")
        sample.append(x)
        sample.append(dx)
        sample.append(h)
        sample.append(t)
        print(time[-1])
        print(t[1]-t[0])
        created_samples.append(sample)
    return created_samples
            
        


def create_diverse_dataset(alpha_max,num_samples,dt,save=True):
    d = len(str(dt).split('.')[1]) 
    maker = threebody(SAMPLES)
    GOOGLES_T = round(SAMPLES["googles"]["T"],d)
    YARN_T = round(SAMPLES["yarn"]["T"],d)
    MOTH_T = round(SAMPLES["moth"]["T"],d)
    FIG8_T = round(SAMPLES["fig8"]["T"],d)
    VIII_T = round(SAMPLES["v810"]["T"],d)
    print(GOOGLES_T,YARN_T,MOTH_T,FIG8_T,VIII_T)
    t_googles = torch.linspace(0,GOOGLES_T-dt,int(GOOGLES_T/dt))
    t_yarn= torch.linspace(0,YARN_T-dt,int(YARN_T/dt))
    t_moth = torch.linspace(0,MOTH_T-dt,int(MOTH_T/dt))
    t_fig8 = torch.linspace(0,FIG8_T-dt,int(FIG8_T/dt))
    t_viii = torch.linspace(0,VIII_T-dt,int(VIII_T/dt))
    list_samples = ["googles","googles","yarn","moth","fig8","v810"]
    t_list = [t_googles,t_yarn,t_moth,t_fig8,t_viii]
    created_samples = []
    if save:
        if not os.path.exists("dat_div"):
            os.makedirs("dat_div")
    for type,time in tqdm(zip(list_samples,t_list)):
        print(type)
        sample = []
        x,dx,t,h = maker.dataset_onekind_t(type,num_samples,time,[0,alpha_max])
        x=transform_dataset(x)
        dx=transform_dataset(dx)
        print(x.shape)
        print(dx.shape)
        print(t.shape)
        print(h.shape)
        if save:
            torch.save(x, "dat_div/"+type+ "_x.pt")
            torch.save(dx,"dat_div/"+type+ "_dx.pt")
            torch.save(h, "dat_div/"+type+ "_h.pt")
            torch.save(t, "dat_div/"+type+ "_t.pt")
        print(time[-1])
        print(t[1]-t[0])
        sample.append(x)
        sample.append(dx)
        sample.append(h)
        sample.append(t)
        created_samples.append(sample)
    return created_samples

def create_dataset(type,alpha_max,num_samples,dt,save=True):
    if type == "dat_div":
        print("div")
        return create_diverse_dataset(alpha_max=alpha_max,num_samples=num_samples,dt=dt,save=save)
    elif type == "dat_fig":
        print("fig")
        return create_fig_dataset(alpha_max=alpha_max,num_samples=num_samples,dt=dt,save=save)
    else:
        print("miss")
        return []

alpha = torch.pi/4 # [0,pi/4]
samples =3
if __name__ == "__main__":

    a = time()    
    data1 = create_dataset("dat_fig",alpha_max=alpha,num_samples=samples,dt =0.01, save= True)
    b = time()
    print(data1[2][3].shape)
    print("time elapsed: {} sec".format(b-a))
    del data1
    a = time()  
    data2 = create_dataset("dat_div",alpha_max=alpha,num_samples=samples,dt =0.01, save= True)
    b = time()
    print("time elapsed: {} sec".format(b-a))
    print(data2[1][1].shape)
    del data2