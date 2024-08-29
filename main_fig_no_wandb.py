from server_funcs import dataset_train
import yaml

with open("configs/fig_cousins.yaml", 'r') as f:
        configs = yaml.load(f, yaml.Loader)
        
dataset_train(configs)


