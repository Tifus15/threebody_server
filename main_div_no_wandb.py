from server_funcs import dataset_train
import yaml

with open("configs/diverse.yaml", 'r') as f:
        configs = yaml.load(f, yaml.Loader)
        
dataset_train(configs)


