import argparse
from load_data import Dataset
from model import *
import yaml
import pickle

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='flchain', help="flchain/support/seer_bc/sac3")
    parser.add_argument("--model", type=str, default='DeepSurv', help="DeepHit/DeepSurv/RSF/DSM")
    parser.add_argument("--config", type=str, default='surv_model_config', help="config file name")
    args = parser.parse_args()

    # Read the config file
    config = read_config('./configs/'+args.config+'.yaml')
    d = Dataset(args.dataset)

    anchor_group_size = [5, 10, 20, 30, 40, 50, 100]
    np.random.seed(1234)
    _ = torch.manual_seed(123)
    random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    J_max = max(anchor_group_size)
    uncensored_idx = np.where(d.events_train == 1)[0]
    anchor_idx = np.random.choice(uncensored_idx, J_max, replace=False)
    anchor2idx = {}
    for J in anchor_group_size:
        anchor2idx[J] = anchor_idx[:J]

    with open('./meta_dataset/anchor2idx_'+args.dataset+'.pt', 'wb') as f:
        pickle.dump(anchor2idx, f)
