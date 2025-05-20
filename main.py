import argparse
from load_data import Dataset
from model import *
import yaml

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='flchain', help="flchain/support/seer_bc/sac3")
    parser.add_argument("--model", type=str, default='DeepHit', help="DeepHit/DeepSurv/RSF/DSM")
    parser.add_argument("--config", type=str, default='surv_model_config', help="config file name")
    parser.add_argument("--training", action='store_false', help="Train or Inference")
    args = parser.parse_args()

    np.random.seed(1234)
    _ = torch.manual_seed(123)
    random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Read the config file
    config = read_config('./configs/'+args.config+'.yaml')
    d = Dataset(args.dataset)
    kwargs = config[args.model][args.dataset]
    kwargs['training'], kwargs['dataset'] = args.training, args.dataset

    model = globals()[args.model](d, **kwargs)
    cindex_pycox, ibs_pycox = model.forward(d)
    print('Dataset:%s' % args.dataset)
    print('pycox:')
    print('cindex:%f, ibs:%f' % (cindex_pycox, ibs_pycox))