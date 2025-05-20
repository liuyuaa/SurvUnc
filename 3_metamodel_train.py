import argparse
from sklearn.ensemble import RandomForestRegressor
from load_data import Dataset
from model import *
import yaml
import pickle
import joblib
import torch
from tqdm import tqdm
from sklearn.neural_network import MLPRegressor

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='flchain', help="flchain/metabric/support/seer_bc/sac3")
    parser.add_argument("--model", type=str, default='DeepSurv', help="DeepHit/DeepSurv/RSF/DSM")
    parser.add_argument("--meta_model", type=str, default='MLP', help="RF/MLP")
    parser.add_argument("--config", type=str, default='surv_model_config', help="config file name")
    parser.add_argument("--training", action='store_true', help="Train or Inference")
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

    with open('./meta_dataset/anchor2idx_'+args.dataset+'.pt', 'rb') as f:
        anchor2idx = pickle.load(f)
    with open('./meta_dataset/'+args.model+'/anchor2metaset_'+args.model+'_'+args.dataset+'.pt', 'rb') as f:
        anchor2meta_train = pickle.load(f)
    with open('./meta_dataset/'+args.model+'/anchor2metaset_val_'+args.model+'_'+args.dataset+'.pt', 'rb') as f:
        anchor2meta_val = pickle.load(f)
    anchor_group_sizes = [5, 10, 20, 30, 40, 50, 100]

    for J in anchor_group_sizes:
        meta_x, meta_y = anchor2meta_train[J]
        meta_x_val, meta_y_val = anchor2meta_val[J]
        meta_x, meta_y, meta_x_val, meta_y_val = np.array(meta_x), np.array(meta_y), np.array(meta_x_val), np.array(meta_y_val)
        if args.meta_model == 'RF':
            meta_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, min_samples_split=10, random_state=42)
            meta_model.fit(meta_x, meta_y)
            mse_train = np.mean((meta_model.predict(meta_x) - meta_y) ** 2)
            mse_val = np.mean((meta_model.predict(meta_x_val) - meta_y_val) ** 2)
            print('Anchor size:%d, MSE_Train:%f, MSE_Valid:%f' % (J, mse_train, mse_val))
        elif args.meta_model == 'MLP':
            meta_model = MLPRegressor(hidden_layer_sizes=[32, 32], batch_size=512, learning_rate_init=0.001, random_state=42, max_iter=500)
            meta_model.fit(meta_x, meta_y)
            mse_train = np.mean((np.abs(np.tanh(meta_model.predict(meta_x ))) - meta_y) ** 2)
            mse_val = np.mean((np.abs(np.tanh(meta_model.predict(meta_x_val))) - meta_y_val) ** 2)
            print('Anchor size:%d, MSE_Train:%f, MSE_Valid:%f' % (J, mse_train, mse_val))

        joblib.dump(meta_model, './meta_model/' + args.model + '/' + args.meta_model + '_' + str(J) + '_'+ args.dataset + '.joblib')
