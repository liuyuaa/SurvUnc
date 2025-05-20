import argparse
from load_data import Dataset
from model import *
import yaml
import pickle
from tqdm import tqdm

def compute_uncertainty_target(model, d, surv_x_all, x_uncensored_all, duration_uncensored_all, x_anchors, durations_anchors, events_anchors):
    J = x_anchors.shape[0]
    risk_x_all = 1 - surv_x_all
    surv_anchors = model.predict_surv(x_anchors, d)
    risk_anchors = 1 - surv_anchors

    time_grid = list(surv_x_all.index)
    x_meta, y_meta = [], []
    for i in range(surv_x_all.shape[1]):
        unc_tar, count = 0, 0
        duration_uncensored = duration_uncensored_all[i]
        for j in range(J):
            if (duration_uncensored < durations_anchors[j]) or (duration_uncensored == durations_anchors[j] and events_anchors[j] == 0):
                count += 1
                time_sel = duration_uncensored
                idx_x = np.searchsorted(time_grid, time_sel, side='right') - 1
                if risk_x_all.iloc[idx_x, i] <= risk_anchors.iloc[idx_x, j]:
                    unc_tar += 1
        if count > 0:
            x_meta.append(x_uncensored_all[i])
            y_meta.append(unc_tar/count)
    return x_meta, y_meta

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='flchain', help="flchain/support/seer_bc/sac3")
    parser.add_argument("--model", type=str, default='DeepSurv', help="DeepHit/DeepSurv/RSF/DSM")
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
    kwargs = config[args.model][args.dataset]
    kwargs['training'], kwargs['dataset'] = False, args.dataset
    model = globals()[args.model](d, **kwargs)

    with open('./meta_dataset/anchor2idx_'+args.dataset+'.pt', 'rb') as f:
        anchor2idx = pickle.load(f)
    anchor_group_sizes = [5, 10, 20, 30, 40, 50, 100]

    # # meta train
    anchor2metaset = {}
    uncensored_idx = np.where(d.events_train == 1)[0]
    x = d.x_train[uncensored_idx]
    duration = d.durations_train[uncensored_idx]
    surv_x_all = model.predict_surv(x, d)
    for J in anchor_group_sizes:
        anchor_idx = anchor2idx[J]
        x_anchors, durations_anchors, events_anchors = d.x_train[anchor_idx], d.durations_train[anchor_idx], d.events_train[anchor_idx]
        x_meta, y_meta = compute_uncertainty_target(model, d, surv_x_all, x, duration, x_anchors, durations_anchors, events_anchors)
        anchor2metaset[J] = [x_meta, y_meta]

    with open('./meta_dataset/'+args.model+'/anchor2metaset_'+args.model+'_'+args.dataset+'.pt', 'wb') as f:
        pickle.dump(anchor2metaset, f)

    # # meta valid
    anchor2metaset = {}
    uncensored_idx = np.where(d.events_val == 1)[0]
    x = d.x_val[uncensored_idx]
    duration = d.durations_val[uncensored_idx]
    surv_x_all = model.predict_surv(x, d)
    for J in anchor_group_sizes:
        anchor_idx = anchor2idx[J]
        x_anchors, durations_anchors, events_anchors = d.x_train[anchor_idx], d.durations_train[anchor_idx], d.events_train[anchor_idx]
        x_meta, y_meta = compute_uncertainty_target(model, d, surv_x_all, x, duration, x_anchors, durations_anchors, events_anchors)
        anchor2metaset[J] = [x_meta, y_meta]

    with open('./meta_dataset/'+args.model+'/'+'anchor2metaset_val_'+args.model+'_'+args.dataset+'.pt', 'wb') as f:
        pickle.dump(anchor2metaset, f)

