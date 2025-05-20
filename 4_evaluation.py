import argparse
from load_data import Dataset
from model import *
import yaml
import pickle
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='flchain', help="flchain/support/seer_bc/sac3")
    parser.add_argument("--model", type=str, default='DeepSurv', help="DeepHit/DeepSurv/RSF/DSM")
    parser.add_argument("--meta_model", type=str, default='RF', help="RF/MLP")
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

    anchor_group_sizes = [5, 10, 20, 30, 40, 50, 100]
    removing_ratios = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    metrics_remove_uncensored = np.zeros((len(anchor_group_sizes), len(removing_ratios), 4))
    metrics_remove_uncensored_only = np.zeros((len(anchor_group_sizes), len(removing_ratios), 5))
    for i_j, J in enumerate(anchor_group_sizes):
        meta_model = joblib.load('./meta_model/' + args.model + '/' + args.meta_model + '_' + str(J) + '_' + args.dataset + '.joblib')
        unc_test = meta_model.predict(d.x_test)
        if args.meta_model == 'MLP':
            unc_test = np.abs(np.tanh(unc_test))

        sorted_idx = np.argsort(unc_test)
        idx_uncensored = [ii for ii, event_ii in enumerate(d.events_test) if event_ii == 1]
        idx_censored = [ii for ii, event_ii in enumerate(d.events_test) if event_ii == 0]
        for i_r, removing_ratio in enumerate(removing_ratios):
            print('anchor size:%d, removing ratio:%f, model:%s, dataset:%s' % (J, removing_ratio, args.model, args.dataset))

            sorted_idx_uncensored = [ii for ii in sorted_idx if ii in idx_uncensored]
            kept_idx_remove_uncensored = sorted_idx_uncensored[:int(len(sorted_idx_uncensored) * (1 - removing_ratio))] + idx_censored

            x_test_remove_uncensored = d.x_test[kept_idx_remove_uncensored, :]
            durations_test_remove_uncensored = d.durations_test[kept_idx_remove_uncensored]
            events_test_remove_uncensored = d.events_test[kept_idx_remove_uncensored]
            cindex_pycox_remove_uncensored, _, _\
                = model.forward_eval(x_test_remove_uncensored, durations_test_remove_uncensored, events_test_remove_uncensored, d)
            metrics_remove_uncensored[i_j, i_r] = [cindex_pycox_remove_uncensored, 0, 0, 0]

            kept_idx_remove_uncensored_only = sorted_idx_uncensored[:int(len(sorted_idx_uncensored) * (1 - removing_ratio))]
            x_test_remove_uncensored_only = d.x_test[kept_idx_remove_uncensored_only, :]
            durations_test_remove_uncensored_only = d.durations_test[kept_idx_remove_uncensored_only]
            events_test_remove_uncensored_only = d.events_test[kept_idx_remove_uncensored_only]
            _, _, surv_uncensored_only\
                = model.forward_eval(x_test_remove_uncensored_only, durations_test_remove_uncensored_only, events_test_remove_uncensored_only, d)

            idx = np.searchsorted(surv_uncensored_only.index, durations_test_remove_uncensored_only, side='right') - 1
            true_label = np.zeros((surv_uncensored_only.shape))
            for i in range(surv_uncensored_only.shape[1]):
                true_label[0:idx[i], i] = 1
            ibs_uncensored_only = np.sum((true_label - surv_uncensored_only.values) ** 2) / surv_uncensored_only.shape[0] / surv_uncensored_only.shape[1]
            metrics_remove_uncensored_only[i_j, i_r] = [0, 0, 0, ibs_uncensored_only]

    with open('./results/1_selective_prediction/' + args.model + '_' + args.meta_model + '_' + args.dataset + '.pkl', 'wb') as f:
        pickle.dump({'metrics_remove_uncensored': metrics_remove_uncensored, 'metrics_remove_uncensored_only': metrics_remove_uncensored_only}, f)
