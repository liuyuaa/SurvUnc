import numpy as np
import pandas as pd
from torch import nn
import torch
import torchtuples as tt
from pycox.evaluation import EvalSurv
from pycox.models import DeepHitSingle, CoxPH
from sksurv.ensemble import RandomSurvivalForest
from dsm_ours import DeepSurvivalMachines
import pickle


class DeepHit(torch.nn.Module):
    def __init__(self, d, **kwargs):
        super(DeepHit, self).__init__()
        num_durations = 100
        labtrans = DeepHitSingle.label_transform(num_durations)
        self.y_train = labtrans.fit_transform(d.durations_train, d.events_train)
        self.y_val = labtrans.transform(d.durations_val, d.events_val)

        out_features = labtrans.out_features
        in_features, batch_norm = d.x_train.shape[1], kwargs['batch_norm']
        dropout, lr, batch_size = kwargs['dropout'], kwargs['lr'], kwargs['batch_size']
        num_nodes_shared = [3 * in_features, 5 * in_features, 3 * in_features]
        self.training, self.kwargs = kwargs['training'], kwargs

        net = CauseSpecificNet(in_features, num_nodes_shared, out_features, batch_norm, dropout)
        optimizer = tt.optim.AdamWR(lr=lr, decoupled_weight_decay=0.01, cycle_eta_multiplier=0.8)
        self.model = DeepHitSingle(net, optimizer, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)
        self.model.optimizer.set_lr(lr)

        if not self.training:
            self.model.load_net('./survival_model_weights/deephit_' + kwargs['dataset'] + '.pt',
                                map_location=torch.device('cpu'))

    def forward(self, d):
        if self.training:
            callbacks = [tt.callbacks.EarlyStopping(patience=10)]
            log = self.model.fit(d.x_train, self.y_train, self.kwargs['batch_size'], epochs=1000, callbacks=callbacks,
                                 verbose=False, val_data=(d.x_val, self.y_val))
            self.model.save_net('./survival_model_weights/deephit_' + self.kwargs['dataset'] + '.pt')

        x_test = d.x_test
        events_train, durations_train, events_test, durations_test = d.events_train, d.durations_train, d.events_test, d.durations_test
        time_grid = np.linspace(durations_test.min(), durations_test.max() - 1, 100)
        # pycox
        surv = self.model.interpolate(10).predict_surv_df(x_test)
        ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
        cindex_pycox = ev.concordance_td('antolini')
        ibs_pycox = ev.integrated_brier_score(time_grid)
        return cindex_pycox, ibs_pycox

    def predict_surv(self, x_test, d):
        surv = self.model.predict_surv_df(x_test)
        return surv

    def forward_eval(self, x_test, durations_test, events_test, d):
        events_train, durations_train = d.events_train, d.durations_train
        time_grid = np.linspace(durations_test.min(), durations_test.max() - 1, 100)
        # pycox
        surv = self.model.interpolate(10).predict_surv_df(x_test)
        ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
        cindex_pycox = ev.concordance_td('antolini')
        ibs_pycox = ev.integrated_brier_score(time_grid)
        surv = self.model.predict_surv_df(x_test)
        return cindex_pycox, ibs_pycox, surv

class DeepSurv(torch.nn.Module):
    def __init__(self, d, **kwargs):
        super(DeepSurv, self).__init__()
        self.y_train = (d.durations_train, d.events_train)
        self.y_val = (d.durations_val, d.events_val)

        out_features = 1
        in_features, batch_norm = d.x_train.shape[1], kwargs['batch_norm']
        dropout, lr, batch_size = kwargs['dropout'], kwargs['lr'], kwargs['batch_size']
        num_nodes_shared = [32]
        self.training, self.kwargs = kwargs['training'], kwargs

        net = CauseSpecificNet_Surv(in_features, num_nodes_shared, out_features, batch_norm, dropout)
        optimizer = tt.optim.AdamWR(lr=lr, decoupled_weight_decay=0.01, cycle_eta_multiplier=0.8)
        self.model = CoxPH(net, optimizer)
        self.model.optimizer.set_lr(lr)
        if not self.training:
            self.model.load_net('./survival_model_weights/deepsurv_' + kwargs['dataset'] + '.pt',
                                map_location=torch.device('cpu'))

    def forward(self, d):
        if self.training:
            callbacks = [tt.callbacks.EarlyStopping(patience=10)]
            log = self.model.fit(d.x_train, self.y_train, self.kwargs['batch_size'], epochs=1000, callbacks=callbacks,
                                 verbose=False, val_data=(d.x_val, self.y_val))
            self.model.save_net('./survival_model_weights/deepsurv_' + self.kwargs['dataset'] + '.pt')

        x_test = d.x_test
        events_train, durations_train, events_test, durations_test = d.events_train, d.durations_train, d.events_test, d.durations_test
        time_grid = np.linspace(durations_test.min(), durations_test.max() - 1, 100)

        _ = self.model.compute_baseline_hazards(d.x_train, self.y_train)
        surv = self.model.predict_surv_df(x_test)
        # pycox
        ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
        cindex_pycox = ev.concordance_td('antolini')
        ibs_pycox = ev.integrated_brier_score(time_grid)
        return cindex_pycox, ibs_pycox

    def predict_surv(self, x_test, d):
        _ = self.model.compute_baseline_hazards(d.x_train, self.y_train)
        surv = self.model.predict_surv_df(x_test)
        return surv

    def forward_eval(self, x_test, durations_test, events_test, d):
        events_train, durations_train = d.events_train, d.durations_train
        time_grid = np.linspace(durations_test.min(), durations_test.max() - 1, 100)

        _ = self.model.compute_baseline_hazards(d.x_train, self.y_train)
        surv = self.model.predict_surv_df(x_test)
        # pycox
        ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
        cindex_pycox = ev.concordance_td('antolini')
        ibs_pycox = ev.integrated_brier_score(time_grid)
        return cindex_pycox, ibs_pycox, surv

class RSF(torch.nn.Module):
    def __init__(self, d, **kwargs):
        super(RSF, self).__init__()
        self.y_train = (d.durations_train, d.events_train)
        self.y_val = (d.durations_val, d.events_val)
        self.training, self.kwargs = kwargs['training'], kwargs

        random_state = 20
        n_estimators, min_samples_split, min_samples_leaf = kwargs['n_estimators'], kwargs['min_samples_split'], kwargs[
            'min_samples_leaf']
        self.model = RandomSurvivalForest(n_estimators=n_estimators, min_samples_split=min_samples_split,
                                          min_samples_leaf=min_samples_leaf, n_jobs=-1, random_state=random_state)

        if not self.training:
            with open('./survival_model_weights/rsf_' + kwargs['dataset'] + '.pkl', 'rb') as file:
                self.model = pickle.load(file)

    def forward(self, d):
        events_train, durations_train, events_test, durations_test = d.events_train, d.durations_train, d.events_test, d.durations_test
        et_train = np.array([(events_train[i], durations_train[i]) for i in range(len(events_train))], dtype=[('e', bool), ('t', float)])
        time_grid = np.linspace(durations_test.min(), durations_test.max() - 1, 100)

        if self.training:
            self.model.fit(d.x_train, et_train)
            with open('./survival_model_weights/rsf_' + self.kwargs['dataset'] + '.pkl', 'wb') as file:
                pickle.dump(self.model, file)

        x_test = d.x_test
        surv_fns = self.model.predict_survival_function(x_test)
        t_set_train = sorted(list(set(durations_train)))
        surv = np.asarray([[fn(t) for t in t_set_train] for fn in surv_fns]).T
        surv = pd.DataFrame(surv, index=t_set_train)
        # pycox
        ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
        cindex_pycox = ev.concordance_td('antolini')
        ibs_pycox = ev.integrated_brier_score(time_grid)
        return cindex_pycox, ibs_pycox

    def predict_surv(self, x_test, d):
        surv_fns = self.model.predict_survival_function(x_test)
        t_set_train = sorted(list(set(d.durations_train)))
        surv = np.asarray([[fn(t) for t in t_set_train] for fn in surv_fns]).T
        surv = pd.DataFrame(surv, index=t_set_train)
        return surv

    def forward_eval(self, x_test, durations_test, events_test, d):
        events_train, durations_train = d.events_train, d.durations_train
        time_grid = np.linspace(durations_test.min(), durations_test.max() - 1, 100)

        surv_fns = self.model.predict_survival_function(x_test)
        t_set_train = sorted(list(set(durations_train)))
        surv = np.asarray([[fn(t) for t in t_set_train] for fn in surv_fns]).T
        surv = pd.DataFrame(surv, index=t_set_train)
        # pycox
        ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
        cindex_pycox = ev.concordance_td('antolini')
        ibs_pycox = ev.integrated_brier_score(time_grid)
        return cindex_pycox, ibs_pycox, surv


class DSM(torch.nn.Module):
    def __init__(self, d, **kwargs):
        super(DSM, self).__init__()
        self.y_train = (d.durations_train, d.events_train)
        self.y_val = (d.durations_val, d.events_val)
        self.training, self.kwargs = kwargs['training'], kwargs
        self.model = DeepSurvivalMachines(k=kwargs['k_dsm'], distribution=kwargs['distribution'], layers=kwargs['layers_dsm'])
        if not self.training:
            self.model = torch.load('./survival_model_weights/dsm_' + kwargs['dataset'] + '.pt')

    def forward(self, d):
        events_train, durations_train, events_test, durations_test = d.events_train, d.durations_train, d.events_test, d.durations_test
        time_grid = np.linspace(durations_test.min(), durations_test.max() - 1, 100)

        if self.training:
            self.model.fit(d.x_train, durations_train, events_train, iters=500, learning_rate=self.kwargs['lr'])
            torch.save(self.model, './survival_model_weights/dsm_' + self.kwargs['dataset'] + '.pt')

        x_test = d.x_test.astype(float)
        t_set_train = sorted(list(set(durations_train)))
        surv = self.model.predict_survival(x_test, t_set_train).T
        surv = pd.DataFrame(surv, index=t_set_train)
        # pycox
        ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
        cindex_pycox = ev.concordance_td('antolini')
        ibs_pycox = ev.integrated_brier_score(time_grid)
        return cindex_pycox, ibs_pycox

    def predict_surv(self, x_test, d):
        t_set_train = sorted(list(set(d.durations_train)))
        surv = self.model.predict_survival(x_test.astype(float), t_set_train).T
        surv = pd.DataFrame(surv, index=t_set_train)
        return surv

    def forward_eval(self, x_test, durations_test, events_test, d):
        events_train, durations_train = d.events_train, d.durations_train
        time_grid = np.linspace(durations_test.min(), durations_test.max() - 1, 100)

        x_test = x_test.astype(float)

        t_set_train = sorted(list(set(durations_train)))
        surv = self.model.predict_survival(x_test, t_set_train).T
        surv = pd.DataFrame(surv, index=t_set_train)
        # pycox
        ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
        cindex_pycox = ev.concordance_td('antolini')
        ibs_pycox = ev.integrated_brier_score(time_grid)
        return cindex_pycox, ibs_pycox, surv

class CauseSpecificNet(torch.nn.Module):
    """Network structure similar to the DeepHit paper, but without the residual connections (for simplicity)."""

    def __init__(self, in_features, num_nodes, out_features, batch_norm=True, dropout=None):
        super().__init__()
        net = []
        neurons_num = [in_features] + num_nodes
        for n_in, n_out in zip(neurons_num[:-1], neurons_num[1:]):
            net.append(nn.Linear(n_in, n_out, bias=True))
            net.append(nn.ReLU())
            if batch_norm == True:
                net.append(nn.BatchNorm1d(n_out))
            if dropout:
                net.append(nn.Dropout(dropout))
        net.append(nn.Linear(neurons_num[-1], out_features, bias=True))
        self.net = nn.Sequential(*net)

    def forward(self, input):
        return self.net(input)

class CauseSpecificNet_Surv(torch.nn.Module):
    """Network structure similar to the DeepHit paper, but without the residual connections (for simplicity)."""

    def __init__(self, in_features, num_nodes, out_features, batch_norm=True, dropout=None):
        super().__init__()
        net = []
        neurons_num = [in_features] + num_nodes
        for n_in, n_out in zip(neurons_num[:-1], neurons_num[1:]):
            net.append(nn.Linear(n_in, n_out, bias=True))
            net.append(nn.ReLU())
            if batch_norm == True:
                net.append(nn.BatchNorm1d(n_out))
            if dropout:
                net.append(nn.Dropout(dropout))
        net.append(nn.Linear(neurons_num[-1], out_features, bias=False))  ### For DeepSurvethe final output has no bias
        self.net = nn.Sequential(*net)

    def forward(self, input):
        return self.net(input)
