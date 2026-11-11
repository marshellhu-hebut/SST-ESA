import csv
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace
import numpy as np
import torch as tr
import torch.nn as nn
import torch.optim as optim
from data_loader import CMPDataIter_graph
from model import ARCHITECTURES, FIXED_SEEDS, SPLIT_SEED, build_model

class Train:

    def __init__(self, args, data_dir, device):
        self.args = args
        self.device = tr.device(device)
        random.seed(SPLIT_SEED)
        data_iter = CMPDataIter_graph(str(data_dir), data_set=args.dataset, max_rul=args.max_rul, seq_len=args.patch_size, time_denpen_len=args.time_denpen_len, window_sample=args.window_sample, net_name=1)
        for target, source in [('train', 'out'), ('val', 'cross_val'), ('test', 'test')]:
            setattr(self, target + '_data', self.cuda_(getattr(data_iter, source + '_x')))
            setattr(self, target + '_ops', self.cuda_(getattr(data_iter, source + '_ops')))
            setattr(self, target + '_label', self.cuda_(getattr(data_iter, source + '_y')))
            data, ops = self.data_preprocess_transpose(getattr(self, target + '_data'), getattr(self, target + '_ops'))
            setattr(self, target + '_data', data)
            setattr(self, target + '_ops', ops)
        self.net = build_model(args.dataset, device=device)
        self.loss_function = nn.MSELoss()
        self.optim = optim.Adam(self.net.parameters(), lr=args.lr)

    def Train_batch(self):
        self.net.train()
        loss_ = 0
        batch_size = self.args.batch_size
        iter = int(self.train_data.size(0) / batch_size)
        remain = self.train_data.size(0) - iter * batch_size
        for i in range(iter):
            data = self.train_data[i * batch_size:(i + 1) * batch_size]
            label = self.train_label[i * batch_size:(i + 1) * batch_size]
            self.optim.zero_grad()
            prediction = self.net(data)
            loss = self.loss_function(prediction, label)
            loss.backward()
            self.optim.step()
            loss_ = loss_ + loss.item()
        if remain != 0:
            data = self.train_data[-remain:]
            label = self.train_label[-remain:]
            self.optim.zero_grad()
            prediction = self.net(data)
            loss = self.loss_function(prediction, label)
            loss.backward()
            self.optim.step()
            loss_ = loss_ + loss.item()
        return loss_

    def Train_model(self):
        epoch = self.args.epoch
        cross_loss = np.inf
        test_RMSE = []
        test_score = []
        RUL_predicted = []
        RUL_real = []
        train_RMSE = []
        train_score = []
        train_RUL_predicted = []
        train_RUL_real = []
        for i in range(epoch):
            loss = self.Train_batch()
            if i % self.args.show_interval == 0:
                loss_val = self.Cross_validation()
                if loss_val < cross_loss:
                    cross_loss = loss_val
                    test_RMSE_, test_score_, test_result_predicted, test_result_real = self.Prediction()
                    train_RMSE_, train_score_, train_result_predicted, train_result_real = self.Prediction_training()
                    print('In the {}th epoch, TESTING RMSE is {}, TESTING Score is {}'.format(i, test_RMSE_, test_score_))
                    test_RMSE.append(test_RMSE_)
                    test_score.append(test_score_[0])
                    RUL_predicted.append(test_result_predicted)
                    RUL_real.append(test_result_real)
                    train_RMSE.append(train_RMSE_)
                    train_score.append(train_score_[0])
                    train_RUL_predicted.append(train_result_predicted)
                    train_RUL_real.append(train_result_real)
        index_best_test = np.argmin(test_RMSE)
        predicted_RUL = RUL_predicted[index_best_test]
        real_RUL = RUL_real[index_best_test]
        index_best_train = np.argmin(train_RMSE)
        train_predicted_RUL = train_RUL_predicted[index_best_train]
        train_real_RUL = train_RUL_real[index_best_train]
        print(test_RMSE[index_best_test])
        print(test_score[index_best_test])
        test_RMSE = np.stack(test_RMSE, 0)
        test_score = np.stack(test_score, 0)
        test_results = np.stack([test_RMSE, test_score], 0)
        np.save('./experiment/{}.npy'.format(self.args.save_name), test_results)

    def cuda_(self, x):
        return tr.as_tensor(np.asarray(x), dtype=tr.float32, device=self.device)

    def data_preprocess_transpose(self, data, ops):
        data = tr.transpose(data, 2, 3)
        ops = tr.transpose(ops, 2, 3)
        return (data, ops)

    def Cross_validation(self):
        self.net.eval()
        loss_ = 0
        batch_size = self.args.batch_size
        iter = int(self.val_data.size(0) / batch_size)
        remain = self.val_data.size(0) - iter * batch_size
        for i in range(iter):
            data = self.val_data[i * batch_size:(i + 1) * batch_size]
            label = self.val_label[i * batch_size:(i + 1) * batch_size]
            prediction = self.net(data)
            loss = self.loss_function(prediction, label)
            loss_ = loss_ + loss.item()
        if remain != 0:
            data = self.val_data[-remain:]
            label = self.val_label[-remain:]
            prediction = self.net(data)
            loss = self.loss_function(prediction, label)
            loss_ = loss_ + loss.item()
        return loss_

    def Prediction(self):
        self.net.eval()
        prediction = self.net(self.test_data)
        predicted_RUL = prediction
        real_RUL = self.test_label
        MSE = self.loss_function(predicted_RUL, real_RUL)
        RMSE = tr.sqrt(MSE) * self.args.max_rul
        score = self.scoring_function(predicted_RUL, real_RUL)
        return (RMSE.detach().cpu().numpy(), score.detach().cpu().numpy(), predicted_RUL.detach().cpu().numpy(), real_RUL.detach().cpu().numpy())

    def Prediction_training(self):
        self.net.eval()
        sample_idx = random.sample(range(len(self.train_data)), 100)
        train_data_sample = self.train_data[sample_idx]
        train_label_sample = self.train_label[sample_idx]
        prediction = self.net(train_data_sample)
        MSE = self.loss_function(prediction, train_label_sample)
        RMSE = tr.sqrt(MSE) * self.args.max_rul
        score = self.scoring_function(prediction, train_label_sample)
        return (RMSE.detach().cpu().numpy(), score.detach().cpu().numpy(), prediction.detach().cpu().numpy(), train_label_sample.detach().cpu().numpy())

    def scoring_function(self, predicted, real):
        score = 0
        num = predicted.size(0)
        for i in range(num):
            if real[i] > predicted[i]:
                score = score + (tr.exp((real[i] * self.args.max_rul - predicted[i] * self.args.max_rul) / 13) - 1)
            elif real[i] <= predicted[i]:
                score = score + (tr.exp((predicted[i] * self.args.max_rul - real[i] * self.args.max_rul) / 10) - 1)
        return score

def make_args(dataset):
    value = ARCHITECTURES[dataset]
    return SimpleNamespace(dataset=dataset, epoch=41, batch_size=100, lr=0.001, max_rul=125, show_interval=1, patch_size=value['input_patch'], time_denpen_len=value['patch_count'], window_sample=value['input_patch'] * value['patch_count'], save_name='source_test_records')

class Trainer(Train):

    def __init__(self, dataset, data_dir, output, device):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / 'experiment').mkdir()
        self.epoch_index = -1
        self.history = []
        self.candidates = []
        self.best_validation = float('inf')
        self.best_test = float('inf')
        super().__init__(make_args(dataset), data_dir, device)

    def Train_batch(self):
        self.epoch_index += 1
        self.started_epoch = time.perf_counter()
        self.train_loss = super().Train_batch()
        return self.train_loss

    def Cross_validation(self):
        with tr.no_grad():
            value = super().Cross_validation()
        self.history.append(dict(epoch=self.epoch_index, train_sum_batch_mse=self.train_loss, validation_sum_batch_mse=value, epoch_seconds=time.perf_counter() - self.started_epoch))
        write_csv(self.output / 'history.csv', self.history)
        print(f'{self.args.dataset} {self.epoch_index + 1}/{self.args.epoch} {self.train_loss:.8f} {value:.8f}', flush=True)
        return value

    def Prediction(self):
        with tr.no_grad():
            value = super().Prediction()
        rmse, score, predicted, truth = value
        record = dict(epoch=self.epoch_index, test_rmse=float(rmse), test_nasa_score=float(np.asarray(score).reshape(-1)[0]), validation_sum_batch_mse=self.history[-1]['validation_sum_batch_mse'])
        state = dict(dataset=self.args.dataset, variant='compact_esa', seed=FIXED_SEEDS[self.args.dataset], split_seed=SPLIT_SEED, epoch=self.epoch_index, model=self.net.state_dict())
        tr.save(state, self.output / 'validation_selected.pt')
        if float(rmse) < self.best_test:
            self.best_test = float(rmse)
            tr.save(state, self.output / 'original_test_selected.pt')
            write_predictions(self.output / 'original_test_selected_predictions.csv', predicted, truth)
        write_predictions(self.output / 'validation_selected_predictions.csv', predicted, truth)
        self.candidates.append(record)
        write_csv(self.output / 'test_candidates.csv', self.candidates)
        return value

    def run(self):
        previous = Path.cwd()
        try:
            os.chdir(self.output)
            self.Train_model()
        finally:
            os.chdir(previous)
        chosen = min(self.candidates, key=lambda value: value['test_rmse'])
        result = dict(dataset=self.args.dataset, seed=FIXED_SEEDS[self.args.dataset], split_seed=SPLIT_SEED, epochs=self.args.epoch, parameters=sum((p.numel() for p in self.net.parameters())), original_test_selected=chosen, validation_selected=self.candidates[-1])
        (self.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        return result

def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def write_predictions(path, predicted, truth):
    predicted = np.asarray(predicted).reshape(-1) * 125
    truth = np.asarray(truth).reshape(-1) * 125
    with Path(path).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['engine', 'predicted_rul', 'true_rul'])
        writer.writerows(((i + 1, float(p), float(y)) for i, (p, y) in enumerate(zip(predicted, truth))))
