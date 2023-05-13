import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import pandas as pd
from multiprocessing.spawn import freeze_support



class Arguments:
    def __init__(self):
        # model setting
        self.is_training = 0
        self.train_only = False
        self.model_id = 'NLinear_test'
        self.model = 'NLinear'
        # data loader
        self.data = 'custom'
        self.root_path = r'dataset/all_six_datasets/bitcoin'
        self.data_path = 'from220330_1200to230330_1422_1hLTSF.csv'
        # features
        self.features = 'M'
        self.target = 'Close price'
        self.freq = 'h'
        self.checkpoints = './checkpoints/'
        self.seq_len = 48
        self.label_len = 24
        self.pred_len = 4
        self.lookback_len = 0
        # training config
        self.itr = 2
        self.train_epochs = 20
        self.num_workers = 10
        self.batch_size = 32
        self.patience = 3
        self.learning_rate = 0.0001
        self.loss = 'mse'

    def set_arguments(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Invalid argument: {key}")
            
    def get_parser(self):
        parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
        # model setting
        parser.add_argument('--is_training', type=int, default=self.is_training, help='status')
        parser.add_argument('--train_only', type=bool, default=self.train_only, help='perform training on full input dataset without validation and testing')
        parser.add_argument('--model_id', type=str, default=self.model_id, help='model id')
        parser.add_argument('--model', type=str, default=self.model, help='model name, options: [Autoformer, Informer, Transformer]')
        # data loader
        parser.add_argument('--data', type=str, default=self.data, help='dataset type')
        parser.add_argument('--root_path', type=str, default=self.root_path, help='root path of the data file')
        parser.add_argument('--data_path', type=str, default=self.data_path, help='data file')
        # features
        parser.add_argument('--features', type=str, default=self.features, help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        parser.add_argument('--target', type=str, default=self.target, help='target feature in S or MS task')
        parser.add_argument('--freq', type=str, default=self.freq, help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        parser.add_argument('--checkpoints', type=str, default=self.checkpoints, help='location of model checkpoints')
        parser.add_argument('--seq_len', type=int, default=self.seq_len, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=self.label_len, help='start token length')
        parser.add_argument('--pred_len', type=int, default=self.pred_len, help='prediction sequence length')
        parser.add_argument('--lookback_len', type=int, default=self.lookback_len, help='lookback length')
        # DLinear
        parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
        # Formers 
        parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
        parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
        parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=7, help='output size')
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
        parser.add_argument('--factor', type=int, default=1, help='attn factor')
        parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
        parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--do_predict', action='store_true', default = True, help='whether to predict unseen future data')
        # Optimization
        parser.add_argument('--num_workers', type=int, default=self.num_workers, help='data loader num workers')
        parser.add_argument('--itr', type=int, default=self.itr, help='experiments times')
        parser.add_argument('--train_epochs', type=int, default=self.train_epochs, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=self.batch_size, help='batch size of train input data')
        parser.add_argument('--patience', type=int, default=self.patience, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=self.learning_rate, help='optimizer learning rate')
        parser.add_argument('--des', type=str, default='test', help='exp description')
        parser.add_argument('--loss', type=str, default=self.loss, help='loss function')
        parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
        # GPU
        parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        parser.add_argument('--gpu', type=int, default=0, help='gpu')
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
        parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
        return parser.parse_args()


class Numpy3DArray():
    def __init__(self):
        self.array_list = []

    def append_2d_array(self, array_2d):
        self.array_list.append(array_2d)

    def to_3d_array(self):
        self.array_3d = np.stack(self.array_list, axis=0)
        print("3D NumPy array:\n", self.array_3d)

    def save_3d_array(self, filename):
        if self.array_3d is not None:
            np.save(filename, self.array_3d)
        else:
            print("No array to save. Please create an array first.")

    def load_3d_array(self, filename):
        self.array_3d = np.load(filename)
        print("Loaded 3D NumPy array:\n", self.array_3d)



class LTSF():
    def __init__(self):
        # 시드 및 Exp 설정
        freeze_support()
        self.set_seed()
        self.Exp = Exp_Main
        # args 선언
        args = Arguments()
        # --is_training default값 1일 경우 training, default값이 0일 경우 모델 load
        args.set_arguments(is_training=0, model_id='MY_REV0',
                           seq_len=48, pred_len = 4, lookback_len = 0)
        self.args = args.get_parser()
        # args 업데이트
        self.args.use_gpu = torch.cuda.is_available() and self.args.use_gpu
        if self.args.use_gpu and self.args.use_multi_gpu:
            self.args.devices = self.args.devices.replace(' ', '')
            device_ids = self.args.devices.split(',')
            self.args.device_ids = [int(id_) for id_ in device_ids]
            self.args.gpu = self.args.device_ids[0]
        print('Args in experiment:')
        print(self.args)

    def set_seed(self):
        seed=2021
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def main(self):
        if self.args.is_training:
            self.run_training()
        else:
            preds = self.run_predict_only()
        print('Args in experiment:')
        print(self.args)
        torch.cuda.empty_cache()
        return preds
        

    def run_training(self):
        for ii in range(self.args.itr):
            print("Train 모드입니다.")
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                self.args.model_id,
                self.args.model,
                self.args.data,
                self.args.features,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.d_model,
                self.args.n_heads,
                self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.factor,
                self.args.embed,
                self.args.distil,
                self.args.des, ii)

            exp = self.Exp(self.args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            if not self.args.train_only:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)

            if self.args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                preds = exp.predict(setting, True)
                return preds

    def run_predict_only(self):
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            self.args.model_id,
            self.args.model,
            self.args.data,
            self.args.features,
            self.args.seq_len,
            self.args.label_len,
            self.args.pred_len,
            self.args.d_model,
            self.args.n_heads,
            self.args.e_layers,
            self.args.d_layers,
            self.args.d_ff,
            self.args.factor,
            self.args.embed,
            self.args.distil,
            self.args.des, ii)

        exp = self.Exp(self.args)

        if self.args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            preds = exp.predict(setting, True)
            return preds
        else:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
    
    def predict_entire(self):
        sequences = Numpy3DArray()
        ground_truth = Numpy3DArray()
        prediction = Numpy3DArray()
        df_raw = pd.read_csv(os.path.join(self.args.root_path,
                                          self.args.data_path))
        df_len = len(df_raw)
        # iter = df_len - self.args.seq_len
        iter = 10
        for i in range(iter):
            print("여기는 진입 하려나?")
            # 현재 까지는 lookback이 문제로 보임
            lookback = df_len - i + self.args.seq_len
            self.args.lookback_len = lookback
            preds = self.main()
            sequences.append_2d_array(df_raw[0:i+self.args.seq_len])
            ground_truth.append_2d_array(df_raw[i+self.args.seq_len:i+self.args.seq_len+self.args.pred_len])
            prediction.append_2d_array(preds)
            print("sequences: ", df_raw[0:i+self.args.seq_len])
            print("ground_truth: ", df_raw[i+self.args.seq_len:i+self.args.seq_len+self.args.pred_len])
            print("prediction: ", preds)
        sequences.to_3d_array()
        ground_truth.to_3d_array()
        prediction.to_3d_array()
        sequences.save_3d_array(os.path.join(self.args.root_path, 'sequences.npy'))
        ground_truth.save_3d_array(os.path.join(self.args.root_path, 'ground_truth.npy'))
        prediction.save_3d_array(os.path.join(self.args.root_path, 'prediction.npy'))
        return sequences, ground_truth, prediction
    



if __name__ == '__main__':
    ltsf = LTSF()
    ltsf.predict_entire()
    # ltsf.main()