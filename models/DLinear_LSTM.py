import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import numpy as np
import pywt
from statsmodels.tsa.stattools import adfuller
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

from scipy.stats import kurtosis, normaltest

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean








class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.count = 1
        self.count_1 = 1
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        self.LSTM_Seasonal_1 = nn.LSTM(input_size=11, hidden_size=64, batch_first=True,dropout=0.2)
        self.LSTM_Seasonal_2 = nn.LSTM(input_size=64, hidden_size=11, batch_first=True,dropout=0.2)
        # self.LSTM_Seasonal_5 = nn.LSTM(input_size=64, hidden_size=11, batch_first=True, dropout=0.2)
        self.LSTM_Seasonal_3 = nn.LSTM(input_size=11, hidden_size=64, batch_first=True, dropout=0.2)
        self.LSTM_Seasonal_4 = nn.LSTM(input_size=64, hidden_size=11, batch_first=True, dropout=0.2)
        # self.LSTM_Seasonal_6 = nn.LSTM(input_size=64, hidden_size=11, batch_first=True, dropout=0.2)
        # Attention Layer for Seasonal Component
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_1 = nn.Linear(64, 11)
        self.Linear_2 = nn.Linear(64, 11)
        self.Linear_3 = nn.Linear(128, 11)
        self.Linear_4 = nn.Linear(self.seq_len, self.pred_len)
        # Use this two lines if you want to visualize the weights
        # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # # ACF
        # if self.count_1 == 1:
        #     data = x[0, :, -1].cpu().numpy()
        #     df = pd.DataFrame(data, columns=["o3"])
        #     df.to_csv("dataset/origin_o3_data.csv", index=False)
        #     self.count_1 = self.count_1+1
        seasonal_init, trend_init = self.decompsition(x)
        # # ACF
        # if self.count == 1:
        #     data = seasonal_init[0, :, -1].cpu().numpy()
        #     df = pd.DataFrame(data, columns=["o3"])
        #     df.to_csv("dataset/o3_data.csv", index=False)
        #     self.count = self.count+1


        # plt.figure(figsize=(12, 6))
        # plt.plot(trend_init[0, :, -1].cpu().numpy(), color='blue')
        # plt.savefig("draw/output_plot.png")
        # plt.show()
        # noise_result = self.detector.detect_noise(trend_init)
        # print("检测结果：", noise_result)
        seasonal_init, (h_1, c_1) = self.LSTM_Seasonal_1(seasonal_init)
        seasonal_init, (h_1, c_1) = self.LSTM_Seasonal_2(seasonal_init)
        # seasonal_init, (h_1, c_1) = self.LSTM_Seasonal_5(seasonal_init)
        # seasonal_init = self.Linear_1(seasonal_init)
        trend_init, (h_2, c_2) = self.LSTM_Seasonal_3(trend_init)
        trend_init, (h_2, c_2) = self.LSTM_Seasonal_4(trend_init)
        # trend_init, (h_2, c_2) = self.LSTM_Seasonal_6(trend_init)
        # trend_init = self.Linear_2(trend_init)
        #
        #
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)  # to [Batch, Output length, Channel]
        # x1 = h_1[-1].unsqueeze(1)
        # x2 = h_2[-1].unsqueeze(1)
        #
        # x = self.Linear_1(x1) + self.Linear_2(x2)


        # print(x.shape)
        #print(target.shape)

        # x, (h_n, c_n) = self.LSTM_Seasonal_1(x)
        # x, (h_n, c_n) = self.LSTM_Seasonal_2(x)


        # x = h_n[-1].unsqueeze(1)
        # x = self.Linear_1(x)
        # x = x.permute(0, 2, 1)
        # x = self.Linear_4(x)
        # x = x.permute(0, 2, 1)
        # print(x.shape)

        return x
