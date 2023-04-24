import torch
import torch.nn as nn
from einops import rearrange # for patching

class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len

        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        batch, ts_len, ts_dim = x.shape

        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len = self.seg_len)
        x_embed = self.linear(x_segment)
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b = batch, d = ts_dim)
        
        return x_embed


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
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.active = nn.LeakyReLU()

        self.BN_seasonal = nn.BatchNorm1d(7)
        self.Linear_Seasonal = nn.Linear(self.seq_len, 2*self.seq_len)
        self.Linear_Seasonal2 = nn.Linear(2*self.seq_len, 2*self.seq_len)
        self.Linear_Seasonal3 = nn.Linear(2*self.seq_len, self.pred_len)
        
        self.BN_Trend = nn.BatchNorm1d(7)
        self.Linear_Trend = nn.Linear(self.seq_len, 2*self.seq_len)
        self.Linear_Trend2 = nn.Linear(2*self.seq_len, 2*self.seq_len)
        self.Linear_Trend3 = nn.Linear(2*self.seq_len, self.pred_len)

        self.dropout = nn.Dropout()
        # Use this two lines if you want to visualize the weights
        # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x, x_mark, y, y_mark):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
   
        seasonal_init
        seasonal_output_inter = self.active(self.Linear_Seasonal(seasonal_init))
        seasonal_output = self.active(self.Linear_Seasonal2(seasonal_output_inter))
        seasonal_output = seasonal_output + self.dropout(seasonal_output_inter)
        seasonal_output = self.BN_seasonal(seasonal_output)
        seasonal_output = self.Linear_Seasonal3(seasonal_output)

        trend_output_inter = self.active(self.Linear_Trend(trend_init))
        trend_output = self.active(self.Linear_Trend2(trend_output_inter))
        trend_output = trend_output + self.dropout(trend_output_inter)
        trend_output = self.BN_Trend(trend_output)
        trend_output = self.Linear_Trend3(trend_output)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]
