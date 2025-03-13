# coding=gbk
import warnings
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from torch import Tensor
from typing import Optional
from sklearn import metrics

from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

warnings.filterwarnings("ignore")

# Create argument namespace
args = SimpleNamespace(
    save_path="TIDER.pt",
    device="cuda",
    eta=1e-2,
    num_epochs=100,
    batch_size=32,
    dim_size=50,
    lag_list='list(range(1))',
    lambda_ar=0.2,
    bias_dimension=1,
    season_num=1,
    seasonality=1,
    learning_rate=0.05,
    lambda_trend=0.1,
    n_test = 4273
)


def evaluated_message(y_test: np.ndarray, y_pred: np.ndarray) -> str:
    rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
    mae  = metrics.mean_absolute_error(y_test, y_pred)
    msk_0=[]
    for i in range(len(y_pred)):
          if y_test[i]!=0:
            msk_0.append(i)
    y_test, y_pred = y_test[msk_0], y_pred[msk_0]
    abs_errors = np.abs(y_test - y_pred)
    mape = np.mean(abs_errors / y_test)
    msg = (f" RMSE: {rmse:7.4} MAE: {mae:7.4}"
           f" Max Error: {np.max(abs_errors):7.4} MAEP: {mape:7.4}")
    return msg



class TIDER(nn.Module):
    def __init__(self, n: int, t: int, hid_size: int,
                 bias_lag_list, bias_dim_, season_num_,
                 seasonality_):
        super(TIDER, self).__init__()
        self.s_embeddings = nn.Embedding(n, hid_size)
        # channel-wise matrix
        self.t_embeddings_trend = nn.Embedding(t, hid_size)
        # trend feature matrix
        self.t_embeddings_season = nn.Embedding(2*season_num_, hid_size)
        # seasonality feature matrix

        self.bias_dim = bias_dim_
        # Db
        self.bias_t = nn.Embedding(t, self.bias_dim)
        # bias feature matrix
        self.hidden_size = hid_size
        # dimension for feature matrix
        self.seasonal = seasonality_
        # seasonality
        self.season_num = season_num_
        # degree of Fourier Series

        self.dims = (n, t)
        # num of channels, temporal length

        self.param=torch.nn.Parameter(torch.zeros(2))
        # adaptive weight for trend and seasonality

        self.modlst = torch.nn.ModuleList([])
        # Autoregressive Linears for bias feature matrix
        self.lag_list = bias_lag_list
        # Lag list for bias matrix
        self.len_lag = len(bias_lag_list)

        for _ in range(self.len_lag):
            self.modlst.append(torch.nn.Linear(self.bias_dim, self.bias_dim, bias=False))
        # one autoregressive matrix corresponds to one time lag

    def _forward(self, roads, times):
        """
        roads (batch_size,): road link ids.
        times (batch_size,): time steps.
        """
        ms = self.getu(roads)
        # ms:(batch_size, dim_size+bias_dim)

        mt = self.getv(times)
        # mt: (T,dim_size+bias_dim)

        return torch.mm(ms, mt.t())
        # (batch_size, T)

    def forward(self, roads):
        """
        roads (batch_size, ): road link ids.
        """
        t = self.dims[1]
        # total time
        times = torch.arange(t, dtype=torch.long, device=roads.device)  # Ensure correct length
        # times:[t]
        return self._forward(roads, times)
        # ret: (batch_size, T)

    def infer(self, n_step):
        """
        Infer the values of last n-step.
        """
        n, t = self.dims
        device_ = self.get_device

        roads = torch.arange(n).to(device_)
        # [N]
        times = torch.arange(t-n_step, t).to(args.device)
        # [n_step]
        # [n_step]


        return self._forward(roads, times)

    def bias_loss(self, times):
        """
        times (batch_size,): time steps.
        maxlag: int
        """

        mt_bias = self.bias_t(times.to(args.device).long())  # Ensure times is long
        # mt_bias:(T,bias_dim)

        max_len = int(max(self.lag_list))

        t = mt_bias.shape[0]
        if isinstance(t, int):
            pass
        else:
            t = int(t)

        tensor_len = t-max_len

        lst_lag = []

        for i in self.lag_list:
            lst_lag.append(mt_bias[max_len-i:max_len-i+tensor_len].clone())
            # for every time lag，we clone one corresponding part for autoregression
            # for example, if temporal length is 50  and time_lag is [1,3,5]
            # then the sequences we clone are (4.49),(2,47),(0,45)，which are all elements effecting (5,50)

        ret_bias_origin = mt_bias[max_len:max_len+tensor_len].clone()

        ret_var = self.modlst[0](lst_lag[0])
        for i in range(1, self.len_lag):
            ret_var = ret_var+self.modlst[i](lst_lag[i])

        return ret_var-ret_bias_origin
        # makes bias feature matrix follow a certain autoregression relationship

    @property
    def get_device(self):
        return self.s_embeddings.weight.device

    def getu(self, roads):
        device_ = self.get_device
        roads = roads.to(device_).long()  # Ensure roads are on the correct device

        ms = self.s_embeddings(roads)
        # ms:(batch_size, dim_size)
        s_batch_size = ms.shape[0]
        # batch_size
        ones_s = torch.ones((s_batch_size, self.bias_dim)).to(device_)

        # Ensure both tensors have the same shape before concatenation
        ms = ms.squeeze(-1) if ms.dim() > 2 else ms
        ones_s = ones_s.squeeze(-1) if ones_s.dim() > 2 else ones_s

        print(f"\t\t\t\tShape of ms (after fix): {ms.shape}")
        print(f"\t\t\t\tShape of ones_s (after fix): {ones_s.shape}")

        # ones_s: (batch_size,bias_dim)
        ms = torch.cat((ms, ones_s), 1).to(device_)
        # ms:(batch_size, dim_size+bias_dim)

        return ms

    def getv(self, times):
        device_ = self.get_device
        times = times.to(device_)  # Ensure it's on the same device

        mt_trend = self.getv_trend(times)
        # mt_trend:(T,dim_size+bias_dim)
        mt_season = self.getv_season(times)
        # mt_season :(T,dim_size+bias_dim)

        softmax_res = torch.softmax(self.param, dim=-1)

        mt = softmax_res[0] * mt_trend + softmax_res[1] * mt_season
        # mt: (T, dim_size)

        mt_bias = self.bias_t(times)
        # mt_bias:(T,bias_dim)

        mt = torch.cat((mt, mt_bias), 1).to(device_)
        # mt: (T,dim_size+bias_dim)

        return mt



    def getv_trend(self, times):
        # Ensure times is a long tensor
        device_ = self.get_device
        times = times.to(device_).long()  # Convert times to long
        mt_trend = self.t_embeddings_trend(times)

        return mt_trend

    def getv_season(self, times):
        # get v'season matrix
        device_ = self.get_device
        ret = torch.zeros(times.shape[0], self.hidden_size).to(device_)
        # [T,dim_size]

        for i in range(self.season_num):
            sin_t, cos_t = self.calc_x_vec(times, i)
            # [T], [T]

            sin_t = sin_t.detach()
            cos_t = cos_t.detach()

            sin_t=sin_t.to(device_)
            cos_t=cos_t.to(device_)

            emb_a = self.t_embeddings_season(torch.LongTensor([i * 2]).to(device_))
            emb_b = self.t_embeddings_season(torch.LongTensor([i * 2 + 1]).to(device_))

            ret = ret+torch.einsum('i,zj->ji',
                                   sin_t,
                                   emb_a).t()
            # ret[T,dim_size]

            ret = ret+torch.einsum('i,zj->ji',
                                   cos_t,
                                   emb_b).t()

        mt_season = ret
        # mt_season:(T, dim_size)
        return mt_season

    def calc_x_vec(self, times, n):
        # vector sin(ωt) and cos(ωt) for seasonality feature matrix
        sin = torch.sin((n*2*np.pi/self.seasonal)*times)
        # sin(n*2Π/T*)
        cos = torch.cos((n*2*np.pi/self.seasonal)*times)
        # cos(n*2Π/T*)

        return sin, cos
        # [T],[T]


def obsloss(model, x, roads):
    """
    X (N, T)
    roads (batch_size,)
    """
    x_ = x[roads]
    # (batch_size, T)
    device_ = model.get_device

    x_, roads = x_.to(device_), roads.to(device_)

    xhat = model(roads)
    # (batch_size, T)
    return obsLossF(xhat, x_)


def spatialloss(model: nn.Module,
                l_: torch.sparse.FloatTensor):
    """
    Laplacian constraint loss Tr(U^T L U).

    Args
        model: MF instance.
        L (N, N): Graph laplacian matrix.
    """
    n, device_ = model.dims[0], model.get_device
    u = model.getu(torch.arange(n).to(device_))
    return torch.einsum('dn,nd->', u.T, l_ @ u) / n


def l2loss(model: nn.Module,
           eta_1: float,
           eta_2: float):
    n, t = model.dims
    device_ = model.get_device
    u = model.getu(torch.arange(n).to(device_))
    v = model.getv(torch.arange(t).to(device_))
    return eta_1*torch.linalg.norm(u) + eta_2*torch.linalg.norm(v)


def arloss_bias(model, lambda_ar_):
    device_ = model.get_device
    n, t = model.dims
    times = torch.arange(t).to(device_)
    y = model.bias_loss(times)
    loss = lambda_ar_*torch.linalg.norm(y)
    return loss


def trend_loss(model, lambda_trend_):
    device_ = model.get_device
    n, t = model.dims
    times = torch.arange(t).to(device_)
    trend = model.t_embeddings_trend(times)

    loss = trend[:, 1:]-trend[:, :-1]
    loss = torch.linalg.norm(loss)

    return lambda_trend_*loss


def train(
    optimizer: torch.optim.Optimizer,
    num_epochs_: int,
    batch_size_: int,
    model: nn.Module,
    x: Tensor,
    x_val_:Tensor,
    eta_: float,
    verbose_: bool,
    lambda_ar_,
    n_test_,
    lambda_trend_,
    x_unobs_
):
    def train_step(i, roads,x,flag_train):


        if eta_ > 0:
            l2_loss = l2loss(model, eta_, eta_)
        else:
            l2_loss = 0.0
        # L2 regression

        if lambda_ar_ > 0:
            loss_bias = arloss_bias(model, lambda_ar_)
        else:
            loss_bias = 0.0
        # loss func for bias feature matrix

        if lambda_trend_ > 0:
            loss_trend = trend_loss(model, lambda_trend_)
        else:
            loss_trend = 0.0
        #loss func for trend feature matrix

        loss = obsloss(model, x, roads)  + l2_loss + loss_bias + loss_trend
        #print(obsloss(model, x, roads)  ,l2_loss , loss_bias , loss_trend ,loss)

        if(flag_train):
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        return loss.item()

    def train_epoch(idx_,x,flag_train):
        model.train()
        np.random.shuffle(idx_)
        # suffle channels
        n, running_losses = len(idx_), []
        for st_idx in range(0, n, batch_size_):
            # randomly choose one batch for training
            end_idx = min(st_idx + batch_size_, n)
            roads = torch.LongTensor(idx_[st_idx:end_idx])
            # [batch_size_]
            loss_this=train_step(st_idx//batch_size_, roads,x,flag_train)
            running_losses.append(loss_this)

        return np.nanmean(running_losses)

    idx = np.arange(x.shape[0])

    min_val_loss=np.inf

    for epoch in range(num_epochs_):
        running_loss = train_epoch(idx,x,True)


        val_loss=train_epoch(idx,x_val_,False)
        if(val_loss <min_val_loss):
          min_val_loss=val_loss
          min_epo=epoch
          torch.save(model.state_dict(),args.save_path)
    print('\t\t\t\tmin epo: ',min_epo)

@torch.no_grad()
def model_test(model, x_unobs, n_test):

    model.eval()

    xhat = model.infer(n_test)



    x = x_unobs[:, -n_test:].to(xhat.device)
    mask = torch.logical_not(x.isnan())


    y_test_, y_pred_ = x[mask], xhat[mask]


    return y_test_.cpu().numpy(), y_pred_.cpu().numpy()


def run_imputation(model, x_unobs):
    """
    Runs the imputation model and returns the complete imputed matrix.

    Args:
        model (TIDER): Trained imputation model.
        x_unobs (Tensor): Input matrix with missing values.

    Returns:
        np.ndarray: Imputed matrix.
    """
    model.eval()
    imputed_matrix = model(x_unobs)  # Directly pass input to model
    return imputed_matrix.cpu().detach().numpy()



def one_snapshot(
    x_obs_: Tensor,
    x_val: Tensor,
    x_unobs_: Tensor,
    n_test_,
    num_epochs_: int,
    batch_size_: int,
    dim_size_: int,
    lag_list_,
    lambda_ar_,
    bias_dim_,
    season_num_,
    seasonality_,
    lr_,
    lambda_trend_,
    device_: torch.device,
    eta_: Optional[float] = 0.0,
    verbose: Optional[bool] = False
):


    model = TIDER(x_obs_.shape[0],
               x_obs_.shape[1],
               dim_size_,
               lag_list_,
               bias_dim_,
               season_num_,
               seasonality_).to(device_)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_)

    train(optimizer,
          num_epochs_,
          batch_size_,
          model,
          x_obs_,
          x_val,
          eta_,
          verbose,
          lambda_ar_,
          n_test_,
          lambda_trend_,
          x_unobs_)

    model = TIDER(x_obs_.shape[0],
               x_obs_.shape[1],
               dim_size_,
               lag_list_,
               bias_dim_,
               season_num_,
               seasonality_).to(device_)
    model.load_state_dict(torch.load(args.save_path))

    softmax_re=torch.softmax(model.param,dim=-1)
    weight_0=softmax_re[0].item()
    weight_1=softmax_re[1].item()
    print("\tweights : ", weight_0, " - ", weight_1)

    return run_imputation(model, x_unobs_)


def update_obs(x_obs_, x_unobs_):
    """
    Merge the observation in `X_unobs` into `X_obs` except the latest time step.
    """
    observed = torch.logical_not(torch.isnan(x_unobs_))
    observed[:, -1] = False
    x_obs_[observed] = x_unobs_[observed]
    return x_obs_

def obsLossF(Xhat: Tensor, X: Tensor):
    """
    Xhat (n, m): Inferred matrix.
    X (n, m): Partially observed matrix with nan indicating missing values.
    """
    mask = torch.logical_not(X.isnan())
    return f.mse_loss(Xhat[mask], X[mask])



def recoveryTIDER(miss_data, eta=0.01, batch_size=32, epochs=5, device="cuda", lr=0.05):

    x = np.copy(miss_data)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    args.eta = eta
    args.batch_size = batch_size
    args.num_epochs = epochs
    args.device = device
    args.learning_rate = lr
    args.dim_size = miss_data.shape[0]
    args.bias_dimension = miss_data.shape[1]

    print("\n\n\t\t\t(PYTHON) TIDER: Matrix Shape: (", x.shape[0], ", ", x.shape[1], ") for",
          " batch_size ", args.batch_size, " eta ", args.eta, " lr ", lr, " epochs ", args.num_epochs ,
          ", device ", args.device,  " bias_dimension ", args.bias_dimension, ", and ", args.dim_size,
          "=================================================\n\n ")

    trn=np.copy(x)
    val=np.copy(x)
    tst=np.copy(x)

    X_obs = torch.FloatTensor(trn).to(device)
    X_val = torch.FloatTensor(val).to(device)
    X_unobs = torch.FloatTensor(tst).to(device)


    eta = args.eta
    lrate = args.learning_rate
    bias_dim = args.bias_dimension
    lambda_trend = args.lambda_trend
    season_num = args.season_num
    seasonality = args.seasonality
    n_test = args.n_test
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    dim_size = args.dim_size
    lag_list = eval(args.lag_list)
    lambda_ar = args.lambda_ar

    imputed_matrix  = one_snapshot(
            X_obs,
            X_val,
            X_unobs,
            n_test,
            num_epochs,
            batch_size,
            dim_size,
            lag_list,
            lambda_ar,
            bias_dim,
            season_num,
            seasonality,
            lrate,
            lambda_trend,
            device,
            eta
        )

    print("? Imputation Complete. Imputed Matrix:")
    print(imputed_matrix)


    print(args)
    print('\n'*10)

    return imputed_matrix


if __name__ == "__main__":
    ts_1 = TimeSeries()

    # 2. load the timeseries from file or from the code
    ts_1.load_series(utils.search_path("eeg-alcohol"), nbr_val=64)  # shape 64x256
    ts_1.normalize(normalizer="min_max")

    # 3. contamination of the data
    x = ts_1.Contamination.missing_completely_at_random(ts_1.data)

    imputation = recoveryTIDER(x)
