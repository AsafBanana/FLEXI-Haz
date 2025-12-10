#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tensorflow.keras import layers, Model, Input
import pandas as pd
import functools

from sklearn.preprocessing import StandardScaler
import scipy
import tensorflow as tf
import os
from scipy.stats import norm
from lifelines import CoxPHFitter
from tensorflow.keras.initializers import Constant
from scipy.optimize import minimize
import numpy as np
import itertools
from tensorflow.keras.optimizers import AdamW, SGD, Adam
import pickle

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Add, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import pickle


from sklearn.model_selection import GroupKFold
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


# In[2]:


class SimStudyNonLinearPH():

    def __init__(self, h0=0.02, right_c=30., c0=30., surv_grid=None):
        self.h0 = h0
        self.right_c = right_c
        self.c0 = c0
        self.surv_grid = surv_grid

    def simulate(self, n, surv_df=False):
        # sample covariates
        X, Z = self.sample_covs(n)
        X = X.astype(np.float64)
        Z = Z.astype(np.float64)

        v = np.random.exponential(size=(n, 1))  # (n,1) so inv_cum_hazard is happyz
        t = self.inv_cum_hazard(v, (X, Z)).reshape(-1)   # event time (n,)
        c = (self.c0 * np.random.exponential(size=n)).astype(np.float64)  # (n,)
        tt = np.minimum(t, c)
        tt = np.minimum(tt, self.right_c)
        d = (t <= np.minimum(c, self.right_c)).astype(np.int32)

        surv_df_out = self.surv_df((X, Z), self.surv_grid) if surv_df else None

        return dict(
            covs_X=X,
            covs_Z=Z,
            durations=tt,                  # observed times (n,)
            events=d,                      # 0/1 (n,)
            surv_df=surv_df_out,
            durations_true=t,             # true event times (n,)
            events_true=np.ones_like(t, dtype=np.int32),
            censor_durations=c,
            censor_events=np.ones_like(c, dtype=np.int32),
        )


    @staticmethod
    def sample_covs(n):
        return np.random.uniform(-1, 1, size=(n, 3)),np.random.uniform(-1, 1, size=(n, 2))

    @staticmethod
    def g(covs):
        x,z = covs
        x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
        beta = 2/3
        linear = SimStudyLinearPH.g(x)
        nonlinear =  2*beta * (x0**2 + x2**2 + x0*x1 + x0*x2 + x1*x2) + 2*z[:,0] - 1*z[:,1]
        return nonlinear


# In[3]:


class SimStudyNonLinearNonPH(SimStudyNonLinearPH):
    '''Survival simulations study for non-linear non-prop. hazard model.
        h(t | x) = h0 * exp[g(t, x)], 
        with constant h_0, and g(t, x) = a(x) + b(x)*t.
        Cumulative hazard:
        H(t | x) = h0 / b(x) * exp[a(x)] * (exp[b(x) * t] - 1)
        Inverse:
        H^{-1}(v, x) = 1/b(x) log{1 +  v * b(x) / h0 exp[-a(x)]}
    Parameters:
        h0: Is baseline constant.
        right_c: Time for right censoring.
    '''
    def __init__(self, h0=0.1, right_c=10., c0=30., surv_grid=None):
        super().__init__(h0, right_c, c0, surv_grid)

    @staticmethod
    def a(x,z):
        x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
        #return 2*z[:,0] - 1*z[:,1] - 2*x0 + 3*x1 +4*x2
        return 2*z[:,0] - 1*z[:,1]

    @staticmethod
    def b(x):
        x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
        return (0.1 + (0.2 * (x0 + x1) + 0.5 * x0 * x1 + x2**2)**2)*1

    @staticmethod
    def g(t, covs):
        x,z = covs
        return SimStudyNonLinearNonPH.a(x,z) + SimStudyNonLinearNonPH.b(x) * t.reshape(-1, )

    def g_real(self,X,Z,T):
        return  SimStudyNonLinearNonPH.g(T,(X,Z)) + np.log(self.h0)

    def g_real_grad(self,X,Z,T):
        return  SimStudyNonLinearNonPH.b(X)

    def inv_cum_hazard(self, v, covs):
        x, z = covs
        v = np.asarray(v, dtype=np.float64).reshape(-1, 1)
        a = self.a(x, z).astype(np.float64).reshape(-1, 1)
        b = self.b(x).astype(np.float64).reshape(-1, 1)

        # t = (1/b) * log(1 + v*b/(h0*exp(a)))
        #denom = np.maximum(b, 1e-12)
        denom = np.maximum(b, 1e-12)
        inside = 1.0 + (v * denom / self.h0) * np.exp(-a)
        inside = np.maximum(inside, 1e-300)  # avoid log(0) if v huge/rounding
        return np.log(inside) / denom  # (n,1)


    def cum_hazard(self, t, covs):
        x, z = covs
        t = np.asarray(t, dtype=np.float64).reshape(-1, 1)     # (n,1)
        a = self.a(x, z).astype(np.float64).reshape(-1, 1)     # (n,1)
        b = self.b(x).astype(np.float64).reshape(-1, 1)        # (n,1)

        # H(t|x) = (h0 * exp(a)/b) * (exp(b t) - 1)
        # use expm1 for small b*t and guard division by ~0
        denom = np.maximum(b, 1e-12)
        return (self.h0 * np.exp(a) * np.expm1(b * t)) / denom  # (n,1)




# In[4]:


class SimStudyNLPH(SimStudyNonLinearPH):
    '''Survival simulations study for non-linear non-prop. hazard model.
        h(t | x) = h0 * exp[g(t, x)], 
        with constant h_0, and g(t, x) = a(x) + b(x)*t.
        Cumulative hazard:
        H(t | x) = h0 / b(x) * exp[a(x)] * (exp[b(x) * t] - 1)
        Inverse:
        H^{-1}(v, x) = 1/b(x) log{1 +  v * b(x) / h0 exp[-a(x)]}
    Parameters:
        h0: Is baseline constant.
        right_c: Time for right censoring.
    '''
    def __init__(self, h0=0.1, right_c=30., c0=30., surv_grid=None):
        super().__init__(h0, right_c, c0, surv_grid)

    @staticmethod
    def a(x,z):
        x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
        return 2*z[:,0] - 1*z[:,1] - 2*x0 + 3*x1 +4*x2 + 3*(0.2 * (x0 + x1) + 0.5 * x0 * x1 + x2**2)**2
        r#eturn 2*z[:,0] - 1*z[:,1] - 2*x0 + 3*x1 +4*x2

    @staticmethod
    def b(x):
        x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
        return 0.05*np.ones(x.shape[0])

    @staticmethod
    def g(t, covs):
        x,z = covs
        return SimStudyNLPH.a(x,z) + SimStudyNLPH.b(x) * t.reshape(-1, )

    def g_real(self,X,Z,T):
        return  SimStudyNLPH.g(T,(X,Z)) + np.log(self.h0)

    def g_real_grad(self,X,Z,T):
        return  SimStudyNLPH.b(X)

    def inv_cum_hazard(self, v, covs):
        x, z = covs
        v = np.asarray(v, dtype=np.float64).reshape(-1, 1)
        a = self.a(x, z).astype(np.float64).reshape(-1, 1)
        b = self.b(x).astype(np.float64).reshape(-1, 1)

        # t = (1/b) * log(1 + v*b/(h0*exp(a)))
        #denom = np.maximum(b, 1e-12)
        denom = np.maximum(b, 1e-12)
        inside = 1.0 + (v * denom / self.h0) * np.exp(-a)
        inside = np.maximum(inside, 1e-300)  # avoid log(0) if v huge/rounding
        return np.log(inside) / denom  # (n,1)


    def cum_hazard(self, t, covs):
        x, z = covs
        t = np.asarray(t, dtype=np.float64).reshape(-1, 1)     # (n,1)
        a = self.a(x, z).astype(np.float64).reshape(-1, 1)     # (n,1)
        b = self.b(x).astype(np.float64).reshape(-1, 1)        # (n,1)

        # H(t|x) = (h0 * exp(a)/b) * (exp(b t) - 1)
        # use expm1 for small b*t and guard division by ~0
        denom = np.maximum(b, 1e-12)
        return (self.h0 * np.exp(a) * np.expm1(b * t)) / denom  # (n,1)


# In[5]:


def simulate(n,sampler,data=None):
    if data == None:
        data = sampler.simulate(n)
    step      = 0.01
    durations = data["durations"]

    t_unique = np.unique(durations)
    t_sorted = np.sort(t_unique)

    X_covs = data["covs_X"]
    Z_covs = data["covs_Z"]
    E = data["events"]
    T = data["durations"]

    X_list, Z_list, E_list, st_list, et_list, id_list = [], [], [], [], [], []

    for i in range(len(X_covs)):
        for j in range(1, len(t_sorted)):
            if t_sorted[j] > T[i]:
                break
            X_list.append(X_covs[i])
            Z_list.append(Z_covs[i])
            E_list.append(E[i] if t_sorted[j] == T[i] else 0)
            st_list.append(t_sorted[j-1])
            et_list.append(t_sorted[j])
            id_list.append(i)

    X = np.stack(X_list)
    Z = np.stack(Z_list)
    E = np.array(E_list).reshape(-1, 1)
    st = np.array(st_list).reshape(-1, 1)
    et = np.array(et_list).reshape(-1, 1)
    id_list = np.array(id_list).reshape(-1, 1)

    return pd.DataFrame(
        np.concatenate([X, Z, E, st, et, id_list], axis=1),
        columns=[f"X{i}" for i in range(X.shape[1])] +
                [f"Z{i}" for i in range(Z.shape[1])] +
                ["E", "st", "et", "id"]
    ) , data


# In[6]:


def loss_beta(beta, Z, g_fixed, et, st, delta):
    beta = beta.reshape(-1, 1)
    linear_term = Z @ beta
    total = g_fixed + linear_term
    integral_term = (et - st) * np.exp(total)
    event_term = delta * total
    return np.sum((integral_term - event_term))

def grad_beta(beta, Z, g_fixed, et, st, delta):
    beta = beta.reshape(-1, 1)
    linear_term = Z @ beta
    total = g_fixed + linear_term
    exp_total = np.exp(total)
    grad = Z * ((et - st) * exp_total - delta)
    return np.sum(grad, axis=0)  # Return 1D array for gradient


def hessp_beta(beta, p, Z, g_fixed, et, st, delta, clip=80.0):
    beta = beta.reshape(-1, 1)
    linear_term = Z @ beta
    total = g_fixed + linear_term
    tot = total
    w = (et - st) * np.exp(np.clip(tot, -clip, clip))   # (n,1)
    p = np.asarray(p).reshape(-1,1)                     # (p,1)
    return (Z.T @ (w * (Z @ p))).ravel()                # (p,)

def loss_beta_sf(beta, Z, g_fixed, et, st, delta):
    x = beta.reshape(-1, 1)
    beta = x[:2]
    shift = x[2]
    scale = x[3]
    linear_term = Z @ beta
    total = g_fixed*scale + shift + linear_term
    integral_term = (et - st) * np.exp(total)
    event_term = delta * total
    return np.sum((integral_term - event_term)) + (tf.reduce_sum(beta**2))*0



# In[7]:


def custom_loss(y_true, y_pred, lmbd_cali, lmbd_cor):
    clip_pred  = 80.0
    eps        = 1e-30
    log_switch = 50.0
    dt_tiny    = 1e-8

    et    = y_true[:, 0:1]
    st    = y_true[:, 1:2]
    delta = y_true[:, 2:3]
    train_phase = y_true[:, 3:4]
    r     = y_true[:, 4:]

    train_phase = tf.reduce_mean(train_phase)


    # y_pred is (B,2): [chi, g]
    chi = y_pred[:, 0:1]            # (B,1)
    g   = y_pred[:, 1:2]            # (B,1)  <-- key change

    ne = tf.reduce_sum(delta) + 1e-12
    ne = tf.maximum(ne, 1.0)
    dt = tf.maximum(et - st, 0.0)

    y_safe = tf.clip_by_value(chi, -clip_pred, clip_pred)
    dt_safe = tf.maximum(dt, eps)
    use_log = tf.logical_or(y_safe > log_switch, dt < dt_tiny)

    integral_term = tf.where(
        use_log,
        tf.exp(tf.math.log(dt_safe) + y_safe),
        dt * tf.exp(y_safe)
    )

    event_term = delta * y_safe
    base = tf.reduce_sum(integral_term - event_term) / 10000

    # calibration penalty (train-only)
    pen_cali = (ne - tf.reduce_sum(integral_term)) / 10000
    pen_cali = lmbd_cali * tf.square(pen_cali)
    pen_cali = pen_cali*train_phase

    # ---- correlation penalty (event-weighted) ----
    w = tf.math.divide_no_nan(delta, ne)        # (B,1)

    # Event-weighted means
    mu_g = tf.reduce_sum(w * g)                 # scalar
    mu_r = tf.reduce_sum(w * r, axis=0)         # (p,)

    # Centered
    g_c = g - mu_g                              # (B,1)
    r_c = r - mu_r[tf.newaxis, :]              # (B,p)

    # Weighted variances
    var_g = tf.reduce_sum(w * tf.square(g_c)) + 1e-12
    var_r = tf.reduce_sum(w * tf.square(r_c), axis=0) + 1e-12  # (p,)

    # Weighted covariances per feature
    cov   = tf.reduce_sum(w * g_c * r_c, axis=0)               # (p,)
    corr2 = tf.square(cov) / (var_g * var_r)                   # (p,)

    corr2 = tf.abs(corr2)

    pen_cor = lmbd_cor * tf.reduce_sum(corr2)
    pen_cor = pen_cor*train_phase

    #tf.print(base,pen_cor)

    return base


# In[8]:


def cox_ph_loss_no_ties(y_true, y_pred, clip_pred=80.0, eps=1e-12):
    """
    Negative Cox partial log-likelihood assuming no ties.
    y_true columns: [et, st, delta]; only et and delta are used.
    y_pred: chi = g(et,X) + beta^T Z, shape (batch, 1).
    IMPORTANT: Use full-batch training (risk sets need all samples).
    """
    # flatten to 1D
    T = tf.reshape(y_true[:, 0], [-1])       # observed times (et)
    d = tf.reshape(y_true[:, 1], [-1])       # event indicator (0/1)
    y = tf.reshape(tf.clip_by_value(y_pred[:,0], -clip_pred, clip_pred), [-1])

    # sort by descending time so risk set at i is prefix [:i+1]
    order = tf.argsort(T, direction='DESCENDING')
    y_sorted = tf.gather(y, order)
    d_sorted = tf.gather(d, order)

    # denom_i = log sum_{j<=i} exp(y_j)  (stable enough with clipping)
    cumexp = tf.cumsum(tf.exp(y_sorted))
    log_denom = tf.math.log(tf.maximum(cumexp, eps))

    # partial loglik = sum_{events i} (y_i - log denom_i)
    contrib = d_sorted * (y_sorted - log_denom)
    # negative average over events (scale-stable)
    events = tf.reduce_sum(d_sorted)
    return -tf.reduce_sum(contrib) / tf.maximum(events, 1.0)


# In[9]:


class L1Var(tf.keras.regularizers.Regularizer):
    def __init__(self, coeff_var): self.c = coeff_var
    def __call__(self, x): return self.c * tf.reduce_sum(tf.abs(x))


def build_model(input_dim_X, input_dim_Z, nn_config, beta_init,lmbd_l1=0):
    X_input  = Input(shape=(input_dim_X,), name="X_input")
    Z_input  = Input(shape=(input_dim_Z,), name="Z_input")
    et_input = Input(shape=(1,),           name="et_input")    # (time) used in g(t,X)


    body_layers = []
    head_layers = []
    reg = L1Var(lmbd_l1)
    g_input = Concatenate(name="concat_xt")([et_input, X_input])
    x = g_input
    for _ in range(nn_config["n_hidden_layers"]):
        tmp = Dense(nn_config["hidden_layers_nodes"], activation='relu',kernel_regularizer=reg,bias_regularizer=reg)
        body_layers += [tmp]
        x = tmp(x)
        x = tf.keras.layers.Activation('relu')(x)

    tmp = Dense(1, activation=None, name="g_out",kernel_regularizer=reg,bias_regularizer=None)
    g_out = tmp(x)
    body_layers += [tmp]
    beta_out = Dense(1, use_bias=False, activation=None, name="beta_layer",kernel_initializer=Constant(beta_init))(Z_input)
    head_layers = [beta_out]
    out = Add(name="chi")([g_out, beta_out])
    chi_g = Concatenate(axis=1, name="chi_g")([out, g_out])


    model = Model(inputs=[X_input, Z_input, et_input], outputs=chi_g)
    return model, body_layers, head_layers, reg

def compile_cox_ph(model,nn_config,optimizer=None,lmbd_cali=0,lmbd_cor=0):
    if optimizer == None:
        optimizer = Adam(learning_rate=nn_config["learning_rate"])
    model.compile(optimizer=optimizer, loss=cox_ph_loss_no_ties, jit_compile=False)
    return model,optimizer


def compile_baseliness(model,nn_config,optimizer=None,lmbd_cali=0,lmbd_cor=0):
    if optimizer == None:
        optimizer = Adam(learning_rate=nn_config["learning_rate"])

    def inner_loss(y_true, y_pred):
        return custom_loss(y_true, y_pred,lmbd_cali=lmbd_cali,lmbd_cor=lmbd_cor)
    model.compile(optimizer=optimizer, loss=inner_loss , jit_compile=False)
    return model, optimizer


def build_mse(input_dim_X, nn_config):
    X_input = Input(shape=(input_dim_X,), name="X_input")
    et_input = Input(shape=(1,), name="et_input")  # Time t used in g(t, X)

    # g(t, X) modeled by feeding [t, X]
    g_input = Concatenate()([et_input, X_input])
    x = g_input
    for _ in range(nn_config["n_hidden_layers"]):
        x = Dense(nn_config["hidden_layers_nodes"], activation=nn_config["activation"])(x)
        x = Dropout(nn_config["dropout"])(x)
    g_out = Dense(1, activation='linear')(x)

    model = Model(inputs=[X_input, et_input], outputs=g_out)

    optimizer = tf.keras.optimizers.get(nn_config["optimizer"])
    optimizer.learning_rate = nn_config["learning_rate"]

    model.compile(optimizer=optimizer, loss="mse", jit_compile=False)
    return model



# In[10]:


def build_mse(input_dim_X, nn_config, lmbd_l1=0):
    X_input = Input(shape=(input_dim_X,), name="X_input")
    et_input = Input(shape=(1,), name="et_input")  # Time t used in g(t, X)

    # g(t, X) modeled by feeding [t, X]
    g_input = Concatenate()([et_input, X_input])
    x = g_input
    reg = L1Var(lmbd_l1)
    for _ in range(nn_config["n_hidden_layers"]):
        x = Dense(nn_config["hidden_layers_nodes"], activation=nn_config["activation"],kernel_regularizer=reg,bias_regularizer=reg)(x)
        x = Dropout(nn_config["dropout"])(x)
    g_out = Dense(1, activation='linear',kernel_regularizer=reg,bias_regularizer=reg)(x)

    model = Model(inputs=[X_input, et_input], outputs=g_out)

    optimizer = tf.keras.optimizers.get(nn_config["optimizer"])
    optimizer.learning_rate = nn_config["learning_rate"]

    model.compile(optimizer=optimizer, loss="mse", jit_compile=False)
    return model


# In[11]:


def crossfit_event_residuals(df, nn_config, K=5, time_scale=30.0):
    """
    OOF residuals at EVENTS ONLY for correlation penalty.
    Trains each r_j on train-fold EVENTS and predicts for held-out fold EVENTS.
    Returns r_oof with shape (n, p), filled on events (delta==1), zeros elsewhere.
    """

    X_cols = [c for c in df.columns if c.startswith("X")]
    Z_cols = [c for c in df.columns if c.startswith("Z")]
    ids = df["id"]

    X = df[X_cols].values
    Z = df[Z_cols].values
    et = df["et"].values.reshape(-1, 1)
    st = df["st"].values.reshape(-1, 1)
    E = df["E"].values.reshape(-1, 1)

    X = np.asarray(X, dtype=np.float32)
    Z = np.asarray(Z, dtype=np.float32)
    et = np.asarray(et, dtype=np.float32).reshape(-1, 1)
    E = np.asarray(E, dtype=np.int32).reshape(-1)
    ids = np.asarray(ids)

    n, p = Z.shape
    r_oof = np.zeros((n, p), dtype=np.float32)

    # split by subject to avoid leakage across the same id
    gkf = GroupKFold(n_splits=min(K, np.unique(ids).size))
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, groups=ids)):
        tr_ev = tr_idx[E[tr_idx] == 1]
        te_ev = te_idx[E[te_idx] == 1]
        if tr_ev.size == 0 or te_ev.size == 0:
            continue  # skip degenerate fold

        for j in range(p):
            rmodel = build_mse(input_dim_X=X.shape[1], nn_config=nn_config)
            es = EarlyStopping(monitor='val_loss', patience=nn_config["patience"],
                               restore_best_weights=True)
            # fit only on EVENTS in the training folds
            rmodel.fit(
                [X[tr_ev], et[tr_ev] / time_scale], Z[tr_ev, j],
                validation_data=([X[te_ev], et[te_ev] / time_scale], Z[te_ev, j]),
                batch_size=nn_config["batch_size"],
                epochs=5000, callbacks=[es], verbose=0
            )
            # predict on EVENTS in the held-out fold
            pred_te_ev = rmodel.predict([X[te_ev], et[te_ev] / time_scale], verbose=0,batch_size=10000).reshape(-1)
            r_oof[te_ev, j] = Z[te_ev, j] - pred_te_ev
    # non-events remain zero; OK because penalty uses event weights
    return r_oof


# In[12]:


#p,in_ci_95,in_ci_90,TRUE_THETA,beta_all,cov_all,val_loss_all,train_loss_all,cox_result,deep_cox_result,r2_all =  pickle.load(open("4k_NLNPH","rb"))


# In[13]:


def run_sample(n,nn_config):

    def get_r2(t,p):
        tot = ((t-np.mean(t))**2).sum()
        cond = ((t-p)**2).sum()
        return 1- (cond/tot)


    epochs = nn_config["epochs"]
    val_frac = 0.33
    TRUE_THETA = np.array([2,-1])

    p = 2
    in_ci_95 = np.zeros(p)
    in_ci_90 = np.zeros(p)

    in_ci_95_opt = np.zeros(p)
    in_ci_90_opt = np.zeros(p)

    time_scale = 30
    sampler = SimStudyNonLinearNonPH()
    df_all, data = simulate(int(n*0.66),sampler)
    df_train = df_all
    df_val, data_val = simulate(int(n*0.33),sampler)
    X_cols = [c for c in df_all.columns if c.startswith("X")]
    Z_cols = [c for c in df_all.columns if c.startswith("Z")]
    #ids = df_all["id"].unique()

    p = len(Z_cols)
    #r_oof = crossfit_event_residuals(df_all,nn_config)
    #cutoff = int(len(ids)*(1-val_frac))
    #train_ids = ids[:cutoff]
    #val_ids = ids[cutoff:]
    #df_train = df_all[df_all["id"].isin(train_ids)]
    #df_val = df_all[df_all["id"].isin(val_ids)]

    # r_mat_train = r_oof[df_all["id"].isin(train_ids)]
    # r_mat_val = r_oof[df_all["id"].isin(val_ids)]
    r_mat_train = np.zeros((df_train.shape[0],p))
    r_mat_val = np.zeros((df_val.shape[0],p))


    # Remove times untill first event
    t_l_bound = df_train[df_train.E == 1].et.min()
    t_h_bound = df_train[df_train.E == 1].et.max()

    r_mat_train = r_mat_train[df_train.st >= t_l_bound]
    r_mat_val = r_mat_val[df_val.st >= t_l_bound]

    df_train = df_train[df_train.st >= t_l_bound]
    df_val = df_val[df_val.st >= t_l_bound]

    navie_cox_df = np.concatenate([data["covs_X"],data["covs_Z"],data["durations"].reshape(-1,1),data["events"].reshape(-1,1)],axis=1)
    navie_cox_df = pd.DataFrame(navie_cox_df,columns=X_cols + Z_cols + ["durations","events"])


    cph = CoxPHFitter()
    cph.fit(navie_cox_df, duration_col="durations", event_col="events", show_progress=False)


    # Extract beta coefficients for Z variables
    coef_series = cph.params_
    beta_init = coef_series[Z_cols].values.reshape(-1, 1)


    X_train = df_train[X_cols].values
    Z_train = df_train[Z_cols].values
    et_train = df_train["et"].values.reshape(-1, 1) / time_scale
    st_train = df_train["st"].values.reshape(-1, 1) / time_scale
    E_train = df_train["E"].values.reshape(-1, 1)


    X_val = df_val[X_cols].values
    Z_val = df_val[Z_cols].values
    et_val = df_val["et"].values.reshape(-1, 1) / time_scale
    st_val = df_val["st"].values.reshape(-1, 1) / time_scale
    E_val = df_val["E"].values.reshape(-1, 1)

    mask_tr = (E_train.reshape(-1,) == 1)
    mask_va = (E_val.reshape(-1,) == 1)

    r_mean_tr = r_mat_train[mask_tr].mean(axis=0, keepdims=True)
    r_mat_train[mask_tr] -= r_mean_tr

    r_mean_va = r_mat_val[mask_va].mean(axis=0, keepdims=True)
    r_mat_val[mask_va] -= r_mean_va
    # then use these in your inputs as you already do:
    train_inp = tuple(np.asarray(arr, dtype=np.float32) for arr in [
        X_train, Z_train, et_train, r_mat_train, E_train
    ])
    val_inp = tuple(np.asarray(arr, dtype=np.float32) for arr in [
        X_val,   Z_val,   et_val,   r_mat_val,   E_val
    ])

    y_train = np.concatenate([et_train,
                              st_train,
                              E_train,
                              np.ones(et_train.shape),
                              r_mat_train], axis=1) 

    y_val = np.concatenate([et_val,
                            st_val,
                            E_val,
                            np.zeros(et_val.shape),
                            r_mat_val], axis=1)

    train_inp = tuple(np.asarray(arr, dtype=np.float32) for arr in [
        X_train,
        Z_train,
        et_train,
    ])

    val_inp = tuple(np.asarray(arr, dtype=np.float32) for arr in [
        X_val,
        Z_val,
        et_val,
    ])

    # g_real returns chi not g
    g_real_train = (sampler.g_real(X_train,Z_train,et_train*time_scale).reshape(-1, 1) - (Z_train @ TRUE_THETA).reshape(-1, 1)).reshape(-1,1) + np.log(time_scale)
    g_real_val = (sampler.g_real(X_val,Z_val,et_val*time_scale).reshape(-1, 1) - (Z_val @ TRUE_THETA).reshape(-1, 1)).reshape(-1,1) + np.log(time_scale)



    res = minimize(
        loss_beta,
        x0=beta_init[:,0],
        args=(Z_train, g_real_train, et_train, st_train, E_train),
        method='L-BFGS-B',
        jac=grad_beta
    )
    beat_opt_real = res.x


    val_score = loss_beta(beat_opt_real, Z_val, g_real_val, et_val, st_val, E_val)
    train_score = loss_beta(beat_opt_real, Z_train, g_real_train, et_train, st_train, E_train)    

    val_score = loss_beta(TRUE_THETA, Z_val, g_real_val, et_val, st_val, E_val)
    train_score = loss_beta(TRUE_THETA, Z_train, g_real_train, et_train, st_train, E_train)   

    optimizer = Adam(learning_rate=0.001)    
    model,body,head,reg = build_model(input_dim_X=X_train.shape[1],
                        input_dim_Z=Z_train.shape[1],
                        nn_config=nn_config,
                        beta_init=beta_init,lmbd_l1=nn_config["lmbd_L1"])

    model, optimizer = compile_baseliness(model,nn_config,optimizer=optimizer,lmbd_cali=nn_config["lmbd_cali"],lmbd_cor=nn_config["lmbd_cor"])
    beta_hat = model.get_layer("beta_layer").get_weights()[0].reshape((p,))
    g_train = model.predict(train_inp, batch_size=100000,verbose=0)[:,1].reshape(-1,1)
    x0 = np.concatenate((beta_hat,[0,1]))
    res = minimize(
        fun=loss_beta_sf,
        x0=x0,
        args=(Z_train.astype(np.float64), g_train.astype(np.float64),
              et_train.astype(np.float64), st_train.astype(np.float64),
              E_train.astype(np.float64)),
        method="L-BFGS-B",
    )
    res.x
    beta_hat = res.x[:2]
    shift = res.x[2]
    scale = res.x[3]


    dense = model.get_layer("g_out")  # Dense(1)
    W, b = dense.get_weights()        # W shape (1,1), b shape (1,)
    dense.set_weights([W * scale, b* scale + shift])

    optimizer.learning_rate = 0.0001
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=200,
        restore_best_weights=True
    )    
    history = model.fit(
        train_inp, y_train,
        validation_data=(val_inp, y_val),
        batch_size=int(X_train.shape[0]/400),
        epochs=epochs,
        callbacks=[early_stop],
        verbose = 0
    )

    beta_hat = model.get_layer("beta_layer").get_weights()[0].reshape((p,))
    #print(beta_hat)

    # optimizer.learning_rate = 0.0001
    # history = model.fit(
    #     train_inp, y_train,
    #     validation_data=(val_inp, y_val),
    #     batch_size=int(X_train.shape[0]/20),
    #     epochs=10,
    #     verbose = 0
    # )
    # beta_hat = model.get_layer("beta_layer").get_weights()[0].reshape((p,))



    g_train = model.predict(train_inp, batch_size=10000,verbose=0)[:,1].reshape(-1,)
    #print(np.mean(r_mat_train[(E_train == 1).reshape(-1,),0]*g_train[E_train == 1]),np.mean(r_mat_train[(E_train == 1).reshape(-1,),1]*g_train[E_train == 1]))
    # On events
    mask = (E_train.ravel()==1)
    e = (g_train.ravel() - g_real_train.ravel())[mask]
    #beta_hat = model.get_layer("beta_layer").get_weights()[0].reshape((p,))
    g_train = model.predict(train_inp, batch_size=10000,verbose=0)[:,1].reshape(-1,1)
    g_val = model.predict(val_inp, batch_size=10000,verbose=0)[:,1].reshape(-1,1)

    val_loss = loss_beta(beta_hat, Z_val, g_val, et_val, st_val, E_val)
    train_loss = loss_beta(beta_hat, Z_train, g_train, et_train, st_train, E_train)
    r2 = get_r2(g_real_val[E_val == 1], g_val[E_val == 1])

    r_mat = r_mat_train - r_mat_train.mean(axis=0)

    I_hat = (r_mat).T @ r_mat / n
    try:
        cov_theta = np.linalg.inv(I_hat) / n
    except np.linalg.LinAlgError:
        ridge = 1e-8 * np.eye(p)
        cov_theta = np.linalg.inv(I_hat + ridge) / n
    se_theta = np.sqrt(np.diag(cov_theta))
    z_crit_95 = norm.ppf(0.975)
    z_crit_90 = norm.ppf(0.95)

    ci_lower_95 = beta_hat - z_crit_95 * se_theta
    ci_upper_95 = beta_hat + z_crit_95 * se_theta

    ci_lower_90 = beta_hat - z_crit_90 * se_theta
    ci_upper_90 = beta_hat + z_crit_90 * se_theta

    p = Z_train.shape[1]
    in_ci_95 = np.zeros(p)
    in_ci_90 = np.zeros(p)
    for k in range(p):
        in_ci_95[k] = (TRUE_THETA[k] > ci_lower_95[k]) and (TRUE_THETA[k] < ci_upper_95[k])
        in_ci_90[k] = (TRUE_THETA[k] > ci_lower_90[k]) and (TRUE_THETA[k] < ci_upper_90[k])

    return val_score, beta_hat, cov_theta, in_ci_90, in_ci_95, r2, beat_opt_real, data


# In[14]:


# nn_config = {
#     "hidden_layers_nodes": 20,
#     "n_hidden_layers": 5,
#     "learning_rate": 0.001,
#     "activation": 'relu', 
#     "optimizer": 'adam',
#     "batch_size": 128,
#     "patience": 50,
#     "dropout": 0.0,

#     "lmbd_L1" : 1e-3,
#     "lmbd_cali": 0,
#     "lmbd_cor": 0,
#     "epochs" : 50
# }
# results = run_sample(100,nn_config)









