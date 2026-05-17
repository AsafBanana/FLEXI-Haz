import random

# ===== Notebook cell 1 =====

# --- Device / runtime config ---
import os

USE_CPU_ONLY = False          # True -> force CPU, False -> allow GPU
SEED = 0
ENABLE_XLA = False            # try True on GPU if your loss stays stable
ENABLE_MIXED_PRECISION = False  # try carefully on GPU only

# IMPORTANT:
# Change this cell, then RESTART THE KERNEL before re-running the notebook.
if USE_CPU_ONLY:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ===== Notebook cell 2 =====

import time
import copy
import pickle
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Add, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, train_test_split
from lifelines import CoxPHFitter
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt

np.random.seed(SEED)
tf.random.set_seed(SEED)

if ENABLE_XLA:
    try:
        tf.config.optimizer.set_jit(True)
        print("XLA JIT enabled")
    except Exception as e:
        print("Could not enable XLA:", repr(e))

if ENABLE_MIXED_PRECISION:
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled:", mixed_precision.global_policy())
    except Exception as e:
        print("Could not enable mixed precision:", repr(e))

# Memory growth if GPU is present
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

print("TensorFlow:", tf.__version__)
print("CPUs:", tf.config.list_physical_devices("CPU"))
print("GPUs:", tf.config.list_physical_devices("GPU"))
print("USE_CPU_ONLY =", USE_CPU_ONLY)


# ===== Notebook cell 3 =====

# --- Timing helpers ---
class StageTimer:
    def __init__(self):
        self.times = {}

    @contextmanager
    def time(self, name):
        st = time.perf_counter()
        yield
        self.times[name] = self.times.get(name, 0.0) + (time.perf_counter() - st)

    def add(self, name, value):
        self.times[name] = self.times.get(name, 0.0) + float(value)

    def as_series(self):
        s = pd.Series(self.times).sort_values(ascending=False)
        return s

    def pretty(self):
        s = self.as_series()
        if len(s) == 0:
            return pd.DataFrame(columns=["stage", "seconds"])
        out = s.reset_index()
        out.columns = ["stage", "seconds"]
        out["seconds"] = out["seconds"].round(3)
        return out

def print_timing_dict(d, title="Timing summary"):
    print(title)
    if not d:
        print("(empty)")
        return
    s = pd.Series(d).sort_values(ascending=False)
    for k, v in s.items():
        print(f"{k:35s} {v:10.3f} sec")


# ===== Notebook cell 4 =====

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
    def g_linear(covs):
        x = covs
        x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
        return 0.44 * x0 + 0.66 * x1 + 0.88 * x2
    
    @staticmethod
    def g(covs):
        x,z = covs
        x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
        beta = 2/3
        linear = SimStudyNonLinearPH.g_linear(x)
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


# ===== Notebook cell 5 =====

def split_subject_data(data, train_ids, test_ids):
    train_ids = np.asarray(train_ids).astype(int)
    test_ids  = np.asarray(test_ids).astype(int)

    data_train = {
        "ids":       train_ids,                 # <-- ADD THIS
        "covs_X":    data["covs_X"][train_ids],
        "covs_Z":    data["covs_Z"][train_ids],
        "durations": data["durations"][train_ids],
        "events":    data["events"][train_ids],
    }
    data_test = {
        "ids":       test_ids,                  # <-- ADD THIS
        "covs_X":    data["covs_X"][test_ids],
        "covs_Z":    data["covs_Z"][test_ids],
        "durations": data["durations"][test_ids],
        "events":    data["events"][test_ids],
    }
    return data_train, data_test


# ===== Notebook cell 6 =====

import numpy as np
import pandas as pd

# 1) Discrete X via truncated Poisson + standardization

def truncated_poisson(n, lam=2.0, k_max=6, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x = rng.poisson(lam=lam, size=n)
    return np.clip(x, 0, k_max)

def sample_covs_discrete_poisson(n, lam=2.0, k_max=6, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    X = np.column_stack([truncated_poisson(n, lam, k_max, rng) for _ in range(3)]).astype(np.float64)
    X = (2*X/k_max) - 1
    #X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-12)
    Z = rng.uniform(-1, 1, size=(n, 2)).astype(np.float64)
    return X, Z


def sample_covs_binary(n, rng=None):
    rng = np.random.default_rng() if rng is None else rng

    X = rng.choice([-1.0, 1.0], size=(n, 3))
    Z = rng.uniform(-1, 1, size=(n, 2)).astype(np.float64)

    return X, Z


# Patch base simulator (inherited by all simulators)
try:
    SimStudyNonLinearPH.sample_covs = staticmethod(lambda n: sample_covs_binary(n))
    print("Patched SimStudyNonLinearPH.sample_covs -> discrete truncated Poisson + standardization")
except NameError:
    print("WARNING: SimStudyNonLinearPH not found. Import/define it, then rerun this cell.")


# ===== Notebook cell 7 =====

import numpy as np
import pandas as pd

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found. Available: {list(df.columns)[:40]}")

def cumtrapz(y, x):
    # y: (m,), x: (m,)
    out = np.zeros_like(y, dtype=np.float64)
    dx = np.diff(x)
    out[1:] = np.cumsum(0.5*(y[1:]+y[:-1])*dx)
    return out

def cumtrapz_vec(Y, x):
    # Y: (m,p) integrates each column
    Y = np.asarray(Y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    m, p = Y.shape
    out = np.zeros((m, p), dtype=np.float64)
    dx = np.diff(x)
    out[1:,:] = np.cumsum(0.5*(Y[1:,:]+Y[:-1,:]) * dx.reshape(-1,1), axis=0)
    return out

def predict_chi(model, X, Z, t_scaled, batch=20000):
    # model output[:,0] = chi, output[:,1] = g_out
    out = model.predict([X, Z, t_scaled], batch_size=batch, verbose=0)
    return out[:,0].astype(np.float64)

def build_strata_from_X(X_subj):
    keys = [tuple(row.tolist()) for row in X_subj.astype(np.float64)]
    uniq = sorted(set(keys))
    key2j = {k:j for j,k in enumerate(uniq)}
    idx = np.array([key2j[k] for k in keys], dtype=int)
    return uniq, key2j, idx


# ===== Notebook cell 8 =====

# --- Patched / faster core functions ---

def predict_chi(model, X, Z, t_scaled, batch=65536):
    """
    Faster than model.predict(...) for repeated calls in this notebook.
    """
    X = np.asarray(X, dtype=np.float32)
    Z = np.asarray(Z, dtype=np.float32)
    t_scaled = np.asarray(t_scaled, dtype=np.float32)

    outs = []
    n = X.shape[0]
    for s in range(0, n, batch):
        e = min(n, s + batch)
        xb = tf.convert_to_tensor(X[s:e], dtype=tf.float32)
        zb = tf.convert_to_tensor(Z[s:e], dtype=tf.float32)
        tb = tf.convert_to_tensor(t_scaled[s:e], dtype=tf.float32)
        out = model([xb, zb, tb], training=False)
        outs.append(out[:, 0].numpy())
    return np.concatenate(outs, axis=0).astype(np.float64)


class L1Var(tf.keras.regularizers.Regularizer):
    def __init__(self, coeff_var):
        self.c = float(coeff_var)

    def __call__(self, x):
        return self.c * tf.reduce_sum(tf.abs(x))

    def get_config(self):
        return {"coeff_var": self.c}


def build_model(input_dim_X, input_dim_Z, nn_config, beta_init, lmbd_l1=0):
    X_input  = Input(shape=(input_dim_X,), name="X_input")
    Z_input  = Input(shape=(input_dim_Z,), name="Z_input")
    et_input = Input(shape=(1,), name="et_input")

    reg = L1Var(lmbd_l1)
    x = Concatenate(name="concat_xt")([et_input, X_input])

    # removed redundant second ReLU
    for _ in range(nn_config["n_hidden_layers"]):
        x = Dense(
            nn_config["hidden_layers_nodes"],
            activation="relu",
            kernel_regularizer=reg,
            bias_regularizer=reg
        )(x)

    g_out = Dense(
        1,
        activation=None,
        name="g_out",
        kernel_regularizer=reg,
        bias_regularizer=None
    )(x)

    beta_out = Dense(
        1,
        use_bias=False,
        activation=None,
        name="beta_layer",
        kernel_initializer=Constant(beta_init)
    )(Z_input)

    out = Add(name="chi")([g_out, beta_out])
    chi_g = Concatenate(axis=1, name="chi_g")([out, g_out])

    model = Model(inputs=[X_input, Z_input, et_input], outputs=chi_g)
    return model, None, None, reg


def compile_baseliness(model, nn_config, optimizer=None, lmbd_cali=0, lmbd_cor=0):
    if optimizer is None:
        optimizer = Adam(learning_rate=nn_config["learning_rate"])

    def inner_loss(y_true, y_pred):
        return custom_loss(y_true, y_pred, lmbd_cali=lmbd_cali, lmbd_cor=lmbd_cor)

    model.compile(
        optimizer=optimizer,
        loss=inner_loss,
        jit_compile=bool(nn_config.get("jit_compile", False))
    )
    return model, optimizer


def build_mse(input_dim_X, nn_config, lmbd_l1=0):
    X_input = Input(shape=(input_dim_X,), name="X_input")
    et_input = Input(shape=(1,), name="et_input")

    reg = L1Var(lmbd_l1)
    x = Concatenate()([et_input, X_input])

    for _ in range(nn_config["n_hidden_layers"]):
        x = Dense(
            nn_config["hidden_layers_nodes"],
            activation=nn_config["activation"],
            kernel_regularizer=reg,
            bias_regularizer=reg
        )(x)
        if nn_config.get("dropout", 0.0) > 0:
            x = Dropout(nn_config["dropout"])(x)

    g_out = Dense(1, activation="linear", kernel_regularizer=reg, bias_regularizer=reg)(x)
    model = Model(inputs=[X_input, et_input], outputs=g_out)

    optimizer = tf.keras.optimizers.get(nn_config["optimizer"])
    optimizer.learning_rate = nn_config["learning_rate"]

    model.compile(
        optimizer=optimizer,
        loss="mse",
        jit_compile=bool(nn_config.get("jit_compile", False))
    )
    return model


def compute_global_beta_init(data):
    X_dim = data["covs_X"].shape[1]
    Z_dim = data["covs_Z"].shape[1]
    X_cols = [f"X{i}" for i in range(X_dim)]
    Z_cols = [f"Z{i}" for i in range(Z_dim)]

    naive_cox_df = np.concatenate([
        data["covs_X"],
        data["covs_Z"],
        data["durations"].reshape(-1, 1),
        data["events"].reshape(-1, 1)
    ], axis=1)
    naive_cox_df = pd.DataFrame(naive_cox_df, columns=X_cols + Z_cols + ["durations", "events"])

    cph = CoxPHFitter()
    cph.fit(naive_cox_df, duration_col="durations", event_col="events", show_progress=False)
    beta_init = cph.params_[Z_cols].values.reshape(-1, 1)
    return beta_init


def crossfit_event_residuals(
    df,
    nn_config,
    K=5,
    time_scale=30.0,
    residual_config=None,
    return_timing=False,
):
    """
    Faster version:
    - allows a smaller dedicated residual config
    - tracks timing
    """
    timer = StageTimer()

    X_cols = [c for c in df.columns if c.startswith("X")]
    Z_cols = [c for c in df.columns if c.startswith("Z")]
    ids = np.asarray(df["id"])

    X = df[X_cols].values.astype(np.float32)
    Z = df[Z_cols].values.astype(np.float32)
    et = df["et"].values.astype(np.float32).reshape(-1, 1)
    E = df["E"].values.astype(np.int32).reshape(-1)

    n, p = Z.shape
    r_oof = np.zeros((n, p), dtype=np.float32)

    cfg = copy.deepcopy(nn_config)
    if residual_config is not None:
        cfg.update(residual_config)

    gkf = GroupKFold(n_splits=min(K, np.unique(ids).size))

    with timer.time("residual_total"):
        for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, groups=ids), start=1):
            tr_ev = tr_idx[E[tr_idx] == 1]
            te_ev = te_idx[E[te_idx] == 1]
            if tr_ev.size == 0 or te_ev.size == 0:
                continue

            for j in range(p):
                with timer.time("residual_fit_models"):
                    rmodel = build_mse(input_dim_X=X.shape[1], nn_config=cfg, lmbd_l1=cfg.get("lmbd_L1", 0.0))
                    es = EarlyStopping(
                        monitor="val_loss",
                        patience=cfg["patience"],
                        restore_best_weights=True
                    )
                    rmodel.fit(
                        [X[tr_ev], et[tr_ev] / time_scale],
                        Z[tr_ev, j],
                        validation_data=([X[te_ev], et[te_ev] / time_scale], Z[te_ev, j]),
                        batch_size=cfg["batch_size"],
                        epochs=cfg["epochs"],
                        callbacks=[es],
                        verbose=0
                    )

                with timer.time("residual_predict"):
                    pred_te_ev = rmodel(
                        [X[te_ev], et[te_ev] / time_scale],
                        training=False
                    ).numpy().reshape(-1)
                    r_oof[te_ev, j] = Z[te_ev, j] - pred_te_ev

    if return_timing:
        return r_oof, timer.times
    return r_oof


def fit_model_on_ids(
    df_all,
    data,
    train_ids,
    val_ids,
    nn_config,
    r_oof,
    beta_init,
    time_scale=30.0,
    sampler=None,
    TRUE_THETA=np.array([2.0, -1.0]),
    t_l_bound=None,
    return_timing=False,
):
    timer = StageTimer()

    with timer.time("fit_model_on_ids_total"):
        df_all = df_all.copy()

        X_cols = [c for c in df_all.columns if c.startswith("X")]
        Z_cols = [c for c in df_all.columns if c.startswith("Z")]

        train_ids = np.asarray(train_ids).astype(int)
        val_ids   = np.asarray(val_ids).astype(int)

        with timer.time("split_data"):
            df_train = df_all[df_all["id"].isin(train_ids)].copy()
            df_val   = df_all[df_all["id"].isin(val_ids)].copy()
            data_train, data_val = split_subject_data(data, train_ids, val_ids)

            r_mat_train = r_oof[df_all["id"].isin(train_ids)]
            r_mat_val   = r_oof[df_all["id"].isin(val_ids)]

        if t_l_bound is None:
            t_l_bound = df_train[df_train.E == 1].et.min()

        with timer.time("trim_rows"):
            mask_tr_rows = (df_train.st.values >= t_l_bound)
            mask_va_rows = (df_val.st.values   >= t_l_bound)

            r_mat_train = r_mat_train[mask_tr_rows]
            r_mat_val   = r_mat_val[mask_va_rows]
            df_train    = df_train.loc[mask_tr_rows].copy()
            df_val      = df_val.loc[mask_va_rows].copy()

        with timer.time("build_arrays"):
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

            if mask_tr.sum() > 0:
                r_mean_tr = r_mat_train[mask_tr].mean(axis=0, keepdims=True)
                r_mat_train[mask_tr] -= r_mean_tr
            if mask_va.sum() > 0:
                r_mean_va = r_mat_val[mask_va].mean(axis=0, keepdims=True)
                r_mat_val[mask_va] -= r_mean_va

            y_train = np.concatenate([et_train, st_train, E_train, np.ones(et_train.shape), r_mat_train], axis=1)
            y_val   = np.concatenate([et_val,   st_val,   E_val,   np.zeros(et_val.shape), r_mat_val], axis=1)

            train_inp = tuple(np.asarray(arr, dtype=np.float32) for arr in [X_train, Z_train, et_train])
            val_inp   = tuple(np.asarray(arr, dtype=np.float32) for arr in [X_val,   Z_val,   et_val])

        with timer.time("build_compile_model"):
            optimizer = Adam(learning_rate=nn_config["learning_rate"])
            model, _, _, reg = build_model(
                input_dim_X=X_train.shape[1],
                input_dim_Z=Z_train.shape[1],
                nn_config=nn_config,
                beta_init=beta_init,
                lmbd_l1=nn_config["lmbd_L1"]
            )
            model, optimizer = compile_baseliness(
                model,
                nn_config,
                optimizer=optimizer,
                lmbd_cali=nn_config["lmbd_cali"],
                lmbd_cor=nn_config["lmbd_cor"]
            )

        with timer.time("main_model_fit"):
            es = EarlyStopping(monitor="val_loss", patience=nn_config["patience"], restore_best_weights=True)
            hist = model.fit(
                train_inp,
                y_train,
                validation_data=(val_inp, y_val),
                batch_size=nn_config["batch_size"],
                epochs=nn_config["epochs"],
                callbacks=[es],
                verbose=0,
            )

    out = {
        "model": model,
        "history": hist.history,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "t_l_bound": t_l_bound,
        "n_train_rows": int(len(df_train)),
        "n_val_rows": int(len(df_val)),
    }

    if return_timing:
        out["timings"] = timer.times
    return out


def prepare_fold_for_one_step(
    df_train_outer,
    data_train_outer,
    df_eval,
    data_eval,
    model,
    t_grid,
    uniq_keys,
    key2j,
    time_scale=30.0,
    ridge=1e-8,
    eps=1e-18,
):
    """
    Same logic as before, but vectorizes I_hat.
    """
    X_tr_subj = data_train_outer["covs_X"].astype(np.float64)
    Z_tr_subj = data_train_outer["covs_Z"].astype(np.float64)
    n_tr = X_tr_subj.shape[0]
    p = Z_tr_subj.shape[1]

    X_ev_subj = data_eval["covs_X"].astype(np.float64)
    Z_ev_subj = data_eval["covs_Z"].astype(np.float64)
    n_ev = X_ev_subj.shape[0]

    subj_str_tr = np.array([key2j[tuple(x.tolist())] for x in X_tr_subj], dtype=int)
    subj_str_ev = np.array([key2j[tuple(x.tolist())] for x in X_ev_subj], dtype=int)

    J = len(uniq_keys)
    pi_hat = np.bincount(subj_str_tr, minlength=J) / max(1, n_tr)

    X_cols = [c for c in df_train_outer.columns if c.startswith("X")]
    Z_cols = [c for c in df_train_outer.columns if c.startswith("Z")]

    tr_ids = np.asarray(data_train_outer["ids"]).astype(int)
    tr_id2pos = {int(sid): i for i, sid in enumerate(tr_ids)}
    sid_tr = np.array([tr_id2pos[int(s)] for s in df_train_outer["id"].values.astype(int)], dtype=int)

    st_tr = df_train_outer["st"].values.astype(np.float64)
    et_tr = df_train_outer["et"].values.astype(np.float64)
    dt_tr = (et_tr - st_tr).astype(np.float64)

    Xr_tr = df_train_outer[X_cols].values.astype(np.float32)
    Zr_tr = df_train_outer[Z_cols].values.astype(np.float32)

    tidx_tr = np.searchsorted(t_grid, et_tr)
    ok_tr = (tidx_tr >= 0) & (tidx_tr < len(t_grid)) & (t_grid[tidx_tr] == et_tr)
    if not np.all(ok_tr):
        tidx_tr = np.clip(tidx_tr, 0, len(t_grid)-1)

    chi_tr = predict_chi(model, Xr_tr, Zr_tr, (et_tr/time_scale).reshape(-1, 1).astype(np.float32))
    lam_tr = np.exp(chi_tr) / time_scale

    row_str_tr = subj_str_tr[sid_tr]
    m = len(t_grid)

    num_A = np.zeros((J, m), dtype=np.float64)
    np.add.at(num_A, (row_str_tr, tidx_tr), lam_tr)
    num_A /= max(1, n_tr)
    A_hat = num_A / (pi_hat[:, None] + eps)

    num_g = np.zeros((J, m, p), dtype=np.float64)
    den_g = np.zeros((J, m), dtype=np.float64)
    for jj in range(p):
        np.add.at(num_g[:, :, jj], (row_str_tr, tidx_tr), lam_tr * Zr_tr[:, jj].astype(np.float64))
    np.add.at(den_g, (row_str_tr, tidx_tr), lam_tr)

    g_star = np.zeros((J, m, p), dtype=np.float64)
    for jj in range(p):
        g_star[:, :, jj] = num_g[:, :, jj] / (den_g + eps)

    g_row_tr = g_star[row_str_tr, tidx_tr, :]
    diff_tr = (Zr_tr.astype(np.float64) - g_row_tr)
    w_tr = (dt_tr * lam_tr).astype(np.float64)

    # vectorized I_hat
    WD = diff_tr * w_tr[:, None]
    I_hat = (WD.T @ diff_tr) / max(1, n_tr)
    I_inv = np.linalg.inv(I_hat + ridge*np.eye(p))

    ev_ids = np.asarray(data_eval["ids"]).astype(int)
    ev_id2pos = {int(sid): i for i, sid in enumerate(ev_ids)}
    sid_ev = np.array([ev_id2pos[int(s)] for s in df_eval["id"].values.astype(int)], dtype=int)

    st_ev = df_eval["st"].values.astype(np.float64)
    et_ev = df_eval["et"].values.astype(np.float64)
    dt_ev = (et_ev - st_ev).astype(np.float64)
    E_ev  = df_eval["E"].values.astype(np.float64)

    Xr_ev = df_eval[X_cols].values.astype(np.float32)
    Zr_ev = df_eval[Z_cols].values.astype(np.float32)

    tidx_ev = np.searchsorted(t_grid, et_ev)
    ok_ev = (tidx_ev >= 0) & (tidx_ev < len(t_grid)) & (t_grid[tidx_ev] == et_ev)
    if not np.all(ok_ev):
        tidx_ev = np.clip(tidx_ev, 0, len(t_grid)-1)

    chi_ev = predict_chi(model, Xr_ev, Zr_ev, (et_ev/time_scale).reshape(-1,1).astype(np.float32))
    lam_ev = np.exp(chi_ev) / time_scale
    dM_ev = E_ev - dt_ev * lam_ev

    row_str_ev = subj_str_ev[sid_ev]
    g_row_ev = g_star[row_str_ev, tidx_ev, :]
    diff_ev = (Zr_ev.astype(np.float64) - g_row_ev)

    ell = np.zeros((n_ev, p), dtype=np.float64)
    for jj in range(p):
        np.add.at(ell[:, jj], sid_ev, diff_ev[:, jj] * dM_ev)

    ell_sum = ell.sum(axis=0)
    ell_outer = ell.T @ ell

    stratum_rows = {}
    for j in range(J):
        subj_idx = np.where(subj_str_ev == j)[0]
        if subj_idx.size == 0:
            continue
        pos_to_local = -np.ones(n_ev, dtype=int)
        pos_to_local[subj_idx] = np.arange(subj_idx.size)

        mask_rows = (row_str_ev == j)
        if mask_rows.sum() == 0:
            stratum_rows[j] = dict(
                subj_idx=subj_idx,
                sid_local=np.zeros((0,), dtype=int),
                tidx=np.zeros((0,), dtype=int),
                dM=np.zeros((0,), dtype=np.float64),
            )
            continue

        sid_local = pos_to_local[sid_ev[mask_rows]]
        stratum_rows[j] = dict(
            subj_idx=subj_idx,
            sid_local=sid_local.astype(int),
            tidx=tidx_ev[mask_rows].astype(int),
            dM=dM_ev[mask_rows].astype(np.float64),
        )

    return dict(
        J=J,
        m=m,
        p=p,
        pi_hat=pi_hat,
        A_hat=A_hat,
        g_star=g_star,
        I_inv=I_inv,
        ell=ell,
        ell_sum=ell_sum,
        ell_outer=ell_outer,
        stratum_rows=stratum_rows,
        n_eval=n_ev,
    )


def fit_discreteX_one_step_crossfit5(
    df_all,
    data,
    nn_config,
    K=5,
    time_scale=30.0,
    inner_val_frac=0.33,
    random_state=0,
    residual_config=None,
):
    """
    Optimized version with stage timings.
    """
    timer = StageTimer()

    with timer.time("total_fit_crossfit"):
        df_all = df_all.copy()
        ids = np.asarray(sorted(df_all["id"].unique())).astype(int)
        n_total = len(ids)

        with timer.time("build_strata"):
            X_full = data["covs_X"].astype(np.float64)
            uniq_keys, key2j, subj_str_full = build_strata_from_X(X_full)
            y_strata = subj_str_full[ids]

        with timer.time("outer_split"):
            try:
                outer_split = StratifiedKFold(n_splits=K, shuffle=True, random_state=random_state)
                fold_splits = list(outer_split.split(ids, y_strata))
            except Exception as e:
                print("WARNING: StratifiedKFold failed; falling back to KFold. Reason:", repr(e))
                outer_split = KFold(n_splits=K, shuffle=True, random_state=random_state)
                fold_splits = list(outer_split.split(ids))

        with timer.time("common_grid_setup"):
            t_l_bounds = []
            t_evals = []
            for tr_idx, te_idx in fold_splits:
                tr_ids = ids[tr_idx]
                df_tr_outer = df_all[df_all["id"].isin(tr_ids)]
                t_l = df_tr_outer[df_tr_outer["E"] == 1]["et"].min()
                if np.isfinite(t_l):
                    t_l_bounds.append(float(t_l))
                t_evals.append(float(data["durations"][tr_ids].max()))
            t_l_bound = float(np.max(t_l_bounds)) if len(t_l_bounds) else float(df_all[df_all["E"]==1]["et"].min())
            t_eval = float(np.min(t_evals)) if len(t_evals) else float(data["durations"].max())
            t_grid = _ensure_time_grid(df_all, t_l_bound=t_l_bound, t_eval=t_eval)

        with timer.time("global_beta_init"):
            beta_init = compute_global_beta_init(data)

        with timer.time("crossfit_event_residuals"):
            r_oof, residual_timings = crossfit_event_residuals(
                df_all,
                nn_config,
                K=K,
                time_scale=time_scale,
                residual_config=residual_config,
                return_timing=True,
            )

        folds = []
        per_fold_timings = []

        for k, (tr_idx, te_idx) in enumerate(fold_splits, start=1):
            fold_timer = StageTimer()

            outer_train_ids = ids[tr_idx]
            outer_eval_ids  = ids[te_idx]

            y_tr = y_strata[tr_idx]
            with fold_timer.time("inner_split"):
                try:
                    tr_in_ids, va_in_ids = train_test_split(
                        outer_train_ids,
                        test_size=inner_val_frac,
                        random_state=random_state + k,
                        shuffle=True,
                        stratify=y_tr
                    )
                except Exception:
                    tr_in_ids, va_in_ids = train_test_split(
                        outer_train_ids,
                        test_size=inner_val_frac,
                        random_state=random_state + k,
                        shuffle=True
                    )

            print(f"Fold {k}/{K}: train_inner={len(tr_in_ids)}, val_inner={len(va_in_ids)}, eval={len(outer_eval_ids)}")

            with fold_timer.time("train_main_model"):
                fit = fit_model_on_ids(
                    df_all=df_all,
                    data=data,
                    train_ids=tr_in_ids,
                    val_ids=va_in_ids,
                    nn_config=nn_config,
                    r_oof=r_oof,
                    beta_init=beta_init,
                    time_scale=time_scale,
                    sampler=None,
                    t_l_bound=t_l_bound,
                    return_timing=True,
                )
                model_k = fit["model"]

            with fold_timer.time("build_outer_data"):
                data_train_outer, data_eval = split_subject_data(data, outer_train_ids, outer_eval_ids)
                df_train_outer = df_all[df_all["id"].isin(outer_train_ids)].copy()
                df_eval = df_all[df_all["id"].isin(outer_eval_ids)].copy()
                df_train_outer = df_train_outer[(df_train_outer["st"] >= t_l_bound) & (df_train_outer["et"] <= t_eval)].copy()
                df_eval = df_eval[(df_eval["st"] >= t_l_bound) & (df_eval["et"] <= t_eval)].copy()

            with fold_timer.time("prepare_fold_for_one_step"):
                fold_obj = prepare_fold_for_one_step(
                    df_train_outer=df_train_outer,
                    data_train_outer=data_train_outer,
                    df_eval=df_eval,
                    data_eval=data_eval,
                    model=model_k,
                    t_grid=t_grid,
                    uniq_keys=uniq_keys,
                    key2j=key2j,
                    time_scale=time_scale
                )

            fold_obj["model"] = model_k
            fold_obj["outer_eval_ids"] = outer_eval_ids
            fold_obj["train_timing"] = fit.get("timings", {})
            fold_obj["fold_timing"] = fold_timer.times
            per_fold_timings.append({
                "fold": k,
                **{f"fit::{kk}": vv for kk, vv in fit.get("timings", {}).items()},
                **{f"fold::{kk}": vv for kk, vv in fold_timer.times.items()},
            })
            folds.append(fold_obj)

    def eval_curve(x0, z0, alpha=0.05):
        x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
        z0 = np.asarray(z0, dtype=np.float64).reshape(-1)
        j0 = key2j[tuple(x0.tolist())]

        m = len(t_grid)
        n = n_total

        H_hat_cf = np.zeros(m, dtype=np.float64)
        sum_phi = np.zeros(m, dtype=np.float64)
        sum_phi2 = np.zeros(m, dtype=np.float64)

        for fold in folds:
            model_k = fold["model"]
            n_ev = fold["n_eval"]

            X0 = np.repeat(x0.reshape(1,-1), m, axis=0).astype(np.float32)
            Z0 = np.repeat(z0.reshape(1,-1), m, axis=0).astype(np.float32)
            chi0 = predict_chi(model_k, X0, Z0, (t_grid/time_scale).reshape(-1,1).astype(np.float32), batch=4096)
            lam0 = np.exp(chi0) / time_scale
            H_hat_k = cumtrapz(lam0, t_grid)
            H_hat_cf += (n_ev / n) * H_hat_k

            pi0 = fold["pi_hat"][j0]
            A0 = fold["A_hat"][j0, :]
            w_time = lam0 / (pi0 * (A0 + 1e-18))

            g0 = fold["g_star"][j0, :, :]
            integrand = lam0.reshape(-1,1) * (z0.reshape(1,-1) - g0)
            B = cumtrapz_vec(integrand, t_grid).T
            U = fold["I_inv"] @ B

            ell_sum = fold["ell_sum"]
            ell_outer = fold["ell_outer"]

            sum_phi_theta = (ell_sum.reshape(1,-1) @ U).reshape(-1)

            sum_phi_theta2 = np.zeros(m, dtype=np.float64)
            for kk in range(m):
                u = U[:,kk]
                sum_phi_theta2[kk] = float(u.T @ ell_outer @ u)

            if j0 in fold["stratum_rows"]:
                info = fold["stratum_rows"][j0]
                subj_idx = info["subj_idx"]
                sid_local = info["sid_local"]
                tidx = info["tidx"]
                dM = info["dM"]
                n0 = subj_idx.size

                diff = np.zeros((n0, m+1), dtype=np.float64)
                if dM.size > 0:
                    contrib = w_time[tidx] * dM
                    np.add.at(diff, (sid_local, tidx), contrib)
                phi_g = np.cumsum(diff[:, :m], axis=1)

                sum_phi_g = phi_g.sum(axis=0)
                sum_phi_g2 = (phi_g**2).sum(axis=0)

                ell_sub = fold["ell"][subj_idx, :]
                cross = np.zeros(m, dtype=np.float64)
                block = 512
                for s in range(0, m, block):
                    e = min(m, s+block)
                    Q = ell_sub @ U[:, s:e]
                    cross[s:e] = np.sum(phi_g[:, s:e] * Q, axis=0)

                sum_phi_fold = sum_phi_g + sum_phi_theta
                sum_phi2_fold = sum_phi_g2 + sum_phi_theta2 + 2.0*cross
            else:
                sum_phi_fold = sum_phi_theta
                sum_phi2_fold = sum_phi_theta2

            sum_phi += sum_phi_fold
            sum_phi2 += sum_phi2_fold

        H1 = H_hat_cf + (sum_phi / n)
        se = np.sqrt(sum_phi2) / n

        zcrit = norm.ppf(1 - alpha/2)
        lo = H1 - zcrit*se
        hi = H1 + zcrit*se

        return dict(
            t_grid=t_grid,
            H_hat=H_hat_cf,
            H1=H1,
            se=se,
            lo=lo,
            hi=hi,
            t_l_bound=t_l_bound,
            t_eval=t_eval,
        )

    return dict(
        t_grid=t_grid,
        folds=folds,
        uniq_keys=uniq_keys,
        key2j=key2j,
        t_l_bound=t_l_bound,
        t_eval=t_eval,
        r_oof=r_oof,
        eval_curve=eval_curve,
        timings=timer.times,
        residual_timings=residual_timings,
        per_fold_timings=pd.DataFrame(per_fold_timings) if len(per_fold_timings) else pd.DataFrame(),
        beta_init=beta_init,
        n_total=n_total,
    )


# ===== Notebook cell 9 =====

# --- Weights-only save / load helpers ---

def _pickle_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def _pickle_load(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def _strip_cf_for_save(cf):
    cf_core = {}
    for k, v in cf.items():
        if k in ("eval_curve", "folds"):
            continue
        cf_core[k] = v

    folds_core = []
    for fold in cf["folds"]:
        fold_core = {}
        for k, v in fold.items():
            if k == "model":
                continue
            fold_core[k] = v
        folds_core.append(fold_core)

    cf_core["folds"] = folds_core
    return cf_core

def _make_eval_curve(cf, time_scale):
    folds = cf["folds"]
    t_grid = np.asarray(cf["t_grid"], dtype=np.float64)
    key2j = cf["key2j"]
    n_total = int(cf["n_total"])

    def eval_curve(x0, z0, alpha=0.05):
        x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
        z0 = np.asarray(z0, dtype=np.float64).reshape(-1)
        j0 = key2j[tuple(x0.tolist())]

        m = len(t_grid)
        n = n_total

        H_hat_cf = np.zeros(m, dtype=np.float64)
        sum_phi = np.zeros(m, dtype=np.float64)
        sum_phi2 = np.zeros(m, dtype=np.float64)

        for fold in folds:
            model_k = fold["model"]
            n_ev = fold["n_eval"]

            X0 = np.repeat(x0.reshape(1, -1), m, axis=0).astype(np.float32)
            Z0 = np.repeat(z0.reshape(1, -1), m, axis=0).astype(np.float32)
            chi0 = predict_chi(model_k, X0, Z0, (t_grid / time_scale).reshape(-1, 1).astype(np.float32), batch=4096)
            lam0 = np.exp(chi0) / time_scale
            H_hat_k = cumtrapz(lam0, t_grid)
            H_hat_cf += (n_ev / n) * H_hat_k

            pi0 = fold["pi_hat"][j0]
            A0 = fold["A_hat"][j0, :]
            w_time = lam0 / (pi0 * (A0 + 1e-18))

            g0 = fold["g_star"][j0, :, :]
            integrand = lam0.reshape(-1, 1) * (z0.reshape(1, -1) - g0)
            B = cumtrapz_vec(integrand, t_grid).T
            U = fold["I_inv"] @ B

            ell_sum = fold["ell_sum"]
            ell_outer = fold["ell_outer"]

            sum_phi_theta = (ell_sum.reshape(1, -1) @ U).reshape(-1)

            sum_phi_theta2 = np.zeros(m, dtype=np.float64)
            for kk in range(m):
                u = U[:, kk]
                sum_phi_theta2[kk] = float(u.T @ ell_outer @ u)

            if j0 in fold["stratum_rows"]:
                info = fold["stratum_rows"][j0]
                subj_idx = info["subj_idx"]
                sid_local = info["sid_local"]
                tidx = info["tidx"]
                dM = info["dM"]
                n0 = subj_idx.size

                diff = np.zeros((n0, m + 1), dtype=np.float64)
                if dM.size > 0:
                    contrib = w_time[tidx] * dM
                    np.add.at(diff, (sid_local, tidx), contrib)
                phi_g = np.cumsum(diff[:, :m], axis=1)

                sum_phi_g = phi_g.sum(axis=0)
                sum_phi_g2 = (phi_g ** 2).sum(axis=0)

                ell_sub = fold["ell"][subj_idx, :]
                cross = np.zeros(m, dtype=np.float64)

                block = 512
                for s in range(0, m, block):
                    e = min(m, s + block)
                    Q = ell_sub @ U[:, s:e]
                    cross[s:e] = np.sum(phi_g[:, s:e] * Q, axis=0)

                sum_phi_fold = sum_phi_g + sum_phi_theta
                sum_phi2_fold = sum_phi_g2 + sum_phi_theta2 + 2.0 * cross
            else:
                sum_phi_fold = sum_phi_theta
                sum_phi2_fold = sum_phi_theta2

            sum_phi += sum_phi_fold
            sum_phi2 += sum_phi2_fold

        H1 = H_hat_cf + (sum_phi / n)
        se = np.sqrt(sum_phi2) / n

        zcrit = norm.ppf(1 - alpha / 2)
        lo = H1 - zcrit * se
        hi = H1 + zcrit * se

        return dict(
            t_grid=t_grid,
            H_hat=H_hat_cf,
            H1=H1,
            se=se,
            lo=lo,
            hi=hi,
            t_l_bound=cf["t_l_bound"],
            t_eval=cf["t_eval"],
        )

    return eval_curve

def save_crossfit_bundle(base_dir, cf, df_all, data, nn_config, time_scale, residual_config=None, extra=None):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for i, fold in enumerate(cf["folds"], start=1):
        weights_path = models_dir / f"fold_{i:02d}.weights.h5"
        fold["model"].save_weights(weights_path)

    cf_core = _strip_cf_for_save(cf)
    cf_core["n_total"] = int(data["covs_X"].shape[0])

    payload = {
        "cf_core": cf_core,
        "df_all": df_all,
        "data": data,
        "nn_config": nn_config,
        "residual_config": residual_config,
        "time_scale": time_scale,
        "extra": extra,
    }
    _pickle_save(payload, base_dir / "bundle.pkl")
    print(f"Saved cross-fit bundle to: {base_dir.resolve()}")

def load_crossfit_bundle(base_dir, compile_models=False):
    base_dir = Path(base_dir)
    payload = _pickle_load(base_dir / "bundle.pkl")

    cf_core = payload["cf_core"]
    df_all = payload["df_all"]
    data = payload["data"]
    nn_config = payload["nn_config"]
    residual_config = payload.get("residual_config", None)
    time_scale = payload["time_scale"]
    extra = payload.get("extra", None)

    input_dim_X = int(data["covs_X"].shape[1])
    input_dim_Z = int(data["covs_Z"].shape[1])
    lmbd_l1 = float(nn_config.get("lmbd_L1", 0.0))

    rebuilt_folds = []
    for i, fold_core in enumerate(cf_core["folds"], start=1):
        model, _, _, _ = build_model(
            input_dim_X=input_dim_X,
            input_dim_Z=input_dim_Z,
            nn_config=nn_config,
            beta_init=0.0,
            lmbd_l1=lmbd_l1,
        )
        weights_path = base_dir / "models" / f"fold_{i:02d}.weights.h5"
        model.load_weights(weights_path)

        if compile_models:
            model, _ = compile_baseliness(model, nn_config)

        fold = dict(fold_core)
        fold["model"] = model
        rebuilt_folds.append(fold)

    cf = dict(cf_core)
    cf["folds"] = rebuilt_folds
    cf["eval_curve"] = _make_eval_curve(cf, time_scale=time_scale)

    return {
        "cf": cf,
        "df_all": df_all,
        "data": data,
        "nn_config": nn_config,
        "residual_config": residual_config,
        "time_scale": time_scale,
        "extra": extra,
    }


# ===== Notebook cell 10 =====

def _ensure_time_grid(df_all, t_l_bound, t_eval):
    df2 = df_all[(df_all["st"] >= t_l_bound) & (df_all["et"] <= t_eval)].copy()
    t_grid = np.unique(df2["et"].values.astype(np.float64))
    t_grid = np.sort(t_grid)
    return t_grid


# ===== Notebook cell 13 =====

# --- Monte Carlo experiment config ---

NN_CONFIG = {
    "hidden_layers_nodes": 64,
    "n_hidden_layers": 4,
    "learning_rate": 0.001,
    "activation": "relu",
    "optimizer": "adam",
    "batch_size": 10000,   # large batch often works much better on GPU here
    "patience": 50,
    "dropout": 0.0,
    "lmbd_L1": 0.0,
    "lmbd_cali": 0.0,
    "lmbd_cor": 0.0,
    "epochs": 2000,
    "jit_compile": ENABLE_XLA,
}

RESIDUAL_CONFIG = {
    "hidden_layers_nodes": 32,
    "n_hidden_layers": 2,
    "learning_rate": 0.001,
    "activation": "relu",
    "optimizer": "adam",
    "batch_size": 10000,
    "patience": 15,
    "dropout": 0.0,
    "lmbd_L1": 0.0,
    "epochs": 300,
    "jit_compile": ENABLE_XLA,
}

# Number of independent Monte Carlo datasets to simulate / fit
N_MONTE_CARLO_RUNS = 5

# Number of NEW patients to sample per Monte Carlo run for saved curves
N_NEW_SAMPLES_PER_RUN = 200

# Dataset size per Monte Carlo run
N_SUBJECTS_PER_RUN = 1000

# Outer cross-fitting folds for the estimator
CROSSFIT_K = 5

# Root directory; a timestamped subdirectory will be created under this path
MONTE_CARLO_ROOT_DIR = "saved_runs/discreteX_crossfit5_mc"

# If True, save wide run artifacts too (cf core, data, configs). This can take more disk.
SAVE_FULL_RUN_BUNDLE = False


# ===== Notebook cell 14 =====

# --- Monte Carlo save / evaluation helpers ---

import json
from pathlib import Path
from datetime import datetime

def make_mc_output_dir(root_dir):
    root = Path(root_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / f"mc_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def save_df_auto(df, path_without_suffix, index=False):
    path_without_suffix = Path(path_without_suffix)
    try:
        out_path = path_without_suffix.with_suffix(".parquet")
        df.to_parquet(out_path, index=index)
        return out_path
    except Exception:
        out_path = path_without_suffix.with_suffix(".csv.gz")
        df.to_csv(out_path, index=index, compression="gzip")
        return out_path

def evaluate_new_samples_survival_curves(cf, sampler, n_new, mc_run):
    t_grid = np.asarray(cf["t_grid"], dtype=np.float64)
    eval_curve = cf["eval_curve"]

    X_new, Z_new = sampler.sample_covs(n_new)
    X_new = np.asarray(X_new, dtype=np.float64)
    Z_new = np.asarray(Z_new, dtype=np.float64)

    rows = []
    run_summary_rows = []

    for sample_id in range(n_new):
        x0 = X_new[sample_id]
        z0 = Z_new[sample_id]
        out = eval_curve(x0, z0)

        H_hat = np.asarray(out["H_hat"], dtype=np.float64).reshape(-1)
        H1    = np.asarray(out["H1"], dtype=np.float64).reshape(-1)
        lo    = np.asarray(out["lo"], dtype=np.float64).reshape(-1)
        hi    = np.asarray(out["hi"], dtype=np.float64).reshape(-1)
        se    = np.asarray(out["se"], dtype=np.float64).reshape(-1)

        H_true = sampler.cum_hazard(
            t_grid,
            (x0.reshape(1, -1), z0.reshape(1, -1))
        ).reshape(-1).astype(np.float64)

        S_true   = np.exp(-H_true)
        S_plugin = np.exp(-H_hat)
        S_onestep = np.exp(-H1)

        # Since H in [lo, hi] implies S in [exp(-hi), exp(-lo)]
        S_lo = np.exp(-hi)
        S_hi = np.exp(-lo)

        covered = ((H_true >= lo) & (H_true <= hi)).astype(np.int8)

        row_df = pd.DataFrame({
            "mc_run": mc_run,
            "sample_id": sample_id,
            "t": t_grid,
            "H_true": H_true,
            "H_plugin": H_hat,
            "H_onestep": H1,
            "H_lo": lo,
            "H_hi": hi,
            "se": se,
            "S_true": S_true,
            "S_plugin": S_plugin,
            "S_onestep": S_onestep,
            "S_lo": S_lo,
            "S_hi": S_hi,
            "covered": covered,
        })

        for j, val in enumerate(x0):
            row_df[f"X{j+1}"] = float(val)
        for j, val in enumerate(z0):
            row_df[f"Z{j+1}"] = float(val)

        rows.append(row_df)

        run_summary_rows.append({
            "mc_run": mc_run,
            "sample_id": sample_id,
            "avg_coverage": float(covered.mean()),
            "final_t": float(t_grid[-1]),
            "final_S_true": float(S_true[-1]),
            "final_S_plugin": float(S_plugin[-1]),
            "final_S_onestep": float(S_onestep[-1]),
            "final_S_lo": float(S_lo[-1]),
            "final_S_hi": float(S_hi[-1]),
        })

    curves_df = pd.concat(rows, axis=0, ignore_index=True)
    patient_summary_df = pd.DataFrame(run_summary_rows)
    return curves_df, patient_summary_df

def run_one_monte_carlo_fit(
    mc_run,
    sampler,
    n_subjects,
    nn_config,
    residual_config,
    K,
    time_scale,
    inner_val_frac=0.33,
):
    np.random.seed(SEED + mc_run)
    tf.keras.utils.set_random_seed(SEED + mc_run)

    df_all_run, data_run = simulate(int(n_subjects), sampler)

    st = time.time()
    cf_run = fit_discreteX_one_step_crossfit5(
        df_all=df_all_run,
        data=data_run,
        nn_config=copy.deepcopy(nn_config),
        K=K,
        time_scale=time_scale,
        inner_val_frac=inner_val_frac,
        random_state=SEED + mc_run,
        residual_config=copy.deepcopy(residual_config),
    )
    elapsed = time.time() - st

    return df_all_run, data_run, cf_run, elapsed


# ===== simlib wrappers for SageMaker / local runs =====

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple


def configure_runtime(
    seed: int = 0,
    use_cpu_only: bool = False,
    enable_xla: bool = False,
    enable_mixed_precision: bool = False,
    verbose: bool = True,
) -> None:
    """
    Best-effort runtime setup. Call early in a fresh process.
    """
    global SEED, USE_CPU_ONLY, ENABLE_XLA, ENABLE_MIXED_PRECISION

    SEED = int(seed)
    USE_CPU_ONLY = bool(use_cpu_only)
    ENABLE_XLA = bool(enable_xla)
    ENABLE_MIXED_PRECISION = bool(enable_mixed_precision)

    if USE_CPU_ONLY:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    np.random.seed(SEED)
    random.seed(SEED)
    try:
        tf.keras.utils.set_random_seed(SEED)
    except Exception:
        tf.random.set_seed(SEED)

    if ENABLE_XLA:
        try:
            tf.config.optimizer.set_jit(True)
        except Exception:
            pass

    if ENABLE_MIXED_PRECISION:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    if verbose:
        print("TensorFlow:", tf.__version__)
        print("CPUs:", tf.config.list_physical_devices("CPU"))
        print("GPUs:", tf.config.list_physical_devices("GPU"))
        print("USE_CPU_ONLY =", USE_CPU_ONLY)
        print("ENABLE_XLA =", ENABLE_XLA)
        print("ENABLE_MIXED_PRECISION =", ENABLE_MIXED_PRECISION)
        print("SEED =", SEED)


@dataclass
class MonteCarloJobConfig:
    seed: int = 0
    n_subjects: int = 6000
    n_new_samples: int = 200
    crossfit_k: int = 5
    time_scale: float = 30.0
    inner_val_frac: float = 0.33
    use_cpu_only: bool = False
    enable_xla: bool = False
    enable_mixed_precision: bool = False
    save_full_run_bundle: bool = False
    sampler_name: str = "nonlinear_nonph"
    output_dir: Optional[str] = None


DEFAULT_NN_CONFIG = {
    "hidden_layers_nodes": 64,
    "n_hidden_layers": 4,
    "learning_rate": 0.001,
    "activation": "relu",
    "optimizer": "adam",
    "batch_size": 10000,
    "patience": 50,
    "dropout": 0.0,
    "lmbd_L1": 0.0,
    "lmbd_cali": 0.0,
    "lmbd_cor": 0.0,
    "epochs": 2000,
    "jit_compile": False,
}

DEFAULT_RESIDUAL_CONFIG = {
    "hidden_layers_nodes": 32,
    "n_hidden_layers": 2,
    "learning_rate": 0.001,
    "activation": "relu",
    "optimizer": "adam",
    "batch_size": 10000,
    "patience": 15,
    "dropout": 0.0,
    "lmbd_L1": 0.0,
    "epochs": 300,
    "jit_compile": False,
}


def build_sampler(name: str = "nonlinear_nonph"):
    name = str(name).lower()
    if name in {"nonlinear_nonph", "default"}:
        return SimStudyNonLinearNonPH()
    if name in {"nlph"}:
        return SimStudyNLPH()
    if name in {"nonlinear_ph"}:
        return SimStudyNonLinearPH()
    raise ValueError(f"Unknown sampler_name={name!r}")


def run_sample(
    n: int,
    nn_config: Optional[Dict[str, Any]] = None,
    *,
    residual_config: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    n_new_samples: int = 200,
    crossfit_k: int = 5,
    time_scale: float = 30.0,
    inner_val_frac: float = 0.33,
    use_cpu_only: bool = False,
    enable_xla: bool = False,
    enable_mixed_precision: bool = False,
    save_full_run_bundle: bool = False,
    sampler_name: str = "nonlinear_nonph",
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run one Monte Carlo training-set draw and evaluate on new sampled patients.
    Returns in-memory results and optionally saves them under output_dir.
    """
    configure_runtime(
        seed=seed,
        use_cpu_only=use_cpu_only,
        enable_xla=enable_xla,
        enable_mixed_precision=enable_mixed_precision,
        verbose=True,
    )

    nn_cfg = copy.deepcopy(DEFAULT_NN_CONFIG)
    if nn_config:
        nn_cfg.update(nn_config)
    nn_cfg["jit_compile"] = bool(enable_xla or nn_cfg.get("jit_compile", False))

    resid_cfg = copy.deepcopy(DEFAULT_RESIDUAL_CONFIG)
    if residual_config:
        resid_cfg.update(residual_config)
    resid_cfg["jit_compile"] = bool(enable_xla or resid_cfg.get("jit_compile", False))

    sampler = build_sampler(sampler_name)

    # preserve notebook behavior: mc_run=0 and base seed=seed
    globals()["SEED"] = int(seed)
    df_all, data, cf, elapsed = run_one_monte_carlo_fit(
        mc_run=0,
        sampler=sampler,
        n_subjects=int(n),
        nn_config=nn_cfg,
        residual_config=resid_cfg,
        K=int(crossfit_k),
        time_scale=float(time_scale),
        inner_val_frac=float(inner_val_frac),
    )

    curves_df, patient_summary_df = evaluate_new_samples_survival_curves(
        cf=cf,
        sampler=sampler,
        n_new=int(n_new_samples),
        mc_run=int(seed),
    )

    result = {
        "seed": int(seed),
        "elapsed_sec": float(elapsed),
        "n_subjects": int(n),
        "n_new_samples": int(n_new_samples),
        "crossfit_k": int(crossfit_k),
        "time_scale": float(time_scale),
        "nn_config": nn_cfg,
        "residual_config": resid_cfg,
        "run_summary": {
            "seed": int(seed),
            "elapsed_sec": float(elapsed),
            "t_grid_size": int(len(cf["t_grid"])),
            "t_l_bound": float(cf["t_l_bound"]),
            "t_eval": float(cf["t_eval"]),
            "mean_patient_curve_coverage": float(patient_summary_df["avg_coverage"].mean()),
        },
        "curves_df": curves_df,
        "patient_summary_df": patient_summary_df,
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        config_payload = {
            "job": asdict(MonteCarloJobConfig(
                seed=seed,
                n_subjects=n,
                n_new_samples=n_new_samples,
                crossfit_k=crossfit_k,
                time_scale=time_scale,
                inner_val_frac=inner_val_frac,
                use_cpu_only=use_cpu_only,
                enable_xla=enable_xla,
                enable_mixed_precision=enable_mixed_precision,
                save_full_run_bundle=save_full_run_bundle,
                sampler_name=sampler_name,
                output_dir=str(out),
            )),
            "nn_config": nn_cfg,
            "residual_config": resid_cfg,
        }
        (out / "config.json").write_text(json.dumps(config_payload, indent=2))
        save_df_auto(curves_df, out / "survival_curves")
        save_df_auto(patient_summary_df, out / "patient_summary")
        with open(out / "run_summary.json", "w") as f:
            json.dump(result["run_summary"], f, indent=2)

        if save_full_run_bundle:
            save_crossfit_bundle(
                out / "bundle",
                cf=cf,
                df_all=df_all,
                data=data,
                nn_config=nn_cfg,
                residual_config=resid_cfg,
                time_scale=time_scale,
                extra={"seed": seed, "elapsed_sec": elapsed},
            )

    return result


def run_local_batch(
    seeds,
    output_root: str,
    *,
    n_subjects: int,
    n_new_samples: int,
    nn_config: Optional[Dict[str, Any]] = None,
    residual_config: Optional[Dict[str, Any]] = None,
    crossfit_k: int = 5,
    time_scale: float = 30.0,
    inner_val_frac: float = 0.33,
    use_cpu_only: bool = False,
    enable_xla: bool = False,
    enable_mixed_precision: bool = False,
    save_full_run_bundle: bool = False,
    sampler_name: str = "nonlinear_nonph",
) -> pd.DataFrame:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        run_dir = output_root / f"seed_{int(seed):05d}"
        print("=" * 100)
        print(f"Running local seed={seed} -> {run_dir}")
        result = run_sample(
            n=n_subjects,
            nn_config=nn_config,
            residual_config=residual_config,
            seed=int(seed),
            n_new_samples=n_new_samples,
            crossfit_k=crossfit_k,
            time_scale=time_scale,
            inner_val_frac=inner_val_frac,
            use_cpu_only=use_cpu_only,
            enable_xla=enable_xla,
            enable_mixed_precision=enable_mixed_precision,
            save_full_run_bundle=save_full_run_bundle,
            sampler_name=sampler_name,
            output_dir=str(run_dir),
        )
        rows.append(result["run_summary"])

    summary_df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    save_df_auto(summary_df, output_root / "run_summary_all")
    return summary_df
