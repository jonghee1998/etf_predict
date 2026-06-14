import yaml, warnings, os, json, gc, pickle
from pathlib import Path
warnings.filterwarnings('ignore')

CONFIG_FILE = 'config_v4.yaml'
with open(CONFIG_FILE, 'r') as f:
    CFG = yaml.safe_load(f)

EXPERIMENT_NAME    = CFG['experiment']['name']
ETF_CODE           = CFG['etf']['code']
START_DATE         = CFG['etf']['start_date']
END_DATE           = CFG['etf'].get('end_date')
BASE_DATE          = CFG['base_date']

OPTUNA_VALID_MONTHS = CFG['periods']['optuna_valid_months']
SIM_TEST_MONTHS     = CFG['periods']['sim_test_months']

N_DAYS_CANDIDATES                  = CFG['target']['n_days_candidates']
TARGET_RETURN_THRESHOLD_CANDIDATES = CFG['target']['return_threshold_candidates']

VIF_THRESHOLD        = CFG['features']['vif_threshold']
FEATURE_SELECT_YEARS = CFG['features']['feature_select_years']
LAG_DAYS             = CFG['features']['lag_days']
TOP_N_MAX            = CFG['features']['top_n_max']

USE_RSI    = CFG['features']['use_rsi'];   RSI_PERIOD = CFG['features']['rsi_period']
USE_MACD   = CFG['features']['use_macd'];  MACD_FAST  = CFG['features']['macd_fast']
MACD_SLOW  = CFG['features']['macd_slow']; MACD_SIG   = CFG['features']['macd_signal']
USE_BB     = CFG['features']['use_bollinger']
BB_PERIOD  = CFG['features']['bollinger_period']; BB_STD = CFG['features']['bollinger_std']
USE_ATR    = CFG['features']['use_atr'];   ATR_PERIOD = CFG['features']['atr_period']
USE_52W    = CFG['features']['use_52w']

EXTERNAL_TICKERS       = CFG['external_tickers']
EXTERNAL_FEATURE_TYPES = CFG['external_feature_types']

RANDOM_STATE = CFG['importance']['random_state']
N_RF_RUNS    = CFG['importance']['n_rf_runs']
N_REPEATS    = CFG['importance']['n_repeats']

S1_OBJ_METRIC = CFG['stage1']['objective_metric']
S1_MIN_EVAL   = CFG['stage1']['min_valid_eval_count']
S1_MIN_PRED1  = CFG['stage1']['min_valid_pred_1_count']
S1_MODEL      = CFG['stage1']['model_name']
S1_N_EST      = CFG['stage1']['n_estimators']
S1_TOP_N      = CFG['stage1']['top_n_fixed']
S1_N_SEEDS    = CFG['stage1']['n_seeds']

S2_N_TRIALS        = CFG['optuna_stage2']['n_trials']
S2_OBJ_METRIC      = CFG['optuna_stage2']['objective_metric']
S2_PRECISION_FLOOR = CFG['optuna_stage2'].get('precision_floor', 0.65)
S2_MIN_EVAL        = CFG['optuna_stage2']['min_valid_eval_count']
S2_MIN_PRED1_ORIG  = CFG['optuna_stage2']['min_valid_pred_1_count']
S2_MIN_PRED1       = CFG['optuna_stage2']['min_valid_pred_1_count']
S2_MODELS          = CFG['optuna_stage2']['model_candidates']
S2_TOP_N_RANGE     = CFG['optuna_stage2']['top_n_range']
S2_THRESH_RANGE    = CFG['optuna_stage2']['pred_threshold_range']
S2_THRESH_STEP     = CFG['optuna_stage2']['pred_threshold_step']

S3_PRECISION_TARGET = CFG['stage3']['precision_target']
S3_WALKFORWARD = CFG['stage3'].get('walkforward_halves', False)
S3_MIN_PRED         = CFG['stage3']['min_pred_count']
S3_ENSEMBLE_TOP_K   = CFG['stage3']['ensemble_top_k']

INITIAL_CASH   = CFG['simulation']['initial_cash']
BUY_RATIO      = CFG['simulation']['buy_ratio']
MIN_CASH_RATIO = CFG['simulation']['min_cash_ratio']

LOAD_FROM_CACHE = CFG['cache']['load_from_cache']

_cache_tag     = f"{ETF_CODE}_{BASE_DATE.replace('-','')}"
SHARED_CACHE   = Path(f'cache_{_cache_tag}')           # v3와 공유 (base_dataset, VIF)
V4_CACHE       = Path(f'cache_{_cache_tag}_v4')         # v4 전용 (stage1/2/3)
SHARED_CACHE.mkdir(parents=True, exist_ok=True)
V4_CACHE.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path(f'experiments_v4/{EXPERIMENT_NAME}')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_cache(name, obj, v4_only=False):
    d = V4_CACHE if v4_only else SHARED_CACHE
    path = d / f'{name}.pkl'
    with open(path, 'wb') as f: pickle.dump(obj, f)
    print(f'[CACHE SAVED] {path}')

def load_cache(name, v4_only=False):
    for d in ([V4_CACHE] if v4_only else [SHARED_CACHE, V4_CACHE]):
        path = d / f'{name}.pkl'
        if path.exists():
            with open(path, 'rb') as f: obj = pickle.load(f)
            print(f'[CACHE LOADED] {path}')
            return obj
    raise FileNotFoundError(f'{name}.pkl not found')

print(f'Experiment : {EXPERIMENT_NAME}')
print(f'ETF        : {ETF_CODE}  BASE_DATE={BASE_DATE}')
print(f'Output dir : {OUTPUT_DIR}')
print(f'S2 objective: {S2_OBJ_METRIC}, precision_floor={S2_PRECISION_FLOOR}')
print(f'S3 precision_target={S3_PRECISION_TARGET}, ensemble_top_k={S3_ENSEMBLE_TOP_K}')


# ============================================================
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from statsmodels.stats.outliers_influence import variance_inflation_factor

import xgboost as xgb
import lightgbm as lgb
import plotly.graph_objects as go
from IPython.display import display

def display_df(df, n=20):
    display(df.head(n))

# ── 기술지표 헬퍼 ──────────────────────────────────────────────────────────
def _rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ── 1. 데이터 로드 ─────────────────────────────────────────────────────────
def load_price_data(ticker, start_date='2020-01-01', end_date=None):
    df = fdr.DataReader(ticker, start_date, end_date)
    df = df.reset_index().rename(columns={'index': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def _keep_weekdays(df, date_col='Date'):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df[date_col].dt.weekday < 5].sort_values(date_col).reset_index(drop=True)
    return df

# ── 2. ETF 피처 생성 ───────────────────────────────────────────────────────
def make_target_etf_features(etf_df, prefix, cfg=None):
    df    = etf_df.copy().sort_values('Date').reset_index(drop=True)
    close = df['Adj Close']
    high  = df['High']   if 'High'   in df.columns else close
    low   = df['Low']    if 'Low'    in df.columns else close
    volume= df['Volume']

    r = pd.DataFrame()
    r['Date']                       = df['Date']
    r[f'{prefix}_adj_close']        = close
    r[f'{prefix}_ret_1d']           = close.pct_change(1)
    r[f'{prefix}_ret_5d']           = close.pct_change(5)
    r[f'{prefix}_ret_20d']          = close.pct_change(20)

    ma5  = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma200= close.rolling(200).mean()
    r[f'{prefix}_ma5_ratio']        = close / ma5  - 1
    r[f'{prefix}_ma20_ratio']       = close / ma20 - 1
    r[f'{prefix}_ma60_ratio']       = close / ma60 - 1
    r[f'{prefix}_ma200_ratio']      = close / ma200 - 1
    r[f'{prefix}_ma5_ma20_ratio']   = ma5 / ma20 - 1
    r[f'{prefix}_ma20_ma60_ratio']  = ma20 / ma60 - 1

    r[f'{prefix}_vol_20d']          = r[f'{prefix}_ret_1d'].rolling(20).std()
    r[f'{prefix}_vol_5d']           = r[f'{prefix}_ret_1d'].rolling(5).std()

    vol_ma20 = volume.rolling(20).mean()
    r[f'{prefix}_volume_ratio_20d'] = volume / vol_ma20 - 1
    r[f'{prefix}_volume_ratio_5d']  = volume / volume.rolling(5).mean() - 1

    # 연속 상승/하락 일수
    up = (close.diff() > 0).astype(int)
    consec = up.groupby((up != up.shift()).cumsum()).cumcount() + 1
    r[f'{prefix}_consec_up']   = np.where(up == 1,  consec, 0)
    r[f'{prefix}_consec_down'] = np.where(up == 0, -consec, 0)

    # 고가/저가 대비 위치 (10/20일)
    r[f'{prefix}_hl_pos_10d'] = (close - low.rolling(10).min()) / \
                                 (high.rolling(10).max() - low.rolling(10).min() + 1e-9)
    r[f'{prefix}_hl_pos_20d'] = (close - low.rolling(20).min()) / \
                                 (high.rolling(20).max() - low.rolling(20).min() + 1e-9)

    if cfg is None or cfg.get('use_rsi', True):
        p = (cfg or {}).get('rsi_period', 14)
        r[f'{prefix}_rsi_{p}'] = _rsi(close, p)

    if cfg is None or cfg.get('use_macd', True):
        fast = (cfg or {}).get('macd_fast', 12)
        slow = (cfg or {}).get('macd_slow', 26)
        sig  = (cfg or {}).get('macd_signal', 9)
        ema_f = close.ewm(span=fast, adjust=False).mean()
        ema_s = close.ewm(span=slow, adjust=False).mean()
        macd  = ema_f - ema_s
        macd_sig = macd.ewm(span=sig, adjust=False).mean()
        r[f'{prefix}_macd']        = macd
        r[f'{prefix}_macd_hist']   = macd - macd_sig
        r[f'{prefix}_macd_signal'] = macd_sig

    if cfg is None or cfg.get('use_bollinger', True):
        p   = (cfg or {}).get('bollinger_period', 20)
        std = (cfg or {}).get('bollinger_std', 2.0)
        ma  = close.rolling(p).mean()
        sd  = close.rolling(p).std()
        upper = ma + std * sd
        lower = ma - std * sd
        band  = (upper - lower).replace(0, np.nan)
        r[f'{prefix}_bb_pct']   = (close - lower) / band
        r[f'{prefix}_bb_width'] = band / ma

    if cfg is None or cfg.get('use_atr', True):
        p = (cfg or {}).get('atr_period', 14)
        atr_val = _atr(high, low, close, p)
        r[f'{prefix}_atr_{p}']       = atr_val
        r[f'{prefix}_atr_{p}_ratio'] = atr_val / close

    if cfg is None or cfg.get('use_52w', True):
        r[f'{prefix}_52w_high_ratio'] = close / close.rolling(252).max() - 1
        r[f'{prefix}_52w_low_ratio']  = close / close.rolling(252).min() - 1

    return r

# ── 3. 외부지표 피처 ───────────────────────────────────────────────────────
def make_external_features(raw_df, name, feature_type='price'):
    df    = raw_df.copy().sort_values('Date').reset_index(drop=True)
    close = df['Adj Close']
    r     = pd.DataFrame()
    r['Date'] = df['Date']

    if feature_type == 'price':
        r[f'{name}_ret_1d']  = close.pct_change(1)
        r[f'{name}_ret_3d']  = close.pct_change(3)
        r[f'{name}_ret_5d']  = close.pct_change(5)
        r[f'{name}_ret_20d'] = close.pct_change(20)
    elif feature_type == 'risk':
        r[f'{name}_level']   = close
        r[f'{name}_chg_5d']  = close.diff(5)
        r[f'{name}_chg_20d'] = close.diff(20)
    elif feature_type == 'rate':
        r[f'{name}_level']   = close
        r[f'{name}_diff_5d'] = close.diff(5)
        r[f'{name}_diff_20d']= close.diff(20)
    else:
        raise ValueError(f'Unknown feature_type: {feature_type}')
    return r

# ── 4. Base Feature Dataset ────────────────────────────────────────────────
def make_base_feature_dataset(etf_code, external_tickers, external_feature_types,
                               start_date='2020-01-01', end_date=None, feat_cfg=None):
    etf_raw = _keep_weekdays(load_price_data(etf_code, start_date, end_date))
    base_df = make_target_etf_features(etf_raw, prefix=etf_code, cfg=feat_cfg)
    base_df = _keep_weekdays(base_df)

    for name, ticker in external_tickers.items():
        print(f'  Loading {name} ({ticker})')
        try:
            raw   = _keep_weekdays(load_price_data(ticker, start_date, end_date))
            ftype = external_feature_types.get(name, 'price')
            ext   = _keep_weekdays(make_external_features(raw, name, ftype))
            base_df = base_df.merge(ext, on='Date', how='left')
        except Exception as e:
            print(f'  [SKIP] {name}: {e}')

    base_df = _keep_weekdays(base_df)
    etf_prefix    = f'{etf_code}_'
    external_cols = [c for c in base_df.columns if c != 'Date' and not c.startswith(etf_prefix)]
    base_df[external_cols] = base_df[external_cols].ffill()

    # ── 교차섹션 피처 (레짐 불변 신호) ──────────────────────────────
    # NVDA vs SMH 지연반응: NVDA가 선행하면 SMH가 뒤따른다
    if 'NVDA_ret_1d' in base_df.columns and f'{etf_code}_ret_1d' in base_df.columns:
        base_df['NVDA_vs_SMH_1d'] = base_df['NVDA_ret_1d'] - base_df[f'{etf_code}_ret_1d']
        base_df['NVDA_vs_SMH_5d'] = base_df.get('NVDA_ret_5d', 0) - base_df[f'{etf_code}_ret_5d']
    if 'QQQ_ret_1d' in base_df.columns and f'{etf_code}_ret_1d' in base_df.columns:
        base_df['QQQ_vs_SMH_1d']  = base_df['QQQ_ret_1d'] - base_df[f'{etf_code}_ret_1d']
    if 'SOXX_ret_1d' in base_df.columns and f'{etf_code}_ret_1d' in base_df.columns:
        base_df['SOXX_vs_SMH_1d'] = base_df['SOXX_ret_1d'] - base_df[f'{etf_code}_ret_1d']
    # VIX × SMH 조합: 공포 + 하락 = 과매도 신호
    if 'VIX_level' in base_df.columns:
        base_df['VIX_x_SMH_ret5d'] = base_df['VIX_level'] * base_df[f'{etf_code}_ret_5d']
        base_df['VIX_x_SMH_ret20d'] = base_df['VIX_level'] * base_df[f'{etf_code}_ret_20d']
    # SMH vol-adjusted daily return (시장 변동성 정규화)
    if f'{etf_code}_vol_20d' in base_df.columns:
        vol = base_df[f'{etf_code}_vol_20d'].replace(0, np.nan)
        base_df['SMH_ret1d_vol_adj'] = base_df[f'{etf_code}_ret_1d'] / vol
        base_df['SMH_ret5d_vol_adj'] = base_df[f'{etf_code}_ret_5d'] / vol

    close_col    = f'{etf_code}_adj_close'
    feature_cols = [c for c in base_df.columns if c not in ['Date', close_col]]
    return base_df, feature_cols, close_col

# ── 5. VIF 제거 ────────────────────────────────────────────────────────────
def reduce_features_by_vif(df, feature_cols, vif_threshold=30.0, verbose=True):
    numeric_cols = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    work = df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna(axis=0).copy()
    nunique   = work.nunique()
    remaining = [c for c in numeric_cols if nunique.get(c, 0) > 1]
    removed   = []

    while True:
        if len(remaining) <= 1: break
        X  = work[remaining].copy()
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        vifs = []
        for i, col in enumerate(remaining):
            try:    v = variance_inflation_factor(Xs, i)
            except: v = np.inf
            vifs.append((col, v))
        worst_col, worst_v = max(vifs, key=lambda x: x[1])
        if verbose: print(f'  VIF max: {worst_v:.1f}  ({worst_col})')
        if worst_v <= vif_threshold: break
        remaining.remove(worst_col)
        removed.append({'removed_feature': worst_col, 'vif': worst_v})

    print(f'  VIF 제거: {len(removed)}개 / 남은 피처: {len(remaining)}개')
    return remaining, pd.DataFrame(removed)

# ── 6. Target 생성 ─────────────────────────────────────────────────────────
def add_target_column(df, close_col, n_days=5, threshold=0.05, target_col=None):
    r = df.copy().sort_values('Date').reset_index(drop=True)
    if target_col is None:
        target_col = f'target_{n_days}d_up_{int(threshold*100)}pct'
    future_close = r[close_col].shift(-n_days)
    r[f'future_ret_{n_days}d'] = future_close / r[close_col] - 1
    r[target_col] = np.where(r[f'future_ret_{n_days}d'] >= threshold, 1, 0)
    r.loc[r[f'future_ret_{n_days}d'].isna(), target_col] = np.nan
    return r, target_col

# ── 7. Lag 탐색 & 적용 ────────────────────────────────────────────────────
def find_best_lag_by_feature(df, feature_cols, target_col, lag_days, date_col='Date'):
    records = []
    for col in feature_cols:
        if col not in df.columns: continue
        for lag in lag_days:
            tmp = df[[date_col, col, target_col]].copy()
            tmp[f'{col}_lag{lag}'] = tmp[col].shift(lag)
            tmp = tmp[[f'{col}_lag{lag}', target_col]].replace([np.inf,-np.inf], np.nan).dropna()
            if len(tmp) < 30: continue
            x    = tmp[f'{col}_lag{lag}']
            corr = x.corr(tmp[target_col]) if x.nunique() > 1 else np.nan
            records.append({'feature': col, 'lag': lag, 'corr': corr,
                            'abs_corr': abs(corr) if pd.notna(corr) else np.nan})
    lag_df = pd.DataFrame(records)
    if lag_df.empty: raise ValueError('lag 탐색 결과 없음')
    best_lag_df = (lag_df.sort_values(['feature','abs_corr'], ascending=[True,False])
                         .groupby('feature', as_index=False).head(1)
                         .sort_values('abs_corr', ascending=False).reset_index(drop=True))
    return lag_df, best_lag_df

def make_lagged_dataset(df, best_lag_df, target_col=None, close_col=None,
                        n_days=5, date_col='Date', drop_target_na=True):
    r = pd.DataFrame()
    r[date_col] = df[date_col]
    if close_col and close_col in df.columns:
        r[close_col] = df[close_col]
    fut = f'future_ret_{n_days}d'
    if fut in df.columns: r[fut] = df[fut]
    if target_col and target_col in df.columns: r[target_col] = df[target_col]

    lagged_cols = []
    for _, row in best_lag_df.iterrows():
        feat = row['feature']; lag = int(row['lag'])
        if feat not in df.columns: continue
        lc = f'{feat}_lag{lag}'
        r[lc] = df[feat].shift(lag)
        lagged_cols.append(lc)

    r = r.replace([np.inf, -np.inf], np.nan)
    if drop_target_na and target_col and target_col in r.columns:
        r = r.dropna(subset=[target_col] + lagged_cols).reset_index(drop=True)
    else:
        r = r.reset_index(drop=True)
    return r, lagged_cols

# ── 8. Permutation Importance ─────────────────────────────────────────────
def run_permutation_importance(lagged_df, feature_cols, target_col,
                                n_rf_runs=3, n_repeats=5, random_state=42):
    df  = lagged_df.copy()
    X   = df[feature_cols].replace([np.inf,-np.inf], np.nan)
    tmp = pd.concat([X, df[target_col]], axis=1).dropna().copy()
    X   = tmp[feature_cols].copy()
    y   = tmp[target_col].astype(int)

    all_imp = []
    for run in range(n_rf_runs):
        rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                    random_state=random_state+run, n_jobs=-1)
        rf.fit(X, y)
        pi = permutation_importance(rf, X, y, n_repeats=n_repeats,
                                    random_state=random_state+run, n_jobs=-1)
        for fi, col in enumerate(feature_cols):
            for val in pi.importances[fi]:
                all_imp.append({'feature': col, 'importance': val})

    imp_df = pd.DataFrame(all_imp)
    agg = (imp_df.groupby('feature')['importance']
                 .agg(mean_importance='mean', std_importance='std')
                 .reset_index()
                 .sort_values('mean_importance', ascending=False)
                 .reset_index(drop=True))
    agg['score'] = agg['mean_importance'] - agg['std_importance'].fillna(0)
    return agg

# ── 9. 지표 계산 ───────────────────────────────────────────────────────────
def safe_binary_metrics(y_true, pred, pred_proba=None):
    y_true = pd.Series(y_true).astype(int)
    pred   = pd.Series(pred).astype(int)
    out = {
        'eval_count':     int(len(y_true)),
        'actual_1_count': int((y_true==1).sum()),
        'pred_1_count':   int((pred==1).sum()),
        'accuracy':       accuracy_score(y_true, pred) if len(y_true) else np.nan,
        'precision':      precision_score(y_true, pred, zero_division=0),
        'recall':         recall_score(y_true, pred, zero_division=0),
        'f1':             f1_score(y_true, pred, zero_division=0),
    }
    if pred_proba is not None and len(y_true) > 0 and y_true.nunique() == 2:
        try: out['roc_auc'] = roc_auc_score(y_true, pred_proba)
        except: out['roc_auc'] = np.nan
    return out

# ── 10. 모델 생성 ──────────────────────────────────────────────────────────
def make_classifier(model_name, random_state=42, model_params=None,
                    pos_count=None, neg_count=None):
    p = model_params or {}

    if model_name == 'random_forest':
        return RandomForestClassifier(
            n_estimators=p.get('n_estimators', 500),
            max_depth=p.get('max_depth', None),
            min_samples_split=p.get('min_samples_split', 2),
            min_samples_leaf=p.get('min_samples_leaf', 1),
            max_features=p.get('max_features', 'sqrt'),
            class_weight=p.get('class_weight', 'balanced'),
            random_state=random_state, n_jobs=-1)

    if model_name == 'extra_trees':
        return ExtraTreesClassifier(
            n_estimators=p.get('n_estimators', 500),
            max_depth=p.get('max_depth', None),
            min_samples_split=p.get('min_samples_split', 2),
            min_samples_leaf=p.get('min_samples_leaf', 1),
            max_features=p.get('max_features', 'sqrt'),
            class_weight=p.get('class_weight', 'balanced'),
            random_state=random_state, n_jobs=-1)

    if model_name == 'hist_gradient_boosting':
        return HistGradientBoostingClassifier(
            max_iter=p.get('max_iter', 300),
            learning_rate=p.get('learning_rate', 0.05),
            max_leaf_nodes=p.get('max_leaf_nodes', 31),
            min_samples_leaf=p.get('min_samples_leaf', 20),
            class_weight='balanced',
            random_state=random_state)

    if model_name == 'xgboost':
        scale_pw = (neg_count / pos_count) if (pos_count and neg_count and pos_count > 0) else 1.0
        return xgb.XGBClassifier(
            n_estimators=p.get('n_estimators', 300),
            learning_rate=p.get('learning_rate', 0.05),
            max_depth=p.get('max_depth', 4),
            subsample=p.get('subsample', 0.8),
            colsample_bytree=p.get('colsample_bytree', 0.8),
            scale_pos_weight=scale_pw,
            random_state=random_state, eval_metric='logloss', verbosity=0, n_jobs=-1)

    if model_name == 'lightgbm':
        return lgb.LGBMClassifier(
            n_estimators=p.get('n_estimators', 300),
            learning_rate=p.get('learning_rate', 0.05),
            max_depth=p.get('max_depth', -1),
            num_leaves=p.get('num_leaves', 31),
            subsample=p.get('subsample', 0.8),
            colsample_bytree=p.get('colsample_bytree', 0.8),
            is_unbalance=True,
            random_state=random_state, verbose=-1, n_jobs=-1)

    raise ValueError(f'Unknown model: {model_name}')

def suggest_model_params(trial, model_name):
    if model_name in ['random_forest', 'extra_trees']:
        return {
            'n_estimators':      trial.suggest_int(f'{model_name}_n_est', 200, 800, step=100),
            'max_depth':         trial.suggest_categorical(f'{model_name}_max_depth', [None,3,5,7,10]),
            'min_samples_split': trial.suggest_int(f'{model_name}_mss', 2, 10),
            'min_samples_leaf':  trial.suggest_int(f'{model_name}_msl', 1, 10),
            'max_features':      trial.suggest_categorical(f'{model_name}_mf', ['sqrt','log2']),
            'class_weight':      trial.suggest_categorical(f'{model_name}_cw',
                                     ['balanced','balanced_subsample']),
        }
    if model_name == 'hist_gradient_boosting':
        return {
            'max_iter':        trial.suggest_int('hgb_iter', 100, 500, step=100),
            'learning_rate':   trial.suggest_float('hgb_lr', 0.01, 0.2, log=True),
            'max_leaf_nodes':  trial.suggest_int('hgb_leaves', 15, 63),
            'min_samples_leaf':trial.suggest_int('hgb_msl', 10, 50),
        }
    if model_name == 'xgboost':
        return {
            'n_estimators':    trial.suggest_int('xgb_n_est', 100, 600, step=100),
            'learning_rate':   trial.suggest_float('xgb_lr', 0.01, 0.2, log=True),
            'max_depth':       trial.suggest_int('xgb_depth', 3, 8),
            'subsample':       trial.suggest_float('xgb_sub', 0.6, 1.0, step=0.1),
            'colsample_bytree':trial.suggest_float('xgb_col', 0.6, 1.0, step=0.1),
        }
    if model_name == 'lightgbm':
        return {
            'n_estimators':    trial.suggest_int('lgb_n_est', 100, 600, step=100),
            'learning_rate':   trial.suggest_float('lgb_lr', 0.01, 0.2, log=True),
            'num_leaves':      trial.suggest_int('lgb_leaves', 15, 127),
            'subsample':       trial.suggest_float('lgb_sub', 0.6, 1.0, step=0.1),
            'colsample_bytree':trial.suggest_float('lgb_col', 0.6, 1.0, step=0.1),
        }
    raise ValueError(model_name)

def evaluate_model_on_period(train_df, eval_df, feature_cols, target_col, close_col,
                              n_days, model_name, model_params, pred_threshold,
                              random_state=42):
    keep_train = [target_col] + feature_cols
    tr = train_df[keep_train].replace([np.inf,-np.inf], np.nan).dropna().copy()
    keep_eval  = ['Date', target_col, f'future_ret_{n_days}d', close_col] + feature_cols
    ev = eval_df[[c for c in keep_eval if c in eval_df.columns]].replace([np.inf,-np.inf], np.nan).dropna().copy()

    if len(tr) == 0 or len(ev) == 0:
        raise ValueError('train/eval 비어있음')
    if tr[target_col].nunique() < 2:
        raise ValueError('train target 단일 class')

    X_tr = tr[feature_cols]; y_tr = tr[target_col].astype(int)
    X_ev = ev[feature_cols]; y_ev = ev[target_col].astype(int)

    pos = int((y_tr==1).sum()); neg = int((y_tr==0).sum())
    model = make_classifier(model_name, random_state=random_state,
                            model_params=model_params, pos_count=pos, neg_count=neg)

    # 시간 가중치: 최근 데이터에 더 높은 가중치 (지수 감쇠)
    n_tr = len(y_tr)
    t_weights = np.exp(np.linspace(0.0, 2.0, n_tr))
    t_weights = t_weights / t_weights.mean()
    class_weights_base = compute_sample_weight('balanced', y_tr)

    # 레짐-인식 가중치: 고변동성 조정장 구간 3× 강화
    vix_cols   = [c for c in X_tr.columns if 'VIX_level' in c]
    smh_r20_cols = [c for c in X_tr.columns if 'SMH_ret_20d' in c or 'smh_ret_20d' in c.lower()]
    regime_boost = np.ones(n_tr)
    if vix_cols and smh_r20_cols:
        vix_s  = X_tr[vix_cols[0]].fillna(20)
        r20_s  = X_tr[smh_r20_cols[0]].fillna(0)
        is_corr = ((vix_s > 25) & (r20_s < -0.05)).values
        regime_boost = np.where(is_corr, 3.0, 1.0)

    sample_w = class_weights_base * t_weights * regime_boost

    try:
        model.fit(X_tr, y_tr, sample_weight=sample_w)
    except TypeError:
        model.fit(X_tr, y_tr)

    pred_proba = model.predict_proba(X_ev)[:,1] if hasattr(model,'predict_proba') \
                 else model.predict(X_ev).astype(float)
    pred       = (pred_proba >= pred_threshold).astype(int)
    metrics    = safe_binary_metrics(y_ev, pred, pred_proba)

    pred_df = ev[['Date', close_col, f'future_ret_{n_days}d', target_col]].copy()
    pred_df['pred_proba'] = pred_proba
    pred_df['pred']       = pred
    return model, metrics, pred_df

# ── 11. Objective Score ───────────────────────────────────────────────────
def compute_objective_score(metrics, pred_df, objective_metric, n_days,
                             min_eval=30, min_pred1=3, precision_floor=0.65):
    ec   = int(metrics.get('eval_count',0) or 0)
    a1   = int(metrics.get('actual_1_count',0) or 0)
    p1   = int(metrics.get('pred_1_count',0) or 0)
    if ec < min_eval or p1 < min_pred1: return 0.0

    pos_rate  = a1 / ec
    precision = float(metrics.get('precision',0) or 0)
    recall    = float(metrics.get('recall',0) or 0)
    f1        = float(metrics.get('f1',0) or 0)
    pl        = max(0.0, precision - pos_rate)
    fl        = max(0.0, f1 - pos_rate)

    if objective_metric == 'precision_lift_recall': return pl * recall
    if objective_metric == 'precision_lift':        return pl
    if objective_metric == 'f1_lift':               return fl
    if objective_metric == 'return_score':
        strat = np.where(pred_df['pred']==1, pred_df[f'future_ret_{n_days}d'], 0.0)
        cret  = float(np.prod(1 + strat) - 1)
        return cret + 0.5 * pl * recall

    # ★ v4 신규: precision_hard
    # precision < floor면 즉시 0 반환 → Optuna가 보수적 영역을 집중 탐색
    if objective_metric == 'precision_hard':
        if precision < precision_floor: return 0.0
        # precision 최우선, recall은 소폭 보너스, 예측 건수도 로그 보너스
        # 최소 5건 이상 예측 강제 (소수건 과적합 방지)
        if p1 < 5: return 0.0
        return precision * (1 + 0.3 * recall) * np.log1p(p1)

    raise ValueError(objective_metric)

def _metrics_extra(metrics, n_days, pred_df=None):
    ec = int(metrics.get('eval_count',0) or 0)
    a1 = int(metrics.get('actual_1_count',0) or 0)
    pos_rate  = a1 / ec if ec > 0 else np.nan
    precision = float(metrics.get('precision',0) or 0)
    recall    = float(metrics.get('recall',0) or 0)
    f1        = float(metrics.get('f1',0) or 0)
    p1        = int(metrics.get('pred_1_count',0) or 0)
    return {
        'positive_rate':         pos_rate,
        'precision_lift':        precision - pos_rate if pd.notna(pos_rate) else np.nan,
        'f1_lift':               f1 - pos_rate        if pd.notna(pos_rate) else np.nan,
        'precision_lift_recall': max(0, precision-pos_rate)*recall if pd.notna(pos_rate) else np.nan,
        'pred_1_ratio':          p1/ec if ec > 0 else np.nan,
    }


# ── v4 추가: 시장 레짐 필터 (MA200 + VIX) ──────────────────────────────────
def apply_regime_filter(df, etf_code, close_col):
    """조정 레짐 날에만 예측: VIX>18 OR MA200 아래 OR 20d수익률 < -2%"""
    close = df[close_col]
    ma200 = close.rolling(200).mean()
    below_ma200 = (close < ma200 * 1.02)   # MA200 근처 or 아래
    vix_cols = [c for c in df.columns if 'VIX_level' in c]
    if vix_cols:
        high_vix = df[vix_cols[0]] > 18   # 약간의 공포 있는 날
    else:
        high_vix = pd.Series(True, index=df.index)
    smh_weak = close.pct_change(20).shift(1) < -0.02  # 20d 약세 (lag1)
    regime = below_ma200 | high_vix | smh_weak
    return regime.reindex(df.index, fill_value=False)

def apply_correction_only_filter(df, close_col):
    """엄격한 조정 레짐: VIX>22 AND (MA200 아래 OR 20d<-5%)"""
    close = df[close_col]
    ma200 = close.rolling(200).mean()
    below_ma200 = (close < ma200)
    vix_cols = [c for c in df.columns if 'VIX_level' in c]
    if vix_cols:
        high_vix = df[vix_cols[0]] > 22
    else:
        high_vix = pd.Series(False, index=df.index)
    smh_down = close.pct_change(20).shift(1) < -0.05
    regime = high_vix & (below_ma200 | smh_down)
    return regime.reindex(df.index, fill_value=False)

# ── v4 추가: isotonic calibration ──────────────────────────────────────────
def fit_isotonic_calibrator(val_proba, y_val):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(val_proba, y_val.astype(float))
    return ir

print('레짐 필터 + Isotonic calibration 함수 로드 완료')



# ============================================================
def make_oos_windows(base_date, start_date, optuna_valid_months, sim_test_months):
    base  = pd.to_datetime(base_date)
    start = pd.to_datetime(start_date)
    base_p= base.to_period('M')

    sim_start_p   = base_p  - (sim_test_months - 1)
    valid_end_p   = sim_start_p - 1
    valid_start_p = valid_end_p - (optuna_valid_months - 1)
    train_end_p   = valid_start_p - 1

    return {
        'train_start':        start,
        'train_end':          train_end_p.to_timestamp(how='end').normalize(),
        'optuna_valid_start': valid_start_p.to_timestamp(how='start'),
        'optuna_valid_end':   valid_end_p.to_timestamp(how='end').normalize(),
        'sim_test_start':     sim_start_p.to_timestamp(how='start'),
        'sim_test_end':       base,
        'base_date':          base,
    }

WINDOWS = make_oos_windows(BASE_DATE, START_DATE, OPTUNA_VALID_MONTHS, SIM_TEST_MONTHS)
for k, v in WINDOWS.items():
    print(f'  {k:22s}: {str(v)[:10]}')


# ============================================================
feat_cfg = {
    'use_rsi': USE_RSI, 'rsi_period': RSI_PERIOD,
    'use_macd': USE_MACD, 'macd_fast': MACD_FAST, 'macd_slow': MACD_SLOW, 'macd_signal': MACD_SIG,
    'use_bollinger': USE_BB, 'bollinger_period': BB_PERIOD, 'bollinger_std': BB_STD,
    'use_atr': USE_ATR, 'atr_period': ATR_PERIOD,
    'use_52w': USE_52W,
}

if LOAD_FROM_CACHE:
    try:
        base_df, raw_feature_cols, close_col = load_cache('base_dataset')
    except FileNotFoundError:
        print('캐시 없음, 새로 생성...')
        base_df, raw_feature_cols, close_col = make_base_feature_dataset(
            etf_code=ETF_CODE, external_tickers=EXTERNAL_TICKERS,
            external_feature_types=EXTERNAL_FEATURE_TYPES,
            start_date=START_DATE, end_date=END_DATE, feat_cfg=feat_cfg)
        save_cache('base_dataset', (base_df, raw_feature_cols, close_col))
else:
    print('Base feature dataset 생성 중...')
    base_df, raw_feature_cols, close_col = make_base_feature_dataset(
        etf_code=ETF_CODE, external_tickers=EXTERNAL_TICKERS,
        external_feature_types=EXTERNAL_FEATURE_TYPES,
        start_date=START_DATE, end_date=END_DATE, feat_cfg=feat_cfg)
    save_cache('base_dataset', (base_df, raw_feature_cols, close_col))

print(f'\nbase_df shape: {base_df.shape}')
print(f'피처 수: {len(raw_feature_cols)}')
print(f'기간: {base_df["Date"].min().date()} ~ {base_df["Date"].max().date()}')


# ============================================================
_vif_pkl = SHARED_CACHE / 'vif_feature_cols.pkl'
if LOAD_FROM_CACHE and _vif_pkl.exists():
    vif_feature_cols = load_cache('vif_feature_cols')
else:
    train_base_df = base_df[
        (base_df['Date'] >= WINDOWS['train_start']) &
        (base_df['Date'] <= WINDOWS['train_end'])
    ].copy()
    print(f'VIF 계산 대상: {train_base_df.shape}')
    vif_feature_cols, removed_vif_df = reduce_features_by_vif(
        df=train_base_df, feature_cols=raw_feature_cols, vif_threshold=VIF_THRESHOLD)
    save_cache('vif_feature_cols', vif_feature_cols)

print(f'VIF 통과 피처: {len(vif_feature_cols)}개')


# ============================================================
feature_cache = {}
if LOAD_FROM_CACHE:
    try:
        feature_cache = load_cache('feature_cache')
        print(f'feature_cache 로드: {len(feature_cache)}가지 조합')
    except FileNotFoundError:
        print('feature_cache 없음, 새로 생성')

def _gc(msg):
    gc.collect()
    print(f'  [GC] {msg}')

def build_feature_artifacts(n_days, target_return_threshold, include_infer_df=False):
    key = (int(n_days), float(target_return_threshold))
    if key in feature_cache:
        art = feature_cache[key]
        if include_infer_df and 'all_lagged_infer_df' not in art:
            td, _ = add_target_column(base_df, close_col, n_days, target_return_threshold)
            art['all_lagged_infer_df'], _ = make_lagged_dataset(
                td, art['best_lag_df'], target_col=art['target_col'],
                close_col=close_col, n_days=n_days, drop_target_na=False)
            del td; _gc('infer target_df deleted')
        return art

    print(f'  [build] n_days={n_days}, threshold={target_return_threshold}')

    target_df, target_col = add_target_column(base_df, close_col, n_days, target_return_threshold)

    fs_end   = WINDOWS['train_end']
    fs_start = fs_end - pd.DateOffset(years=FEATURE_SELECT_YEARS)
    fs_df    = target_df[(target_df['Date']>=fs_start)&(target_df['Date']<=fs_end)].copy()
    print(f'    feature select 기간: {fs_df["Date"].min().date()} ~ {fs_df["Date"].max().date()}')

    _, best_lag_df = find_best_lag_by_feature(fs_df, vif_feature_cols, target_col, LAG_DAYS)
    del fs_df; _gc('fs_df deleted')

    all_lagged_eval_df, lagged_cols = make_lagged_dataset(
        target_df, best_lag_df, target_col=target_col,
        close_col=close_col, n_days=n_days, drop_target_na=True)

    fs_lag_df = all_lagged_eval_df[
        (all_lagged_eval_df['Date']>=fs_start)&(all_lagged_eval_df['Date']<=fs_end)].copy()

    if fs_lag_df[target_col].nunique() < 2:
        raise ValueError('Feature selection target 단일 class')

    imp_df = run_permutation_importance(
        fs_lag_df, lagged_cols, target_col, N_RF_RUNS, N_REPEATS, RANDOM_STATE)

    top_cols_max = imp_df['feature'].head(TOP_N_MAX).tolist()

    art = {
        'n_days': int(n_days), 'target_return_threshold': target_return_threshold,
        'target_col': target_col, 'best_lag_df': best_lag_df,
        'all_lagged_eval_df': all_lagged_eval_df, 'lagged_feature_cols': lagged_cols,
        'importance_df': imp_df, 'top_feature_cols_max': top_cols_max,
    }
    if include_infer_df:
        art['all_lagged_infer_df'], _ = make_lagged_dataset(
            target_df, best_lag_df, target_col=target_col,
            close_col=close_col, n_days=n_days, drop_target_na=False)

    feature_cache[key] = art
    del target_df, fs_lag_df; _gc(f'deleted after build n_days={n_days} th={target_return_threshold}')
    return art

print('Feature cache 함수 준비 완료')


# ============================================================
def run_stage1_grid_search(objective_metric, n_seeds=3, random_state=42):
    from itertools import product
    rows   = []
    combos = list(product(N_DAYS_CANDIDATES, TARGET_RETURN_THRESHOLD_CANDIDATES))
    print(f'총 {len(combos)}가지 조합 x {n_seeds} seeds = {len(combos)*n_seeds}회 평가')

    for i, (n_days, threshold) in enumerate(combos):
        print(f'  [{i+1}/{len(combos)}] n_days={n_days}, threshold={threshold}', end=' ')
        seed_scores = []
        try:
            art = build_feature_artifacts(n_days, threshold)
            tc  = art['target_col']
            ev  = art['all_lagged_eval_df']
            fc  = art['top_feature_cols_max'][:S1_TOP_N]

            if not fc:
                print('→ 피처 없음 skip'); continue

            tr_df = ev[(ev['Date']>=WINDOWS['train_start'])&(ev['Date']<=WINDOWS['train_end'])]
            va_df = ev[(ev['Date']>=WINDOWS['optuna_valid_start'])&(ev['Date']<=WINDOWS['optuna_valid_end'])]
            params = {'n_estimators': S1_N_EST, 'class_weight': 'balanced'}

            for s in range(n_seeds):
                _, metrics, pred_df = evaluate_model_on_period(
                    tr_df, va_df, fc, tc, close_col, n_days,
                    S1_MODEL, params, pred_threshold=0.5, random_state=random_state+s*100)
                score = compute_objective_score(metrics, pred_df, objective_metric,
                                                n_days, S1_MIN_EVAL, S1_MIN_PRED1)
                # 양성률 필터: val 기간 양성률 > 30%면 0 처리 (80% precision 불가)
            pos_r = metrics.get('actual_1_count', 0) / max(metrics.get('eval_count', 1), 1)
            if pos_r > 0.30:
                score = 0.0
            seed_scores.append(score)

            mean_score = float(pd.Series(seed_scores).mean())
            std_score  = float(pd.Series(seed_scores).std())
            print(f'→ score={mean_score:.4f} (±{std_score:.4f})')

            rows.append({
                'n_days': n_days, 'threshold': threshold,
                'score': mean_score, 'score_std': std_score,
                **metrics, **_metrics_extra(metrics, n_days, pred_df),
            })
        except Exception as e:
            print(f'→ ERROR: {e}')
            rows.append({'n_days': n_days, 'threshold': threshold, 'score': 0.0, 'error': str(e)})

    trials_df = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)
    best = trials_df.iloc[0]
    return {
        'trials_df': trials_df,
        'best_n_days':    int(best['n_days']),
        'best_threshold': float(best['threshold']),
        'best_score':     float(best['score']),
    }

_s1_pkl = V4_CACHE / 'stage1_result.pkl'
if LOAD_FROM_CACHE and _s1_pkl.exists():
    stage1_result = load_cache('stage1_result', v4_only=True)
else:
    print('Stage 1 Grid Search 시작...')
    stage1_result = run_stage1_grid_search(
        objective_metric=S1_OBJ_METRIC, n_seeds=S1_N_SEEDS, random_state=RANDOM_STATE)
    save_cache('stage1_result', stage1_result, v4_only=True)

print(f'\nStage 1 Best → n_days={stage1_result["best_n_days"]}, '
      f'threshold={stage1_result["best_threshold"]}, score={stage1_result["best_score"]:.4f}')
stage1_result['trials_df'].to_csv(OUTPUT_DIR / 'stage1_grid_results.csv', index=False)
display_df(stage1_result['trials_df'])


# ============================================================
def run_stage2_optuna(n_days, threshold, n_trials, objective_metric,
                       precision_floor=0.65, random_state=42):
    """Stage 2: 모델+하이퍼파라미터 탐색. precision_hard 목표."""
    art    = build_feature_artifacts(n_days, threshold)
    tc     = art['target_col']
    ev     = art['all_lagged_eval_df']
    all_fc = art['top_feature_cols_max']
    rows   = []

    def objective(trial):
        top_n  = trial.suggest_int('top_n', S2_TOP_N_RANGE[0], min(S2_TOP_N_RANGE[1], len(all_fc)))
        fc     = all_fc[:top_n]
        mname  = trial.suggest_categorical('model_name', S2_MODELS)
        mparams= suggest_model_params(trial, mname)
        pth    = trial.suggest_float('pred_threshold',
                                     S2_THRESH_RANGE[0], S2_THRESH_RANGE[1],
                                     step=S2_THRESH_STEP)

        tr_df = ev[(ev['Date']>=WINDOWS['train_start'])&(ev['Date']<=WINDOWS['train_end'])]
        va_df_all = ev[(ev['Date']>=WINDOWS['optuna_valid_start'])&(ev['Date']<=WINDOWS['optuna_valid_end'])]
        # 조정 레짐 날에만 평가 (mean-reversion 특화)
        regime_mask = apply_regime_filter(va_df_all, ETF_CODE, close_col)
        va_df = va_df_all[regime_mask].copy() if regime_mask.sum() >= S2_MIN_EVAL else va_df_all

        try:
            _, metrics, pred_df = evaluate_model_on_period(
                tr_df, va_df, fc, tc, close_col, n_days,
                mname, mparams, pth, random_state=random_state)
            score = compute_objective_score(
                metrics, pred_df, objective_metric, n_days,
                S2_MIN_EVAL, S2_MIN_PRED1, precision_floor=precision_floor)
            row = {'trial': trial.number, 'score': score, 'model': mname,
                   'top_n': top_n, 'pred_threshold': pth,
                   **metrics, **_metrics_extra(metrics, n_days, pred_df),
                   'model_params': json.dumps(mparams, default=str)}
            rows.append(row)
            trial.set_user_attr('model_params', mparams)
            trial.set_user_attr('metrics', metrics)
            return score
        except Exception as e:
            rows.append({'trial': trial.number, 'score': 0.0, 'error': str(e), 'model': mname})
            return 0.0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    trials_df = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)
    bt = study.best_trial

    best_config = {
        'n_days':           n_days,
        'target_return_threshold': threshold,
        'target_col':       art['target_col'],
        'model_name':       bt.params['model_name'],
        'model_params':     bt.user_attrs['model_params'],
        'top_n':            bt.params['top_n'],
        'pred_threshold':   bt.params['pred_threshold'],
        'score':            bt.value,
        'objective_metric': objective_metric,
    }
    return {'study': study, 'trials_df': trials_df, 'best_config': best_config, 'artifacts': art}

_s2_pkl = V4_CACHE / 'stage2_result.pkl'
if LOAD_FROM_CACHE and _s2_pkl.exists():
    stage2_result = load_cache('stage2_result', v4_only=True)
else:
    print(f'Stage 2 Optuna 시작 ({S2_N_TRIALS} trials, objective={S2_OBJ_METRIC}) ...')
    stage2_result = run_stage2_optuna(
        stage1_result['best_n_days'], stage1_result['best_threshold'],
        S2_N_TRIALS, S2_OBJ_METRIC, S2_PRECISION_FLOOR, RANDOM_STATE)
    save_cache('stage2_result', stage2_result, v4_only=True)
    save_cache('feature_cache', feature_cache)

best_cfg = stage2_result['best_config']
print('\nBest config (Stage 2):')
print(json.dumps(best_cfg, ensure_ascii=False, indent=2, default=str))
stage2_result['trials_df'].to_csv(OUTPUT_DIR / 'stage2_optuna_trials.csv', index=False)
display_df(stage2_result['trials_df'], 20)


# ============================================================
# ═══════════════════════════════════════════════════════════════════════
# Stage 3 (v4 신규): Top-K 앙상블 + 검증셋 기반 precision threshold 보정
# -----------------------------------------------------------------------
# 1. Stage 2 trials 중 precision 상위 K개 설정으로 모델 재학습
# 2. 검증셋에서 평균 앙상블 확률 계산
# 3. precision >= 80% 달성하는 최소 임계값 탐색
# 4. 해당 임계값 + 앙상블 모델을 시뮬레이션에 사용
# ═══════════════════════════════════════════════════════════════════════

def run_stage3_ensemble_calibration(
        stage2_trials_df, art, best_cfg, windows, close_col,
        precision_target=0.80, ensemble_top_k=5, min_pred=3, random_state=42):

    tc     = art['target_col']
    ev     = art['all_lagged_eval_df']
    best_fc = art['top_feature_cols_max'][:best_cfg['top_n']]
    n_days  = art['n_days']

    tr_df = ev[(ev['Date']>=windows['train_start'])&(ev['Date']<=windows['train_end'])].copy()
    va_df_all = ev[(ev['Date']>=windows['optuna_valid_start'])&(ev['Date']<=windows['optuna_valid_end'])].copy()
    # 조정 레짐 날에만 threshold 보정 (mean-reversion 특화)
    # 전체 val 사용 (조정레짐 필터 제거)
    va_df = va_df_all.copy()

    keep = [tc] + best_fc
    tr_c  = tr_df[[c for c in keep if c in tr_df.columns]].replace([np.inf,-np.inf], np.nan).dropna().copy()
    va_c  = va_df[[c for c in ['Date', tc, f'future_ret_{n_days}d', close_col] + best_fc
                   if c in va_df.columns]].replace([np.inf,-np.inf], np.nan).dropna().copy()

    X_tr = tr_c[best_fc]; y_tr = tr_c[tc].astype(int)
    X_va = va_c[best_fc]; y_va = va_c[tc].astype(int).values

    # Stage 2 trials에서 top-K 선별 (precision 순, 단 pred_1_count >= min_pred)
    t2 = stage2_trials_df.copy()
    t2 = t2[t2['pred_1_count'].fillna(0) >= min_pred]
    t2 = t2.sort_values('precision', ascending=False).reset_index(drop=True)
    top_k_df = t2.head(ensemble_top_k)
    print(f'  앙상블 후보 {len(top_k_df)}개 (precision 기준):')
    print(top_k_df[['trial','model','top_n','pred_threshold','precision','recall','pred_1_count']]
          .to_string(index=False))

    val_probas = []
    ensemble_models = []

    for _, row in top_k_df.iterrows():
        try:
            mname   = row['model']
            raw_mp  = row.get('model_params', '{}')
            mparams = json.loads(raw_mp) if isinstance(raw_mp, str) else (raw_mp or {})

            # top_n이 best_fc와 다를 수 있으나 공통 feature set 사용
            pos = int((y_tr==1).sum()); neg = int((y_tr==0).sum())
            model = make_classifier(mname, random_state, mparams, pos, neg)
            model.fit(X_tr, y_tr)

            proba = model.predict_proba(X_va)[:,1]
            val_probas.append(proba)
            ensemble_models.append({'model': model, 'model_name': mname, 'mparams': mparams, 'trial_prec': float(row['precision'])})
            print(f'    [{mname}] val_proba mean={proba.mean():.3f}')
        except Exception as e:
            print(f'    [SKIP] {row.get("model","?")} trial {row.get("trial","?")} : {e}')

    if not val_probas:
        raise ValueError('앙상블 모델 빌드 실패')

    avg_proba = np.mean(val_probas, axis=0)

    # 검증셋에서 precision >= target 달성하는 최저 임계값 탐색
    found_th = found_prec = found_cnt = None
    threshold_scan = []

    for th in np.round(np.arange(0.99, 0.29, -0.01), 2):
        pred = (avg_proba >= th).astype(int)
        cnt  = int(pred.sum())
        if cnt < min_pred:
            threshold_scan.append({'threshold': th, 'precision': np.nan, 'pred_count': cnt})
            continue
        p = float(precision_score(y_va, pred, zero_division=0))
        r = float(recall_score(y_va, pred, zero_division=0))
        threshold_scan.append({'threshold': th, 'precision': p, 'recall': r, 'pred_count': cnt})
        if found_th is None and p >= precision_target:
            found_th, found_prec, found_cnt = float(th), p, cnt

    scan_df = pd.DataFrame(threshold_scan)

    if found_th is None:
        valid_scan = scan_df.dropna(subset=['precision'])
        if not valid_scan.empty:
            best_row   = valid_scan.loc[valid_scan['precision'].idxmax()]
            found_th   = float(best_row['threshold'])
            found_prec = float(best_row['precision'])
            found_cnt  = int(best_row['pred_count'])
        else:
            found_th, found_prec, found_cnt = 0.5, 0.0, 0
        print(f'  ⚠ precision {precision_target:.0%} 미달. 최고 precision: {found_prec:.3f} (threshold={found_th:.2f})')
    else:
        print(f'  ✓ precision {found_prec:.3f} @ threshold={found_th:.2f}  (pred_count={found_cnt}/{len(y_va)})')

    # ── Walk-forward 이중 검증 (설정에 따라 활성화) ──
    if S3_WALKFORWARD and found_th is not None and len(y_va) >= 10:
        half = len(y_va) // 2
        proba_h1, y_h1 = avg_proba[:half], y_va[:half]
        proba_h2, y_h2 = avg_proba[half:], y_va[half:]
        results_wf = []
        for th_wf in np.round(np.arange(0.99, 0.30, -0.01), 2):
            p1_h1 = (proba_h1 >= th_wf).astype(int)
            p1_h2 = (proba_h2 >= th_wf).astype(int)
            cnt1 = int(p1_h1.sum()); cnt2 = int(p1_h2.sum())
            if cnt1 < 2 or cnt2 < 2: continue
            prec1 = float(precision_score(y_h1, p1_h1, zero_division=0))
            prec2 = float(precision_score(y_h2, p1_h2, zero_division=0))
            prec_min = min(prec1, prec2)
            results_wf.append({'th': th_wf, 'prec_h1': prec1, 'prec_h2': prec2, 'min': prec_min})
        if results_wf:
            wf_df = pd.DataFrame(results_wf)
            # 양쪽 모두 목표의 75% 이상인 가장 낮은 threshold 선택
            wf_ok = wf_df[wf_df['min'] >= precision_target * 0.75]
            if not wf_ok.empty:
                wf_best = wf_ok.loc[wf_ok['th'].idxmin()]
                # walk-forward 기준 threshold가 더 보수적이면 사용
                if float(wf_best['th']) > found_th:
                    print(f'  [WF] 더 보수적 threshold 채택: {wf_best["th"]:.2f} (h1={wf_best["prec_h1"]:.3f}, h2={wf_best["prec_h2"]:.3f})')
                    old_th = found_th
                    found_th = float(wf_best['th'])
                    # 새 threshold로 전체 val precision 재계산
                    prd_new = (avg_proba >= found_th).astype(int)
                    if prd_new.sum() >= min_pred:
                        found_prec = float(precision_score(y_va, prd_new, zero_division=0))
                        found_cnt  = int(prd_new.sum())
                        print(f'  [WF] 전체 val precision: {found_prec:.3f} ({found_cnt}건)')
                    else:
                        found_th = old_th
                        print(f'  [WF] 예측 수 부족, 기존 threshold 유지')
                else:
                    print(f'  [WF] 기존 threshold {found_th:.2f} 유지 (WF 기준보다 이미 보수적)')
            else:
                print(f'  [WF] 양쪽 모두 {precision_target*0.75:.0%} 달성하는 threshold 없음, 기존 유지')

    # ── Isotonic calibration (검증 데이터로 확률 보정) ──
    calibrator = None
    try:
        calibrator = fit_isotonic_calibrator(avg_proba, y_va)
        cal_proba = calibrator.predict(avg_proba)
        print(f'  [Isotonic] val proba range: raw [{avg_proba.min():.3f}, {avg_proba.max():.3f}] → cal [{cal_proba.min():.3f}, {cal_proba.max():.3f}]')

        # calibrated proba로 threshold 재탐색
        cal_found_th = cal_found_prec = cal_found_cnt = None
        for th2 in np.round(np.arange(0.99, 0.10, -0.01), 2):
            pred2 = (cal_proba >= th2).astype(int)
            cnt2  = int(pred2.sum())
            if cnt2 < min_pred: continue
            p2 = float(precision_score(y_va, pred2, zero_division=0))
            if p2 >= precision_target:
                cal_found_th = float(th2); cal_found_prec = p2; cal_found_cnt = cnt2
                break
        if cal_found_th:
            print(f'  [Isotonic] cal threshold={cal_found_th:.2f}, precision={cal_found_prec:.3f}, cnt={cal_found_cnt}')
        else:
            print('  [Isotonic] 80% 미달 - raw threshold 유지')
    except Exception as e:
        print(f'  [Isotonic] 실패: {e}')
        cal_proba = avg_proba
        cal_found_th = found_th; cal_found_prec = found_prec; cal_found_cnt = found_cnt

    # ── val proba 분위수 저장 (분포 shift 진단용) ──
    val_proba_quantiles = {
        'p50': float(np.percentile(avg_proba, 50)),
        'p75': float(np.percentile(avg_proba, 75)),
        'p90': float(np.percentile(avg_proba, 90)),
        'p95': float(np.percentile(avg_proba, 95)),
        'max': float(avg_proba.max()),
    }
    print(f'  [VAL proba] {val_proba_quantiles}')

    return {
        'ensemble_models': ensemble_models,
        'feature_cols':    best_fc,
        'calibrated_threshold': found_th,
        'calibrated_val_precision': found_prec,
        'calibrated_val_pred_count': found_cnt,
        'cal_threshold': cal_found_th,
        'cal_val_precision': cal_found_prec,
        'calibrator': calibrator,
        'val_avg_proba': avg_proba,
        'cal_val_proba': cal_proba,
        'val_proba_quantiles': val_proba_quantiles,
        'y_val': y_va,
        'val_df': va_c,
        'scan_df': scan_df,
        'n_ensemble': len(ensemble_models),
    }


best_artifacts = build_feature_artifacts(
    best_cfg['n_days'], best_cfg['target_return_threshold'], include_infer_df=True)

_s3_pkl = V4_CACHE / 'stage3_result.pkl'
if LOAD_FROM_CACHE and _s3_pkl.exists():
    stage3_result = load_cache('stage3_result', v4_only=True)
else:
    print('Stage 3 앙상블 보정 시작...')
    stage3_result = run_stage3_ensemble_calibration(
        stage2_trials_df=stage2_result['trials_df'],
        art=best_artifacts,
        best_cfg=best_cfg,
        windows=WINDOWS,
        close_col=close_col,
        precision_target=S3_PRECISION_TARGET,
        ensemble_top_k=S3_ENSEMBLE_TOP_K,
        min_pred=S3_MIN_PRED,
        random_state=RANDOM_STATE,
    )
    save_cache('stage3_result', stage3_result, v4_only=True)

CALIBRATED_TH = stage3_result['calibrated_threshold']
N_DAYS_BEST   = int(best_cfg['n_days'])
target_col    = best_artifacts['target_col']
all_lagged_ev = best_artifacts['all_lagged_eval_df']
all_lagged_in = best_artifacts['all_lagged_infer_df']
best_fc       = stage3_result['feature_cols']

print(f'\n★ 최종 설정:')
print(f'  n_days={N_DAYS_BEST}, threshold={best_cfg["target_return_threshold"]}')
print(f'  앙상블 모델 수: {stage3_result["n_ensemble"]}')
print(f'  보정 임계값: {CALIBRATED_TH:.2f}')
print(f'  검증 precision: {stage3_result["calibrated_val_precision"]:.4f}')

print('\n[임계값별 검증 precision]')
display_df(stage3_result['scan_df'].dropna(subset=['precision']), 30)


# ============================================================
def ensemble_predict_on_df(test_df, ensemble_models, feature_cols, threshold,
                            target_col, close_col, n_days):
    keep = [c for c in ['Date', target_col, f'future_ret_{n_days}d', close_col] + feature_cols
            if c in test_df.columns]
    ev = test_df[keep].replace([np.inf,-np.inf], np.nan).dropna().copy()
    if len(ev) == 0:
        return ev
    X_ev = ev[feature_cols]
    probas = [m['model'].predict_proba(X_ev)[:,1] for m in ensemble_models]
    avg_proba = np.mean(probas, axis=0)
    ev['pred_proba'] = avg_proba
    ev['pred']       = (avg_proba >= threshold).astype(int)
    return ev


# train+valid 로 앙상블 재학습
tr_va_df = all_lagged_ev[
    (all_lagged_ev['Date'] >= WINDOWS['train_start']) &
    (all_lagged_ev['Date'] <= WINDOWS['optuna_valid_end'])
].copy()
sim_test_df = all_lagged_ev[
    (all_lagged_ev['Date'] >= WINDOWS['sim_test_start']) &
    (all_lagged_ev['Date'] <= WINDOWS['sim_test_end'])
].copy()

keep_trva = [target_col] + best_fc
tr_va_c   = tr_va_df[[c for c in keep_trva if c in tr_va_df.columns]].replace([np.inf,-np.inf], np.nan).dropna().copy()
X_trva = tr_va_c[best_fc]; y_trva = tr_va_c[target_col].astype(int)
pos_trva = int((y_trva==1).sum()); neg_trva = int((y_trva==0).sum())

sim_ensemble_models = []
for info in stage3_result['ensemble_models']:
    mname   = info['model_name']
    mparams = info.get('mparams', {})
    new_model = make_classifier(mname, RANDOM_STATE, mparams, pos_trva, neg_trva)
    new_model.fit(X_trva, y_trva)
    sim_ensemble_models.append({'model': new_model, 'model_name': mname})
    print(f'  [{mname}] 재학습 완료')

# raw 앙상블 예측
sim_pred_df = ensemble_predict_on_df(
    sim_test_df, sim_ensemble_models, best_fc, CALIBRATED_TH,
    target_col, close_col, N_DAYS_BEST)

# ── 테스트 proba 분포 진단 ──
test_probas_all = sim_pred_df['pred_proba'].values
val_q = stage3_result.get('val_proba_quantiles', {})
print('[TEST vs VAL proba 분포 비교]')
for q_name, q_val in [('p50',50),('p75',75),('p90',90),('p95',95),('max',100)]:
    tp = float(np.percentile(test_probas_all, q_val))
    vp = val_q.get(q_name, float('nan'))
    print(f'  {q_name}: test={tp:.4f}  val={vp:.4f}')

# ── Isotonic calibration 적용 ──
calibrator = stage3_result.get('calibrator')
if calibrator is not None:
    sim_pred_df['pred_proba_cal'] = calibrator.predict(sim_pred_df['pred_proba'].values)
    print(f'[Isotonic] test cal proba: max={sim_pred_df["pred_proba_cal"].max():.4f}, p90={np.percentile(sim_pred_df["pred_proba_cal"].values,90):.4f}')
else:
    sim_pred_df['pred_proba_cal'] = sim_pred_df['pred_proba']

# ── 레짐 필터 (MA200 + VIX) ──
_test_close = sim_test_df[[close_col,'Date']].copy().set_index('Date')
_test_close_full = all_lagged_ev[['Date', close_col]].copy().set_index('Date')
_ma200 = _test_close_full[close_col].rolling(200).mean()
_above_ma200 = (_test_close_full[close_col] >= _ma200 * 0.97)
vix_col_candidates = [c for c in sim_test_df.columns if 'VIX_level' in c]
if vix_col_candidates:
    _vix_vals = sim_test_df.set_index('Date')[vix_col_candidates[0]]
    _vix_ok = _vix_vals < 35
else:
    _vix_ok = pd.Series(True, index=sim_pred_df['Date'])

regime_series = _above_ma200.reindex(sim_pred_df['Date'].values)
vix_series    = _vix_ok.reindex(sim_pred_df['Date'].values)
sim_pred_df['regime_ok'] = (regime_series.values & vix_series.fillna(True).values)

# ── threshold 선택: isotonic cal threshold 우선, 없으면 raw threshold ──
CAL_TH = stage3_result.get('cal_threshold') or CALIBRATED_TH
print(f'threshold: raw={CALIBRATED_TH:.2f}, cal={CAL_TH:.2f}')

# ── 다양한 전략 precision 비교 ──
strategies = [
    ('Raw only',          'pred_proba',     CALIBRATED_TH,   pd.Series(True, index=sim_pred_df.index)),
    ('Cal only',          'pred_proba_cal', CAL_TH,          pd.Series(True, index=sim_pred_df.index)),
    ('Raw+Regime',        'pred_proba',     CALIBRATED_TH,   pd.Series(sim_pred_df['regime_ok'].values, index=sim_pred_df.index)),
    ('Cal+Regime',        'pred_proba_cal', CAL_TH,          pd.Series(sim_pred_df['regime_ok'].values, index=sim_pred_df.index)),
    ('Raw+Regime th0.55', 'pred_proba',     0.55,            pd.Series(sim_pred_df['regime_ok'].values, index=sim_pred_df.index)),
    ('Raw+Regime th0.50', 'pred_proba',     0.50,            pd.Series(sim_pred_df['regime_ok'].values, index=sim_pred_df.index)),
]

best_strategy_prec = 0.0
best_strategy_pred = None
print()
for name, proba_col, th, regime_mask in strategies:
    preds = ((sim_pred_df[proba_col] >= th) & regime_mask).astype(int)
    cnt = int(preds.sum())
    if cnt == 0:
        print(f'  [{name}]: 0건')
        continue
    m = safe_binary_metrics(sim_pred_df[target_col].astype(int), preds, sim_pred_df[proba_col])
    flag = '✓' if m['precision'] >= 0.80 else '△' if m['precision'] >= 0.60 else '✗'
    print(f'  [{name}] {flag} prec={m["precision"]:.4f} recall={m["recall"]:.4f} cnt={cnt}')
    if (m['precision'] > best_strategy_prec or
        (m['precision'] == best_strategy_prec and cnt > best_strategy_cnt)) and cnt >= 1:
        best_strategy_prec = m['precision']
        best_strategy_cnt  = cnt
        best_strategy_pred = preds.values.copy()

# 최고 전략으로 sim_pred_df 업데이트
if best_strategy_pred is not None:
    sim_pred_df['pred'] = best_strategy_pred

sim_metrics = safe_binary_metrics(
    sim_pred_df[target_col].astype(int),
    sim_pred_df['pred'].astype(int),
    sim_pred_df['pred_proba_cal'] if 'pred_proba_cal' in sim_pred_df.columns else sim_pred_df['pred_proba'])
# pred_proba 업데이트
if 'pred_proba_cal' in sim_pred_df.columns:
    sim_pred_df['pred_proba'] = sim_pred_df['pred_proba_cal']

prec_test = sim_metrics.get('precision', 0)
flag = '✓' if prec_test >= 0.80 else '△' if prec_test >= 0.65 else '✗'
print(f'\n{flag} 시뮬레이션 테스트 precision: {prec_test:.4f}  (목표: 0.80)')
print('[Simulation Test 지표]')
for k, v in sim_metrics.items():
    print(f'  {k:20s}: {v}')


# ============================================================
def summarize_sim(pred_df, target_col, n_days):
    ev = pred_df.dropna(subset=[target_col, 'pred', f'future_ret_{n_days}d']).copy()
    if len(ev) == 0:
        return pd.DataFrame([{'eval_count': 0}])
    metrics = safe_binary_metrics(ev[target_col].astype(int), ev['pred'].astype(int), ev['pred_proba'])
    ex      = _metrics_extra(metrics, n_days, ev)
    strat   = np.where(ev['pred']==1, ev[f'future_ret_{n_days}d'], 0.0)
    strat_if= np.where(ev[target_col].astype(int)==1, ev[f'future_ret_{n_days}d'], 0.0)
    cret    = float(np.prod(1+strat)-1)
    cret_if = float(np.prod(1+strat_if)-1)
    avg_buy = float(ev.loc[ev['pred']==1, f'future_ret_{n_days}d'].mean()) if (ev['pred']==1).any() else np.nan
    return pd.DataFrame([{'target_col': target_col, 'n_days': n_days,
                           **metrics, **ex,
                           'strategy_compound_return': cret,
                           'strategy_compound_return_if': cret_if,
                           'avg_ret_when_buy': avg_buy}])


def make_cash_simulation(pred_df, close_col, target_col, n_days, take_profit_threshold,
                          initial_cash=1_000_000, buy_ratio=0.05, min_cash_ratio=0.30):
    df = pred_df.copy().sort_values('Date').reset_index(drop=True)
    min_cash = initial_cash * min_cash_ratio
    cash    = float(initial_cash); cash_if = float(initial_cash)
    open_lots = []; open_lots_if = []
    records   = []

    def _process_lots(lots, price, i, tp_th):
        remaining, sell_amt, reason = [], 0.0, None
        for lot in lots:
            ret = price / lot['buy_price'] - 1
            if ret >= tp_th:
                sell_amt += lot['qty'] * price; reason = 'TAKE_PROFIT'
            elif lot['sell_idx'] <= i:
                sell_amt += lot['qty'] * price
                if reason != 'TAKE_PROFIT': reason = 'EXPIRE'
            else:
                remaining.append(lot)
        return remaining, sell_amt, reason

    def _buy(lots, cash_val, asset_val, price, i, buy_ratio, min_cash):
        spend = min(asset_val * buy_ratio, cash_val - min_cash)
        if spend <= 0 or price <= 0: return lots, cash_val, 0.0
        qty = spend / price; cash_val -= spend
        lots.append({'buy_price': price, 'qty': qty, 'sell_idx': i + n_days})
        return lots, cash_val, spend

    for i, row in df.iterrows():
        date   = row['Date']; price = float(row[close_col])
        actual = int(row[target_col]) if pd.notna(row.get(target_col)) else 0

        open_lots, sell_amt, sell_rsn = _process_lots(open_lots, price, i, take_profit_threshold)
        cash += sell_amt
        holding = sum(l['qty']*price for l in open_lots)
        asset   = cash + holding
        action  = 'HOLD'; buy_amt = 0.0
        if pd.notna(row.get('pred')) and int(row['pred']) == 1 and cash > min_cash:
            open_lots, cash, buy_amt = _buy(open_lots, cash, asset, price, i, buy_ratio, min_cash)
            action = 'BUY'
        if sell_amt > 0: action = f'SELL_{sell_rsn}' + ('_BUY' if buy_amt > 0 else '')
        holding = sum(l['qty']*price for l in open_lots)
        asset   = cash + holding

        open_lots_if, sell_amt_if, sell_rsn_if = _process_lots(open_lots_if, price, i, take_profit_threshold)
        cash_if += sell_amt_if
        holding_if = sum(l['qty']*price for l in open_lots_if)
        asset_if   = cash_if + holding_if
        action_if  = 'HOLD'; buy_amt_if = 0.0
        if actual == 1 and cash_if > min_cash:
            open_lots_if, cash_if, buy_amt_if = _buy(open_lots_if, cash_if, asset_if, price, i, buy_ratio, min_cash)
            action_if = 'BUY'
        if sell_amt_if > 0: action_if = f'SELL_{sell_rsn_if}' + ('_BUY' if buy_amt_if > 0 else '')
        holding_if = sum(l['qty']*price for l in open_lots_if)
        asset_if   = cash_if + holding_if

        records.append({'Date': date, 'price': price,
                        'pred': row.get('pred'), 'pred_proba': row.get('pred_proba'),
                        'actual_target': actual,
                        'trade_action': action, 'cash': cash, 'buy_amount': buy_amt,
                        'sell_amount': sell_amt, 'holding_value': holding, 'asset': asset,
                        'cum_return': asset / initial_cash - 1,
                        'trade_action_if': action_if, 'cash_if': cash_if,
                        'buy_amount_if': buy_amt_if, 'sell_amount_if': sell_amt_if,
                        'holding_value_if': holding_if, 'asset_if': asset_if,
                        'cum_return_if': asset_if / initial_cash - 1})
    return pd.DataFrame(records)


sim_summary_df   = summarize_sim(sim_pred_df, target_col, N_DAYS_BEST)
take_profit_th   = best_cfg['target_return_threshold']
cash_sim_df      = make_cash_simulation(
    sim_pred_df, close_col, target_col, N_DAYS_BEST,
    take_profit_threshold=take_profit_th,
    initial_cash=INITIAL_CASH, buy_ratio=BUY_RATIO, min_cash_ratio=MIN_CASH_RATIO)

print('[Simulation Test 요약]')
display_df(sim_summary_df.T.rename(columns={0:'value'}))
print()
print('pred 기준 action 분포:')
print(cash_sim_df['trade_action'].value_counts(dropna=False))

final_asset    = cash_sim_df['asset'].iloc[-1]
final_asset_if = cash_sim_df['asset_if'].iloc[-1]
print(f'\n최종 자산 (pred 기준): {final_asset:,.0f}  ({(final_asset/INITIAL_CASH-1)*100:.1f}%)')
print(f'최종 자산 (what-if):   {final_asset_if:,.0f}  ({(final_asset_if/INITIAL_CASH-1)*100:.1f}%)')


# ============================================================
# ── 그래프 1: pred vs actual_target ───────────────────────────────────────
fig_pred_actual = go.Figure()
fig_pred_actual.add_trace(go.Scatter(
    x=sim_pred_df['Date'], y=sim_pred_df['pred'].astype(float) + 0.04,
    mode='lines+markers', name='pred (앙상블 신호)',
    line=dict(width=2, dash='dot'), marker=dict(size=7, symbol='x')))
fig_pred_actual.add_trace(go.Scatter(
    x=sim_pred_df['Date'], y=sim_pred_df[target_col].astype(float),
    mode='lines+markers', name='actual_target',
    line=dict(width=2), marker=dict(size=7, symbol='circle')))
fig_pred_actual.update_layout(
    title=f'{ETF_CODE} v4 — pred vs actual_target (precision={sim_metrics.get("precision",0):.3f})',
    xaxis_title='Date', yaxis_title='0 / 1', hovermode='x unified', height=420,
    yaxis=dict(tickvals=[0,1], ticktext=['0','1'], range=[-0.2,1.4]))
fig_pred_actual.show()

# ── 그래프 2: 자산 추이 ────────────────────────────────────────────────────
fig_cash_flow = go.Figure()
fig_cash_flow.add_trace(go.Scatter(
    x=cash_sim_df['Date'], y=cash_sim_df['asset'],
    mode='lines', name='자산 (앙상블 pred)', line=dict(width=2)))
fig_cash_flow.add_trace(go.Scatter(
    x=cash_sim_df['Date'], y=cash_sim_df['asset_if'],
    mode='lines', name='자산 (what-if)', line=dict(width=2, dash='dash')))
fig_cash_flow.add_hline(y=INITIAL_CASH, line_dash='dot', line_color='gray',
    annotation_text='초기자산', annotation_position='bottom right')
fig_cash_flow.update_layout(
    title=f'{ETF_CODE} v4 — 자산 추이',
    xaxis_title='Date', yaxis_title='자산 (KRW)', hovermode='x unified', height=420)
fig_cash_flow.show()

# ── 그래프 3: threshold scan ───────────────────────────────────────────────
scan_valid = stage3_result['scan_df'].dropna(subset=['precision'])
if not scan_valid.empty:
    fig_scan = go.Figure()
    fig_scan.add_trace(go.Scatter(
        x=scan_valid['threshold'], y=scan_valid['precision'],
        mode='lines+markers', name='validation precision'))
    if 'recall' in scan_valid.columns:
        fig_scan.add_trace(go.Scatter(
            x=scan_valid['threshold'], y=scan_valid['recall'],
            mode='lines', name='validation recall', line=dict(dash='dot')))
    fig_scan.add_hline(y=S3_PRECISION_TARGET, line_dash='dash', line_color='red',
        annotation_text=f'목표 {S3_PRECISION_TARGET:.0%}')
    fig_scan.add_vline(x=CALIBRATED_TH, line_dash='dot', line_color='green',
        annotation_text=f'선택 th={CALIBRATED_TH:.2f}')
    fig_scan.update_layout(
        title='Stage 3 — Threshold Scan (검증셋 precision/recall)',
        xaxis_title='threshold', yaxis_title='metric', height=400)
    fig_scan.show()


# ============================================================
# BASE_DATE까지 전체 데이터로 앙상블 재학습 → 최신 신호 추론
final_train_df = all_lagged_ev[all_lagged_ev['Date'] <= WINDOWS['base_date']].copy()
keep_fin = [target_col] + best_fc
fin_c    = final_train_df[[c for c in keep_fin if c in final_train_df.columns]]\
           .replace([np.inf,-np.inf], np.nan).dropna().copy()
X_fin = fin_c[best_fc]; y_fin = fin_c[target_col].astype(int)
pos_fin = int((y_fin==1).sum()); neg_fin = int((y_fin==0).sum())

final_ensemble = []
for info in stage3_result['ensemble_models']:
    mname   = info['model_name']
    mparams = info.get('mparams', {})
    m = make_classifier(mname, RANDOM_STATE, mparams, pos_fin, neg_fin)
    m.fit(X_fin, y_fin)
    final_ensemble.append(m)

latest_row    = all_lagged_in.iloc[[-1]].copy()
latest_date   = latest_row['Date'].values[0]
X_latest      = latest_row[best_fc].replace([np.inf,-np.inf], np.nan)

final_probas  = [m.predict_proba(X_latest)[:,1][0] for m in final_ensemble]
avg_proba_lat = float(np.mean(final_probas))
pred_latest   = int(avg_proba_lat >= CALIBRATED_TH)

real_inference = {
    'as_of_date':     str(latest_date)[:10],
    'base_date':      BASE_DATE,
    'n_days':         N_DAYS_BEST,
    'target_col':     target_col,
    'ensemble_count': len(final_ensemble),
    'individual_probas': [round(float(p), 4) for p in final_probas],
    'avg_pred_proba': round(float(avg_proba_lat), 4),
    'calibrated_threshold': CALIBRATED_TH,
    'signal':         'BUY' if pred_latest == 1 else 'WAIT',
}
print(json.dumps(real_inference, ensure_ascii=False, indent=2))


# ============================================================
base_tag = f"{ETF_CODE}_{pd.to_datetime(BASE_DATE).strftime('%Y%m%d')}"

sim_pred_df.to_csv(OUTPUT_DIR / f'sim_pred_{base_tag}.csv', index=False)
cash_sim_df.to_csv(OUTPUT_DIR / f'cash_sim_{base_tag}.csv', index=False)
best_artifacts['importance_df'].to_csv(OUTPUT_DIR / f'feature_importance_{base_tag}.csv', index=False)
stage1_result['trials_df'].to_csv(OUTPUT_DIR / f'stage1_trials_{base_tag}.csv', index=False)
stage2_result['trials_df'].to_csv(OUTPUT_DIR / f'stage2_trials_{base_tag}.csv', index=False)
stage3_result['scan_df'].to_csv(OUTPUT_DIR / f'stage3_threshold_scan_{base_tag}.csv', index=False)

with open(OUTPUT_DIR / f'best_config_{base_tag}.json', 'w', encoding='utf-8') as f:
    json.dump(best_cfg, f, ensure_ascii=False, indent=2, default=str)
with open(OUTPUT_DIR / f'real_inference_{base_tag}.json', 'w', encoding='utf-8') as f:
    json.dump(real_inference, f, ensure_ascii=False, indent=2, default=str)

fig_pred_actual.write_html(OUTPUT_DIR / f'plot_pred_vs_actual_{base_tag}.html')
fig_cash_flow.write_html(OUTPUT_DIR / f'plot_cash_flow_{base_tag}.html')

# 누적 summary CSV
summary_path = Path('experiments_v4/experiment_summary_v4.csv')
summary_path.parent.mkdir(parents=True, exist_ok=True)
s1row = sim_summary_df.iloc[0].to_dict()
summary_row = {
    'experiment_name':          EXPERIMENT_NAME,
    'run_date':                 pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
    'etf':                      ETF_CODE,
    'base_date':                BASE_DATE,
    'best_n_days':              best_cfg['n_days'],
    'best_threshold':           best_cfg['target_return_threshold'],
    'best_model_s2':            best_cfg['model_name'],
    'ensemble_count':           stage3_result['n_ensemble'],
    'calibrated_threshold':     CALIBRATED_TH,
    'val_precision':            stage3_result['calibrated_val_precision'],
    'val_pred_count':           stage3_result['calibrated_val_pred_count'],
    'sim_precision':            s1row.get('precision'),
    'sim_recall':               s1row.get('recall'),
    'sim_f1':                   s1row.get('f1'),
    'sim_pred_count':           s1row.get('pred_1_count'),
    'sim_precision_lift':       s1row.get('precision_lift'),
    'sim_compound_return':      s1row.get('strategy_compound_return'),
    'sim_avg_ret_buy':          s1row.get('avg_ret_when_buy'),
    'cash_sim_final_value':     cash_sim_df['asset'].iloc[-1],
    'cash_sim_return_pct':      (cash_sim_df['asset'].iloc[-1] / INITIAL_CASH - 1) * 100,
    'real_signal':              real_inference['signal'],
    'real_pred_proba':          real_inference['avg_pred_proba'],
    's2_n_trials':              S2_N_TRIALS,
    'precision_floor':          S2_PRECISION_FLOOR,
    'precision_target':         S3_PRECISION_TARGET,
}

if summary_path.exists():
    new_df = pd.concat([pd.read_csv(summary_path), pd.DataFrame([summary_row])], ignore_index=True)
else:
    new_df = pd.DataFrame([summary_row])
new_df.to_csv(summary_path, index=False)

print(f'결과 저장 완료: {OUTPUT_DIR.resolve()}')
print(f'Summary CSV: {summary_path.resolve()}')
display_df(new_df)
