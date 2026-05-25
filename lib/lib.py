import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import FinanceDataReader as fdr

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix
)
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

## 함수

################################################################################################################################################
## 1. Base Feature Dataset 생성
################################################################################################################################################
def load_price_data(ticker, start_date="2020-01-01", end_date=None):
    """
    FinanceDataReader로 가격 데이터를 가져온다.
    Date 컬럼을 일반 컬럼으로 유지한다.
    """
    df = fdr.DataReader(ticker, start_date, end_date)
    df = df.reset_index().rename(columns={"index": "Date"})

    # 컬럼명 정리
    df["Date"] = pd.to_datetime(df["Date"])

    return df

def make_target_etf_features(etf_df, prefix):
    """
    예측 대상 ETF용 feature 생성.
    target은 여기서 만들지 않는다.
    """
    df = etf_df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    close = df["Adj Close"]
    volume = df["Volume"]

    result = pd.DataFrame()
    result["Date"] = df["Date"]

    # target 만들 때 필요하므로 Close는 반드시 보존
    result[f"{prefix}_adj_close"] = close

    # 수익률
    result[f"{prefix}_ret_1d"] = close.pct_change(1)
    result[f"{prefix}_ret_5d"] = close.pct_change(5)
    result[f"{prefix}_ret_20d"] = close.pct_change(20)

    # 이동평균 대비 위치
    ma_5 = close.rolling(5).mean()
    ma_20 = close.rolling(20).mean()
    ma_60 = close.rolling(60).mean()

    result[f"{prefix}_ma5_ratio"] = close / ma_5 - 1
    result[f"{prefix}_ma20_ratio"] = close / ma_20 - 1
    result[f"{prefix}_ma60_ratio"] = close / ma_60 - 1

    # 변동성
    result[f"{prefix}_vol_20d"] = result[f"{prefix}_ret_1d"].rolling(20).std()

    # 거래량 비율
    vol_ma20 = volume.rolling(20).mean()
    result[f"{prefix}_volume_ratio_20d"] = volume / vol_ma20 - 1

    return result

def make_external_features(raw_df, name, feature_type="price"):
    """
    외부 지표용 최소 파생변수 생성.
    feature_type:
        - price: 일반 가격형 지표
        - risk: VIX 같은 리스크 레벨 지표
        - rate: 금리 지표
    """
    df = raw_df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    close = df["Adj Close"]

    result = pd.DataFrame()
    result["Date"] = df["Date"]

    if feature_type == "price":
        result[f"{name}_ret_5d"] = close.pct_change(5)
        result[f"{name}_ret_20d"] = close.pct_change(20)

    elif feature_type == "risk":
        result[f"{name}_level"] = close
        result[f"{name}_chg_5d"] = close.diff(5)
        result[f"{name}_chg_20d"] = close.diff(20)

    elif feature_type == "rate":
        result[f"{name}_level"] = close
        result[f"{name}_diff_5d"] = close.diff(5)
        result[f"{name}_diff_20d"] = close.diff(20)

    else:
        raise ValueError("feature_type must be one of ['price', 'risk', 'rate']")

    return result

def make_base_feature_dataset(
    etf_code,
    external_tickers,
    external_feature_types,
    start_date="2020-01-01",
    end_date=None
):
    """
    target 없는 기본 feature dataset 생성.
    이 함수는 느린 작업이므로 한 번만 실행하는 것을 목표로 한다.
    """
    # 1. ETF 본체 로드
    etf_raw = load_price_data(etf_code, start_date, end_date)

    # 2. ETF 본체 feature 생성
    base_df = make_target_etf_features(etf_raw, prefix=etf_code)

    # 3. 외부 지표 붙이기
    for name, ticker in external_tickers.items():
        print(f"Loading external ticker: {name} / {ticker}")

        try:
            raw = load_price_data(ticker, start_date, end_date)

            feature_type = external_feature_types.get(name, "price")

            ext_feat = make_external_features(
                raw_df=raw,
                name=name,
                feature_type=feature_type
            )

            base_df = base_df.merge(ext_feat, on="Date", how="left")

        except Exception as e:
            print(f"[SKIP] {name} / {ticker} 로드 실패:", e)

    # 4. 날짜 정렬
    base_df = base_df.sort_values("Date").reset_index(drop=True)

    # 5. feature_cols 정리
    close_col = f"{etf_code}_adj_close"

    feature_cols = [
        col for col in base_df.columns
        if col not in ["Date", close_col]
    ]

    return base_df, feature_cols, close_col

def make_base_feature_dataset(
    etf_code,
    external_tickers,
    external_feature_types,
    start_date="2020-01-01",
    end_date=None
):
    """
    target 없는 기본 feature dataset 생성.
    이 함수는 느린 작업이므로 한 번만 실행하는 것을 목표로 한다.

    주말 데이터가 섞여 NA가 늘어나는 문제를 막기 위해
    ETF / 외부 ticker 모두 영업일(월~금)만 사용한다.
    """

    # =========================================================
    # 0. 영업일 필터 함수
    # =========================================================
    def keep_weekdays_only(df, date_col="Date"):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df[df[date_col].dt.weekday < 5]  # 월=0, 금=4
        df = df.sort_values(date_col).reset_index(drop=True)
        return df

    # =========================================================
    # 1. ETF 본체 로드
    # =========================================================
    etf_raw = load_price_data(etf_code, start_date, end_date)

    # 주말 제거
    etf_raw = keep_weekdays_only(etf_raw, date_col="Date")

    # =========================================================
    # 2. ETF 본체 feature 생성
    # =========================================================
    base_df = make_target_etf_features(etf_raw, prefix=etf_code)

    # feature 생성 후에도 혹시 모르니 다시 주말 제거
    base_df = keep_weekdays_only(base_df, date_col="Date")

    # =========================================================
    # 3. 외부 지표 붙이기
    # =========================================================
    for name, ticker in external_tickers.items():
        print(f"Loading external ticker: {name} / {ticker}")

        try:
            raw = load_price_data(ticker, start_date, end_date)

            # 외부 ticker도 주말 제거
            raw = keep_weekdays_only(raw, date_col="Date")

            feature_type = external_feature_types.get(name, "price")

            ext_feat = make_external_features(
                raw_df=raw,
                name=name,
                feature_type=feature_type
            )

            # 외부 feature 생성 후에도 다시 주말 제거
            ext_feat = keep_weekdays_only(ext_feat, date_col="Date")

            # ETF 거래일 기준으로 붙임
            base_df = base_df.merge(ext_feat, on="Date", how="left")

        except Exception as e:
            print(f"[SKIP] {name} / {ticker} 로드 실패:", e)

    # =========================================================
    # 4. 날짜 정렬 + 주말 최종 제거
    # =========================================================
    base_df = keep_weekdays_only(base_df, date_col="Date")
    
    # =========================================================
    # 4.1. 외부 지표 NA 보정
    # - ETF 본체는 건드리지 않고
    # - 외부 지표만 ffill
    # =========================================================
    etf_prefix = f"{etf_code}_"

    external_cols = [
        col for col in base_df.columns
        if col != "Date" and not col.startswith(etf_prefix)
    ]

    base_df[external_cols] = base_df[external_cols].ffill()

    # =========================================================
    # 5. feature_cols 정리
    # =========================================================
    close_col = f"{etf_code}_adj_close"

    feature_cols = [
        col for col in base_df.columns
        if col not in ["Date", close_col]
    ]

    return base_df, feature_cols, close_col

################################################################################################################################################
## 2. VIF 기반 불필요 칼럼 제거
################################################################################################################################################
def reduce_features_by_vif(
    df,
    feature_cols,
    vif_threshold=30.0,
    date_col="Date",
    verbose=True
):
    """
    VIF 기준으로 다중공선성이 높은 feature를 반복 제거한다.

    주의:
    - target 생성 전 단계에서 실행한다.
    - lag 생성 전 단계에서 실행한다.
    - Date, adj_close 등 보존 컬럼은 feature_cols에 넣지 않는 것을 전제로 한다.
    """

    # 1. 숫자형 feature만 사용
    numeric_feature_cols = [
        col for col in feature_cols
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    work_df = df[numeric_feature_cols].copy()

    # 2. inf 처리
    work_df = work_df.replace([np.inf, -np.inf], np.nan)

    # 3. VIF 계산용 결측 제거
    #    여기서는 VIF 계산에만 dropna를 쓰고,
    #    원본 df 자체를 줄이지는 않는다.
    vif_calc_df = work_df.dropna(axis=0).copy()

    print("VIF 계산 대상 row 수:", len(vif_calc_df))
    print("VIF 계산 대상 feature 수:", len(numeric_feature_cols))

    if len(vif_calc_df) == 0:
        raise ValueError("VIF 계산 가능한 데이터가 없습니다. 결측값을 확인하세요.")

    # 4. 상수 컬럼 제거
    nunique = vif_calc_df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()

    if len(constant_cols) > 0:
        print("상수 컬럼 제거:", constant_cols)

    remaining_cols = [
        col for col in numeric_feature_cols
        if col not in constant_cols
    ]

    removed_records = []

    # 5. VIF 반복 제거
    while True:
        if len(remaining_cols) <= 1:
            break

        X = vif_calc_df[remaining_cols].copy()

        # 표준화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        vif_values = []

        for i, col in enumerate(remaining_cols):
            try:
                vif = variance_inflation_factor(X_scaled, i)
            except Exception:
                vif = np.inf

            vif_values.append({
                "feature": col,
                "vif": vif
            })

        vif_df = pd.DataFrame(vif_values).sort_values("vif", ascending=False)

        max_vif_row = vif_df.iloc[0]
        max_feature = max_vif_row["feature"]
        max_vif = max_vif_row["vif"]

        if verbose:
            print(f"현재 max VIF: {max_vif:.2f} / feature: {max_feature}")

        if max_vif <= vif_threshold:
            break

        # 가장 VIF 높은 컬럼 제거
        remaining_cols.remove(max_feature)

        removed_records.append({
            "removed_feature": max_feature,
            "vif": max_vif,
            "remaining_feature_count": len(remaining_cols)
        })

    removed_vif_df = pd.DataFrame(removed_records)

    # 6. 최종 VIF 테이블 계산
    final_vif_records = []

    if len(remaining_cols) > 1:
        X_final = vif_calc_df[remaining_cols].copy()

        scaler = StandardScaler()
        X_final_scaled = scaler.fit_transform(X_final)

        for i, col in enumerate(remaining_cols):
            try:
                vif = variance_inflation_factor(X_final_scaled, i)
            except Exception:
                vif = np.inf

            final_vif_records.append({
                "feature": col,
                "vif": vif
            })

        final_vif_df = pd.DataFrame(final_vif_records).sort_values("vif", ascending=False)

    else:
        final_vif_df = pd.DataFrame({
            "feature": remaining_cols,
            "vif": [np.nan] * len(remaining_cols)
        })

    print()
    print("========== VIF 제거 결과 ==========")
    print("초기 feature 수:", len(numeric_feature_cols))
    print("상수 제거 feature 수:", len(constant_cols))
    print("VIF 제거 feature 수:", len(removed_records))
    print("최종 feature 수:", len(remaining_cols))
    print("===================================")

    return remaining_cols, removed_vif_df, final_vif_df


################################################################################################################################################
## 3. Target 변수 생성
################################################################################################################################################
def add_target_column(
    df,
    close_col,
    n_days=5,
    threshold=0.05,
    target_col=None
):
    """
    현재 시점 기준 n_days 뒤 수익률이 threshold 이상이면 1, 아니면 0인 target 생성.

    예:
    n_days=5, threshold=0.05
    → 5거래일 뒤 수익률이 +5% 이상이면 target=1
    """

    result = df.copy()
    result = result.sort_values("Date").reset_index(drop=True)

    if target_col is None:
        target_col = f"target_{n_days}d_up_{int(threshold * 100)}pct"

    # 미래 가격
    future_close = result[close_col].shift(-n_days)

    # 미래 수익률
    result[f"future_ret_{n_days}d"] = future_close / result[close_col] - 1

    # target 생성
    result[target_col] = np.where(
        result[f"future_ret_{n_days}d"] >= threshold,
        1,
        0
    )

    # 마지막 n_days개는 미래 가격이 없으므로 제거
    result.loc[result[f"future_ret_{n_days}d"].isna(), target_col] = np.nan

    return result, target_col


################################################################################################################################################
## 4. Lag 생성 후 변수별 최적 lag 탐색
################################################################################################################################################

def find_best_lag_by_feature(
    df,
    feature_cols,
    target_col,
    lag_days=[1, 3, 5, 10, 20],
    date_col="Date",
    method="corr"
):
    """
    각 feature별로 target과 가장 관계가 강한 선행 lag를 찾는다.

    lag 의미:
    - lag=1  : feature의 1거래일 전 값으로 오늘 target 설명
    - lag=5  : feature의 5거래일 전 값으로 오늘 target 설명
    - lag=20 : feature의 20거래일 전 값으로 오늘 target 설명

    즉, feature가 먼저 움직이고 나중에 target이 움직이는 구조만 본다.
    """

    records = []

    for col in feature_cols:
        if col not in df.columns:
            continue

        for lag in lag_days:
            temp = df[[date_col, col, target_col]].copy()

            # 선행변수 구조
            temp[f"{col}_lag{lag}"] = temp[col].shift(lag)

            temp = temp[[f"{col}_lag{lag}", target_col]].replace(
                [np.inf, -np.inf],
                np.nan
            ).dropna()

            if len(temp) < 30:
                continue

            x = temp[f"{col}_lag{lag}"]
            y = temp[target_col]

            if x.nunique() <= 1:
                corr = np.nan
            else:
                corr = x.corr(y)

            records.append({
                "feature": col,
                "lag": lag,
                "corr": corr,
                "abs_corr": abs(corr) if pd.notna(corr) else np.nan,
                "n_rows": len(temp)
            })

    lag_result_df = pd.DataFrame(records)

    if lag_result_df.empty:
        raise ValueError("lag 탐색 결과가 비어 있습니다. feature_cols 또는 target_col을 확인하세요.")

    # feature별 abs_corr가 가장 큰 lag 선택
    best_lag_df = (
        lag_result_df
        .sort_values(["feature", "abs_corr"], ascending=[True, False])
        .groupby("feature", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    best_lag_df = best_lag_df.sort_values("abs_corr", ascending=False).reset_index(drop=True)

    return lag_result_df, best_lag_df

def make_lagged_dataset_by_best_lag(
    df,
    best_lag_df,
    target_col,
    close_col,
    n_days=5,
    date_col="Date"
):
    """
    best_lag_df 기준으로 feature별 최적 lag를 적용한 최종 모델용 데이터셋 생성.
    """

    result = pd.DataFrame()
    result[date_col] = df[date_col]
    result[close_col] = df[close_col]

    # 확인용 미래수익률 보존
    future_ret_col = f"future_ret_{n_days}d"
    if future_ret_col in df.columns:
        result[future_ret_col] = df[future_ret_col]

    # target 보존
    result[target_col] = df[target_col]

    lagged_feature_cols = []

    for _, row in best_lag_df.iterrows():
        feature = row["feature"]
        lag = int(row["lag"])

        if feature not in df.columns:
            continue

        lagged_col = f"{feature}_lag{lag}"
        result[lagged_col] = df[feature].shift(lag)
        lagged_feature_cols.append(lagged_col)

    # 결측/무한값 제거
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.dropna().reset_index(drop=True)

    return result, lagged_feature_cols



################################################################################################################################################
## 5. RandomForest in-sample 학습 + permutation importance
################################################################################################################################################

def run_rf_permutation_importance_in_sample(
    lagged_df,
    feature_cols,
    target_col,
    date_col="Date",
    close_col=None,
    n_rf_runs=3,
    n_repeats=10,
    random_state=42
):
    """
    lagged_df 기준으로 RandomForestClassifier를 in-sample 학습한 뒤
    permutation importance를 반복 계산한다.

    핵심:
    - RF를 n_rf_runs번 학습
    - 각 RF마다 permutation을 n_repeats번 수행
    - feature별 importance raw 값을 전부 저장
    - 최종 importance_df는 총 n_rf_runs * n_repeats개의 raw importance 기준으로 계산

    예:
    n_rf_runs=3, n_repeats=10이면
    feature별 importance 값 30개를 기반으로 평균/표준편차/스코어 계산
    """

    df = lagged_df.copy()

    # =====================================================
    # 1. 모델 input / target 분리
    # =====================================================

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X = X.replace([np.inf, -np.inf], np.nan)

    model_df = pd.concat([X, y], axis=1).dropna().copy()

    X = model_df[feature_cols].copy()
    y = model_df[target_col].astype(int).copy()


    # =====================================================
    # 2. 여러 RF run + permutation raw importance 저장
    # =====================================================

    importance_records = []
    baseline_records = []

    for run in range(n_rf_runs):

        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state + run,
            n_jobs=-1
        )

        rf.fit(X, y)

        # =================================================
        # 3. in-sample 예측 성능 확인
        # =================================================

        pred = rf.predict(X)
        pred_proba = rf.predict_proba(X)[:, 1]

        acc = accuracy_score(y, pred)
        precision = precision_score(y, pred, zero_division=0)
        recall = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        loss = log_loss(y, pred_proba)

        cm = confusion_matrix(y, pred)

        baseline_records.append({
            "run": run + 1,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "log_loss": loss,
            "pred_1_count": int((pred == 1).sum()),
            "actual_1_count": int((y == 1).sum())
        })

        # =================================================
        # 4. permutation importance
        # =================================================

        perm = permutation_importance(
            rf,
            X,
            y,
            scoring="neg_log_loss",
            n_repeats=n_repeats,
            random_state=random_state + run,
            n_jobs=-1
        )

        # 핵심:
        # perm.importances shape = (n_features, n_repeats)
        # 여기서 반복별 raw importance를 전부 저장한다.
        for i, col in enumerate(feature_cols):
            for repeat_idx, importance_value in enumerate(perm.importances[i]):
                importance_records.append({
                    "run": run + 1,
                    "repeat": repeat_idx + 1,
                    "feature": col,
                    "importance": importance_value
                })

    # =====================================================
    # 5. 결과 정리
    # =====================================================

    raw_importance_df = pd.DataFrame(importance_records)
    baseline_df = pd.DataFrame(baseline_records)

    importance_df = (
        raw_importance_df
        .groupby("feature", as_index=False)
        .agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
            importance_var=("importance", "var"),
            importance_min=("importance", "min"),
            importance_max=("importance", "max"),
            run_count=("run", "nunique"),
            repeat_count=("repeat", "count")
        )
    )

    # 안정성 점수
    # 평균 중요도는 높고, 30회 전체 기준 표준편차는 낮을수록 높게
    importance_df["importance_score"] = (
        importance_df["importance_mean"]
        - importance_df["importance_std"].fillna(0)
    )

    importance_df = importance_df.sort_values(
        "importance_score",
        ascending=False
    ).reset_index(drop=True)

    return importance_df, raw_importance_df, baseline_df

def split_lagged_feature_name(feature_name):
    """
    feature_lag20 형태의 컬럼명을 원본 feature와 lag로 분리한다.
    """

    if "_lag" not in feature_name:
        return feature_name, np.nan

    base_name = feature_name.rsplit("_lag", 1)[0]
    lag = feature_name.rsplit("_lag", 1)[1]

    try:
        lag = int(lag)
    except:
        lag = np.nan

    return base_name, lag

