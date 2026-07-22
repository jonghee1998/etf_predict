"""Quant Manager Korea static data generator.

KOSPI market-cap leaders and major non-leveraged Korean-listed ETFs are
downloaded from KRX listings (FinanceDataReader) and Yahoo Finance prices.
"""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).parent
OUT = ROOT / "data.js"
SIM_OUT = ROOT / "data_sim.js"
TOP_N = 50
STOCK_UNIVERSE = 200
ETF_UNIVERSE = 120
PERIODS = {"m1": 21, "m3": 63, "m6": 126, "m12": 252}
NULL_STOCK = ["pe", "fpe", "pbr", "psr", "evebitda", "peg", "fcfy", "divy",
              "roe", "roa", "gm", "opm", "npm", "curr", "d2e", "revg", "earng",
              "beta", "sfloat", "inst", "insider", "tgt", "rec", "nanal", "earnings"]
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; QuantManagerKR/1.0)"}
FNGUIDE_BASE = "https://comp.fnguide.com/SVO2/ASP"


def _number(value):
    """HTML/DART 숫자를 안전하게 float로 바꾼다."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "--", "N/A", "nan"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    try:
        value = float(text)
        return -value if negative else value
    except ValueError:
        return None


def _label(value):
    return re.sub(r"계산에 참여한 계정 펼치기|\s+", "", str(value))


def _row(table, *names):
    labels = table.iloc[:, 0].map(_label)
    for name in names:
        wanted = _label(name)
        # 펼친 상세행에는 "PER순이익/..."처럼 산식 설명이 라벨 뒤에 붙는다.
        hit = table.loc[(labels == wanted) | labels.str.startswith(wanted)]
        if not hit.empty:
            return hit.iloc[0]
    return None


def _value(table, name, column, *aliases):
    row = _row(table, name, *aliases)
    return None if row is None else _number(row.get(column))


def _annual_columns(table):
    # 첫 날짜열의 월이 해당 회사 결산월이다. 마지막에는 최신 누적 분기가
    # 붙기도 하므로 같은 월만 고른다. 이 방식은 12월/3월 결산법인을 모두 처리한다.
    dated = [c for c in table.columns[1:] if re.fullmatch(r"20\d\d/\d\d", str(c))]
    if not dated:
        return []
    fiscal_month = str(dated[0])[-2:]
    return [c for c in dated if str(c).endswith(f"/{fiscal_month}")]


def _ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator not in (None, 0) else None


def _growth(current, previous):
    return current / previous - 1 if current is not None and previous not in (None, 0) else None


def _fnguide_tables(code, page):
    params = {"pGB": 1, "gicode": f"A{code}", "cID": "", "MenuYn": "Y",
              "ReportGB": "", "NewMenuID": 103 if page == "SVD_Finance" else 105,
              "stkGb": 701}
    response = requests.get(f"{FNGUIDE_BASE}/{page}.asp", params=params,
                            headers=HTTP_HEADERS, timeout=20)
    response.raise_for_status()
    # 명시적으로 lxml을 사용하면 html5lib가 없는 최소 실행환경에서도 동작한다.
    # FnGuide는 세부 계정을 접힌 행으로 제공한다. displayed_only=False가 없으면
    # 이익잉여금·장기차입금 같은 Z/F-Score 필수 계정이 조용히 누락된다.
    return pd.read_html(StringIO(response.text), flavor="lxml", displayed_only=False)


def _piotroski_from_fnguide(inc, bal, cf, annual):
    """FnGuide 연결 연간표로 계산 가능한 Piotroski 8개 항목을 계산한다."""
    if len(annual) < 2:
        return None, 0
    old, new = annual[-2], annual[-1]
    ni0, ni1 = _value(inc, "당기순이익", new), _value(inc, "당기순이익", old)
    ta0, ta1 = _value(bal, "자산", new), _value(bal, "자산", old)
    cfo0 = _value(cf, "영업활동으로인한현금흐름", new)
    rev0, rev1 = _value(inc, "매출액", new), _value(inc, "매출액", old)
    gp0, gp1 = _value(inc, "매출총이익", new), _value(inc, "매출총이익", old)
    debt0 = _value(bal, "장기차입금", new, "장기금융부채")
    debt1 = _value(bal, "장기차입금", old, "장기금융부채")
    ca0, ca1 = _value(bal, "유동자산", new), _value(bal, "유동자산", old)
    cl0, cl1 = _value(bal, "유동부채", new), _value(bal, "유동부채", old)
    checks = []
    if ni0 is not None and ta0: checks.append(ni0 / ta0 > 0)
    if cfo0 is not None: checks.append(cfo0 > 0)
    if None not in (ni0, ni1) and ta0 and ta1: checks.append(ni0 / ta0 > ni1 / ta1)
    if None not in (cfo0, ni0): checks.append(cfo0 > ni0)
    if None not in (debt0, debt1) and ta0 and ta1: checks.append(debt0 / ta0 <= debt1 / ta1)
    if None not in (ca0, ca1) and cl0 and cl1: checks.append(ca0 / cl0 > ca1 / cl1)
    if None not in (gp0, gp1) and rev0 and rev1: checks.append(gp0 / rev0 > gp1 / rev1)
    if None not in (rev0, rev1) and ta0 and ta1: checks.append(rev0 / ta0 > rev1 / ta1)
    return (int(sum(checks)), len(checks)) if checks else (None, 0)


def collect_fnguide(code, mcap):
    """FnGuide 국내 연결 재무/투자지표. 금액 원 단위, fin만 화면용 십억원 단위."""
    finance = _fnguide_tables(code, "SVD_Finance")
    invest = _fnguide_tables(code, "SVD_Invest")
    if len(finance) < 5:
        raise ValueError("FnGuide financial tables missing")
    inc, qinc, bal, cf = finance[0], finance[1], finance[2], finance[4]
    annual = _annual_columns(inc)[-4:]
    if not annual:
        raise ValueError("FnGuide annual columns missing")
    latest = annual[-1]
    revenue = _value(inc, "매출액", latest)
    gross = _value(inc, "매출총이익", latest)
    op = _value(inc, "영업이익", latest)
    net = _value(inc, "당기순이익", latest)
    assets = _value(bal, "자산", latest)
    equity = _value(bal, "자본", latest)
    liabilities = _value(bal, "부채", latest)
    current_assets = _value(bal, "유동자산", latest)
    current_liabilities = _value(bal, "유동부채", latest)

    out = {"fin": {"years": [str(c)[:4] for c in annual]}}
    for key, name in (("revenue", "매출액"), ("op", "영업이익"), ("net", "당기순이익")):
        # FnGuide 금액 단위는 억원. 차트 공통 단위인 십억원으로 변환한다.
        out["fin"][key] = [None if (v := _value(inc, name, c)) is None else round(v / 10, 3)
                            for c in annual]
    out.update(roe=_ratio(net, equity), roa=_ratio(net, assets), gm=_ratio(gross, revenue),
               opm=_ratio(op, revenue), npm=_ratio(net, revenue),
               curr=_ratio(current_assets, current_liabilities),
               d2e=None if liabilities is None or not equity else liabilities / equity * 100)

    qcols = [c for c in qinc.columns[1:] if re.fullmatch(r"20\d\d/\d\d", str(c))]
    if qcols:
        qcol = qcols[-1]
        out["revg"] = _growth(_value(qinc, "매출액", qcol), _value(qinc, "매출액", "전년동기"))
        out["earng"] = _growth(_value(qinc, "당기순이익", qcol), _value(qinc, "당기순이익", "전년동기"))

    out["fscore"], out["fmax"] = _piotroski_from_fnguide(inc, bal, cf, annual)
    retained = _value(bal, "이익잉여금(결손금)", latest, "이익잉여금")
    if None not in (assets, liabilities, current_assets, current_liabilities,
                    retained, op, revenue) and assets and liabilities and mcap:
        # 재무제표와 동일한 억원 단위로 시가총액을 맞춘 원형 Altman Z-Score.
        mcap_eok = mcap / 100_000_000
        out["zscore"] = round(1.2 * (current_assets-current_liabilities) / assets
                              + 1.4 * retained / assets + 3.3 * op / assets
                              + 0.6 * mcap_eok / liabilities + revenue / assets, 2)
    else:
        out["zscore"] = None

    if len(invest) > 1:
        inv = invest[1]
        inv_cols = [c for c in inv.columns[1:] if re.fullmatch(r"20\d\d/\d\d", str(c))]
        inv_col = latest if latest in inv_cols else inv_cols[-1]
        for key, name in (("pe", "PER"), ("pbr", "PBR"), ("psr", "PSR"),
                          ("evebitda", "EV/EBITDA")):
            out[key] = _value(inv, name, inv_col)
        dps = _value(inv, "DPS(보통주,현금)(원)", inv_col)
        out["divy"] = None  # 현재가가 수집된 뒤 아래에서 계산
        out["_dps"] = dps
        fcff = _value(inv, "FCFF", inv_col)
        out["fcfy"] = None if fcff is None or not mcap else (fcff * 100_000_000) / mcap
    return {k: round(v, 6) if isinstance(v, float) and np.isfinite(v) else v for k, v in out.items()}


def collect_financials(stocks, workers=8):
    """네트워크 지연을 줄이되 공급사에 무리가 가지 않는 제한된 병렬 수집."""
    meta = stocks.set_index("Code")
    results, failures = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(collect_fnguide, code, int(meta.loc[code, "Marcap"])): code
                for code in stocks.Code}
        for n, future in enumerate(as_completed(jobs), 1):
            code = jobs[future]
            try:
                results[code] = future.result()
            except Exception as exc:
                failures[code] = str(exc)[:100]
            if n % 25 == 0:
                print(f"   FnGuide {n}/{len(jobs)}")
            time.sleep(0.02)
    if failures:
        print(f"   ⚠ FnGuide 일부 실패 {len(failures)}개: {list(failures)[:10]}")
    return results


def ticker(code):
    return f"{str(code).zfill(6)}.KS"


def clean_code(symbol):
    return str(symbol).split(".")[0].zfill(6)


def price_frame(symbols, period):
    raw = yf.download([ticker(s) for s in symbols], period=period, auto_adjust=True,
                      progress=False, threads=True)
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(ticker(symbols[0]))
    close.columns = [clean_code(c) for c in close.columns]
    return close.dropna(axis=1, how="all").dropna(axis=0, how="all")


def metrics(s):
    s = s.dropna()
    out = {}
    for key, days in PERIODS.items():
        out[key] = float(s.iloc[-1] / s.iloc[-1-days] - 1) if len(s) > days else None
    out["mom121"] = float(s.iloc[-22] / s.iloc[-253] - 1) if len(s) > 253 else None
    ret = s.pct_change().dropna()
    r60, r12 = ret.iloc[-60:], ret.iloc[-252:]
    out["vol60"] = float(r60.std() * np.sqrt(252)) if len(r60) >= 40 else None
    out["sharpe12"] = float(r12.mean()/r12.std()*np.sqrt(252)) if len(r12) >= 120 and r12.std() else None
    eq = (1+r12).cumprod()
    out["mdd12"] = float((eq/eq.cummax()-1).min()) if len(eq) else None
    delta = s.diff(); gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain/loss.replace(0, np.nan)
    out["rsi14"] = round(float((100-100/(1+rs)).iloc[-1]), 1)
    tail = s.iloc[-252:]
    out["hi52"], out["lo52"], out["last"] = float(tail.max()), float(tail.min()), float(s.iloc[-1])
    out["from_high"] = float(s.iloc[-1]/tail.max()-1)
    ma = s.rolling(200).mean().iloc[-1]
    out["above_ma200"] = None if pd.isna(ma) else bool(s.iloc[-1] > ma)
    return {k: round(v, 6) if isinstance(v, float) and np.isfinite(v) else v for k, v in out.items()}


def px_payload(s, tail=270, decimals=2):
    s = s.dropna().iloc[-tail:]
    return {"d": [int(x.strftime("%Y%m%d")) for x in s.index],
            "c": [round(float(x), decimals) for x in s.values]}


def ranks(rows):
    return {key: sorted((s for s, m in rows.items() if m.get(key) is not None),
                        key=lambda s: rows[s][key], reverse=True)[:TOP_N]
            for key in PERIODS}


def etf_category(name):
    if re.search(r"채권|국고채|국채|회사채|단기채|통안채|CD금리|KOFR|머니마켓", name): return "채권·금리"
    if re.search(r"금현물|골드|은선물|원유|구리|농산물|원자재", name): return "원자재"
    if re.search(r"리츠|부동산|인프라", name): return "리츠·인프라"
    if re.search(r"배당|커버드콜|고배당", name): return "배당·인컴"
    if re.search(r"미국|S&P|나스닥|NASDAQ|다우|글로벌|차이나|중국|일본|인도|베트남", name): return "해외 주식"
    if re.search(r"반도체|2차전지|바이오|헬스|AI|로봇|자동차|은행|증권|보험|조선|방산|화장품|게임|미디어|에너지", name): return "산업·테마"
    if re.search(r"가치|성장|퀄리티|모멘텀|저변동|동일가중", name): return "팩터·스타일"
    return "국내 지수"


def stock_sector(industry):
    text = str(industry or "")
    rules = [
        ("정보기술·반도체", r"반도체|소프트웨어|컴퓨터|통신|전자부품|영상·음향"),
        ("금융", r"은행|금융|보험|신탁|증권"),
        ("헬스케어", r"의약|의료|병원|생물학|의료용"),
        ("산업재·조선·방산", r"조선|항공|기계|장비|운송|건설|엔지니어링"),
        ("자동차", r"자동차|차체|트레일러"),
        ("에너지·화학", r"석유|화학|가스|전기|에너지|배터리"),
        ("소재·철강", r"철강|금속|비금속|시멘트|유리|종이"),
        ("소비재·유통", r"음식|섬유|의류|화장품|유통|소매|숙박|여행"),
        ("미디어·엔터", r"영화|방송|오디오|출판|오락|게임"),
    ]
    for name, pattern in rules:
        if re.search(pattern, text): return name
    return "기타 KOSPI"


def load_universes():
    kospi = fdr.StockListing("KOSPI").copy()
    kospi["Code"] = kospi["Code"].astype(str).str.zfill(6)
    kospi = kospi[~kospi["Name"].str.contains(r"우$|우B$|우C$|스팩|리츠", regex=True, na=False)]
    desc = fdr.StockListing("KRX-DESC")[["Code", "Industry", "Products"]].copy()
    desc["Code"] = desc["Code"].astype(str).str.zfill(6)
    stocks = kospi.sort_values("Marcap", ascending=False).head(STOCK_UNIVERSE).merge(desc, on="Code", how="left")

    etfs = fdr.StockListing("ETF/KR").copy()
    etfs["Symbol"] = etfs["Symbol"].astype(str).str.zfill(6)
    banned = r"레버리지|인버스|곱버스|선물인버스|2X|2배|합성-인버스"
    etfs = etfs[~etfs["Name"].str.contains(banned, case=False, regex=True, na=False)]
    etfs = etfs.sort_values("MarCap", ascending=False).drop_duplicates("Name").head(ETF_UNIVERSE)
    return stocks, etfs


def collect_describe(stocks, etfs):
    symbols = list(stocks.Code) + list(etfs.Symbol)
    close = price_frame(symbols, "15mo")
    print("국내 연결 재무·투자지표 수집 (FnGuide)...")
    financials = collect_financials(stocks)
    stock_meta = stocks.set_index("Code").to_dict("index")
    etf_meta = etfs.set_index("Symbol").to_dict("index")

    stock_rows = {}
    for code in stocks.Code:
        if code not in close or close[code].count() < 40: continue
        meta = stock_meta[code]; industry = meta.get("Industry")
        industry = "한국 유가증권시장" if pd.isna(industry) else str(industry)
        products = meta.get("Products")
        products = "한국거래소 KOSPI 상장 기업." if pd.isna(products) else str(products)
        row = {"sym": code, "name": meta["Name"], "sector": stock_sector(industry),
            "industry": industry, "mcap": int(meta["Marcap"]), "summary": products,
            "fin": None, "fscore": None, "fmax": 0, "zscore": None}
        row.update({k: None for k in NULL_STOCK})
        row.update(financials.get(code, {}))
        row.update(metrics(close[code]))
        dps = row.pop("_dps", None)
        if dps is not None and row["last"]:
            row["divy"] = round(dps / row["last"] * 100, 4)
        row["px"] = px_payload(close[code])
        stock_rows[code] = row

    etf_rows = {}
    for code in etfs.Symbol:
        if code not in close or close[code].count() < 40: continue
        meta = etf_meta[code]; name = meta["Name"]; cat = etf_category(name)
        row = {"sym": code, "name": name, "cat": cat, "aum": int(meta["MarCap"])*100_000_000,
               "expense": None, "divy": None, "mgr": name.split()[0], "summary": "",
               "desc_kr": f"한국거래소 상장 {cat} ETF. 상품명: {name}."}
        row.update(metrics(close[code])); row["px"] = px_payload(close[code]); etf_rows[code] = row

    top, etop = ranks(stock_rows), ranks(etf_rows)
    sectors = {}
    for k, vals in top.items():
        cnt = {}
        for s in vals: cnt[stock_rows[s]["sector"]] = cnt.get(stock_rows[s]["sector"], 0)+1
        sectors[k] = cnt
    ecats = {}
    for k, vals in etop.items():
        cnt = {}
        for s in vals: cnt[etf_rows[s]["cat"]] = cnt.get(etf_rows[s]["cat"], 0)+1
        ecats[k] = cnt
    asof = str(close.index.max().date())
    return {"asof": asof, "generated": str(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")),
            "universe": len(stock_rows), "top": top, "sectors": sectors, "stocks": stock_rows,
            "etf_asof": asof, "etf_universe": len(etf_rows), "etop": etop, "ecats": ecats, "etfs": etf_rows}


def collect_sim(etfs):
    codes = list(etfs.Symbol)
    close = price_frame(codes, "max").loc["2002-01-01":]
    meta = etfs.set_index("Symbol")
    cats, px = {}, {}
    for code in codes:
        if code not in close or close[code].count() < 130: continue
        cats[code] = etf_category(meta.loc[code, "Name"])
        px[code] = px_payload(close[code], tail=len(close), decimals=2)
    payload = {"asof": str(close.index.max().date()), "generated": str(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")),
               "cats": cats, "px": px}
    SIM_OUT.write_text("window.QSIM = "+json.dumps(payload, ensure_ascii=False, separators=(",", ":"))+";", encoding="utf-8")
    print(f"saved {SIM_OUT} ({len(px)} ETFs, {SIM_OUT.stat().st_size/1024/1024:.1f}MB)")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", choices=["all", "describe", "sim"], default="all")
    args = ap.parse_args(); stocks, etfs = load_universes()
    print(f"universe: KOSPI {len(stocks)}, ETF {len(etfs)}")
    if args.only in ("all", "describe"):
        payload = collect_describe(stocks, etfs)
        OUT.write_text("window.QDATA = "+json.dumps(payload, ensure_ascii=False, separators=(",", ":"))+";", encoding="utf-8")
        print(f"saved {OUT} ({OUT.stat().st_size/1024/1024:.1f}MB)")
    if args.only in ("all", "sim"): collect_sim(etfs)


if __name__ == "__main__": main()
