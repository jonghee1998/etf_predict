"""Quant Manager Korea static data generator.

KOSPI market-cap leaders and major non-leveraged Korean-listed ETFs are
downloaded from KRX listings (FinanceDataReader) and Yahoo Finance prices.
"""
import argparse
import json
import re
from pathlib import Path

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
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
        row.update({k: None for k in NULL_STOCK}); row.update(metrics(close[code])); row["px"] = px_payload(close[code])
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
