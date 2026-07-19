# quant_manager 데이터 수집: S&P500 주식 + 주요 ETF → data.js
# 실행: .venv/bin/python3 quant_manager/update_data.py [--only all|stocks|etf]
#       (매월 1회 권장, 전체 10~18분 / etf만 4~7분)
import argparse
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "data.js"
TOP_N = 50
CHART_DAYS = 270          # 상세 차트용 일봉 (약 13개월)
PERIODS = {"m1": 21, "m3": 63, "m6": 126, "m12": 252}

# ═══════════════ ETF 유니버스 (큐레이션, 레버리지·인버스 제외) ═══════════════
ETFS = {}
def _cat(cat, syms): ETFS.update({s: cat for s in syms.split()})
_cat("대형 지수",     "SPY IVV VOO VTI QQQ QQQM DIA RSP VT ACWI")
_cat("중소형 지수",   "IWM IJH IJR MDY VB")
_cat("섹터",         "XLK XLF XLV XLE XLI XLY XLP XLU XLB XLRE XLC KRE ITB XME JETS ITA IYT PAVE IBB XBI IHI")
_cat("테크·반도체",   "SMH SOXX IGV VGT FDN SKYY CIBR HACK BOTZ AIQ")
_cat("팩터·스타일",   "MTUM QUAL USMV VLUE SPLV SPHQ VUG VTV IWF IWD VBK VBR IWO IWN")
_cat("배당·인컴",     "SCHD VYM DVY HDV NOBL SDY DGRO JEPI JEPQ QYLD")
_cat("국제·지역",     "EFA VEA VGK EZU EWG EWJ EWY EWT IEMG EEM VWO FXI MCHI KWEB EWZ INDA")
_cat("채권·금리",     "AGG BND TLT IEF SHY BIL SGOV LQD HYG JNK TIP EMB MUB")
_cat("원자재·귀금속", "GLD IAU SLV GDX GDXJ USO UNG DBC PDBC URA LIT COPX")
_cat("리츠",         "VNQ IYR SCHH")
_cat("암호화폐",      "IBIT FBTC BITO")
_cat("테마·혁신",     "ARKK ARKG ARKW ICLN TAN MAGS")

# ETF 한글 설명 (상세 화면 '상품 개요' 우측에 표시)
ETF_DESC = {
    "SPY": "미국 대형주 500개(S&P 500)를 통째로 사는 가장 유명한 ETF. 세계 최대 규모·최고 유동성.",
    "IVV": "SPY와 같은 S&P 500 추종. 블랙록 운용으로 보수가 더 저렴해 장기 보유에 유리.",
    "VOO": "SPY와 같은 S&P 500 추종. 뱅가드 운용, 저보수로 장기 투자자에게 인기.",
    "VTI": "S&P 500을 넘어 미국 상장 주식 거의 전부(약 3,500개)를 담는 '미국 전체' ETF.",
    "QQQ": "나스닥 상위 100개 기술·성장주. 애플·MS·엔비디아 비중이 커 S&P보다 공격적.",
    "QQQM": "QQQ와 동일 지수를 더 싼 보수로 추종하는 장기 보유용 버전.",
    "DIA": "다우존스 30개 전통 우량주. 오래된 대형 산업·소비 기업 중심.",
    "RSP": "S&P 500을 시가총액이 아닌 동일 비중으로 담아 대형 빅테크 쏠림을 줄인 버전.",
    "VT": "미국+선진국+신흥국, 전 세계 주식을 하나로 사는 '지구 전체' ETF.",
    "ACWI": "VT와 유사한 전 세계 주식 ETF (블랙록 운용).",
    "IWM": "미국 소형주 2,000개(러셀 2000). 경기 민감도가 높아 경기 회복기에 강함.",
    "IJH": "미국 중형주 400개(S&P MidCap). 대형주와 소형주의 중간 성격.",
    "IJR": "미국 소형주 600개(S&P SmallCap). IWM보다 우량 소형주 위주.",
    "MDY": "IJH와 같은 S&P 중형주 400 추종의 원조 격 ETF.",
    "VB": "뱅가드의 미국 소형주 ETF. 저보수 분산형.",
    "XLK": "S&P 500 중 기술 업종만: 애플·MS·엔비디아 등. 기술주 집중 투자.",
    "XLF": "S&P 500 중 금융 업종만: JP모건·버크셔·비자 등 은행·보험·결제.",
    "XLV": "S&P 500 중 헬스케어 업종만: 제약·바이오·의료기기·보험.",
    "XLE": "S&P 500 중 에너지 업종만: 엑슨모빌·셰브론 등 석유·가스. 유가에 민감.",
    "XLI": "S&P 500 중 산업재 업종만: 항공·방산·기계·운송.",
    "XLY": "S&P 500 중 경기소비재만: 아마존·테슬라·홈디포 등 '여유 있을 때 쓰는 소비'.",
    "XLP": "S&P 500 중 필수소비재만: P&G·코카콜라 등 '불황에도 사는 것들'. 방어적.",
    "XLU": "S&P 500 중 유틸리티(전력·가스)만. 배당 높고 방어적, 금리에 민감.",
    "XLB": "S&P 500 중 소재 업종만: 화학·광산·건자재.",
    "XLRE": "S&P 500 중 부동산(리츠) 업종만.",
    "XLC": "S&P 500 중 커뮤니케이션 업종만: 메타·알파벳·넷플릭스·통신사.",
    "KRE": "미국 지역은행 모음. 금리·부동산 경기에 매우 민감 (2023 SVB 사태의 그 지역은행들).",
    "ITB": "미국 주택건설사 모음. 주택 경기·모기지 금리에 민감.",
    "XME": "미국 금속·광산 기업 모음. 원자재 가격에 연동.",
    "JETS": "미국 항공사 모음. 여행 수요·유가에 민감.",
    "ITA": "미국 방위산업·항공우주 모음: 록히드마틴·RTX 등.",
    "IYT": "미국 운송(철도·트럭·물류) 모음. 경기 선행지표로도 쓰임.",
    "PAVE": "미국 인프라 건설 수혜주 모음: 자재·장비·엔지니어링.",
    "IBB": "나스닥 바이오테크 모음. 신약 개발 대형 바이오 중심.",
    "XBI": "미국 바이오테크 동일가중 모음. IBB보다 중소형 바이오 비중이 커 변동성 큼.",
    "IHI": "미국 의료기기 기업 모음.",
    "SMH": "글로벌 반도체 대표 25개: 엔비디아·TSMC·브로드컴. 반도체 사이클에 올인.",
    "SOXX": "SMH와 유사한 반도체 ETF (필라델피아 반도체지수 기반).",
    "IGV": "미국 소프트웨어 기업 모음: MS·오라클·세일즈포스 등.",
    "VGT": "뱅가드 기술 업종 ETF. XLK보다 종목 수가 많고 저보수.",
    "FDN": "인터넷 기업 모음: 아마존·메타·넷플릭스 등.",
    "SKYY": "클라우드 컴퓨팅 기업 모음.",
    "CIBR": "사이버보안 기업 모음 (HACK보다 규모 큼).",
    "HACK": "사이버보안 기업 모음의 원조 ETF.",
    "BOTZ": "로봇·AI 기업 모음 (글로벌).",
    "AIQ": "인공지능·빅데이터 관련 기업 모음.",
    "MTUM": "최근 6~12개월 상승세가 강한 미국 대형주만 골라 담는 모멘텀 팩터 ETF.",
    "QUAL": "재무가 탄탄한(고ROE·저부채) 미국 우량주만 담는 퀄리티 팩터 ETF.",
    "USMV": "출렁임이 적은 종목만 골라 담는 저변동성 ETF. 하락장 방어형.",
    "VLUE": "이익·자산 대비 싼 종목을 담는 밸류 팩터 ETF.",
    "SPLV": "S&P 500 중 변동성 낮은 100개. USMV와 유사한 방어형.",
    "SPHQ": "S&P 500 중 퀄리티 상위 종목. QUAL과 유사.",
    "VUG": "미국 대형 성장주(매출·이익 고성장) 모음. 기술주 비중 큼.",
    "VTV": "미국 대형 가치주(저PER·고배당) 모음. 금융·헬스케어 비중 큼.",
    "IWF": "러셀 1000 성장주. VUG와 유사.",
    "IWD": "러셀 1000 가치주. VTV와 유사.",
    "VBK": "미국 소형 성장주.",
    "VBR": "미국 소형 가치주.",
    "IWO": "러셀 2000 소형 성장주.",
    "IWN": "러셀 2000 소형 가치주.",
    "SCHD": "배당을 10년 이상 늘려온 미국 우량 배당주 100개. 한국 투자자에게 가장 인기 있는 배당 ETF.",
    "VYM": "배당수익률 높은 미국 대형주 광범위 모음.",
    "DVY": "고배당 100개, 유틸리티 비중 높음.",
    "HDV": "재무 건전성을 거른 고배당 75개.",
    "NOBL": "25년 연속 배당을 늘린 '배당 귀족'만 모음.",
    "SDY": "20년 이상 배당 성장 기업 모음.",
    "DGRO": "5년 이상 배당 성장 기업을 넓게 담은 배당 성장형.",
    "JEPI": "S&P 500 보유 + 콜옵션 판매로 월배당을 만드는 인컴형. 상승장 수익 일부를 배당과 맞바꿈.",
    "JEPQ": "JEPI의 나스닥 버전. 월배당 + 기술주 노출.",
    "QYLD": "나스닥 100 커버드콜. 월배당이 높은 대신 상승 여력을 대부분 포기.",
    "EFA": "미국 제외 선진국(유럽·일본 등) 대형주.",
    "VEA": "EFA와 유사한 선진국 ETF, 뱅가드 저보수.",
    "VGK": "유럽 주식 모음.",
    "EZU": "유로존(유로화 사용국) 주식 모음.",
    "EWG": "독일 대표 기업 모음.",
    "EWJ": "일본 대표 기업 모음.",
    "EWY": "한국 대표 기업 모음 (삼성전자·SK하이닉스 등).",
    "EWT": "대만 대표 기업 모음 (TSMC 비중 매우 큼).",
    "IEMG": "신흥국(중국·인도·대만·브라질 등) 광범위 모음.",
    "EEM": "IEMG의 원조 격 신흥국 ETF.",
    "VWO": "뱅가드 신흥국 ETF. 저보수.",
    "FXI": "중국 대형주 50개 (홍콩 상장).",
    "MCHI": "중국 주식 광범위 모음.",
    "KWEB": "중국 인터넷 기업 모음: 알리바바·텐센트·핀둬둬 등.",
    "EWZ": "브라질 대표 기업 모음.",
    "INDA": "인도 대표 기업 모음.",
    "AGG": "미국 투자등급 채권 전체(국채+회사채+MBS). 채권의 기본형.",
    "BND": "AGG와 유사한 종합 채권, 뱅가드.",
    "TLT": "미국 장기 국채(20년+). 금리 하락에 크게 오르고 상승에 크게 하락. 주식 헤지용.",
    "IEF": "미국 중기 국채(7~10년). TLT보다 금리 민감도 절반.",
    "SHY": "미국 단기 국채(1~3년). 금리 영향 작은 안전 자산.",
    "BIL": "초단기 국채(1~3개월). 사실상 현금 대용, 금리만큼 이자.",
    "SGOV": "BIL과 유사한 초단기 국채. 현금 파킹용으로 인기.",
    "LQD": "미국 투자등급 회사채.",
    "HYG": "미국 하이일드(투기등급) 채권. 이자 높지만 불황에 주식처럼 하락.",
    "JNK": "HYG와 유사한 하이일드 채권.",
    "TIP": "물가연동 국채. 인플레이션이 오르면 원금이 불어남.",
    "EMB": "신흥국 정부 달러채권.",
    "MUB": "미국 지방채 (미국인에게 면세 혜택).",
    "GLD": "금 현물 가격 추종. 인플레·위기 헤지의 대표.",
    "IAU": "GLD와 같은 금 현물, 보수가 더 저렴.",
    "SLV": "은 현물 가격 추종. 금보다 변동성 큼.",
    "GDX": "금광 기업 모음. 금값의 레버리지 성격.",
    "GDXJ": "중소형 금광 기업. GDX보다 더 공격적.",
    "USO": "원유(WTI) 선물 추종. 장기 보유 시 선물 롤오버 비용 주의.",
    "UNG": "천연가스 선물 추종. 변동성 매우 큼, 장기 보유 부적합.",
    "DBC": "원유·금속·농산물 등 원자재 종합 바스켓.",
    "PDBC": "DBC와 유사한 원자재 바스켓 (세금 서류 간소화 버전).",
    "URA": "우라늄 채굴·원자력 관련 기업 모음.",
    "LIT": "리튬 채굴·배터리 기업 모음. 전기차 밸류체인.",
    "COPX": "구리 채굴 기업 모음. 경기·전기화 수요에 연동.",
    "VNQ": "미국 리츠(부동산 투자회사) 광범위 모음. 배당 높고 금리에 민감.",
    "IYR": "VNQ와 유사한 미국 부동산 ETF.",
    "SCHH": "슈왑의 미국 리츠 ETF, 저보수.",
    "IBIT": "비트코인 현물 ETF (블랙록). 비트코인 가격을 직접 추종.",
    "FBTC": "비트코인 현물 ETF (피델리티).",
    "BITO": "비트코인 선물 ETF. 현물보다 추적 오차·롤오버 비용 있음.",
    "ARKK": "캐시 우드의 파괴적 혁신 기업 액티브 ETF. 고위험 고변동.",
    "ARKG": "ARK의 유전체·바이오 혁신 버전.",
    "ARKW": "ARK의 인터넷·핀테크 혁신 버전.",
    "ICLN": "글로벌 클린에너지(태양광·풍력) 기업 모음.",
    "TAN": "태양광 기업 모음. 정책·금리에 민감, 변동성 큼.",
    "MAGS": "매그니피센트 7 (애플·MS·구글·아마존·엔비디아·메타·테슬라)만 담은 ETF.",
}


# ═══════════════ 공통 지표 ═══════════════
def rsi14(s, period=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = g / l.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def compute_metrics(s):
    """일별 종가 시리즈 → 수익률·기술지표 (주식/ETF 공통)"""
    m = {}
    for k, d in PERIODS.items():
        m[k] = float(s.iloc[-1] / s.iloc[-1 - d] - 1) if len(s) > d else None
    m["mom121"] = float(s.iloc[-22] / s.iloc[-253] - 1) if len(s) > 253 else None
    r = s.pct_change().dropna()
    m["vol60"] = float(r.iloc[-60:].std() * np.sqrt(252)) if len(r) >= 60 else None
    r12 = r.iloc[-252:]
    m["sharpe12"] = float(r12.mean() / r12.std() * np.sqrt(252)) if len(r12) >= 120 and r12.std() > 0 else None
    eq = (1 + r12).cumprod()
    m["mdd12"] = float((eq / eq.cummax() - 1).min()) if len(eq) else None
    m["rsi14"] = round(rsi14(s), 1)
    hi52 = float(s.iloc[-252:].max()) if len(s) >= 20 else float(s.max())
    lo52 = float(s.iloc[-252:].min()) if len(s) >= 20 else float(s.min())
    m["from_high"] = float(s.iloc[-1] / hi52 - 1)
    m["hi52"], m["lo52"] = hi52, lo52
    m["last"] = float(s.iloc[-1])
    ma200 = s.rolling(200).mean()
    m["above_ma200"] = bool(s.iloc[-1] > ma200.iloc[-1]) if not np.isnan(ma200.iloc[-1]) else None
    return m


def px_payload(s):
    s = s.dropna().iloc[-CHART_DAYS:]
    return {"d": [int(d.strftime("%Y%m%d")) for d in s.index],
            "c": [round(float(v), 2) for v in s.values]}


def top_ranks(metrics):
    top = {}
    for k in PERIODS:
        ranked = sorted((t for t in metrics if metrics[t][k] is not None),
                        key=lambda t: metrics[t][k], reverse=True)
        top[k] = ranked[:TOP_N]
    return top


# ═══════════════ 재무제표 팩터 (주식 전용) ═══════════════
def row_at(df, name, i):
    try:
        v = df.loc[name].iloc[i]
        return None if pd.isna(v) else float(v)
    except Exception:
        return None


def piotroski(inc, bal, cf):
    """Piotroski F-Score (2000): 9개 항목 중 계산 가능한 항목의 통과 수. (score, 계산가능수)"""
    checks = []
    ni0, ni1 = row_at(inc, "Net Income", 0), row_at(inc, "Net Income", 1)
    ta0, ta1 = row_at(bal, "Total Assets", 0), row_at(bal, "Total Assets", 1)
    cfo0 = row_at(cf, "Operating Cash Flow", 0)
    rev0, rev1 = row_at(inc, "Total Revenue", 0), row_at(inc, "Total Revenue", 1)
    gp0, gp1 = row_at(inc, "Gross Profit", 0), row_at(inc, "Gross Profit", 1)
    ltd0, ltd1 = row_at(bal, "Long Term Debt", 0), row_at(bal, "Long Term Debt", 1)
    ca0, ca1 = row_at(bal, "Current Assets", 0), row_at(bal, "Current Assets", 1)
    cl0, cl1 = row_at(bal, "Current Liabilities", 0), row_at(bal, "Current Liabilities", 1)
    sh0, sh1 = row_at(bal, "Ordinary Shares Number", 0), row_at(bal, "Ordinary Shares Number", 1)
    if ni0 is not None and ta0:
        checks.append(ni0 / ta0 > 0)
    if cfo0 is not None:
        checks.append(cfo0 > 0)
    if None not in (ni0, ni1) and ta0 and ta1:
        checks.append(ni0 / ta0 > ni1 / ta1)
    if None not in (cfo0, ni0):
        checks.append(cfo0 > ni0)
    if ta0 and ta1:
        checks.append((ltd0 or 0) / ta0 <= (ltd1 or 0) / ta1)
    if None not in (ca0, ca1) and cl0 and cl1:
        checks.append(ca0 / cl0 > ca1 / cl1)
    if None not in (sh0, sh1):
        checks.append(sh0 <= sh1 * 1.02)
    if None not in (gp0, gp1) and rev0 and rev1:
        checks.append(gp0 / rev0 > gp1 / rev1)
    if None not in (rev0, rev1) and ta0 and ta1:
        checks.append(rev0 / ta0 > rev1 / ta1)
    return (int(sum(checks)), len(checks)) if checks else (None, 0)


def altman_z(inc, bal, mcap):
    """Altman Z-Score (1968). 금융주에는 부적합 — 표시단에서 제외 처리."""
    ta = row_at(bal, "Total Assets", 0)
    tl = row_at(bal, "Total Liabilities Net Minority Interest", 0)
    ca, cl = row_at(bal, "Current Assets", 0), row_at(bal, "Current Liabilities", 0)
    re_ = row_at(bal, "Retained Earnings", 0)
    ebit = row_at(inc, "EBIT", 0)
    rev = row_at(inc, "Total Revenue", 0)
    if None in (ta, tl, ca, cl, re_, ebit, rev) or not ta or not tl or not mcap:
        return None
    z = 1.2 * (ca - cl) / ta + 1.4 * re_ / ta + 3.3 * ebit / ta + 0.6 * mcap / tl + 1.0 * rev / ta
    return round(float(z), 2)


INFO_KEYS = dict(mcap="marketCap", pe="trailingPE", fpe="forwardPE", pbr="priceToBook",
                 psr="priceToSalesTrailing12Months", evebitda="enterpriseToEbitda",
                 peg="trailingPegRatio", fcf="freeCashflow",
                 roe="returnOnEquity", roa="returnOnAssets", gm="grossMargins",
                 opm="operatingMargins", npm="profitMargins", curr="currentRatio",
                 d2e="debtToEquity", revg="revenueGrowth", earng="earningsQuarterlyGrowth",
                 divy="dividendYield", beta="beta",
                 sfloat="shortPercentOfFloat", inst="heldPercentInstitutions",
                 insider="heldPercentInsiders",
                 tgt="targetMeanPrice", rec="recommendationMean", nanal="numberOfAnalystOpinions",
                 avgvol="averageVolume")
FIN_ROWS = dict(revenue="Total Revenue", op="Operating Income", net="Net Income")


def collect_stocks():
    print("① S&P 500 목록...")
    listing = fdr.StockListing("S&P500")
    # 야후 심볼 변환: FDR은 점 없이(BRKB), 야후는 대시(BRK-B)를 쓴다
    FIX = {"BRKB": "BRK-B", "BFB": "BF-B", "BRK.B": "BRK-B", "BF.B": "BF-B"}
    listing["Yahoo"] = listing["Symbol"].str.replace(".", "-", regex=False).replace(FIX)
    meta = listing.set_index("Yahoo")[["Symbol", "Name", "Sector", "Industry"]].to_dict("index")
    symbols = listing["Yahoo"].tolist()
    print(f"   {len(symbols)}개 티커")

    print("② 전체 가격 일괄 다운로드 (15개월)...")
    raw = yf.download(symbols, period="15mo", auto_adjust=True, progress=False, threads=True)
    close = raw["Close"].dropna(axis=1, how="all").dropna(axis=0, how="all")
    print(f"   {close.shape[1]}개 티커 × {close.shape[0]}일, 최신 {close.index.max().date()}")

    print("③ 수익률·기술지표·랭킹 계산...")
    metrics = {}
    for t in close.columns:
        s = close[t].dropna()
        if len(s) >= 40:
            metrics[t] = compute_metrics(s)
    top = top_ranks(metrics)
    selected = sorted(set().union(*top.values()))
    print(f"   기준별 상위 {TOP_N} 합집합: {len(selected)}개 종목 → 재무·팩터 수집")

    print("④ 재무·기업정보·퀀트팩터 수집 (yfinance, 종목당 ~3초)...")
    data, fails = {}, []
    for i, t in enumerate(selected, 1):
        md = meta.get(t, {})
        row = dict(sym=md.get("Symbol", t), name=md.get("Name", t),
                   sector=md.get("Sector", "?"), industry=md.get("Industry", "?"))
        row.update({k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics[t].items()})
        try:
            tk = yf.Ticker(t)
            info = tk.info or {}
            for k, ik in INFO_KEYS.items():
                v = info.get(ik)
                row[k] = round(float(v), 6) if isinstance(v, (int, float)) else None
            row["summary"] = (info.get("longBusinessSummary") or "")[:600]

            inc, bal, cf = tk.income_stmt, tk.balance_sheet, tk.cashflow
            if inc is not None and len(inc.columns):
                cols = list(inc.columns)[:4][::-1]
                fdat = {"years": [str(c.year) for c in cols]}
                for k, rname in FIN_ROWS.items():
                    if rname in inc.index:
                        vals = [inc.loc[rname, c] for c in cols]
                        fdat[k] = [None if pd.isna(v) else round(float(v) / 1e9, 3) for v in vals]
                    else:
                        fdat[k] = [None] * len(cols)
                row["fin"] = fdat
            else:
                row["fin"] = None
            if inc is not None and bal is not None and cf is not None and len(inc.columns) >= 2:
                fs, fn = piotroski(inc, bal, cf)
                row["fscore"], row["fmax"] = fs, fn
                row["zscore"] = altman_z(inc, bal, row.get("mcap"))
            else:
                row["fscore"], row["fmax"], row["zscore"] = None, 0, None
            row["fcfy"] = round(row["fcf"] / row["mcap"], 4) if row.get("fcf") and row.get("mcap") else None
            try:
                cal = tk.calendar
                ed = cal.get("Earnings Date") if isinstance(cal, dict) else None
                row["earnings"] = str(ed[0]) if ed else None
            except Exception:
                row["earnings"] = None
        except Exception as e:
            fails.append((t, str(e)[:60]))
            row.setdefault("summary", ""); row.setdefault("fin", None)
            row.setdefault("fscore", None); row.setdefault("fmax", 0); row.setdefault("zscore", None)
            row.setdefault("fcfy", None); row.setdefault("earnings", None)
            for k in INFO_KEYS: row.setdefault(k, None)
        row["px"] = px_payload(close[t])
        data[t] = row
        if i % 20 == 0:
            print(f"   {i}/{len(selected)}...")
        time.sleep(0.15)
    if fails:
        print(f"   ⚠️ {len(fails)}개 종목 정보 일부 실패: {[f[0] for f in fails]}")

    sector_counts = {}
    for k in PERIODS:
        cnt = {}
        for t in top[k]:
            sec = data[t]["sector"]
            cnt[sec] = cnt.get(sec, 0) + 1
        sector_counts[k] = cnt
    return dict(asof=str(close.index.max().date()), universe=len(close.columns),
                top=top, sectors=sector_counts, stocks=data)


def collect_etfs():
    print("⑥ ETF 가격 일괄 다운로드 (15개월)...")
    syms = list(ETFS)
    raw = yf.download(syms, period="15mo", auto_adjust=True, progress=False, threads=True)
    close = raw["Close"].dropna(axis=1, how="all").dropna(axis=0, how="all")
    print(f"   {close.shape[1]}/{len(syms)}개 ETF × {close.shape[0]}일, 최신 {close.index.max().date()}")

    print("⑦ ETF 지표·정보 수집 (yfinance, 개당 ~1.5초)...")
    metrics = {t: compute_metrics(close[t].dropna()) for t in close.columns
               if len(close[t].dropna()) >= 40}
    etop = top_ranks(metrics)
    data, fails = {}, []
    for i, t in enumerate(sorted(metrics), 1):
        row = dict(sym=t, cat=ETFS.get(t, "?"))
        row.update({k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics[t].items()})
        try:
            info = yf.Ticker(t).info or {}
            row["name"] = info.get("longName") or info.get("shortName") or t
            row["aum"] = info.get("totalAssets")
            exp = info.get("netExpenseRatio")
            if exp is None:
                exp = info.get("annualReportExpenseRatio")
            row["expense"] = round(float(exp), 4) if isinstance(exp, (int, float)) else None
            dv = info.get("dividendYield")
            if dv is None and isinstance(info.get("yield"), (int, float)):
                dv = info["yield"] * 100
            row["divy"] = round(float(dv), 3) if isinstance(dv, (int, float)) else None
            row["mgr"] = info.get("fundFamily")
            row["summary"] = (info.get("longBusinessSummary") or "")[:500]
            row["desc_kr"] = ETF_DESC.get(t, "")
        except Exception as e:
            fails.append((t, str(e)[:50]))
            row.setdefault("name", t)
            for k in ["aum", "expense", "divy", "mgr"]: row.setdefault(k, None)
            row.setdefault("summary", "")
            row.setdefault("desc_kr", ETF_DESC.get(t, ""))
        row["px"] = px_payload(close[t])
        data[t] = row
        if i % 25 == 0:
            print(f"   {i}/{len(metrics)}...")
        time.sleep(0.12)
    if fails:
        print(f"   ⚠️ {len(fails)}개 ETF 정보 일부 실패: {[f[0] for f in fails]}")

    cat_counts = {}
    for k in PERIODS:
        cnt = {}
        for t in etop[k]:
            c = data[t]["cat"]
            cnt[c] = cnt.get(c, 0) + 1
        cat_counts[k] = cnt
    return dict(etf_asof=str(close.index.max().date()), etf_universe=len(close.columns),
                etop=etop, ecats=cat_counts, etfs=data)


def collect_sim():
    """ETF_Simulation 탭용: 전 ETF의 상장 이후 전체 일봉 종가 → data_sim.js (별도 파일)"""
    print("⑨ 시뮬레이션용 장기 가격 다운로드 (상장 이후 전체)...")
    syms = list(ETFS)
    raw = yf.download(syms, period="max", auto_adjust=True, progress=False, threads=True)
    close = raw["Close"].dropna(axis=1, how="all")
    close = close.loc[close.index >= "1998-01-01"].dropna(axis=0, how="all")
    out = {}
    for t in close.columns:
        s = close[t].dropna()
        out[t] = {"d": [int(d.strftime("%Y%m%d")) for d in s.index],
                  "c": [round(float(v), 3) for v in s.values]}
    p = Path(__file__).parent / "data_sim.js"
    payload = dict(asof=str(close.index.max().date()),
                   generated=str(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")),
                   cats=ETFS, px=out)
    p.write_text("window.QSIM = " + json.dumps(payload, ensure_ascii=False) + ";", encoding="utf-8")
    print(f"   저장: {p} ({p.stat().st_size/1024/1024:.1f}MB, {len(out)}개 ETF, 최신 {payload['asof']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["all", "stocks", "etf", "sim"], default="all")
    ONLY = ap.parse_args().only

    if ONLY != "sim":
        payload = {}
        if OUT.exists() and ONLY != "all":
            payload = json.loads(re.sub(r"^window\.QDATA = |;$", "", OUT.read_text(encoding="utf-8")))
        if ONLY in ("all", "stocks"):
            payload.update(collect_stocks())
        if ONLY in ("all", "etf"):
            payload.update(collect_etfs())
        payload["generated"] = str(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
        print("⑧ data.js 저장...")
        OUT.write_text("window.QDATA = " + json.dumps(payload, ensure_ascii=False) + ";",
                       encoding="utf-8")
        print(f"   저장: {OUT} ({OUT.stat().st_size/1024:.0f}KB, 주식 {payload.get('asof')} / ETF {payload.get('etf_asof')})")

    if ONLY in ("all", "sim"):
        collect_sim()
