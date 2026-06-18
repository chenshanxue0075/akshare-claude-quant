# -*- coding: utf-8 -*-
"""
实时量化看板 v3.0
- 东方财富 AKShare 真实数据，24小时接入
- Claude AI 深度个股分析，支持多轮对话
- 全市场主板实时扫描推荐
- 回测：双均线/RSI/布林带/MACD/KDJ，支持自定义时间段
"""
import os, json, math, time, random, asyncio
import datetime as dt
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

USE_REAL        = os.getenv("USE_REAL", "true").lower() == "true"
PORT            = int(os.getenv("PORT", 8088))
HOST            = "0.0.0.0"
MAX_CHANGE_PCT  = 5.0
TOP_N           = 8
SPOT_CACHE_SEC  = 10
MAX_HIST_SCAN   = 40
MAIN_BOARD_PREFIX = ("600", "601", "603", "605", "000", "001")

def is_main_board(code: str) -> bool:
    return str(code).split(".")[0].startswith(MAIN_BOARD_PREFIX)

def is_trading_time() -> bool:
    now = dt.datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dt.time(9,30) <= t <= dt.time(11,30) or dt.time(13,0) <= t <= dt.time(15,0)


# ========================= 数据适配器 =========================
class EastMoneyAdapter:
    def __init__(self, use_real: bool = True):
        self.use_real    = use_real
        self._ak         = None
        self._spot_df    = None
        self._spot_ts    = 0
        self._name_map   = {}
        self._mock_state = {}
        if use_real:
            try:
                import akshare as ak
                self._ak = ak
                print("akshare loaded — 东方财富真实数据")
            except Exception as e:
                print(f"akshare failed, fallback mock: {e}")
                self.use_real = False
        if not self.use_real:
            self._build_mock_universe()

    def _get_spot(self) -> pd.DataFrame:
        now = time.time()
        ttl = SPOT_CACHE_SEC if is_trading_time() else 60
        if self._spot_df is not None and now - self._spot_ts < ttl:
            return self._spot_df
        df = self._ak.stock_zh_a_spot_em()
        rename = {"代码":"code","名称":"name","最新价":"price","涨跌幅":"change_pct",
                  "昨收":"pre_close","量比":"volume_ratio","换手率":"turnover_rate",
                  "成交额":"amount","总市值":"market_cap","市盈率-动态":"pe"}
        df = df.rename(columns=rename)
        for c in ["price","change_pct","pre_close","volume_ratio","turnover_rate","amount","market_cap","pe"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["code","price"])
        for _, r in df.iterrows():
            self._name_map[str(r["code"])] = str(r["name"])
        self._spot_df = df
        self._spot_ts = now
        return df

    @staticmethod
    def _row_to_quote(r) -> Dict:
        ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        def g(k, d=0.0):
            v = r.get(k, d)
            try:
                if v is None or (isinstance(v, float) and math.isnan(v)): return d
                return float(v)
            except: return d
        return {"code":str(r["code"]),"name":str(r.get("name",r["code"])),
                "price":round(g("price"),2),"pre_close":round(g("pre_close"),2),
                "change_pct":round(g("change_pct"),2),"volume_ratio":round(g("volume_ratio"),2),
                "turnover_rate":round(g("turnover_rate"),2),"amount":g("amount"),
                "market_cap":g("market_cap"),"pe":g("pe"),"ts":ts}

    def get_realtime_quotes(self, codes: List[str] = None) -> List[Dict]:
        if not self.use_real: return self._mock_realtime(codes)
        df = self._get_spot()
        if codes is not None:
            want = {str(c).split(".")[0] for c in codes}
            df = df[df["code"].isin(want)]
        return [self._row_to_quote(r) for _, r in df.iterrows()]

    def get_history(self, code: str, days: int = 500) -> pd.DataFrame:
        if not self.use_real: return self._mock_history(code, days)
        bare  = str(code).split(".")[0]
        end   = dt.date.today()
        start = end - dt.timedelta(days=int(days * 1.8) + 60)
        df = self._ak.stock_zh_a_hist(symbol=bare, period="daily",
                                      start_date=start.strftime("%Y%m%d"),
                                      end_date=end.strftime("%Y%m%d"), adjust="qfq")
        rename = {"日期":"date","开盘":"open","收盘":"close","最高":"high",
                  "最低":"low","成交量":"volume","成交额":"amount"}
        df = df.rename(columns=rename)
        keep = ["date","open","high","low","close","volume","amount"]
        for c in keep:
            if c not in df.columns: df[c] = np.nan
        df = df[keep].dropna(subset=["close"]).reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df.tail(days).reset_index(drop=True)

    def get_universe(self) -> List[str]:
        if not self.use_real: return [c for c in self._mock_universe if is_main_board(c)]
        df = self._get_spot()
        return [c for c in df["code"].tolist() if is_main_board(c)]

    def get_name(self, code: str) -> str:
        return self._name_map.get(str(code).split(".")[0], str(code))

    def get_index_quote(self) -> Dict:
        if not self.use_real: return self._mock_index()
        df  = self._get_spot()
        chg = df["change_pct"]
        adv, dec = int((chg>0).sum()), int((chg<0).sum())
        lu,  ld  = int((chg>=9.8).sum()), int((chg<=-9.8).sum())
        amount_yi = round(float(df["amount"].sum())/1e8, 0)
        idx_chg = self._try_index_change()
        if idx_chg is None: idx_chg = round(float(chg.median()), 2)
        return {"name":"上证指数","change_pct":round(idx_chg,2),
                "advance":adv,"decline":dec,"limit_up":lu,"limit_down":ld,
                "amount":amount_yi,"trading":is_trading_time()}

    def _try_index_change(self):
        try:
            d = self._ak.stock_zh_index_spot_em(symbol="上证系列指数")
            row = d[d["代码"]=="000001"]
            if not row.empty: return float(row.iloc[0]["涨跌幅"])
        except: pass
        try:
            d = self._ak.stock_zh_index_spot_em()
            row = d[d["代码"]=="000001"]
            if not row.empty: return float(row.iloc[0]["涨跌幅"])
        except: pass
        return None

    def _build_mock_universe(self):
        names = {"600519":"贵州茅台","600036":"招商银行","601318":"中国平安",
                 "000333":"美的集团","000651":"格力电器","000858":"五粮液",
                 "600276":"恒瑞医药","601012":"隆基绿能","600887":"伊利股份",
                 "603288":"海天味业","600030":"中信证券","601899":"紫金矿业"}
        for c,n in names.items():
            self._mock_state[c] = {"name":n,"pre_close":round(random.uniform(8,1700),2)}
            self._name_map[c]   = n
        self._mock_universe = list(names.keys())

    def _mock_realtime(self, codes):
        codes = codes or self._mock_universe
        ts    = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        out   = []
        for c in codes:
            st  = self._mock_state.setdefault(c,{"name":c,"pre_close":round(random.uniform(8,100),2)})
            pre = st["pre_close"]
            chg = float(np.clip(random.gauss(0.3,2.5),-10,10))
            out.append({"code":c,"name":st["name"],"price":round(pre*(1+chg/100),2),
                        "pre_close":pre,"change_pct":round(chg,2),
                        "volume_ratio":round(abs(random.gauss(1.2,0.8))+0.3,2),
                        "turnover_rate":round(abs(random.gauss(2.0,1.5))+0.2,2),
                        "amount":round(abs(random.gauss(8e8,5e8)),0),
                        "market_cap":round(abs(random.gauss(1e11,5e10)),0),
                        "pe":round(abs(random.gauss(20,10)),1),"ts":ts})
        return out

    def _mock_history(self, code, days):
        st    = self._mock_state.get(str(code).split(".")[0]) or {"pre_close":50.0}
        dates = pd.bdate_range(end=dt.date.today(), periods=days)
        price = st["pre_close"]; rows=[]
        for d in dates:
            o=price; c_=max(0.5,o*(1+random.gauss(0.0005,0.02)))
            h=max(o,c_)*(1+abs(random.gauss(0,0.008)))
            l=min(o,c_)*(1-abs(random.gauss(0,0.008)))
            v=abs(random.gauss(1e7,4e6))
            rows.append([d.date(),round(o,2),round(h,2),round(l,2),round(c_,2),int(v),v*c_])
            price=c_
        return pd.DataFrame(rows,columns=["date","open","high","low","close","volume","amount"])

    def _mock_index(self):
        return {"name":"上证指数","change_pct":round(random.uniform(-1.5,1.5),2),
                "advance":random.randint(1500,3500),"decline":random.randint(1500,3500),
                "limit_up":random.randint(20,90),"limit_down":random.randint(0,30),
                "amount":round(random.uniform(7000,12000),0),"trading":False}


# ========================= 技术指标 =========================
def _ma(s, n):  return s.rolling(n).mean()
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()

def _macd(close, fast=12, slow=26, signal=9):
    dif = _ema(close,fast) - _ema(close,slow)
    dea = _ema(dif,signal)
    return dif, dea, (dif-dea)*2

def _rsi(close, n=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100/(1+rs)

def _boll(close, n=20, k=2):
    mid = _ma(close, n)
    std = close.rolling(n).std()
    return mid+k*std, mid, mid-k*std

def _kdj(high, low, close, n=9):
    ll  = low.rolling(n).min()
    hh  = high.rolling(n).max()
    rsv = (close-ll)/(hh-ll+1e-9)*100
    K   = rsv.ewm(com=2, adjust=False).mean()
    D   = K.ewm(com=2, adjust=False).mean()
    J   = 3*K - 2*D
    return K, D, J

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l = df["close"], df["high"], df["low"]
    df["ma5"],  df["ma10"]  = _ma(c,5),  _ma(c,10)
    df["ma20"], df["ma60"]  = _ma(c,20), _ma(c,60)
    df["ma120"] = _ma(c,120)
    df["dif"], df["dea"], df["macd_hist"] = _macd(c)
    df["rsi"]  = _rsi(c)
    df["boll_up"], df["boll_mid"], df["boll_dn"] = _boll(c)
    df["K"], df["D"], df["J"] = _kdj(h,l,c)
    df["vol_ma5"]  = _ma(df["volume"],5)
    df["vol_ma20"] = _ma(df["volume"],20)
    prev = df["close"].shift(1)
    tr   = pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    return df

def bullish_alignment(row) -> bool:
    try: return row["ma5"]>row["ma10"]>row["ma20"]
    except: return False


# ========================= 数据分析师 =========================
class DataAnalyst:
    def __init__(self, adapter): self.adapter = adapter

    def analyze(self, code: str) -> dict:
        df = enrich(self.adapter.get_history(code, 300))
        if len(df) < 30:
            return {"code":code,"name":self.adapter.get_name(code),"tech_score":50,
                    "trend":"数据不足","bullish_alignment":False,"rsi":0,"macd_hist":0,
                    "ma5":0,"ma10":0,"ma20":0,"ma60":0,"boll_up":0,"boll_dn":0,
                    "K":0,"D":0,"J":0,"atr":0,"signals":[],"detail":{},
                    "kline":{"dates":[],"close":[],"ma5":[],"ma20":[],"ma60":[],"boll_up":[],"boll_dn":[]},
                    "summary":"历史数据不足，无法分析。"}
        last    = df.iloc[-1]
        signals = self._signals(df)
        score   = self._tech_score(df, signals)
        trend   = self._trend(df)
        return {"code":str(code).split(".")[0],"name":self.adapter.get_name(code),
                "tech_score":score,"trend":trend,
                "bullish_alignment":bool(bullish_alignment(last)),
                "rsi":round(float(last["rsi"]),1),
                "macd_hist":round(float(last["macd_hist"]),3),
                "ma5":round(float(last["ma5"]),2),"ma10":round(float(last["ma10"]),2),
                "ma20":round(float(last["ma20"]),2),"ma60":round(float(last["ma60"]),2),
                "boll_up":round(float(last["boll_up"]),2),
                "boll_dn":round(float(last["boll_dn"]),2),
                "K":round(float(last["K"]),1),"D":round(float(last["D"]),1),
                "J":round(float(last["J"]),1),"atr":round(float(last["atr"]),3),
                "signals":signals,"detail":self._detail(df),
                "kline":self._kline(df),"summary":self._summary(score,trend,signals)}

    def _signals(self, df):
        sig=[]; last=df.iloc[-1]; prev=df.iloc[-2]
        if prev["dif"]<=prev["dea"] and last["dif"]>last["dea"]:
            sig.append({"type":"buy","name":"MACD金叉","strength":"中"})
        if prev["dif"]>=prev["dea"] and last["dif"]<last["dea"]:
            sig.append({"type":"sell","name":"MACD死叉","strength":"中"})
        if prev["close"]<=prev["ma20"] and last["close"]>last["ma20"]:
            sig.append({"type":"buy","name":"上穿MA20","strength":"强"})
        if prev["close"]>=prev["ma20"] and last["close"]<last["ma20"]:
            sig.append({"type":"sell","name":"跌破MA20","strength":"强"})
        if prev["close"]<=prev["ma60"] and last["close"]>last["ma60"]:
            sig.append({"type":"buy","name":"上穿MA60","strength":"强"})
        if last["rsi"]<30: sig.append({"type":"buy","name":"RSI超卖(<30)","strength":"弱"})
        if last["rsi"]>75: sig.append({"type":"sell","name":"RSI超买(>75)","strength":"弱"})
        if last["J"]<10:   sig.append({"type":"buy","name":"KDJ超卖(J<10)","strength":"中"})
        if last["J"]>90:   sig.append({"type":"sell","name":"KDJ超买(J>90)","strength":"中"})
        if last["close"]<last["boll_dn"]:
            sig.append({"type":"buy","name":"跌破布林下轨","strength":"中"})
        if last["close"]>last["boll_up"]:
            sig.append({"type":"sell","name":"突破布林上轨","strength":"中"})
        if last["volume"]>last["vol_ma5"]*1.8 and last["close"]>prev["close"]:
            sig.append({"type":"buy","name":"放量上涨","strength":"中"})
        if last["volume"]>last["vol_ma5"]*1.8 and last["close"]<prev["close"]:
            sig.append({"type":"sell","name":"放量下跌","strength":"中"})
        return sig

    def _tech_score(self, df, signals):
        last=df.iloc[-1]; s=50
        if bullish_alignment(last):      s+=15
        if last["close"]>last["ma60"]:   s+=8
        if last["close"]>last["ma120"]:  s+=5
        if last["macd_hist"]>0:          s+=7
        if 40<=last["rsi"]<=65:          s+=8
        elif last["rsi"]>80 or last["rsi"]<20: s-=8
        if last["K"]>last["D"]:          s+=5
        s += sum(6 if x["type"]=="buy" else -6 for x in signals)
        return int(max(0,min(100,s)))

    def _trend(self, df):
        last=df.iloc[-1]
        if last["ma5"]>last["ma20"] and last["close"]>last["ma20"]: return "多头"
        if last["ma5"]<last["ma20"] and last["close"]<last["ma20"]: return "空头"
        return "震荡"

    def _detail(self, df):
        last=df.iloc[-1]; recent=df.tail(20)
        ret20 = (last["close"]/df.iloc[-21]["close"]-1)*100 if len(df)>21 else 0
        vol_ratio = last["volume"]/last["vol_ma5"] if last["vol_ma5"]>0 else 1
        return {"ret_20d":round(ret20,2),"vol_ratio_today":round(vol_ratio,2),
                "high_20d":round(float(recent["high"].max()),2),
                "low_20d":round(float(recent["low"].min()),2),
                "dist_from_ma20":round((last["close"]-last["ma20"])/last["ma20"]*100,2) if last["ma20"]>0 else 0}

    def _summary(self, score, trend, signals):
        buy=[s["name"] for s in signals if s["type"]=="buy"]
        sell=[s["name"] for s in signals if s["type"]=="sell"]
        parts=[f"技术评分{score}分，{trend}格局。"]
        if buy:  parts.append("买入信号：" + "、".join(buy) + "。")
        if sell: parts.append("卖出信号：" + "、".join(sell) + "。")
        if not buy and not sell: parts.append("暂无明确信号，建议观望。")
        return "".join(parts)

    def _kline(self, df, n=120):
        d=df.tail(n)
        return {"dates":[str(x) for x in d["date"].tolist()],
                "close":d["close"].round(2).tolist(),
                "open":d["open"].round(2).tolist(),
                "high":d["high"].round(2).tolist(),
                "low":d["low"].round(2).tolist(),
                "volume":d["volume"].fillna(0).astype(int).tolist(),
                "ma5":d["ma5"].round(2).fillna(0).tolist(),
                "ma20":d["ma20"].round(2).fillna(0).tolist(),
                "ma60":d["ma60"].round(2).fillna(0).tolist(),
                "boll_up":d["boll_up"].round(2).fillna(0).tolist(),
                "boll_dn":d["boll_dn"].round(2).fillna(0).tolist()}


# ========================= 情绪分析师 =========================
class SentimentAnalyst:
    def __init__(self, adapter): self.adapter = adapter

    def market_sentiment(self) -> dict:
        idx  = self.adapter.get_index_quote()
        adv, dec  = idx["advance"], idx["decline"]
        breadth   = adv / max(1, adv+dec)
        limit_net = idx["limit_up"] - idx["limit_down"]
        score = 50 + idx["change_pct"]*6 + (breadth-0.5)*60 + min(20, limit_net*0.4)
        score = int(max(0,min(100,score)))
        mood  = ("亢奋" if score>=75 else "偏暖" if score>=60 else
                 "中性" if score>=45 else "偏冷" if score>=30 else "恐慌")
        status = "交易中" if idx.get("trading") else "已收盘"
        return {"score":score,"mood":mood,"index":idx,"breadth":round(breadth*100,1),
                "status":status,
                "summary":(f"[{status}] 上证{idx['change_pct']:+.2f}%，"
                           f"涨{adv}/跌{dec}，涨停{idx['limit_up']}/跌停{idx['limit_down']}，"
                           f"成交{idx['amount']}亿。情绪{mood}({score}分)")}

    def stock_sentiment(self, code: str) -> dict:
        qs = self.adapter.get_realtime_quotes([code])
        if not qs: return {"code":code,"name":self.adapter.get_name(code),"score":50,
                           "level":"未知","change_pct":0,"volume_ratio":0,"price":0,
                           "turnover_rate":0,"summary":"未获取到实时数据"}
        q   = qs[0]; mkt = self.market_sentiment()
        score = (50 + q["change_pct"]*3 + min(20,(q["volume_ratio"]-1)*15)
                 + (q["change_pct"]-mkt["index"]["change_pct"])*2)
        score = int(max(0,min(100,score)))
        hot   = ("过热" if score>=80 else "活跃" if score>=60 else "平稳" if score>=40 else "低迷")
        return {"code":q["code"],"name":q["name"],"score":score,"level":hot,
                "change_pct":q["change_pct"],"volume_ratio":q["volume_ratio"],
                "price":q["price"],"turnover_rate":q["turnover_rate"],
                "summary":(f"{q['name']} {q['change_pct']:+.2f}%，量比{q['volume_ratio']}，"
                           f"换手{q['turnover_rate']}%，情绪{hot}。")}


# ========================= 推荐系统 =========================
class Recommender:
    def __init__(self, adapter, analyst, sentiment):
        self.adapter, self.analyst, self.sentiment = adapter, analyst, sentiment

    def daily_picks(self, min_volume_ratio=1.2, min_turnover=1.5, require_bullish=True):
        universe = set(self.adapter.get_universe())
        quotes   = self.adapter.get_realtime_quotes()
        mkt      = self.sentiment.market_sentiment()
        pre = [q for q in quotes if
               q["code"] in universe and
               q["change_pct"] < MAX_CHANGE_PCT and q["change_pct"] > -3 and
               q["volume_ratio"] >= min_volume_ratio and
               q["turnover_rate"] >= min_turnover and q["price"] > 0]
        pre.sort(key=lambda x:(x["volume_ratio"],x["change_pct"]), reverse=True)
        pre = pre[:MAX_HIST_SCAN]
        candidates=[]
        for q in pre:
            try:
                df = enrich(self.adapter.get_history(q["code"],120))
                if len(df)<30: continue
                last = df.iloc[-1]
                if require_bullish and not bullish_alignment(last): continue
                tech = self.analyst.analyze(q["code"])
                sent = self.sentiment.stock_sentiment(q["code"])
                score= int(tech["tech_score"]*0.6 + sent["score"]*0.4)
                candidates.append({
                    "code":q["code"],"name":q["name"],"price":q["price"],
                    "change_pct":q["change_pct"],"volume_ratio":q["volume_ratio"],
                    "turnover_rate":q["turnover_rate"],"amount":q["amount"],
                    "score":score,"tech_score":tech["tech_score"],"sentiment_score":sent["score"],
                    "trend":tech["trend"],"rsi":tech["rsi"],"macd_hist":tech["macd_hist"],
                    "signals":[s["name"] for s in tech["signals"] if s["type"]=="buy"],
                    "reason":self._reason(tech,sent,q,mkt),
                    "timing":self._timing(q,last)})
            except: continue
        candidates.sort(key=lambda x:x["score"], reverse=True)
        return {"market":mkt,"picks":candidates[:TOP_N],
                "scan_count":len(pre),"pass_count":len(candidates),
                "disclaimer":"本列表为算法量化输出，不构成投资建议。股市有风险，入市需谨慎。"}

    def _reason(self, tech, sent, q, mkt):
        bits=[]
        if tech["bullish_alignment"]:   bits.append("均线多头排列")
        if tech["macd_hist"]>0:         bits.append("MACD红柱")
        if q["volume_ratio"]>1.5:       bits.append(f"量比{q['volume_ratio']}x放大")
        if sent["score"]>=60:           bits.append("个股情绪活跃")
        if tech["rsi"]<50:              bits.append(f"RSI{tech['rsi']}未超买")
        bits.append(f"大盘{mkt['mood']}")
        return "、".join(bits) if bits else "技术面均衡"

    def _timing(self, q, last):
        if q["change_pct"]>3: return "今日涨幅偏高，不追高，等回踩MA5附近分批介入"
        if q["price"]<float(last["ma5"]): return f"现价MA5下方，逢低关注，站上MA5({round(float(last['ma5']),2)})确认后介入"
        return f"现价附近可分批建仓，跌破MA10({round(float(last['ma10']),2)})止损"


# ========================= 多策略回测 =========================
class Backtester:
    def __init__(self, adapter): self.adapter = adapter

    STRATEGIES = {
        "ma":   "双均线策略",
        "rsi":  "RSI超卖策略",
        "boll": "布林带策略",
        "macd": "MACD策略",
        "kdj":  "KDJ策略",
    }

    def run(self, code, strategy="ma", fast=5, slow=20,
            init_cash=100000, fee=0.0013, days=500,
            start_date=None, end_date=None):
        try:
            df = enrich(self.adapter.get_history(code, max(days,1300))).dropna().reset_index(drop=True)
        except Exception as e:
            return {"error": f"取数失败: {e}"}
        if start_date:
            sd = pd.to_datetime(start_date).date()
            df = df[df["date"]>=sd].reset_index(drop=True)
        if end_date:
            ed = pd.to_datetime(end_date).date()
            df = df[df["date"]<=ed].reset_index(drop=True)
        if len(df) < 30:
            return {"error": "所选时间段数据不足（少于30个交易日）"}

        if strategy=="ma":    df = self._sig_ma(df,fast,slow)
        elif strategy=="rsi": df = self._sig_rsi(df)
        elif strategy=="boll":df = self._sig_boll(df)
        elif strategy=="macd":df = self._sig_macd(df)
        elif strategy=="kdj": df = self._sig_kdj(df)
        else:                 df = self._sig_ma(df,fast,slow)

        df["pos"]  = df["sig"].shift(1).fillna(0)
        df["ret"]  = df["close"].pct_change().fillna(0)
        df["trade"]= df["pos"].diff().abs().fillna(0)
        df["strat_ret"] = df["pos"]*df["ret"] - df["trade"]*fee
        df["equity"]    = (1+df["strat_ret"]).cumprod()*init_cash
        df["bh"]        = (1+df["ret"]).cumprod()*init_cash

        eq        = df["equity"]
        total_ret = eq.iloc[-1]/init_cash - 1
        years     = max(len(df)/244, 0.1)
        cagr      = (eq.iloc[-1]/init_cash)**(1/years) - 1
        peak      = eq.cummax(); mdd = ((eq-peak)/peak).min()
        sharpe    = (df["strat_ret"].mean()/(df["strat_ret"].std()+1e-9))*np.sqrt(244)
        trades    = int(df["trade"].sum())
        wins      = (df.loc[df["trade"]>0,"strat_ret"]>0).sum()
        bh_ret    = df["bh"].iloc[-1]/init_cash - 1

        return {"code":str(code).split(".")[0],"name":self.adapter.get_name(code),
                "strategy":strategy,"strategy_name":self.STRATEGIES.get(strategy,"自定义"),
                "period":f"{df['date'].iloc[0]} ~ {df['date'].iloc[-1]}",
                "trading_days":len(df),
                "total_return":round(total_ret*100,2),
                "bh_return":round(bh_ret*100,2),
                "cagr":round(cagr*100,2),
                "max_drawdown":round(float(mdd)*100,2),
                "sharpe":round(float(sharpe),2),
                "trades":trades,
                "win_rate":round(float(wins)/max(1,trades)*100,1),
                "curve":{"dates":[str(x) for x in df["date"].tolist()],
                         "equity":eq.round(0).tolist(),
                         "benchmark":df["bh"].round(0).tolist()}}

    def _sig_ma(self, df, fast, slow):
        df["sig"] = (df["close"].rolling(fast).mean()>df["close"].rolling(slow).mean()).astype(int)
        return df

    def _sig_rsi(self, df, oversold=30, overbought=70):
        sig=pd.Series(0,index=df.index); pos=0
        for i in range(1,len(df)):
            r=df["rsi"].iloc[i]
            if r<oversold: pos=1
            elif r>overbought: pos=0
            sig.iloc[i]=pos
        df["sig"]=sig; return df

    def _sig_boll(self, df):
        sig=pd.Series(0,index=df.index); pos=0
        for i in range(1,len(df)):
            c=df["close"].iloc[i]; bu=df["boll_up"].iloc[i]; bd=df["boll_dn"].iloc[i]
            if c<=bd: pos=1
            elif c>=bu: pos=0
            sig.iloc[i]=pos
        df["sig"]=sig; return df

    def _sig_macd(self, df):
        sig=pd.Series(0,index=df.index); pos=0
        for i in range(1,len(df)):
            if df["dif"].iloc[i]>df["dea"].iloc[i]: pos=1
            elif df["dif"].iloc[i]<df["dea"].iloc[i]: pos=0
            sig.iloc[i]=pos
        df["sig"]=sig; return df

    def _sig_kdj(self, df, buy=20, sell=80):
        sig=pd.Series(0,index=df.index); pos=0
        for i in range(1,len(df)):
            j=df["J"].iloc[i]
            if j<buy: pos=1
            elif j>sell: pos=0
            sig.iloc[i]=pos
        df["sig"]=sig; return df


# ========================= Claude AI 交易员 =========================
SYSTEM_PROMPT = """你是一名专业的A股量化交易分析师，拥有10年主板投资经验。

你的核心能力：
1. 深度解读技术指标（MACD、RSI、KDJ、布林带、均线系统）
2. 结合量价关系分析主力资金动向
3. 给出具体的买入区间、止损位、目标价
4. 评估当前大盘环境对个股的影响

回答规范：
- 必须结合系统提供的实时数据，引用具体数值
- 每次针对个股的回答需包含：①趋势判断 ②关键支撑/压力位 ③操作建议 ④风险提示
- 买入建议必须明确：建仓区间、止损位（具体价格或跌破哪条均线止损）、目标位
- 字数200-400字，结构清晰，分要点
- 结尾必须注明：以上分析仅供参考，不构成投资建议

禁止事项：
- 不推荐当日涨幅超过5%的标的
- 不忽略风险提示"""


class AIAssistant:
    def __init__(self, adapter, analyst, sentiment):
        self.adapter, self.analyst, self.sentiment = adapter, analyst, sentiment
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._history: List[Dict] = []

    def _build_context(self, code=None) -> str:
        ctx = {"market":self.sentiment.market_sentiment(),
               "time":dt.datetime.now().strftime("%Y-%m-%d %H:%M")}
        if code:
            try:
                tech = self.analyst.analyze(code)
                sent = self.sentiment.stock_sentiment(code)
                qs   = self.adapter.get_realtime_quotes([code])
                ctx["stock"] = {
                    "name":tech["name"],"code":code,
                    "price":qs[0]["price"] if qs else 0,
                    "change_pct":qs[0]["change_pct"] if qs else 0,
                    "tech_score":tech["tech_score"],"trend":tech["trend"],
                    "rsi":tech["rsi"],"macd_hist":tech["macd_hist"],
                    "ma5":tech["ma5"],"ma20":tech["ma20"],"ma60":tech["ma60"],
                    "boll_up":tech["boll_up"],"boll_dn":tech["boll_dn"],
                    "K":tech["K"],"D":tech["D"],"J":tech["J"],
                    "volume_ratio":qs[0]["volume_ratio"] if qs else 0,
                    "turnover_rate":qs[0]["turnover_rate"] if qs else 0,
                    "signals":[s["name"] for s in tech["signals"]],
                    "detail":tech["detail"],"sentiment":sent["level"],
                }
            except Exception as e:
                ctx["stock_error"] = str(e)
        return json.dumps(ctx, ensure_ascii=False, default=str)

    async def chat(self, message: str, code: Optional[str]=None, reset: bool=False) -> dict:
        if reset:
            self._history = []
            return {"reply":"对话已重置，开始新的分析吧！","mode":"reset","history_len":0}

        context = self._build_context(code)
        system  = SYSTEM_PROMPT + f"\n\n【实时市场数据】\n{context}"

        if not self.anthropic_key:
            reply = self._rule_reply(message, code, context)
            return {"reply":reply,"mode":"rule","history_len":0}

        self._history.append({"role":"user","content":message})
        messages = self._history[-20:]

        payload = {
            "model":     "claude-sonnet-4-6",
            "max_tokens": 1200,
            "system":    system,
            "messages":  messages
        }
        headers = {
            "x-api-key":         self.anthropic_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json"
        }
        try:
            async with httpx.AsyncClient(timeout=45) as client:
                r = await client.post("https://api.anthropic.com/v1/messages",
                                      json=payload, headers=headers)
                r.raise_for_status()
                reply = r.json()["content"][0]["text"]
                self._history.append({"role":"assistant","content":reply})
                return {"reply":reply,"mode":"claude","history_len":len(self._history)//2}
        except httpx.HTTPStatusError as e:
            err = f"API错误({e.response.status_code})"
            fallback = self._rule_reply(message, code, context)
            return {"reply":f"❌ {err}，切换本地分析：\n\n{fallback}","mode":"error"}
        except Exception as e:
            fallback = self._rule_reply(message, code, context)
            return {"reply":f"❌ 网络超时，本地分析：\n\n{fallback}","mode":"error"}

    def _rule_reply(self, message, code, context_str):
        try:
            ctx = json.loads(context_str)
            mkt = ctx.get("market",{})
            stk = ctx.get("stock",{})
        except: mkt={}; stk={}
        mood  = mkt.get("mood","未知")
        score = mkt.get("score",50)
        lines = [f"【本地规则模式】大盘情绪：{mood}（{score}分）"]
        if stk:
            lines += [
                f"\n{stk.get('name',code)}（{code}）实时数据：",
                f"价格：{stk.get('price',0)} （{stk.get('change_pct',0):+.2f}%）",
                f"技术评分：{stk.get('tech_score',0)}分，趋势：{stk.get('trend','未知')}",
                f"RSI={stk.get('rsi',0)}，MACD柱={stk.get('macd_hist',0)}",
                f"MA5={stk.get('ma5',0)}，MA20={stk.get('ma20',0)}，MA60={stk.get('ma60',0)}",
                f"布林上轨={stk.get('boll_up',0)}，布林下轨={stk.get('boll_dn',0)}",
                f"KDJ：K={stk.get('K',0)},D={stk.get('D',0)},J={stk.get('J',0)}",
                f"信号：{', '.join(stk.get('signals',[])) or '暂无'}",
            ]
        lines.append("\n📌 配置 ANTHROPIC_API_KEY 环境变量可启用Claude完整分析。")
        return "\n".join(lines)


# ========================= 组装 =========================
adapter     = EastMoneyAdapter(use_real=USE_REAL)
analyst     = DataAnalyst(adapter)
sentiment   = SentimentAnalyst(adapter)
recommender = Recommender(adapter, analyst, sentiment)
backtester  = Backtester(adapter)
ai          = AIAssistant(adapter, analyst, sentiment)

app = FastAPI(title="实时量化看板 v3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ========================= API =========================
@app.get("/api/recommendations")
def recommendations(min_volume_ratio:float=1.2, min_turnover:float=1.5, require_bullish:bool=True):
    try:    return recommender.daily_picks(min_volume_ratio, min_turnover, require_bullish)
    except Exception as e: return {"market":{},"picks":[],"error":str(e)}

@app.get("/api/analyze/{code}")
def analyze(code:str):
    try:    return analyst.analyze(code)
    except Exception as e: return {"error":str(e)}

@app.get("/api/sentiment/market")
def market_sentiment(): return sentiment.market_sentiment()

@app.get("/api/backtest/{code}")
def backtest(code:str, strategy:str="ma", fast:int=5, slow:int=20,
             days:int=500, start_date:str=None, end_date:str=None):
    return backtester.run(code, strategy, fast, slow, days=days,
                          start_date=start_date, end_date=end_date)

class ChatReq(BaseModel):
    message: str
    code: Optional[str] = None
    reset: bool = False

@app.post("/api/ai/chat")
async def ai_chat(req:ChatReq):
    return await ai.chat(req.message, req.code, req.reset)

@app.websocket("/ws")
async def ws(websocket:WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                mkt    = sentiment.market_sentiment()
                quotes = adapter.get_realtime_quotes()[:25]
            except Exception as e:
                mkt    = {"summary":f"数据获取中...","score":50}
                quotes = []
            await websocket.send_text(json.dumps(
                {"type":"tick","quotes":quotes,"market":mkt},
                ensure_ascii=False, default=str))
            await asyncio.sleep(10)
    except WebSocketDisconnect: pass

@app.get("/", response_class=HTMLResponse)
def index(): return HTML_PAGE


# ========================= 前端 =========================
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>实时量化看板 v3</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#08111f;--card:#0f1e35;--card2:#162540;--line:#1e3050;
      --green:#00d48a;--red:#ff4d6d;--gold:#f0b429;--blue:#3b82f6;--purple:#8b5cf6;
      --txt:#dce8f5;--sub:#6b8aaa;}
*{box-sizing:border-box;margin:0;padding:0;font-family:-apple-system,"PingFang SC",sans-serif;}
body{background:var(--bg);color:var(--txt);max-width:480px;margin:0 auto;padding-bottom:72px;}
.topbar{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;
        font-size:12px;color:var(--sub);border-bottom:1px solid var(--line);}
.logo{font-size:14px;font-weight:700;color:var(--blue);}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;margin:10px 12px;padding:14px;}
.card-title{font-size:14px;font-weight:700;margin-bottom:12px;}
.row{display:flex;justify-content:space-between;align-items:flex-start;
     padding:10px 0;border-bottom:1px solid var(--line);}
.row:last-child{border-bottom:none;}
.stk-name{font-size:15px;font-weight:700;}
.stk-code{font-size:11px;color:var(--sub);margin-left:4px;}
.up{color:var(--red);} .dn{color:var(--green);}
.score-badge{font-size:11px;background:rgba(59,130,246,.15);color:var(--blue);
             border-radius:6px;padding:2px 7px;font-weight:600;}
.reason{font-size:11px;color:var(--sub);margin-top:4px;line-height:1.5;}
.timing{font-size:12px;color:#93c5fd;background:rgba(59,130,246,.08);
        border-radius:8px;padding:8px 10px;margin-top:6px;line-height:1.6;}
.sig-buy{font-size:11px;color:var(--red);background:rgba(255,77,109,.1);
         border-radius:5px;padding:2px 6px;margin:2px;display:inline-block;}
.sig-sell{font-size:11px;color:var(--green);background:rgba(0,212,138,.1);
          border-radius:5px;padding:2px 6px;margin:2px;display:inline-block;}
.field{margin:8px 0;}
.field label{font-size:12px;color:var(--sub);display:block;margin-bottom:4px;}
input,select{background:var(--card2);border:1px solid var(--line);
             color:var(--txt);border-radius:10px;padding:10px 12px;font-size:14px;}
input:focus,select:focus{outline:1px solid var(--blue);}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
.btn{width:100%;background:linear-gradient(135deg,#1e3a8a,#2563eb);color:#fff;border:none;
     border-radius:12px;padding:13px;font-size:15px;font-weight:700;margin-top:10px;cursor:pointer;}
.btn:active{opacity:.8;}
.btn-sm{padding:9px 12px;font-size:13px;margin-top:0;border-radius:9px;width:auto;}
.mood-bar{height:5px;border-radius:3px;background:var(--card2);overflow:hidden;margin-top:8px;}
.mood-fill{height:100%;background:linear-gradient(90deg,var(--green),var(--gold),var(--red));transition:width .6s;}
.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px;}
.stat-box{background:var(--card2);border-radius:10px;padding:10px;text-align:center;}
.stat-val{font-size:18px;font-weight:700;}
.stat-lbl{font-size:11px;color:var(--sub);margin-top:2px;}
.tabs{position:fixed;bottom:0;left:50%;transform:translateX(-50%);
      max-width:480px;width:100%;display:flex;background:#08111f;
      border-top:1px solid var(--line);z-index:100;}
.tab{flex:1;text-align:center;padding:10px 0 8px;font-size:11px;color:var(--sub);cursor:pointer;
     display:flex;flex-direction:column;align-items:center;gap:2px;}
.tab .ico{font-size:18px;} .tab.active{color:var(--blue);}
.page{display:none;} .page.active{display:block;}
.chat-box{height:340px;overflow-y:auto;background:var(--card2);border-radius:12px;
          padding:12px;font-size:13px;line-height:1.65;}
.msg-u{text-align:right;margin:8px 0;}
.msg-u span{background:var(--blue);color:#fff;border-radius:12px 12px 0 12px;
             padding:8px 12px;display:inline-block;max-width:85%;text-align:left;}
.msg-a{text-align:left;margin:8px 0;}
.msg-a span{background:var(--card);border:1px solid var(--line);color:var(--txt);
             border-radius:12px 12px 12px 0;padding:8px 12px;display:inline-block;
             max-width:94%;white-space:pre-wrap;word-break:break-word;text-align:left;}
.chat-quick{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;}
.qbtn{font-size:12px;color:var(--blue);background:rgba(59,130,246,.1);
      border:1px solid rgba(59,130,246,.25);border-radius:8px;padding:5px 10px;cursor:pointer;}
.check{display:flex;gap:8px;align-items:center;font-size:13px;}
.conn-dot{width:7px;height:7px;border-radius:50%;background:var(--sub);display:inline-block;margin-right:4px;}
.conn-dot.live{background:var(--green);}
.loading{font-size:13px;color:var(--sub);text-align:center;padding:20px;}
.disc{font-size:10px;color:var(--sub);text-align:center;padding:8px 12px;line-height:1.5;}
</style>
</head>
<body>

<div class="topbar">
  <span class="logo">📈 量化看板 v3</span>
  <span>
    <span class="conn-dot" id="dot"></span>
    <span id="conn">连接中</span>
    &nbsp;|&nbsp;<span id="clock">--:--</span>
  </span>
</div>

<!-- 推荐页 -->
<div class="page active" id="page-rec">
  <div class="card">
    <div class="card-title">🌐 大盘实时情绪</div>
    <div id="mkt-summary" style="font-size:13px;color:var(--sub);line-height:1.6;">加载中...</div>
    <div class="mood-bar"><div class="mood-fill" id="mkt-bar" style="width:50%"></div></div>
    <div class="stat-grid">
      <div class="stat-box"><div class="stat-val up" id="mkt-adv">--</div><div class="stat-lbl">上涨</div></div>
      <div class="stat-box"><div class="stat-val dn" id="mkt-dec">--</div><div class="stat-lbl">下跌</div></div>
      <div class="stat-box"><div class="stat-val up" id="mkt-lu">--</div><div class="stat-lbl">涨停</div></div>
      <div class="stat-box"><div class="stat-val dn" id="mkt-ld">--</div><div class="stat-lbl">跌停</div></div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">🔍 选股参数</div>
    <div class="grid2">
      <div class="field"><label>最小量比</label><input id="f-vr" value="1.2" style="width:100%"></div>
      <div class="field"><label>最小换手率(%)</label><input id="f-to" value="1.5" style="width:100%"></div>
    </div>
    <div class="check" style="margin:8px 0;">
      <input type="checkbox" id="f-bull" checked>
      <label>要求均线多头排列(MA5>MA10>MA20)</label>
    </div>
    <button class="btn" onclick="loadRecs()">⚡ 全市场扫描 · 生成推荐</button>
    <div class="disc">扫描主板全市场，剔除涨幅≥5%，保留量比换手达标标的</div>
  </div>
  <div class="card" id="rec-card" style="display:none">
    <div class="card-title">🚀 今日推荐买入</div>
    <div id="scan-info" style="font-size:11px;color:var(--sub);margin-bottom:8px;"></div>
    <div id="rec-list"></div>
  </div>
  <div class="disc" id="disc-main"></div>
</div>

<!-- 个股分析页 -->
<div class="page" id="page-analyze">
  <div class="card">
    <div class="card-title">📊 个股深度分析</div>
    <div style="display:flex;gap:8px;">
      <input style="flex:1;" id="an-code" placeholder="股票代码，如 600519">
      <button class="btn btn-sm" onclick="analyze()">分析</button>
    </div>
    <div id="an-result" style="margin-top:12px;"></div>
    <div id="an-indicators" style="margin-top:8px;"></div>
    <canvas id="an-chart" height="180" style="margin-top:12px;display:none;"></canvas>
  </div>
</div>

<!-- 回测页 -->
<div class="page" id="page-bt">
  <div class="card">
    <div class="card-title">🧪 多策略回测</div>
    <div class="field"><label>股票代码</label><input id="bt-code" value="600519" style="width:100%"></div>
    <div class="field">
      <label>策略选择</label>
      <select id="bt-strategy" style="width:100%">
        <option value="ma">双均线策略（趋势跟踪）</option>
        <option value="macd">MACD策略（金叉/死叉）</option>
        <option value="rsi">RSI超卖策略（均值回归）</option>
        <option value="boll">布林带策略（通道突破）</option>
        <option value="kdj">KDJ策略（超买超卖）</option>
      </select>
    </div>
    <div class="grid2" id="ma-params">
      <div class="field"><label>快线周期</label><input id="bt-fast" value="5" style="width:100%"></div>
      <div class="field"><label>慢线周期</label><input id="bt-slow" value="20" style="width:100%"></div>
    </div>
    <div class="field">
      <label>回测时间段</label>
      <div class="grid2">
        <input id="bt-start" type="date" style="width:100%">
        <input id="bt-end" type="date" style="width:100%">
      </div>
      <div class="chat-quick" style="margin-top:6px;">
        <span class="qbtn" onclick="setPeriod(1)">近1年</span>
        <span class="qbtn" onclick="setPeriod(3)">近3年</span>
        <span class="qbtn" onclick="setPeriod(5)">近5年</span>
        <span class="qbtn" onclick="setPeriod(0)">全部</span>
      </div>
    </div>
    <button class="btn" onclick="runBacktest()">🚀 运行回测</button>
    <div id="bt-stats" style="margin-top:12px;"></div>
    <canvas id="bt-chart" height="200" style="margin-top:12px;display:none;"></canvas>
  </div>
</div>

<!-- AI页 -->
<div class="page" id="page-ai">
  <div class="card">
    <div class="card-title">🤖 Claude AI 量化交易员</div>
    <div style="margin-bottom:8px;">
      <input id="ai-code" placeholder="关联股票代码（选填，如 600519）"
             style="width:100%;font-size:13px;">
    </div>
    <div class="chat-box" id="chat">
      <div class="msg-a"><span>你好！我是Claude AI量化交易员。

你可以问我：
• 输入股票代码后问买卖时机、止损位
• 大盘走势分析
• 某只股票的技术面解读
• 如何看RSI/MACD/KDJ等指标

我会结合实时数据给你详细分析。</span></div>
    </div>
    <div class="chat-quick">
      <span class="qbtn" onclick="quickAsk('分析当前大盘走势和操作建议')">大盘分析</span>
      <span class="qbtn" onclick="quickAsk('这只股票现在适合买入吗？给我具体的买入价位和止损位')">买卖点</span>
      <span class="qbtn" onclick="quickAsk('分析这只股票的技术面，告诉我各指标含义')">技术面</span>
      <span class="qbtn" onclick="quickAsk('现在大盘情绪如何？适合建仓吗？')">建仓时机</span>
    </div>
    <div style="display:flex;gap:8px;margin-top:8px;">
      <input style="flex:1;font-size:14px;" id="chat-in" placeholder="输入问题，回车发送...">
      <button class="btn btn-sm" onclick="sendChat()">发送</button>
    </div>
    <div style="display:flex;justify-content:flex-end;margin-top:6px;">
      <span class="qbtn" onclick="resetChat()" style="color:var(--sub);border-color:rgba(107,138,170,.3);">🔄 新对话</span>
    </div>
    <div class="disc">AI分析仅供参考，不构成投资建议。由 Anthropic Claude 驱动。</div>
  </div>
</div>

<div class="tabs">
  <div class="tab active" data-p="rec" onclick="switchTab('rec')">
    <span class="ico">⚡</span>推荐
  </div>
  <div class="tab" data-p="analyze" onclick="switchTab('analyze')">
    <span class="ico">📊</span>个股
  </div>
  <div class="tab" data-p="bt" onclick="switchTab('bt')">
    <span class="ico">🧪</span>回测
  </div>
  <div class="tab" data-p="ai" onclick="switchTab('ai')">
    <span class="ico">🤖</span>AI
  </div>
</div>

<script>
let anChart=null, btChart=null;
const $=id=>document.getElementById(id);
const cls=p=>p>=0?'up':'dn';
const fmt=p=>(p>=0?'+':'')+Number(p).toFixed(2)+'%';

setInterval(()=>$('clock').textContent=
  new Date().toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'}),1000);

function connectWS(){
  const proto=location.protocol==='https:'?'wss://':'ws://';
  const ws=new WebSocket(proto+location.host+'/ws');
  ws.onopen=()=>{$('dot').classList.add('live');$('conn').textContent='实时';};
  ws.onclose=()=>{$('dot').classList.remove('live');$('conn').textContent='重连';
    setTimeout(connectWS,4000);};
  ws.onmessage=e=>{
    const d=JSON.parse(e.data);
    if(d.type==='tick'&&d.market) updateMarket(d.market);
  };
}

function updateMarket(m){
  if(!m) return;
  $('mkt-summary').textContent=m.summary||'';
  $('mkt-bar').style.width=(m.score||50)+'%';
  const idx=m.index||{};
  $('mkt-adv').textContent=idx.advance||'--';
  $('mkt-dec').textContent=idx.decline||'--';
  $('mkt-lu').textContent=idx.limit_up||'--';
  $('mkt-ld').textContent=idx.limit_down||'--';
}

// ===== 推荐 =====
async function loadRecs(){
  const vr=$('f-vr').value, to=$('f-to').value, bull=$('f-bull').checked;
  $('rec-card').style.display='block';
  $('rec-list').innerHTML='<div class="loading">🔍 全市场扫描中，首次约需15-30秒...</div>';
  try{
    const r=await fetch(`/api/recommendations?min_volume_ratio=${vr}&min_turnover=${to}&require_bullish=${bull}`);
    const d=await r.json();
    if(d.market) updateMarket(d.market);
    if(d.error){$('rec-list').innerHTML=`<div class="loading">❌ ${d.error}</div>`;return;}
    $('scan-info').textContent=`扫描${d.scan_count||'?'}只 → 通过${d.pass_count||'?'}只 → 精选${(d.picks||[]).length}只`;
    if(!d.picks||!d.picks.length){
      $('rec-list').innerHTML='<div class="loading">当前条件无符合标的，可放宽参数</div>';return;}
    $('rec-list').innerHTML=d.picks.map(p=>`
      <div class="row" style="flex-direction:column;align-items:stretch;gap:4px;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <div>
            <span class="stk-name">${p.name}</span>
            <span class="stk-code">${p.code}</span>
            <span class="score-badge" style="margin-left:6px;">评分${p.score}</span>
          </div>
          <div style="text-align:right;">
            <div style="font-size:17px;font-weight:700;" class="${cls(p.change_pct)}">${p.price}</div>
            <div style="font-size:12px;" class="${cls(p.change_pct)}">${fmt(p.change_pct)}</div>
          </div>
        </div>
        <div class="reason">量比${p.volume_ratio}x · 换手${p.turnover_rate}% · ${p.trend}格局 · ${p.reason}</div>
        ${p.signals&&p.signals.length?`<div>${p.signals.map(s=>`<span class="sig-buy">${s}</span>`).join('')}</div>`:''}
        <div class="timing">${p.timing}</div>
      </div>`).join('');
    $('disc-main').textContent=d.disclaimer||'';
  }catch(e){$('rec-list').innerHTML=`<div class="loading">❌ ${e}</div>`;}
}

// ===== 个股分析 =====
async function analyze(){
  const code=$('an-code').value.trim(); if(!code)return;
  $('an-result').innerHTML='<div class="loading">⏳ 分析中...</div>';
  $('an-indicators').innerHTML=''; $('an-chart').style.display='none';
  try{
    const r=await fetch(`/api/analyze/${code}`);
    const d=await r.json();
    if(d.error){$('an-result').innerHTML=`<div class="loading">❌ ${d.error}</div>`;return;}
    const buys=(d.signals||[]).filter(s=>s.type==='buy');
    const sells=(d.signals||[]).filter(s=>s.type==='sell');
    $('an-result').innerHTML=`
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
        <div>
          <span class="stk-name">${d.name}</span>
          <span class="stk-code">${d.code}</span>
        </div>
        <span class="score-badge" style="font-size:13px;">评分${d.tech_score} · ${d.trend}</span>
      </div>
      <div style="background:var(--card2);border-radius:10px;padding:10px;font-size:13px;line-height:1.8;margin-bottom:8px;">
        ${d.summary}
      </div>
      <div>
        ${buys.map(s=>`<span class="sig-buy">${s.name}(${s.strength})</span>`).join('')}
        ${sells.map(s=>`<span class="sig-sell">${s.name}(${s.strength})</span>`).join('')}
      </div>`;
    const det=d.detail||{};
    $('an-indicators').innerHTML=`
      <div class="stat-grid">
        <div class="stat-box"><div class="stat-val">${d.rsi}</div><div class="stat-lbl">RSI(14)</div></div>
        <div class="stat-box"><div class="stat-val">${d.macd_hist>=0?'▲':'▼'}${Math.abs(d.macd_hist)}</div><div class="stat-lbl">MACD柱</div></div>
        <div class="stat-box"><div class="stat-val">${d.K}/${d.D}</div><div class="stat-lbl">KDJ K/D</div></div>
        <div class="stat-box"><div class="stat-val ${cls(det.ret_20d||0)}">${det.ret_20d||0}%</div><div class="stat-lbl">近20日涨跌</div></div>
      </div>
      <div style="font-size:12px;color:var(--sub);margin-top:8px;line-height:1.8;">
        MA5=${d.ma5} &nbsp;MA20=${d.ma20} &nbsp;MA60=${d.ma60}<br>
        布林上=${d.boll_up} &nbsp;布林下=${d.boll_dn}<br>
        20日高=${det.high_20d} &nbsp;低=${det.low_20d} &nbsp;偏离MA20: ${det.dist_from_ma20}%
      </div>`;
    const k=d.kline;
    if(k&&k.dates&&k.dates.length){
      $('an-chart').style.display='block';
      if(anChart) anChart.destroy();
      anChart=new Chart($('an-chart'),{type:'line',
        data:{labels:k.dates,datasets:[
          {label:'收盘',data:k.close,borderColor:'#dce8f5',pointRadius:0,borderWidth:2,tension:.2},
          {label:'MA5', data:k.ma5, borderColor:'#f0b429',pointRadius:0,borderWidth:1,tension:.2},
          {label:'MA20',data:k.ma20,borderColor:'#3b82f6',pointRadius:0,borderWidth:1,tension:.2},
          {label:'MA60',data:k.ma60,borderColor:'#8b5cf6',pointRadius:0,borderWidth:1,tension:.2},
          {label:'布林上',data:k.boll_up,borderColor:'rgba(255,77,109,.4)',pointRadius:0,borderWidth:1,borderDash:[3,3]},
          {label:'布林下',data:k.boll_dn,borderColor:'rgba(0,212,138,.4)',pointRadius:0,borderWidth:1,borderDash:[3,3]},
        ]},
        options:{animation:{duration:300},responsive:true,
          plugins:{legend:{labels:{color:'#6b8aaa',font:{size:10},boxWidth:12}}},
          scales:{x:{ticks:{color:'#6b8aaa',maxTicksLimit:6},grid:{color:'#1e3050'}},
                  y:{ticks:{color:'#6b8aaa'},grid:{color:'#1e3050'}}}}});
    }
  }catch(e){$('an-result').innerHTML=`<div class="loading">❌ ${e}</div>`;}
}

// ===== 回测 =====
function setPeriod(years){
  const end=new Date(); $('bt-end').value=end.toISOString().slice(0,10);
  if(years===0){$('bt-start').value='';return;}
  const s=new Date(end); s.setFullYear(s.getFullYear()-years);
  $('bt-start').value=s.toISOString().slice(0,10);
}
setPeriod(1);

$('bt-strategy').addEventListener('change',function(){
  $('ma-params').style.display=this.value==='ma'?'grid':'none';
});

async function runBacktest(){
  const code=$('bt-code').value.trim(), strat=$('bt-strategy').value,
        fast=$('bt-fast').value, slow=$('bt-slow').value,
        start=$('bt-start').value, end=$('bt-end').value;
  $('bt-stats').innerHTML='<div class="loading">⏳ 回测计算中...</div>';
  $('bt-chart').style.display='none';
  let url=`/api/backtest/${code}?strategy=${strat}&fast=${fast}&slow=${slow}`;
  if(start) url+=`&start_date=${start}`;
  if(end)   url+=`&end_date=${end}`;
  try{
    const r=await fetch(url); const d=await r.json();
    if(d.error){$('bt-stats').innerHTML=`<div class="loading">❌ ${d.error}</div>`;return;}
    const beat=d.total_return-d.bh_return;
    $('bt-stats').innerHTML=`
      <div style="font-size:12px;color:var(--sub);margin-bottom:10px;">
        ${d.name}(${d.code}) · ${d.strategy_name}<br>${d.period} · ${d.trading_days}交易日
      </div>
      <div class="stat-grid">
        <div class="stat-box"><div class="stat-val ${cls(d.total_return)}">${d.total_return}%</div><div class="stat-lbl">策略总收益</div></div>
        <div class="stat-box"><div class="stat-val ${cls(d.bh_return)}">${d.bh_return}%</div><div class="stat-lbl">买入持有</div></div>
        <div class="stat-box"><div class="stat-val ${cls(beat)}">${beat>=0?'+':''}${beat.toFixed(2)}%</div><div class="stat-lbl">超额收益</div></div>
        <div class="stat-box"><div class="stat-val dn">${d.max_drawdown}%</div><div class="stat-lbl">最大回撤</div></div>
        <div class="stat-box"><div class="stat-val ${cls(d.cagr)}">${d.cagr}%</div><div class="stat-lbl">年化收益</div></div>
        <div class="stat-box"><div class="stat-val">${d.sharpe}</div><div class="stat-lbl">夏普率</div></div>
        <div class="stat-box"><div class="stat-val">${d.trades}</div><div class="stat-lbl">交易次数</div></div>
        <div class="stat-box"><div class="stat-val">${d.win_rate}%</div><div class="stat-lbl">胜率</div></div>
      </div>`;
    $('bt-chart').style.display='block';
    if(btChart) btChart.destroy();
    btChart=new Chart($('bt-chart'),{type:'line',
      data:{labels:d.curve.dates,datasets:[
        {label:'策略净值',data:d.curve.equity,borderColor:'#00d48a',pointRadius:0,borderWidth:2},
        {label:'买入持有',data:d.curve.benchmark,borderColor:'#6b8aaa',pointRadius:0,borderWidth:1.5,borderDash:[4,4]}]},
      options:{animation:{duration:300},responsive:true,
        plugins:{legend:{labels:{color:'#6b8aaa',font:{size:11},boxWidth:14}}},
        scales:{x:{ticks:{color:'#6b8aaa',maxTicksLimit:8},grid:{color:'#1e3050'}},
                y:{ticks:{color:'#6b8aaa'},grid:{color:'#1e3050'}}}}});
  }catch(e){$('bt-stats').innerHTML=`<div class="loading">❌ ${e}</div>`;}
}

// ===== AI =====
function appendMsg(role,text){
  const box=$('chat');
  const div=document.createElement('div');
  div.className=role==='user'?'msg-u':'msg-a';
  const sp=document.createElement('span');
  sp.textContent=text;
  div.appendChild(sp);
  box.appendChild(div);
  box.scrollTop=box.scrollHeight;
}

async function sendChat(){
  const inp=$('chat-in'), msg=inp.value.trim(); if(!msg) return;
  const code=$('ai-code').value.trim()||null;
  appendMsg('user',msg); inp.value=''; inp.disabled=true;

  const ld=document.createElement('div');
  ld.className='msg-a'; ld.id='ai-ld';
  ld.innerHTML='<span>⏳ Claude 分析中...</span>';
  $('chat').appendChild(ld); $('chat').scrollTop=$('chat').scrollHeight;

  try{
    const r=await fetch('/api/ai/chat',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:msg,code:code,reset:false})});
    const d=await r.json();
    document.getElementById('ai-ld')?.remove();
    appendMsg('ai',d.reply||(d.error||'未知错误'));
  }catch(e){
    document.getElementById('ai-ld')?.remove();
    appendMsg('ai','❌ 请求失败: '+e);
  }
  inp.disabled=false; inp.focus();
}

function quickAsk(q){ $('chat-in').value=q; sendChat(); }

async function resetChat(){
  await fetch('/api/ai/chat',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({message:'',code:null,reset:true})});
  $('chat').innerHTML='<div class="msg-a"><span>对话已重置，开始新的分析！</span></div>';
}

document.addEventListener('DOMContentLoaded',()=>{
  $('chat-in').addEventListener('keydown',e=>{
    if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendChat();}
  });
  $('an-code').addEventListener('keydown',e=>{if(e.key==='Enter')analyze();});
});

function switchTab(p){
  document.querySelectorAll('.page').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  $('page-'+p).classList.add('active');
  document.querySelector(`.tab[data-p="${p}"]`).classList.add('active');
}

connectWS();
fetch('/api/sentiment/market').then(r=>r.json()).then(updateMarket).catch(()=>{});
</script>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn
    print(f"启动 http://localhost:{PORT}")
    print(f"数据源: {'东方财富真实数据' if adapter.use_real else '模拟数据'}")
    print(f"AI模式: {'Claude (Anthropic)' if os.getenv('ANTHROPIC_API_KEY') else '本地规则'}")
    uvicorn.run(app, host=HOST, port=PORT)
