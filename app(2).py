# -*- coding: utf-8 -*-
"""
实时量化看板 · 单文件版(东方财富真实数据 / akshare)
运行: pip install -r requirements.txt
      python app.py  ->  http://0.0.0.0:8088
AI:   设置环境变量 ANTHROPIC_API_KEY 即可启用 Claude AI 对话
免责: 算法量化输出,不构成投资建议,入市有风险
"""
import os, json, math, time, random, asyncio
import datetime as dt
from typing import List, Dict
import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

USE_REAL   = os.getenv("USE_REAL", "true").lower() == "true"
PORT       = int(os.getenv("PORT", 8088))
HOST       = "0.0.0.0"
MAX_CHANGE_PCT  = 5.0
TOP_N           = 5
SPOT_CACHE_SEC  = 8
MAX_HIST_SCAN   = 30
MAIN_BOARD_PREFIX = ("600", "601", "603", "605", "000", "001")


def is_main_board(code: str) -> bool:
    return str(code).split(".")[0].startswith(MAIN_BOARD_PREFIX)


# ========================= 数据适配器(东财/akshare) =========================
class EastMoneyAdapter:
    def __init__(self, use_real: bool = True):
        self.use_real = use_real
        self._ak = None
        self._spot_df = None
        self._spot_ts = 0
        self._name_map = {}
        self._mock_state = {}
        if use_real:
            try:
                import akshare as ak
                self._ak = ak
                print("akshare 已加载,使用东方财富真实数据")
            except Exception as e:
                print(f"[警告] akshare 导入失败,回退模拟数据: {e}")
                self.use_real = False
        if not self.use_real:
            self._build_mock_universe()

    def _get_spot(self) -> pd.DataFrame:
        now = time.time()
        if self._spot_df is not None and now - self._spot_ts < SPOT_CACHE_SEC:
            return self._spot_df
        df = self._ak.stock_zh_a_spot_em()
        rename = {"代码": "code", "名称": "name", "最新价": "price",
                  "涨跌幅": "change_pct", "昨收": "pre_close", "量比": "volume_ratio",
                  "换手率": "turnover_rate", "成交额": "amount"}
        df = df.rename(columns=rename)
        for c in ["price", "change_pct", "pre_close", "volume_ratio", "turnover_rate", "amount"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["code", "price"])
        for _, r in df.iterrows():
            self._name_map[str(r["code"])] = str(r["name"])
        self._spot_df = df
        self._spot_ts = now
        return df

    @staticmethod
    def _row_to_quote(r) -> Dict:
        now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        def g(k, d=0.0):
            v = r.get(k, d)
            try:
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    return d
                return float(v)
            except Exception:
                return d
        return {"code": str(r["code"]), "name": str(r.get("name", r["code"])),
                "price": round(g("price"), 2), "pre_close": round(g("pre_close"), 2),
                "change_pct": round(g("change_pct"), 2),
                "volume_ratio": round(g("volume_ratio"), 2),
                "turnover_rate": round(g("turnover_rate"), 2),
                "amount": g("amount"), "ts": now}

    def get_realtime_quotes(self, codes: List[str] = None) -> List[Dict]:
        if not self.use_real:
            return self._mock_realtime(codes)
        df = self._get_spot()
        if codes is not None:
            want = {str(c).split(".")[0] for c in codes}
            df = df[df["code"].isin(want)]
        return [self._row_to_quote(r) for _, r in df.iterrows()]

    def get_history(self, code: str, days: int = 250) -> pd.DataFrame:
        if not self.use_real:
            return self._mock_history(code, days)
        bare = str(code).split(".")[0]
        end  = dt.date.today()
        start = end - dt.timedelta(days=int(days * 1.8) + 40)
        df = self._ak.stock_zh_a_hist(symbol=bare, period="daily",
                                      start_date=start.strftime("%Y%m%d"),
                                      end_date=end.strftime("%Y%m%d"), adjust="qfq")
        rename = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
                  "最低": "low", "成交量": "volume", "成交额": "amount"}
        df = df.rename(columns=rename)
        keep = ["date", "open", "high", "low", "close", "volume", "amount"]
        for c in keep:
            if c not in df.columns:
                df[c] = np.nan
        df = df[keep].dropna(subset=["close"]).reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df.tail(days).reset_index(drop=True)

    def get_universe(self) -> List[str]:
        if not self.use_real:
            return [c for c in self._mock_universe if is_main_board(c)]
        df = self._get_spot()
        return [c for c in df["code"].tolist() if is_main_board(c)]

    def get_name(self, code: str) -> str:
        return self._name_map.get(str(code).split(".")[0], str(code))

    def get_index_quote(self, index_code: str = "000001") -> Dict:
        if not self.use_real:
            return self._mock_index(index_code)
        df = self._get_spot()
        chg = df["change_pct"]
        adv, dec = int((chg > 0).sum()), int((chg < 0).sum())
        lu, ld   = int((chg >= 9.8).sum()), int((chg <= -9.8).sum())
        amount_yi = round(float(df["amount"].sum()) / 1e8, 0)
        idx_chg = self._try_index_change()
        if idx_chg is None:
            idx_chg = round(float(chg.median()), 2)
        return {"code": index_code, "name": "上证指数", "price": 0.0,
                "change_pct": round(idx_chg, 2), "advance": adv, "decline": dec,
                "limit_up": lu, "limit_down": ld, "amount": amount_yi}

    def _try_index_change(self):
        ak = self._ak
        try:
            d   = ak.stock_zh_index_spot_em(symbol="上证系列指数")
            row = d[d["代码"] == "000001"]
            if not row.empty:
                return float(row.iloc[0]["涨跌幅"])
        except Exception:
            pass
        try:
            d   = ak.stock_zh_index_spot_em()
            row = d[d["代码"] == "000001"]
            if not row.empty:
                return float(row.iloc[0]["涨跌幅"])
        except Exception:
            pass
        return None

    def _build_mock_universe(self):
        names = {"600519": "贵州茅台", "600036": "招商银行", "601318": "中国平安",
                 "000333": "美的集团", "000651": "格力电器", "000858": "五粮液",
                 "600276": "恒瑞医药", "601012": "隆基绿能", "600887": "伊利股份",
                 "603288": "海天味业", "600030": "中信证券", "601899": "紫金矿业"}
        for c, n in names.items():
            self._mock_state[c] = {"name": n, "pre_close": round(random.uniform(8, 1700), 2)}
            self._name_map[c]   = n
        self._mock_universe = list(names.keys())

    def _mock_realtime(self, codes):
        codes = codes or self._mock_universe
        now   = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        out   = []
        for c in codes:
            st  = self._mock_state.get(c) or {"name": c, "pre_close": round(random.uniform(8, 100), 2)}
            self._mock_state.setdefault(c, st)
            pre = st["pre_close"]
            chg = float(np.clip(random.gauss(0.3, 2.5), -10, 10))
            out.append({"code": c, "name": st["name"], "price": round(pre * (1 + chg / 100), 2),
                        "pre_close": pre, "change_pct": round(chg, 2),
                        "volume_ratio": round(abs(random.gauss(1.2, 0.8)) + 0.3, 2),
                        "turnover_rate": round(abs(random.gauss(2.0, 1.5)) + 0.2, 2),
                        "amount": round(abs(random.gauss(8e8, 5e8)), 0), "ts": now})
        return out

    def _mock_history(self, code, days):
        st    = self._mock_state.get(str(code).split(".")[0]) or {"pre_close": 50.0}
        dates = pd.bdate_range(end=dt.date.today(), periods=days)
        price = st["pre_close"]; rows = []
        for d in dates:
            o = price; c = max(0.5, o * (1 + random.gauss(0.0005, 0.02)))
            h = max(o, c) * (1 + abs(random.gauss(0, 0.008))); l = min(o, c) * (1 - abs(random.gauss(0, 0.008)))
            v = abs(random.gauss(1e7, 4e6))
            rows.append([d.date(), round(o,2), round(h,2), round(l,2), round(c,2), int(v), v*c])
            price = c
        return pd.DataFrame(rows, columns=["date","open","high","low","close","volume","amount"])

    def _mock_index(self, index_code):
        return {"code": index_code, "name": "上证指数",
                "price": round(3100 + random.uniform(-40, 40), 2),
                "change_pct": round(random.uniform(-1.5, 1.5), 2),
                "advance": random.randint(1500, 3500), "decline": random.randint(1500, 3500),
                "limit_up": random.randint(20, 90), "limit_down": random.randint(0, 30),
                "amount": round(random.uniform(7000, 12000), 0)}


# ============================= 技术指标 =============================
def _ma(s, n): return s.rolling(n).mean()
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()


def _macd(close, fast=12, slow=26, signal=9):
    dif = _ema(close, fast) - _ema(close, slow)
    dea = _ema(dif, signal)
    return dif, dea, (dif - dea) * 2


def _rsi(close, n=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c  = df["close"]
    df["ma5"],  df["ma10"]      = _ma(c, 5),  _ma(c, 10)
    df["ma20"], df["ma60"]      = _ma(c, 20), _ma(c, 60)
    df["dif"], df["dea"], df["macd_hist"] = _macd(c)
    df["rsi"]    = _rsi(c)
    df["vol_ma5"] = _ma(df["volume"], 5)
    return df


def bullish_alignment(row) -> bool:
    try:
        return row["ma5"] > row["ma10"] > row["ma20"]
    except Exception:
        return False


# ============================= 数据分析师 =============================
class DataAnalyst:
    def __init__(self, adapter): self.adapter = adapter

    def analyze(self, code: str) -> dict:
        df = enrich(self.adapter.get_history(code, 250))
        if len(df) < 30:
            return {"code": code, "name": self.adapter.get_name(code), "tech_score": 50,
                    "trend": "数据不足", "bullish_alignment": False, "rsi": 0, "macd_hist": 0,
                    "ma5": 0, "ma10": 0, "ma20": 0, "signals": [],
                    "kline": {"dates": [], "close": [], "ma5": [], "ma20": []},
                    "summary": "历史数据不足,无法分析。"}
        last    = df.iloc[-1]
        signals = self._signals(df)
        score   = self._tech_score(df, signals)
        trend   = self._trend(df)
        return {"code": str(code).split(".")[0], "name": self.adapter.get_name(code),
                "tech_score": score, "trend": trend,
                "bullish_alignment": bool(bullish_alignment(last)),
                "rsi": round(float(last["rsi"]), 1), "macd_hist": round(float(last["macd_hist"]), 3),
                "ma5": round(float(last["ma5"]), 2), "ma10": round(float(last["ma10"]), 2),
                "ma20": round(float(last["ma20"]), 2), "signals": signals,
                "kline": self._kline(df), "summary": self._summary(score, trend, signals)}

    def _signals(self, df):
        sig        = []
        last, prev = df.iloc[-1], df.iloc[-2]
        if prev["dif"] <= prev["dea"] and last["dif"] > last["dea"]:
            sig.append({"type": "buy",  "name": "MACD金叉", "strength": "中"})
        if prev["dif"] >= prev["dea"] and last["dif"] < last["dea"]:
            sig.append({"type": "sell", "name": "MACD死叉", "strength": "中"})
        if prev["close"] <= prev["ma20"] and last["close"] > last["ma20"]:
            sig.append({"type": "buy",  "name": "上穿MA20", "strength": "强"})
        if prev["close"] >= prev["ma20"] and last["close"] < last["ma20"]:
            sig.append({"type": "sell", "name": "跌破MA20", "strength": "强"})
        if last["rsi"] < 30:
            sig.append({"type": "buy",  "name": "RSI超卖",  "strength": "弱"})
        if last["rsi"] > 75:
            sig.append({"type": "sell", "name": "RSI超买",  "strength": "弱"})
        if last["volume"] > last["vol_ma5"] * 1.8 and last["close"] > prev["close"]:
            sig.append({"type": "buy",  "name": "放量上涨", "strength": "中"})
        return sig

    def _tech_score(self, df, signals):
        last = df.iloc[-1]; s = 50
        if bullish_alignment(last):       s += 15
        if last["close"] > last["ma60"]:  s += 8
        if last["macd_hist"] > 0:         s += 7
        if 40 <= last["rsi"] <= 65:       s += 8
        elif last["rsi"] > 80 or last["rsi"] < 20: s -= 8
        s += sum(6 if x["type"] == "buy" else -6 for x in signals)
        return int(max(0, min(100, s)))

    def _trend(self, df):
        last = df.iloc[-1]
        if last["ma5"] > last["ma20"] and last["close"] > last["ma20"]: return "多头"
        if last["ma5"] < last["ma20"] and last["close"] < last["ma20"]: return "空头"
        return "震荡"

    def _summary(self, score, trend, signals):
        buy  = [s["name"] for s in signals if s["type"] == "buy"]
        sell = [s["name"] for s in signals if s["type"] == "sell"]
        parts = [f"技术面评分 {score},当前{trend}格局。"]
        if buy:  parts.append("买入信号:" + "、".join(buy) + "。")
        if sell: parts.append("卖出信号:" + "、".join(sell) + "。")
        if not buy and not sell: parts.append("暂无明确买卖信号,建议观望。")
        return "".join(parts)

    def _kline(self, df, n=120):
        d = df.tail(n)
        return {"dates": [str(x) for x in d["date"].tolist()],
                "close": d["close"].round(2).tolist(),
                "ma5":   d["ma5"].round(2).fillna(0).tolist(),
                "ma20":  d["ma20"].round(2).fillna(0).tolist()}


# ============================= 情绪分析师 =============================
class SentimentAnalyst:
    def __init__(self, adapter): self.adapter = adapter

    def market_sentiment(self) -> dict:
        idx     = self.adapter.get_index_quote("000001")
        adv, dec = idx["advance"], idx["decline"]
        breadth  = adv / max(1, adv + dec)
        limit_net = idx["limit_up"] - idx["limit_down"]
        score    = 50 + idx["change_pct"] * 6 + (breadth - 0.5) * 60 + min(20, limit_net * 0.4)
        score    = int(max(0, min(100, score)))
        mood     = ("亢奋" if score >= 75 else "偏暖" if score >= 60 else
                    "中性" if score >= 45 else "偏冷" if score >= 30 else "恐慌")
        return {"score": score, "mood": mood, "index": idx,
                "breadth": round(breadth * 100, 1),
                "summary": (f"上证{idx['change_pct']:+.2f}%,涨跌家数 {adv}/{dec},"
                            f"涨停{idx['limit_up']}家、跌停{idx['limit_down']}家。"
                            f"市场情绪{mood}(评分{score})。")}

    def stock_sentiment(self, code: str) -> dict:
        qs = self.adapter.get_realtime_quotes([code])
        if not qs:
            return {"code": code, "name": self.adapter.get_name(code), "score": 50,
                    "level": "未知", "change_pct": 0, "volume_ratio": 0,
                    "summary": "未获取到该股实时数据。"}
        q    = qs[0]
        mkt  = self.market_sentiment()
        score = (50 + q["change_pct"] * 3 + min(20, (q["volume_ratio"] - 1) * 15)
                 + (q["change_pct"] - mkt["index"]["change_pct"]) * 2)
        score = int(max(0, min(100, score)))
        hot   = ("过热" if score >= 80 else "活跃" if score >= 60 else
                 "平稳" if score >= 40 else "低迷")
        return {"code": q["code"], "name": q["name"], "score": score, "level": hot,
                "change_pct": q["change_pct"], "volume_ratio": q["volume_ratio"],
                "summary": (f"{q['name']} 当日{q['change_pct']:+.2f}%,量比{q['volume_ratio']},"
                            f"情绪{hot}。结合大盘{mkt['mood']}环境综合判断。")}


# ============================= 每日推荐 =============================
class Recommender:
    def __init__(self, adapter, analyst, sentiment):
        self.adapter, self.analyst, self.sentiment = adapter, analyst, sentiment

    def daily_picks(self, min_volume_ratio=1.2, min_turnover=1.5, require_bullish=True):
        universe = set(self.adapter.get_universe())
        quotes   = self.adapter.get_realtime_quotes()
        mkt      = self.sentiment.market_sentiment()
        pre      = []
        for q in quotes:
            if q["code"] not in universe:          continue
            if q["change_pct"] >= MAX_CHANGE_PCT:  continue
            if q["change_pct"] < -3:               continue
            if q["volume_ratio"] < min_volume_ratio: continue
            if q["turnover_rate"] < min_turnover:  continue
            if q["price"] <= 0:                    continue
            pre.append(q)
        pre.sort(key=lambda x: (x["volume_ratio"], x["change_pct"]), reverse=True)
        pre = pre[:MAX_HIST_SCAN]
        candidates = []
        for q in pre:
            try:
                df = enrich(self.adapter.get_history(q["code"], 120))
                if len(df) < 30: continue
            except Exception:
                continue
            last = df.iloc[-1]
            if require_bullish and not bullish_alignment(last): continue
            tech = self.analyst.analyze(q["code"])
            sent = self.sentiment.stock_sentiment(q["code"])
            score = int(tech["tech_score"] * 0.6 + sent["score"] * 0.4)
            candidates.append({
                "code": q["code"], "name": q["name"], "price": q["price"],
                "change_pct": q["change_pct"], "volume_ratio": q["volume_ratio"],
                "turnover_rate": q["turnover_rate"], "score": score,
                "tech_score": tech["tech_score"], "sentiment_score": sent["score"],
                "reason": self._reason(tech, sent, q, mkt),
                "timing": self._timing(q, last)})
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return {"market": mkt, "picks": candidates[:TOP_N],
                "filters": {"max_change_pct": MAX_CHANGE_PCT,
                            "min_volume_ratio": min_volume_ratio,
                            "min_turnover": min_turnover,
                            "require_bullish": require_bullish},
                "disclaimer": "本列表为算法量化输出,不构成投资建议。股市有风险,入市需谨慎。"}

    def _reason(self, tech, sent, q, mkt):
        bits = []
        if tech["bullish_alignment"]:    bits.append("均线多头排列")
        if tech["macd_hist"] > 0:        bits.append("MACD红柱")
        if q["volume_ratio"] > 1.5:      bits.append(f"量比{q['volume_ratio']}放大")
        if sent["score"] >= 60:          bits.append("个股情绪活跃")
        bits.append(f"大盘{mkt['mood']}")
        return "、".join(bits) if bits else "技术面均衡"

    def _timing(self, q, last):
        if q["change_pct"] > 3:
            return "今日涨幅偏高,不建议追高,待回踩MA5附近分批介入"
        if q["price"] < last["ma5"]:
            return "现价处于MA5下方,可逢低关注"
        return "现价附近可分批建仓,跌破MA10止损"


# ============================= 回测引擎 =============================
class Backtester:
    def __init__(self, adapter): self.adapter = adapter

    def run(self, code, fast=5, slow=20, init_cash=100000, fee=0.0013, days=250):
        try:
            df = enrich(self.adapter.get_history(code, days)).dropna().reset_index(drop=True)
        except Exception as e:
            return {"error": f"取数失败: {e}"}
        if len(df) < slow + 5:
            return {"error": "历史数据不足以回测"}
        df["fast"] = df["close"].rolling(fast).mean()
        df["slow"] = df["close"].rolling(slow).mean()
        df["pos"]  = (df["fast"] > df["slow"]).astype(int).shift(1).fillna(0)
        df["ret"]  = df["close"].pct_change().fillna(0)
        df["trade"] = df["pos"].diff().abs().fillna(0)
        df["strat_ret"] = df["pos"] * df["ret"] - df["trade"] * fee
        df["equity"] = (1 + df["strat_ret"]).cumprod() * init_cash
        df["bh"]     = (1 + df["ret"]).cumprod() * init_cash
        eq        = df["equity"]
        total_ret = eq.iloc[-1] / init_cash - 1
        years     = max(len(df) / 244, 0.1)
        cagr      = (eq.iloc[-1] / init_cash) ** (1 / years) - 1
        peak = eq.cummax(); mdd = ((eq - peak) / peak).min()
        sharpe = (df["strat_ret"].mean() / (df["strat_ret"].std() + 1e-9)) * np.sqrt(244)
        trades = int(df["trade"].sum())
        wins   = (df.loc[df["trade"] > 0, "strat_ret"] > 0).sum()
        return {"code": str(code).split(".")[0], "name": self.adapter.get_name(code),
                "total_return": round(total_ret * 100, 2), "cagr": round(cagr * 100, 2),
                "max_drawdown": round(mdd * 100, 2), "sharpe": round(float(sharpe), 2),
                "trades": trades, "win_rate": round(float(wins) / max(1, trades) * 100, 1),
                "curve": {"dates":     [str(x) for x in df["date"].tolist()],
                          "equity":    eq.round(0).tolist(),
                          "benchmark": df["bh"].round(0).tolist()}}


# ============================= AI交易员助手(Claude) =============================
SYSTEM_PROMPT = (
    "你是一名专业的A股量化交易助手,只服务于主板股票。"
    "回答要结合提供的实时数据(市场情绪、个股技术面)。"
    "强调风险控制,任何建议都要提示不构成投资建议。"
    "当日涨幅超过5%的标的不建议追高。"
    "回答简洁精炼,控制在300字以内。")


class AIAssistant:
    def __init__(self, adapter, analyst, sentiment):
        self.adapter, self.analyst, self.sentiment = adapter, analyst, sentiment
        # 支持 Anthropic Claude API
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        # 兼容旧版 OpenAI 配置
        self.openai_key    = os.getenv("LLM_API_KEY")
        self.openai_url    = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.openai_model  = os.getenv("LLM_MODEL", "gpt-4o-mini")

    def _context(self, code=None):
        ctx = {"market": self.sentiment.market_sentiment()}
        if code:
            try:
                ctx["stock_tech"]      = self.analyst.analyze(code)
                ctx["stock_sentiment"] = self.sentiment.stock_sentiment(code)
            except Exception:
                pass
        return json.dumps(ctx, ensure_ascii=False, default=str)[:4000]

    async def chat(self, message, code=None):
        context = self._context(code)

        # ---- 优先使用 Anthropic Claude ----
        if self.anthropic_key:
            return await self._claude_chat(message, context)

        # ---- 回退 OpenAI 兼容接口 ----
        if self.openai_key:
            return await self._openai_chat(message, context)

        # ---- 本地规则兜底 ----
        return {"reply": self._fallback(message, code, context), "mode": "local-rule"}

    async def _claude_chat(self, message, context):
        """调用 Anthropic Claude API (Messages API)"""
        payload = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT + f"\n\n实时数据上下文：{context}",
            "messages": [{"role": "user", "content": message}]
        }
        headers = {
            "x-api-key": self.anthropic_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    json=payload, headers=headers)
                r.raise_for_status()
                data  = r.json()
                reply = data["content"][0]["text"]
                return {"reply": reply, "mode": "claude"}
        except Exception as e:
            return {"reply": self._fallback(message, None, context),
                    "mode": "local-rule", "error": str(e)}

    async def _openai_chat(self, message, context):
        payload = {
            "model": self.openai_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": f"实时数据上下文:{context}"},
                {"role": "user",   "content": message}],
            "temperature": 0.3}
        headers = {"Authorization": f"Bearer {self.openai_key}"}
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(f"{self.openai_url}/chat/completions",
                                      json=payload, headers=headers)
                r.raise_for_status()
                return {"reply": r.json()["choices"][0]["message"]["content"], "mode": "llm"}
        except Exception as e:
            return {"reply": self._fallback(message, None, context),
                    "mode": "local-rule", "error": str(e)}

    def _fallback(self, message, code, context):
        try:
            ctx = json.loads(context)
            mkt = ctx.get("market", {})
        except Exception:
            mkt = {}
        mood  = mkt.get("mood", "未知")
        score = mkt.get("score", 50)
        base  = f"【规则模式·未配置AI密钥】当前大盘情绪{mood}(评分{score})。"
        if "推荐" in message or "买" in message:
            base += "请使用「推荐」标签页获取今日推荐列表。"
        elif "卖" in message or "止损" in message:
            base += "持仓遇到卖出信号时,建议严格止损,不抱侥幸心理。"
        else:
            base += "如需个股分析,请在「个股」标签页输入股票代码。"
        base += " 配置 ANTHROPIC_API_KEY 环境变量可启用完整 Claude AI 对话能力。"
        return base


# ============================= 组装 + 路由 =============================
adapter    = EastMoneyAdapter(use_real=USE_REAL)
analyst    = DataAnalyst(adapter)
sentiment  = SentimentAnalyst(adapter)
recommender = Recommender(adapter, analyst, sentiment)
backtester  = Backtester(adapter)
ai          = AIAssistant(adapter, analyst, sentiment)

app = FastAPI(title="实时量化看板")


@app.get("/api/recommendations")
def recommendations(min_volume_ratio: float = 1.2, min_turnover: float = 1.5,
                    require_bullish: bool = True):
    try:
        return recommender.daily_picks(min_volume_ratio, min_turnover, require_bullish)
    except Exception as e:
        return {"market": {}, "picks": [], "error": str(e),
                "disclaimer": "数据获取异常,请稍后重试。"}


@app.get("/api/analyze/{code}")
def analyze(code: str):
    try:
        return analyst.analyze(code)
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/sentiment/market")
def market_sentiment():
    return sentiment.market_sentiment()


@app.get("/api/sentiment/stock/{code}")
def stock_sentiment(code: str):
    return sentiment.stock_sentiment(code)


@app.get("/api/backtest/{code}")
def backtest(code: str, fast: int = 5, slow: int = 20):
    return backtester.run(code, fast, slow)


class ChatReq(BaseModel):
    message: str
    code: str = None


@app.post("/api/ai/chat")
async def ai_chat(req: ChatReq):
    return await ai.chat(req.message, req.code)


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                mkt    = sentiment.market_sentiment()
                quotes = adapter.get_realtime_quotes()[:20]
            except Exception as e:
                mkt, quotes = {"summary": f"数据获取中... ({e})", "score": 50}, []
            await websocket.send_text(json.dumps(
                {"type": "tick", "quotes": quotes, "market": mkt},
                ensure_ascii=False, default=str))
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        pass


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


# ============================= 前端页面 =============================
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>实时量化看板</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root{--bg:#0a0e1a;--card:#131a2b;--card2:#1a2236;--line:#222c44;
        --green:#26d07c;--red:#ff5b5b;--gold:#f5b942;--blue:#3b82f6;
        --txt:#e6ecf5;--sub:#8b96ad;}
  *{box-sizing:border-box;margin:0;padding:0;font-family:-apple-system,"PingFang SC",sans-serif;}
  body{background:var(--bg);color:var(--txt);max-width:480px;margin:0 auto;padding-bottom:70px;}
  .topbar{display:flex;justify-content:space-between;align-items:center;padding:14px 16px;font-size:13px;color:var(--sub);}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;margin:10px 12px;padding:14px;}
  .card h3{font-size:15px;margin-bottom:12px;display:flex;align-items:center;gap:6px;}
  .row{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid var(--line);}
  .row:last-child{border-bottom:none;}
  .stk-name{font-size:15px;font-weight:600;}
  .stk-code{font-size:11px;color:var(--sub);margin-left:6px;}
  .price{font-size:15px;font-weight:600;text-align:right;}
  .up{color:var(--red);} .down{color:var(--green);}
  .score{font-size:11px;color:var(--gold);text-align:right;}
  .reason{font-size:11px;color:var(--sub);margin-top:3px;}
  .timing{font-size:12px;color:var(--blue);background:var(--card2);border-radius:8px;padding:8px;margin-top:6px;line-height:1.5;}
  .field{margin:8px 0;}
  .field label{font-size:12px;color:var(--sub);display:block;margin-bottom:4px;}
  .field input,.field select{width:100%;background:var(--card2);border:1px solid var(--line);
    color:var(--txt);border-radius:8px;padding:9px;font-size:14px;}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
  .btn{width:100%;background:linear-gradient(90deg,#1a3a6b,#2563eb);color:#fff;border:none;
    border-radius:10px;padding:12px;font-size:15px;font-weight:600;margin-top:10px;cursor:pointer;}
  .check{display:flex;gap:8px;align-items:center;}
  .mood-bar{height:6px;border-radius:3px;background:var(--card2);overflow:hidden;margin-top:6px;}
  .mood-fill{height:100%;background:linear-gradient(90deg,var(--green),var(--gold),var(--red));transition:width .5s;}
  .tabs{position:fixed;bottom:0;left:50%;transform:translateX(-50%);max-width:480px;width:100%;
    display:flex;background:#0d1322;border-top:1px solid var(--line);}
  .tab{flex:1;text-align:center;padding:12px 0;font-size:12px;color:var(--sub);cursor:pointer;}
  .tab.active{color:var(--blue);}
  .page{display:none;} .page.active{display:block;}
  .chat-box{height:300px;overflow-y:auto;background:var(--card2);border-radius:10px;padding:10px;font-size:13px;line-height:1.6;}
  .msg-u{text-align:right;color:var(--blue);margin:6px 0;}
  .msg-a{text-align:left;color:var(--txt);margin:6px 0;white-space:pre-wrap;}
  .disclaimer{font-size:10px;color:var(--sub);text-align:center;padding:10px;line-height:1.5;}
  .tag{font-size:10px;padding:2px 6px;border-radius:4px;background:var(--card2);color:var(--sub);}
  .loading{font-size:12px;color:var(--sub);text-align:center;padding:10px;}
  input:focus,select:focus{outline:1px solid var(--blue);}
</style>
</head>
<body>
<div class="topbar">
  <span id="clock">--:--</span>
  <span>实时量化看板 · 东财数据 <span class="tag" id="conn">连接中</span></span>
</div>

<div class="page active" id="page-rec">
  <div class="card">
    <h3>🌐 大盘环境 · 情绪分析师</h3>
    <div class="row"><span id="mkt-summary" style="font-size:13px;color:var(--sub);">加载中...</span></div>
    <div class="mood-bar"><div class="mood-fill" id="mkt-bar" style="width:50%"></div></div>
  </div>
  <div class="card">
    <h3>🚀 今日推荐买入(全市场扫描)</h3>
    <div id="rec-list"><div class="loading">扫描中,首次约需10-30秒...</div></div>
  </div>
  <div class="card">
    <h3>🎯 建议买入时机</h3>
    <div id="timing-list"></div>
  </div>
  <div class="card">
    <h3>⚠️ 持仓卖出信号</h3>
    <div class="field"><div style="display:flex;gap:8px;">
      <input id="sell-code" placeholder="输入持仓代码,如 600519">
      <button class="btn" style="width:80px;margin-top:0;" onclick="checkSell()">检查</button>
    </div></div>
    <div id="sell-result"></div>
  </div>
  <div class="card">
    <h3>🔍 智慧选股 · 全市场技术面筛选器</h3>
    <div class="grid2">
      <div class="field"><label>最小量比</label><input id="f-vr" value="1.2"></div>
      <div class="field"><label>最小换手率(%)</label><input id="f-to" value="1.5"></div>
    </div>
    <div class="field check"><input type="checkbox" id="f-bull" checked>
      <label style="margin:0">要求均线多头排列(MA5>MA10>MA20)</label></div>
    <button class="btn" onclick="loadRecs()">一键生成推荐列表</button>
    <div class="disclaimer">扫描范围:全A股主板。已剔除ST、创业板、科创板、北交所;
      仅保留当日涨幅低于5%的标的,遵守不追高原则。</div>
  </div>
  <div class="disclaimer" id="disc"></div>
</div>

<div class="page" id="page-analyze">
  <div class="card">
    <h3>📈 个股分析 · 数据分析师</h3>
    <div style="display:flex;gap:8px;">
      <input class="field" style="flex:1" id="an-code" placeholder="股票代码 如 600519">
      <button class="btn" style="width:80px;margin:0;" onclick="analyze()">分析</button>
    </div>
    <div id="an-result" style="margin-top:10px;"></div>
    <canvas id="an-chart" style="margin-top:10px;"></canvas>
  </div>
</div>

<div class="page" id="page-bt">
  <div class="card">
    <h3>🧪 策略回测(双均线)</h3>
    <div class="grid2">
      <div class="field"><label>股票代码</label><input id="bt-code" value="600519"></div>
      <div class="field"><label>快线/慢线</label>
        <div style="display:flex;gap:6px;"><input id="bt-fast" value="5"><input id="bt-slow" value="20"></div>
      </div>
    </div>
    <button class="btn" onclick="runBacktest()">运行回测</button>
    <div id="bt-stats" style="margin-top:10px;"></div>
    <canvas id="bt-chart" style="margin-top:10px;"></canvas>
  </div>
</div>

<div class="page" id="page-ai">
  <div class="card">
    <h3>🤖 Claude AI 交易员助手</h3>
    <div class="chat-box" id="chat"></div>
    <div style="display:flex;gap:8px;margin-top:8px;">
      <input class="field" style="flex:1;margin:0;" id="chat-in"
             placeholder="问当前行情、个股分析、买卖点...">
      <button class="btn" style="width:70px;margin:0;" onclick="sendChat()">发送</button>
    </div>
    <div class="disclaimer">AI输出仅供参考,不构成投资建议。由 Anthropic Claude 提供支持。</div>
  </div>
</div>

<div class="tabs">
  <div class="tab active" data-p="rec"     onclick="switchTab('rec')">⚡推荐</div>
  <div class="tab"        data-p="analyze" onclick="switchTab('analyze')">📊个股</div>
  <div class="tab"        data-p="bt"      onclick="switchTab('bt')">🧪回测</div>
  <div class="tab"        data-p="ai"      onclick="switchTab('ai')">🤖AI</div>
</div>

<script>
const API="";let anChart,btChart;
setInterval(()=>{document.getElementById('clock').textContent=
  new Date().toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'});},1000);

function connectWS(){
  const proto=location.protocol==='https:'?'wss://':'ws://';
  const ws=new WebSocket(proto+location.host+'/ws');
  ws.onopen=()=>document.getElementById('conn').textContent='● 实时';
  ws.onclose=()=>{document.getElementById('conn').textContent='○ 断开';setTimeout(connectWS,3000);};
  ws.onmessage=e=>{const d=JSON.parse(e.data);if(d.type==='tick')updateMarket(d.market);};
}
function updateMarket(m){
  if(m&&m.summary)document.getElementById('mkt-summary').textContent=m.summary;
  if(m&&m.score!=null)document.getElementById('mkt-bar').style.width=m.score+'%';
}
const cls=p=>p>=0?'up':'down';
const fmt=p=>(p>=0?'+':'')+Number(p).toFixed(2)+'%';

async function loadRecs(){
  const vr=document.getElementById('f-vr').value,
        to=document.getElementById('f-to').value,
        bull=document.getElementById('f-bull').checked;
  document.getElementById('rec-list').innerHTML='<div class="loading">全市场扫描中,请稍候(首次较慢)...</div>';
  try{
    const r=await fetch(`${API}/api/recommendations?min_volume_ratio=${vr}&min_turnover=${to}&require_bullish=${bull}`);
    const d=await r.json();
    if(d.market)updateMarket(d.market);
    const list=document.getElementById('rec-list'),timing=document.getElementById('timing-list');
    if(d.error){list.innerHTML='<div class="loading">出错:'+d.error+'</div>';return;}
    if(!d.picks||!d.picks.length){
      list.innerHTML='<div class="reason">当前条件下无符合标的,可放宽筛选。</div>';
      timing.innerHTML='';return;
    }
    list.innerHTML=d.picks.map(p=>`<div class="row"><div>
      <div><span class="stk-name">${p.name}</span><span class="stk-code">${p.code}</span></div>
      <div class="reason">${p.reason}</div></div>
      <div><div class="price ${cls(p.change_pct)}">${p.price} ${fmt(p.change_pct)}</div>
      <div class="score">评分 ${p.score}</div></div></div>`).join('');
    timing.innerHTML=d.picks.map(p=>`<div class="timing"><b>${p.name}</b>(量比${p.volume_ratio}):${p.timing}</div>`).join('');
    document.getElementById('disc').textContent=d.disclaimer||'';
  }catch(e){document.getElementById('rec-list').innerHTML='<div class="loading">请求失败:'+e+'</div>';}
}

async function checkSell(){
  const code=document.getElementById('sell-code').value.trim();if(!code)return;
  const r=await fetch(`${API}/api/analyze/${code}`);const d=await r.json();
  if(d.error){document.getElementById('sell-result').innerHTML='<div class="reason">'+d.error+'</div>';return;}
  const sells=(d.signals||[]).filter(s=>s.type==='sell');
  document.getElementById('sell-result').innerHTML=`<div class="row"><div>
    <span class="stk-name">${d.name}</span><span class="stk-code">${code}</span>
    <div class="reason">${d.summary}</div></div>
    <div class="score" style="color:${sells.length?'var(--red)':'var(--green)'}">
    ${sells.length?'⚠️ '+sells.map(s=>s.name).join('/'):'✓ 暂无卖出信号'}</div></div>`;
}

async function analyze(){
  const code=document.getElementById('an-code').value.trim();if(!code)return;
  document.getElementById('an-result').innerHTML='<div class="loading">分析中...</div>';
  const r=await fetch(`${API}/api/analyze/${code}`);const d=await r.json();
  if(d.error){document.getElementById('an-result').innerHTML='<div class="reason">'+d.error+'</div>';return;}
  document.getElementById('an-result').innerHTML=`
    <div class="row"><span class="stk-name">${d.name}<span class="stk-code">${code}</span></span>
      <span class="score">技术评分 ${d.tech_score} · ${d.trend}</span></div>
    <div class="reason">RSI ${d.rsi} · MACD ${d.macd_hist} · MA5 ${d.ma5}/MA20 ${d.ma20}</div>
    <div class="timing">${d.summary}</div>
    <div style="margin-top:6px;">${(d.signals||[]).map(s=>
      `<span class="tag" style="color:${s.type==='buy'?'var(--red)':'var(--green)'}">${s.name}(${s.strength})</span>`).join(' ')}</div>`;
  const k=d.kline;if(!k||!k.dates.length)return;
  if(anChart)anChart.destroy();
  anChart=new Chart(document.getElementById('an-chart'),{type:'line',
    data:{labels:k.dates,datasets:[
      {label:'收盘',data:k.close,borderColor:'#e6ecf5',pointRadius:0,borderWidth:1.5},
      {label:'MA5', data:k.ma5, borderColor:'#f5b942',pointRadius:0,borderWidth:1},
      {label:'MA20',data:k.ma20,borderColor:'#3b82f6',pointRadius:0,borderWidth:1}]},
    options:{plugins:{legend:{labels:{color:'#8b96ad',font:{size:10}}}},
      scales:{x:{ticks:{color:'#8b96ad',maxTicksLimit:6}},y:{ticks:{color:'#8b96ad'}}}}});
}

async function runBacktest(){
  const code=document.getElementById('bt-code').value.trim(),
        fast=document.getElementById('bt-fast').value,
        slow=document.getElementById('bt-slow').value;
  document.getElementById('bt-stats').innerHTML='<div class="loading">回测中...</div>';
  const r=await fetch(`${API}/api/backtest/${code}?fast=${fast}&slow=${slow}`);
  const d=await r.json();
  if(d.error){document.getElementById('bt-stats').innerHTML=`<div class="reason">${d.error}</div>`;return;}
  document.getElementById('bt-stats').innerHTML=`
    <div class="row"><span>总收益</span><span class="${cls(d.total_return)}">${d.total_return}%</span></div>
    <div class="row"><span>年化</span><span class="${cls(d.cagr)}">${d.cagr}%</span></div>
    <div class="row"><span>最大回撤</span><span class="down">${d.max_drawdown}%</span></div>
    <div class="row"><span>夏普</span><span>${d.sharpe}</span></div>
    <div class="row"><span>交易次数/胜率</span><span>${d.trades} / ${d.win_rate}%</span></div>`;
  if(btChart)btChart.destroy();
  btChart=new Chart(document.getElementById('bt-chart'),{type:'line',
    data:{labels:d.curve.dates,datasets:[
      {label:'策略',    data:d.curve.equity,    borderColor:'#26d07c',pointRadius:0,borderWidth:1.5},
      {label:'买入持有',data:d.curve.benchmark,borderColor:'#8b96ad',pointRadius:0,borderWidth:1}]},
    options:{plugins:{legend:{labels:{color:'#8b96ad',font:{size:10}}}},
      scales:{x:{ticks:{color:'#8b96ad',maxTicksLimit:6}},y:{ticks:{color:'#8b96ad'}}}}});
}

async function sendChat(){
  const inp=document.getElementById('chat-in'),msg=inp.value.trim();if(!msg)return;
  const box=document.getElementById('chat');
  box.innerHTML+=`<div class="msg-u">${msg}</div>`;
  inp.value='';inp.disabled=true;
  box.scrollTop=box.scrollHeight;
  box.innerHTML+='<div class="msg-a" id="thinking">🤖 思考中...</div>';
  const m=msg.match(/\d{6}/);
  try{
    const r=await fetch(`${API}/api/ai/chat`,{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:msg,code:m?m[0]:null})});
    const d=await r.json();
    document.getElementById('thinking').remove();
    box.innerHTML+=`<div class="msg-a">🤖 ${d.reply}</div>`;
  }catch(e){
    document.getElementById('thinking').textContent='🤖 请求失败,请重试';
  }
  inp.disabled=false;inp.focus();
  box.scrollTop=box.scrollHeight;
}

// Enter键发送
document.addEventListener('DOMContentLoaded',()=>{
  document.getElementById('chat-in').addEventListener('keydown',e=>{
    if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendChat();}
  });
});

function switchTab(p){
  document.querySelectorAll('.page').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  document.getElementById('page-'+p).classList.add('active');
  document.querySelector(`.tab[data-p="${p}"]`).classList.add('active');
}
connectWS();loadRecs();
</script>
</body>
</html>"""


# ============================= 启动 =============================
if __name__ == "__main__":
    import uvicorn
    print(f"启动中... 打开 http://localhost:{PORT}")
    print(f"数据源: {'东方财富(真实)' if adapter.use_real else '模拟数据'}")
    print(f"AI模式: {'Claude (Anthropic)' if os.getenv('ANTHROPIC_API_KEY') else '本地规则兜底'}")
    uvicorn.run(app, host=HOST, port=PORT)
