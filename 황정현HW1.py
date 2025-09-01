# 허용 패키지: numpy, pandas, matplotlib, seaborn(옵션), mplfinance(옵션), SciPy, 표준라이브러리
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 캔들차트
try:
    import mplfinance as mpf
    HAS_MPLFINANCE = True
except Exception:
    HAS_MPLFINANCE = False

# === 경로 설정 ===
SP500_CSV = "data/S&P 500 Historical Data.csv"      # Investing.com S&P 500
AAPL_CSV  = "data/Apple Stock Price History.csv"     # Investing.com AAPL
OUT_DIR   = "figs"
SHOW_FIGS = True   # 창으로도 보고 싶으면 True
os.makedirs(OUT_DIR, exist_ok=True)

# === Investing.com CSV 로더 ===
def load_investing_csv(path):
    # CSV 읽기
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Date 파싱 및 정렬
    if "Date" not in df.columns:
        raise ValueError(f"'Date' column not found in {path}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    # Price → Adj Close (숫자화)
    if "Price" not in df.columns:
        raise ValueError(f"'Price' column not found in {path}")
    df["Price"] = (df["Price"].astype(str)
                             .str.replace(",", "", regex=False)
                             .str.replace("%", "", regex=False))
    df["Adj Close"] = pd.to_numeric(df["Price"], errors="coerce")

    # O/H/L/Close 숫자화(있으면)
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            s = df[col].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
            df[col] = pd.to_numeric(s, errors="coerce")

    # 거래량 'K','M','B' 처리(있으면)
    if "Vol." in df.columns:
        v = df["Vol."].astype(str).str.replace(",", "", regex=False).str.upper()
        mult = np.where(v.str.endswith("K"), 1e3,
               np.where(v.str.endswith("M"), 1e6,
               np.where(v.str.endswith("B"), 1e9, 1.0)))
        v_num = pd.to_numeric(v.str.replace("K","",regex=False)
                                .str.replace("M","",regex=False)
                                .str.replace("B","",regex=False),
                              errors="coerce")
        df["Volume"] = v_num * mult

    # 결측 제거
    df = df.dropna(subset=["Adj Close"])

    # === 캔들차트 호환: Close가 없으면 Adj Close로 대체 ===
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    return df

# === 피처 추가 ===
def add_features(df):
    px = df["Adj Close"]
    df["MA5"]  = px.rolling(5).mean()
    df["MA20"] = px.rolling(20).mean()
    df["MA60"] = px.rolling(60).mean()
    df["Ret"]  = px.pct_change()
    df["Vol20"] = df["Ret"].rolling(20).std() * np.sqrt(252)  # 연율화 변동성(근사)
    cummax = px.cummax()
    df["Drawdown"] = px / cummax - 1.0
    return df

# === 크로스오버 ===
def find_crossovers(df, s="MA5", m="MA20"):
    prev = df[s].shift(1) - df[m].shift(1)
    now  = df[s] - df[m]
    golden = df.index[(prev < 0) & (now > 0)]
    dead   = df.index[(prev > 0) & (now < 0)]
    return golden, dead

# === 베타/상관 ===
def beta_and_corr(spx_ret, aapl_ret):
    ret = pd.concat([spx_ret.rename("SPX"), aapl_ret.rename("AAPL")], axis=1).dropna()
    slope, intercept, r, p, se = stats.linregress(ret["SPX"].values, ret["AAPL"].values)
    return slope, intercept, r, p, ret

# === MA 기울기(최근 구간) ===
def ma_slope(series, lookback=60):
    s = series.dropna().tail(lookback)
    if len(s) < 2:
        return np.nan
    x = np.arange(len(s))
    m, b = np.polyfit(x, s.values, 1)
    return m

# === 시각화 ===
def plot_price_ma(df, title, fname):
    plt.figure(figsize=(11,6))
    plt.plot(df.index, df["Adj Close"], label="Adj Close", alpha=0.9)
    plt.plot(df.index, df["MA5"],  label="MA5",  linewidth=1.5)
    plt.plot(df.index, df["MA20"], label="MA20", linewidth=1.5)
    plt.plot(df.index, df["MA60"], label="MA60", linewidth=1.5)
    golden, dead = find_crossovers(df)
    plt.scatter(golden, df.loc[golden, "Adj Close"], marker="^", s=50, label="Golden X", zorder=3)
    plt.scatter(dead,   df.loc[dead,   "Adj Close"], marker="v", s=50, label="Dead X",   zorder=3)
    plt.title(title); plt.xlabel("Date"); plt.ylabel("Price")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200)
    if SHOW_FIGS: plt.show()
    plt.close()

def plot_beta(ret, beta, intercept, fname):
    plt.figure(figsize=(7,6))
    plt.scatter(ret["SPX"], ret["AAPL"], s=8, alpha=0.6, label="Daily returns")
    xline = np.linspace(ret["SPX"].min(), ret["SPX"].max(), 200)
    yline = intercept + beta * xline
    plt.plot(xline, yline, label=f"AAPL = {intercept:.4f} + {beta:.2f}·SPX")
    plt.title("AAPL vs. SPX Daily Returns (β)")
    plt.xlabel("SPX daily return"); plt.ylabel("AAPL daily return")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200)
    if SHOW_FIGS: plt.show()
    plt.close()

def plot_drawdown_vol(df, title, fname):
    fig, ax1 = plt.subplots(figsize=(11,6))
    ax1.plot(df.index, df["Drawdown"], label="Drawdown", alpha=0.9)
    ax1.set_ylabel("Drawdown"); ax1.axhline(0, linewidth=0.8)
    ax2 = ax1.twinx()
    ax2.plot(df.index, df["Vol20"], label="Vol20 (annualized)", alpha=0.7)
    ax2.set_ylabel("Volatility")
    fig.suptitle(title)
    ax1.grid(True); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=200)
    if SHOW_FIGS: plt.show()
    plt.close(fig)

def maybe_candle_chart(df, title, fname):
    """OHLC가 충분하면 캔들차트 저장 후 True, 아니면 False 반환"""
    if not HAS_MPLFINANCE:
        return False
    needed = {"Open","High","Low","Close"}
    if not needed.issubset(df.columns):
        return False

    # OHLC 프레임(숫자/결측 정리)
    c = df[["Open","High","Low","Close"]].copy()
    c = c.apply(pd.to_numeric, errors="coerce").dropna()
    if c.empty:
        return False

    # 유효값(비-NaN)이 1개 이상 있는 MA만 추가 (빈 시리즈 방어)
    ap = []
    for ma_col, color in [("MA5","b"), ("MA20","g"), ("MA60","r")]:
        if ma_col in df.columns:
            s = df[ma_col].reindex(c.index)
            if s.notna().sum() > 0:
                ap.append(mpf.make_addplot(s, color=color))

    save_path = os.path.join(OUT_DIR, fname)
    if SHOW_FIGS:
        mpf.plot(c, type="candle", addplot=ap, title=title, style="yahoo",
                 savefig=save_path)
    else:
        fig, _ = mpf.plot(c, type="candle", addplot=ap, title=title, style="yahoo",
                          savefig=save_path, returnfig=True)
        plt.close(fig)
    return True

def main():
    spx  = add_features(load_investing_csv(SP500_CSV))
    aapl = add_features(load_investing_csv(AAPL_CSV))

    beta, intercept, r, p, ret = beta_and_corr(spx["Ret"], aapl["Ret"])

    # 지표 계산
    spx_g, spx_d = find_crossovers(spx)
    a_g, a_d = find_crossovers(aapl)
    metrics = {
        "Period": f"{spx.index.min().date()} ~ {spx.index.max().date()}",
        "Corr(SPX,AAPL)": round(ret["SPX"].corr(ret["AAPL"]), 3),
        "Beta(AAPL~SPX)": round(beta, 2),
        "SPX Golden/Dead": f"{len(spx_g)}/{len(spx_d)}",
        "AAPL Golden/Dead": f"{len(a_g)}/{len(a_d)}",
        "SPX MA20 slope": round(ma_slope(spx["MA20"]), 6),
        "AAPL MA20 slope": round(ma_slope(aapl["MA20"]), 6),
        "SPX MaxDD": round(spx["Drawdown"].min(), 3),
        "AAPL MaxDD": round(aapl["Drawdown"].min(), 3),
    }

    # 시각화
    plot_price_ma(spx,  "S&P 500: Price + 5/20/60MA + Crossovers", "spx_ma.png")
    plot_price_ma(aapl, "Apple (AAPL): Price + 5/20/60MA + Crossovers", "aapl_ma.png")
    plot_beta(ret, beta, intercept, "beta_scatter.png")
    plot_drawdown_vol(spx,  "S&P 500: Drawdown & 20d Vol", "spx_dd_vol.png")
    plot_drawdown_vol(aapl, "Apple: Drawdown & 20d Vol",   "aapl_dd_vol.png")

    spx_candle_ok  = maybe_candle_chart(spx,  "S&P 500 Candlestick (with MAs)", "spx_candle.png")
    aapl_candle_ok = maybe_candle_chart(aapl, "AAPL Candlestick (with MAs)",    "aapl_candle.png")

    # 요약 저장
    lines = [
        "=== Summary (S&P 500 + Apple) ===",
        f"Period: {metrics['Period']}",
        f"Corr(SPX,AAPL): {metrics['Corr(SPX,AAPL)']}",
        f"Beta(AAPL~SPX): {metrics['Beta(AAPL~SPX)']}",
        f"SPX Golden/Dead: {metrics['SPX Golden/Dead']}",
        f"AAPL Golden/Dead: {metrics['AAPL Golden/Dead']}",
        f"SPX MA20 slope (per day): {metrics['SPX MA20 slope']}",
        f"AAPL MA20 slope (per day): {metrics['AAPL MA20 slope']}",
        f"SPX Max Drawdown: {metrics['SPX MaxDD']}",
        f"AAPL Max Drawdown: {metrics['AAPL MaxDD']}",
    ]
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    pd.DataFrame([metrics]).to_csv(os.path.join(OUT_DIR, "summary_table.csv"), index=False)

    # 실제 저장된 파일 목록 표시
    saved = ["spx_ma.png", "aapl_ma.png", "beta_scatter.png", "spx_dd_vol.png", "aapl_dd_vol.png"]
    if spx_candle_ok:  saved.append("spx_candle.png")
    if aapl_candle_ok: saved.append("aapl_candle.png")

    print("\n".join(lines))
    print(f"\n[Saved figures] {os.path.abspath(OUT_DIR)} -> " + ", ".join(saved))

if __name__ == "__main__":
    main()
