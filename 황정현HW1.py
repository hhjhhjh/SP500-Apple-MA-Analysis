import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

try:
    import mplfinance as mpf
    HAS_MPLFINANCE = True
except Exception:
    HAS_MPLFINANCE = False

SP500_CSV = "data/S&P 500 Historical Data.csv"
AAPL_CSV  = "data/Apple Stock Price History.csv"
OUT_DIR   = "figs"
SHOW_FIGS = True
os.makedirs(OUT_DIR, exist_ok=True)

def load_investing_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "Date" not in df.columns:
        raise ValueError(f"'Date' column not found in {path}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    if "Price" not in df.columns:
        raise ValueError(f"'Price' column not found in {path}")
    df["Price"] = (df["Price"].astype(str)
                             .str.replace(",", "", regex=False)
                             .str.replace("%", "", regex=False))
    df["Adj Close"] = pd.to_numeric(df["Price"], errors="coerce")

    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            s = df[col].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
            df[col] = pd.to_numeric(s, errors="coerce")

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

    df = df.dropna(subset=["Adj Close"])
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    return df

def add_features(df):
    px = df["Adj Close"]
    df["MA5"]   = px.rolling(5).mean()
    df["MA20"]  = px.rolling(20).mean()
    df["MA60"]  = px.rolling(60).mean()
    df["Ret"]   = px.pct_change()
    df["LogRet"] = np.log(px / px.shift(1))             
    df["Vol20"] = df["LogRet"].rolling(20).std() * np.sqrt(252)
    cummax = px.cummax()
    df["Drawdown"] = px / cummax - 1.0
    return df

def find_crossovers(df, s="MA5", m="MA20"):
    prev = df[s].shift(1) - df[m].shift(1)
    now  = df[s] - df[m]
    golden = df.index[(prev < 0) & (now > 0)]
    dead   = df.index[(prev > 0) & (now < 0)]
    return golden, dead

def beta_and_corr(spx_ret, aapl_ret):
    ret = pd.concat([spx_ret.rename("SPX"), aapl_ret.rename("AAPL")], axis=1).dropna()
    slope, intercept, r, p, se = stats.linregress(ret["SPX"].values, ret["AAPL"].values)
    return slope, intercept, r, p, ret

def ma_slope(series, lookback=60):
    s = series.dropna().tail(lookback)
    if len(s) < 2:
        return np.nan
    x = np.arange(len(s))
    m, b = np.polyfit(x, s.values, 1)
    return m

def plot_price_ma(df, title, fname):
    plt.figure(figsize=(11,6))
    plt.plot(df.index, df["Adj Close"], color="#666666", label="Adj Close", alpha=0.9)
    plt.plot(df.index, df["MA5"],  label="MA5 (Short-term)",   linewidth=1.8, color="tab:blue")
    plt.plot(df.index, df["MA20"], label="MA20 (Medium-term)", linewidth=1.8, color="tab:green")
    plt.plot(df.index, df["MA60"], label="MA60 (Long-term)",   linewidth=1.8, color="tab:red")

    golden, dead = find_crossovers(df)
    plt.scatter(golden, df.loc[golden, "Adj Close"], marker="^", s=55,
                label="Golden Cross", color="tab:purple", zorder=3)
    plt.scatter(dead,   df.loc[dead,   "Adj Close"], marker="v", s=55,
                label="Dead Cross",   color="tab:orange", zorder=3)

    plt.title(title); plt.xlabel("Date"); plt.ylabel("Price")
    leg = plt.legend(title="Moving Averages & Crossovers")
    if leg: leg._legend_box.align = "left"
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200)
    if SHOW_FIGS: plt.show()
    plt.close()

def plot_beta(ret, beta, intercept, r, fname):
    plt.figure(figsize=(7,6))
    plt.scatter(ret["SPX"], ret["AAPL"], s=8, alpha=0.6, label="Daily returns")
    xline = np.linspace(ret["SPX"].min(), ret["SPX"].max(), 200)
    yline = intercept + beta * xline
    plt.plot(xline, yline, label=f"AAPL = {intercept:.4f} + {beta:.2f}·SPX")
    plt.scatter([], [], label=f"corr r = {r:.2f}")
    plt.title("AAPL vs. SPX Daily Returns (Beta)")
    plt.xlabel("SPX daily return"); plt.ylabel("AAPL daily return")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200)
    if SHOW_FIGS: plt.show()
    plt.close()

def plot_drawdown_vol(df, title, fname):
    fig, ax1 = plt.subplots(figsize=(11,6))
    c1, c2 = "tab:blue", "tab:orange"

    l1, = ax1.plot(df.index, df["Drawdown"], color=c1, label="Drawdown", alpha=0.9)
    ax1.set_ylabel("Drawdown", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.axhline(0, color=c1, linewidth=0.8, alpha=0.6)

    ax2 = ax1.twinx()
    l2, = ax2.plot(df.index, df["Vol20"], color=c2, label="Volatility (20d, annualized)", alpha=0.9)
    ax2.set_ylabel("Volatility", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)

    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper right")
    fig.suptitle(title)
    ax1.grid(True); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=200)
    if SHOW_FIGS: plt.show()
    plt.close(fig)

def maybe_candle_chart(df, title, fname):
    if not HAS_MPLFINANCE:
        return False
    needed = {"Open","High","Low","Close"}
    if not needed.issubset(df.columns):
        return False

    c = df[["Open","High","Low","Close"]].apply(pd.to_numeric, errors="coerce").dropna()
    if c.empty:
        return False

    ap = []
    for ma_col, color in [("MA5","tab:blue"), ("MA20","tab:green"), ("MA60","tab:red")]:
        if ma_col in df.columns:
            s = df[ma_col].reindex(c.index)
            if s.notna().sum() > 0:
                ap.append(mpf.make_addplot(s, color=color, width=1.2))

    save_path = os.path.join(OUT_DIR, fname)
    fig, axlist = mpf.plot(
        c, type="candle", addplot=ap, title=title, style="yahoo",
        returnfig=True, savefig=save_path
    )

    ax = axlist[0]
    ax.plot([], [], color="green", label="Bullish candle")
    ax.plot([], [], color="red",   label="Bearish candle")
    ax.plot([], [], color="tab:blue",  label="MA5 (Short-term)")
    ax.plot([], [], color="tab:green", label="MA20 (Medium-term)")
    ax.plot([], [], color="tab:red",   label="MA60 (Long-term)")
    ax.legend(loc="upper left", frameon=True, title="Legend")

    if SHOW_FIGS: plt.show()
    plt.close(fig)
    return True

def main():
    spx  = add_features(load_investing_csv(SP500_CSV))
    aapl = add_features(load_investing_csv(AAPL_CSV))

    beta, intercept, r, p, ret = beta_and_corr(spx["LogRet"], aapl["LogRet"])

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

    plot_price_ma(spx,  "S&P 500: Price + MAs + Crossovers", "spx_ma.png")
    plot_price_ma(aapl, "Apple (AAPL): Price + MAs + Crossovers", "aapl_ma.png")
    plot_beta(ret, beta, intercept, r, "beta_scatter.png")
    plot_drawdown_vol(spx,  "S&P 500: Drawdown & 20d Volatility", "spx_dd_vol.png")
    plot_drawdown_vol(aapl, "Apple: Drawdown & 20d Volatility",   "aapl_dd_vol.png")

    spx_candle_ok  = maybe_candle_chart(spx,  "S&P 500 Candlestick (with MAs)", "spx_candle.png")
    aapl_candle_ok = maybe_candle_chart(aapl, "Apple Candlestick (with MAs)",   "aapl_candle.png")

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

    saved = ["spx_ma.png", "aapl_ma.png", "beta_scatter.png", "spx_dd_vol.png", "aapl_dd_vol.png"]
    if spx_candle_ok:  saved.append("spx_candle.png")
    if aapl_candle_ok: saved.append("aapl_candle.png")

    print("\n".join(lines))
    print(f"\n[Saved figures] {os.path.abspath(OUT_DIR)} -> " + ", ".join(saved))

if __name__ == "__main__":
    main()
