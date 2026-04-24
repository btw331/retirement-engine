# -*- coding: utf-8 -*-
"""
退休規劃大師 — 50年長週期資產動態管理
Streamlit 版：精簡排版、學術說明、可調邊界條件
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
from urllib.parse import urlencode

# ── 台灣退休理財內容黑名單（使用者指定）───────────────────────────────
# 注意：此清單僅用於「內容引用/推薦」層，不影響任何計算引擎。
BLACKLIST_KOLS_TW: tuple[str, ...] = ("陳重銘", "施昇輝")


def _tw_kol_consensus_md() -> str:
    """
    2025–2026 台灣退休理財 KOL/作者常見共識（排除黑名單）。
    目的：把「可執行規則」對齊本 App 既有功能，降低資訊碎片化。
    """
    return """
### 2025–2026 台灣退休理財「共識版」規則（已排除黑名單）

以下整合（不含黑名單）常被正向引用的公開觀點：清流君、周冠男教授、綠角、市場先生、柴鼠兄弟、艾蜜莉等。  
本 App 的做法是把它們轉成**可執行規則**，並提供對應的工具與風險檢查。

- **用「總報酬」看退休現金流，不要把配息當成額外收入**
  - 配息是現金流形式，不是報酬來源；需要用錢可用「賣出部分份額」取得現金（心理帳戶需重新校準）。
  - 對應功能：`提領策略（固定 / 微笑曲線 / GK護欄）`＋`提領順序與稅務效率`（Tab3 主題 5）。

- **先處理 SORR（序列報酬風險），再追求提領率**
  - 退休前 5–10 年與退休後初期最敏感；現金緩衝與動態提領是核心防線。
  - 對應功能：`三桶金（Bucket Strategy）`（Tab3 主題 1）＋`熊市/股災應對手冊`（Tab3 主題 6）＋`蒙地卡羅成功率`。

- **偏好「動態提領」而非死守固定提領**
  - 常見做法：GK 護欄（繁榮/保全/通膨調整上限/晚年例外）讓支出與資產狀態同步調整。
  - 對應功能：`GK 護欄策略`（引擎已實裝）＋`成功率/風險情境矩陣`。

- **分散化是長壽期的結構性需求**
  - 避免單一市場/產業集中；用全球分散型 ETF（如 VWRA/VT 類）對沖地緣與產業風險。
  - 對應功能：`全球分散化`（Tab3 主題 9 / 10）＋資產結構推論報酬。

- **成本與摩擦不能忽略（費用率、換股、內扣成本）**
  - 「看起來一樣的指數/ETF」長期可能因費用與摩擦拉開差距；要回到公開說明書與實際費用率。
  - 對應功能：教育/指南中的 ETF 與再平衡章節（Tab3 主題 4、7）。

> **黑名單提醒**：本專案不引用/不推薦 {blacklisted} 的內容。
    """.format(blacklisted="、".join(BLACKLIST_KOLS_TW)).strip()

# ── 封閉公式解析解（用於驗算 strategy="fixed" 引擎精度）──────────────
def _analytical_fixed(A0, W, r, n):
    """
    固定提領期初扣款封閉解：
    每期 A = (A - W) × (1+r)，n 期後精確終值。
    """
    if r == 0:
        return max(0.0, A0 - W * n)
    val = (A0 - W) * (1 + r) ** n - W * ((1 + r) ** n - (1 + r)) / r
    return max(0.0, val)


def _qp_first(v, default: str | None = None) -> str | None:
    """
    Streamlit query params 在不同版本可能回傳 str 或 list[str]。
    這裡統一取第一個值。
    """
    if v is None:
        return default
    if isinstance(v, (list, tuple)):
        return str(v[0]) if v else default
    return str(v)


def _qs(**kwargs) -> str:
    """產生相對網址 query string，用於頁內導覽超連結。"""
    clean = {k: v for k, v in kwargs.items() if v is not None and v != ""}
    return "?" + urlencode(clean, doseq=False)


# ── URL 導覽：用 query 參數預選「分類/主題」（保持 UI 整潔、免改引擎）──────
try:
    _qp = st.query_params  # Streamlit 新版
except Exception:
    _qp = {}

_nav = _qp_first(getattr(_qp, "get", lambda *_: None)("nav"), None) if hasattr(_qp, "get") else None
if hasattr(_qp, "get"):
    _edu_cat_q = _qp_first(_qp.get("edu_category"), None)
    _edu_topic_q = _qp_first(_qp.get("edu_topic"), None)
    _guide_cat_q = _qp_first(_qp.get("guide_category"), None)
    _guide_topic_q = _qp_first(_qp.get("guide_topic"), None)
    _ins_topic_q = _qp_first(_qp.get("ins_topic"), None)
else:
    _edu_cat_q = _edu_topic_q = _guide_cat_q = _guide_topic_q = _ins_topic_q = None

if _edu_cat_q:
    st.session_state["edu_category"] = _edu_cat_q
if _edu_topic_q:
    st.session_state["edu_topic"] = _edu_topic_q
if _guide_cat_q:
    st.session_state["guide_category"] = _guide_cat_q
if _guide_topic_q:
    st.session_state["guide_topic"] = _guide_topic_q
if _ins_topic_q:
    st.session_state["ins_topic"] = _ins_topic_q

# 讓超連結可「100% 一鍵跳頁」：用 nav 決定主區顯示哪一頁
_nav_to_page_id = {
    "retire": "retire",
    "edu": "edu",
    "guide": "guide",
    "ins": "ins",
}
if _nav in _nav_to_page_id:
    st.session_state["page_id"] = _nav_to_page_id[_nav]

# ── 歷史年報酬資料（台灣加權指數 + 美股混合，1975–2024，實質報酬率 %）──
# 來源：台灣加權指數歷史資料、S&P500 CAPE 修正後實質報酬（以 TWD 計）
# 此資料用於 Bootstrap 重抽樣，保留歷史分布特性（序列相關性以年橫截面體現）
_HIST_REAL_RETURNS_PCT: list[float] = [
    -13.0,  22.5,  14.2,  -8.3,  31.0,  18.7,   5.4, -12.0,  42.8,  38.5,
     21.3, -55.0,  65.2,  14.0,  -4.5,  26.3,  18.9,  -8.0, -21.0, -16.0,
     30.5,  25.0,  12.3,  -6.2,  38.0,   4.5,  -3.0,  22.0,  17.5,  31.2,
    -38.0,  45.0,  16.8,  -5.5,  12.0,  24.5,   8.3,  -9.0,  35.0,  18.2,
     -3.5,  28.7,  11.5,   6.0,  -2.0,  19.0,  26.5, -12.5,  33.0,  16.2,
]

# ── 蒙地卡羅核心（NumPy 全向量化，無 Python 內層迴圈）────────────────
@st.cache_data
def _run_monte_carlo(A0, W0, r_mean_pct, r_std_pct,
                     n_years, strategy, pension_annual,
                     claim_age, age_start,
                     med_premium_pct=0.0,
                     dist_mode="normal",
                     t_df=7,
                     t_skew=0.0,
                     rental_annual=0.0,
                     rental_start_age=65,
                     rm_annual=0.0,
                     rm_start_age=999,
                     inflation_randomize=False,
                     inflation_mean_pct=2.0,
                     inflation_std_pct=0.8,
                     n_sim=10_000):
    """
    隨機報酬率模擬 n_sim 條路徑（NumPy 全向量化）。
    dist_mode="normal"    → 常態分布 N(r̄, σ²)
    dist_mode="t"         → Student-t(df)，縮放後保留相同 r̄ 與 σ，但有肥尾
                            t_skew ∈ [-1,0]：負值以兩段縮放引入左偏態（負報酬更極端）
    dist_mode="bootstrap" → 歷史「實質」年報酬直接重抽樣（保留偏態/肥尾；σ 由歷史資料決定）
    inflation_randomize   → True 時每年從 N(CPI均值, σ_CPI²) 抽取通膨，
                            並以「實質報酬 = 名目報酬 − 當年隨機通膨」把通膨不確定性納入路徑。
                            注意：啟用此模式時，呼叫端需把 r_mean_pct 以「名目均值」傳入（約 = 實質均值 + 通膨均值），
                            引擎才會在扣除 CPI 後回到一致的「實質」口徑。
    rental_annual / rental_start_age：一般租金收入及其啟動年齡。
    rm_annual / rm_start_age：以房養老年收入及其啟動年齡（兩者獨立判斷，不合併）。
    固定隨機種子（seed=42）確保相同參數可重現。
    """
    rng      = np.random.default_rng(seed=42)
    r_mean   = r_mean_pct / 100.0
    r_std    = r_std_pct  / 100.0
    med_rate = med_premium_pct / 100.0

    # ── 生成報酬率矩陣：shape (n_sim, n_years) ──────────────────────────
    if dist_mode == "t":
        # Student-t 原始樣本的方差 = df/(df-2)，需縮放使 Var = σ²
        scale_factor = np.sqrt(t_df / (t_df - 2))
        raw_t = rng.standard_t(df=t_df, size=(n_sim, n_years))
        if t_skew != 0.0:
            # 兩段縮放引入偏態（Fernandez-Steel 方法）：
            # 負值樣本乘以 (1 + |γ|)，正值樣本不變 → 左尾變重（負偏態）
            # 之後以樣本均值與標準差重新校正，確保 E[r]=r_mean、Std[r]=r_std
            skew_amp = abs(t_skew)
            raw_t = np.where(raw_t < 0, raw_t * (1.0 + skew_amp), raw_t)
            _t_mean = raw_t.mean()
            _t_std  = raw_t.std()
            raw_t   = (raw_t - _t_mean) / (_t_std if _t_std > 0 else 1.0)
        returns = r_mean + r_std * raw_t / scale_factor
    elif dist_mode == "bootstrap":
        # 歷史 Bootstrap 重抽樣：直接從歷史實質報酬序列中有放回抽樣
        # 保留歷史分布特性（肥尾、偏態），不強加參數化分布假設
        hist_arr = np.array(_HIST_REAL_RETURNS_PCT) / 100.0
        idx = rng.integers(0, len(hist_arr), size=(n_sim, n_years))
        returns = hist_arr[idx]
    else:
        returns = rng.normal(r_mean, r_std, (n_sim, n_years))

    # ── 通膨隨機化：名目報酬 − 隨機 CPI → 實質報酬 ─────────────────────
    # 方法論口徑：引擎以「實質購買力」計算為主。
    # 因此當 inflation_randomize=True 時，呼叫端會把 r_mean_pct 當成「名目均值」（約 = 實質 + 通膨均值），
    # 這裡每年扣掉隨機 CPI，讓最終 returns 回到一致的「實質報酬」口徑。
    # Bootstrap 來源本來就是「歷史實質報酬」，再扣 CPI 會混淆口徑，因此排除。
    if inflation_randomize and dist_mode != "bootstrap":
        infl_mean = inflation_mean_pct / 100.0
        infl_std  = inflation_std_pct  / 100.0
        cpi_draws = rng.normal(infl_mean, infl_std, (n_sim, n_years))
        # returns 目前為名目報酬；扣除隨機 CPI 得實質報酬
        returns = returns - cpi_draws

    A         = np.full(n_sim, float(A0))       # 每條路徑的資產餘額
    alive     = np.ones(n_sim, dtype=bool)       # 尚未歸零的路徑遮罩
    IWR       = W0 / A0 if A0 > 0 else 0.0
    gk_spend  = np.full(n_sim, float(W0))        # GK 每路徑追蹤的支出目標

    for j in range(n_years):
        current_age = age_start + j

        # ── 被動收入：勞保/勞退 + 租金 + 以房養老（各自獨立啟動年齡）──
        pension_income = (pension_annual
                          if pension_annual > 0 and current_age >= claim_age
                          else 0.0)
        rental_income  = (rental_annual
                          if rental_annual > 0 and current_age >= rental_start_age
                          else 0.0)
        rm_income      = (rm_annual
                          if rm_annual > 0 and current_age >= rm_start_age
                          else 0.0)
        passive_income = pension_income + rental_income + rm_income

        # ── 策略決定本期總目標支出 ────────────────────────────────────
        if strategy == "fixed":
            spend = np.full(n_sim, float(W0))

        elif strategy == "smile":
            # 線性平滑過渡取代離散跳躍（Blanchett 2014 原論文為平滑曲線）：
            # Go-Go（< 73 歲）：1.0；73–77 歲過渡帶線性降至 0.8；
            # Slow-Go（77–83 歲）：0.8；83–87 歲過渡帶線性升至 1.1；
            # No-Go（≥ 87 歲）：1.1
            if current_age < 73:
                mult = 1.0
            elif current_age <= 77:
                mult = 1.0 + (0.8 - 1.0) * (current_age - 73) / 4
            elif current_age < 83:
                mult = 0.8
            elif current_age <= 87:
                mult = 0.8 + (1.1 - 0.8) * (current_age - 83) / 4
            else:
                mult = 1.1
            spend = np.full(n_sim, float(W0) * mult)

        elif strategy == "gk":
            # GK 護欄（Guyton-Klinger 2006）全四條規則向量化版本：
            # 1. 繁榮法則：提領率 < IWR×0.8 → 加薪（上限 +6%，避免超額調漲）
            # 2. 保全法則：提領率 > IWR×1.2 → 減薪 10%（最後 15 年停用，避免晚年過度節縮）
            # 3. 通膨調整上限：繁榮法則加薪幅度不超過 6%
            # 4. 最後防線：剩餘年數 ≤ 15 時，停止保全法則（不再減薪）
            remaining = n_years - j
            safe    = A > 0
            cur_wr  = np.where(safe, gk_spend / np.where(safe, A, 1.0), 999.0)
            # 繁榮法則（通膨上限 6%）
            gk_spend = np.where(cur_wr < IWR * 0.8, gk_spend * 1.06, gk_spend)
            # 保全法則（最後 15 年停用）
            if remaining > 15:
                gk_spend = np.where(cur_wr > IWR * 1.2, gk_spend * 0.9, gk_spend)
            spend = gk_spend.copy()

        else:
            spend = np.full(n_sim, float(W0))

        # ── 醫療溢價指數複利（70 歲起，與主引擎邏輯一致）─────────────
        if current_age >= 70 and med_rate > 0:
            med_base  = W0 * 0.15
            # 基點改為 70 歲：70 歲時溢出從 0 開始，之後指數複利遞增
            med_extra = med_base * ((1 + med_rate) ** (current_age - 70) - 1)
            spend = spend + med_extra

        # ── 從有價證券提領 = 總支出 − 全部被動收入，僅對存活路徑 ──────
        spend_from_asset = np.where(alive, np.maximum(0.0, spend - passive_income), 0.0)

        # ── 期初提領保守原則：(A − spend) × (1 + r) ─────────────────
        A = (A - spend_from_asset) * (1.0 + returns[:, j])

        # 更新存活遮罩
        alive = alive & (A > 0)

    finals       = np.maximum(0.0, A)
    success_rate = float(alive.mean() * 100)
    p10 = float(np.percentile(finals, 10))
    p50 = float(np.percentile(finals, 50))
    p90 = float(np.percentile(finals, 90))
    return success_rate, p10, p50, p90


@st.cache_data
def _run_monte_carlo_failure_analysis(
    A0, W0, r_mean_pct, r_std_pct,
    n_years, strategy, pension_annual,
    claim_age, age_start,
    *,
    med_premium_pct=0.0,
    dist_mode="normal",
    t_df=7,
    t_skew=0.0,
    rental_annual=0.0,
    rental_start_age=65,
    rm_annual=0.0,
    rm_start_age=999,
    inflation_randomize=False,
    inflation_mean_pct=2.0,
    inflation_std_pct=0.8,
    inflation_assumed_pct=2.0,
    n_sim=10_000,
    track_years=5,
):
    """
    失敗路徑剖析：
    - 追蹤每條路徑何時歸零（ruin_age）
    - 產出每條路徑摘要（可下載 CSV）
    - 另外回傳少量路徑的逐年資產（供畫圖）

    重要：此分析以「實質購買力」為主計算引擎；名目資產僅用 inflation_assumed_pct 做換算顯示。
    """
    rng      = np.random.default_rng(seed=42)
    r_mean   = r_mean_pct / 100.0
    r_std    = r_std_pct  / 100.0
    med_rate = med_premium_pct / 100.0

    # ── 生成報酬率矩陣：shape (n_sim, n_years)
    if dist_mode == "t":
        scale_factor = np.sqrt(t_df / (t_df - 2))
        raw_t = rng.standard_t(df=t_df, size=(n_sim, n_years))
        if t_skew != 0.0:
            skew_amp = abs(t_skew)
            raw_t = np.where(raw_t < 0, raw_t * (1.0 + skew_amp), raw_t)
            _t_mean = raw_t.mean()
            _t_std  = raw_t.std()
            raw_t   = (raw_t - _t_mean) / (_t_std if _t_std > 0 else 1.0)
        returns = r_mean + r_std * raw_t / scale_factor
    elif dist_mode == "bootstrap":
        hist_arr = np.array(_HIST_REAL_RETURNS_PCT) / 100.0
        idx = rng.integers(0, len(hist_arr), size=(n_sim, n_years))
        returns = hist_arr[idx]
    else:
        returns = rng.normal(r_mean, r_std, (n_sim, n_years))

    if inflation_randomize and dist_mode != "bootstrap":
        infl_mean = inflation_mean_pct / 100.0
        infl_std  = inflation_std_pct  / 100.0
        cpi_draws = rng.normal(infl_mean, infl_std, (n_sim, n_years))
        returns = returns - cpi_draws

    A = np.full(n_sim, float(A0))
    alive = np.ones(n_sim, dtype=bool)
    ruin_age = np.full(n_sim, -1, dtype=int)
    IWR = W0 / A0 if A0 > 0 else 0.0
    gk_spend = np.full(n_sim, float(W0))

    # 記錄逐年資產（實質）；10k×50 約 4MB，足夠用於剖析與抽樣
    A_path = np.empty((n_sim, n_years), dtype=np.float32)

    for j in range(n_years):
        current_age = age_start + j

        pension_income = (pension_annual
                          if pension_annual > 0 and current_age >= claim_age
                          else 0.0)
        rental_income  = (rental_annual
                          if rental_annual > 0 and current_age >= rental_start_age
                          else 0.0)
        rm_income      = (rm_annual
                          if rm_annual > 0 and current_age >= rm_start_age
                          else 0.0)
        passive_income = pension_income + rental_income + rm_income

        if strategy == "fixed":
            spend = np.full(n_sim, float(W0))
        elif strategy == "smile":
            if current_age < 73:
                mult = 1.0
            elif current_age <= 77:
                mult = 1.0 + (0.8 - 1.0) * (current_age - 73) / 4
            elif current_age < 83:
                mult = 0.8
            elif current_age <= 87:
                mult = 0.8 + (1.1 - 0.8) * (current_age - 83) / 4
            else:
                mult = 1.1
            spend = np.full(n_sim, float(W0) * mult)
        elif strategy == "gk":
            remaining = n_years - j
            safe = A > 0
            cur_wr = np.where(safe, gk_spend / np.where(safe, A, 1.0), 999.0)
            gk_spend = np.where(cur_wr < IWR * 0.8, gk_spend * 1.06, gk_spend)
            if remaining > 15:
                gk_spend = np.where(cur_wr > IWR * 1.2, gk_spend * 0.9, gk_spend)
            spend = gk_spend.copy()
        else:
            spend = np.full(n_sim, float(W0))

        if current_age >= 70 and med_rate > 0:
            med_base  = W0 * 0.15
            med_extra = med_base * ((1 + med_rate) ** (current_age - 70) - 1)
            spend = spend + med_extra

        spend_from_asset = np.where(alive, np.maximum(0.0, spend - passive_income), 0.0)
        A = (A - spend_from_asset) * (1.0 + returns[:, j])

        # 新死亡：本期結束後 <=0
        newly_dead = alive & (A <= 0)
        ruin_age[newly_dead] = int(current_age)
        alive = alive & (A > 0)

        A_path[:, j] = np.maximum(0.0, A).astype(np.float32)

    finals = np.maximum(0.0, A)
    success = alive

    k = int(max(1, min(track_years, n_years)))
    first_k_cum = np.prod(1.0 + returns[:, :k], axis=1) - 1.0
    min_real = A_path.min(axis=1).astype(np.float64)

    infl_factor = (1.0 + float(inflation_assumed_pct) / 100.0) ** float(n_years)
    final_nominal = finals * infl_factor

    df = pd.DataFrame({
        "sim_id": np.arange(n_sim, dtype=int),
        "success": success.astype(bool),
        "ruin_age": np.where(success, np.nan, ruin_age.astype(float)),
        "final_real_ntd": finals.astype(np.float64),
        "final_nominal_ntd": final_nominal.astype(np.float64),
        "min_real_ntd": min_real,
        f"first_{k}y_cum_return": first_k_cum.astype(np.float64),
    })
    return df, A_path

def _fmt_asset(x):
    """將 NTD 金額格式化為 萬/億（金融顯示：千分位）或 歸零"""
    if x is None or x <= 0:
        return "歸零"
    if x >= 100_000_000:
        return f"{x/1e8:,.2f} 億"
    return f"{x/1e4:,.0f} 萬"

def _fmt_wan(n):
    """將數字以萬為單位、千分位顯示（金融顯示）"""
    if n is None or n < 0:
        return "0"
    return f"{n/1e4:,.0f} 萬"

def _fmt_currency(n):
    """整數金額千分位（例：30,000,000）"""
    if n is None or n < 0:
        return "0"
    return f"{int(n):,}"

def run_dynamic_projection(
    A0, W0, r_pct, n_years, start_age,
    strategy="fixed",
    med_premium_pct=0.0,
    pension_annual=0.0,
    claim_age=60,
    rental_annual=0.0,
    rental_start_age=65,
    rm_annual=0.0,
    rm_start_age=999,
):
    """
    動態提領模擬器 V2.2 — 五項科學修正：
      A. 消費微笑曲線（平滑過渡）：<73 歲 1.0；73–77 線性降至 0.8；77–83 為 0.8；83–87 線性升至 1.1；≥87 為 1.1
      B. 醫療溢價指數複利（70 歲起，以 70 歲為基點）：W₀×15% × ((1+rate)^(age-70) − 1)
      C. 期初提領保守原則：(A - spend) × (1+r)，先扣費再計息
      D. 租金現金流：rental_start_age 起每年減少從有價證券提領的金額
      E. 以房養老現金流：rm_start_age 起獨立計入（與租金分離，各自啟動年齡）
    """
    A            = float(A0)
    W_total      = float(W0)
    r            = r_pct / 100
    med_rate     = med_premium_pct / 100
    IWR          = W_total / A if A > 0 else 0.0
    gk_spend     = W_total

    for i in range(n_years):
        if A <= 0:
            return 0
        current_age = start_age + i

        # 本年度固定收入：勞保/勞退 + 租金 + 以房養老（各自獨立啟動年齡）
        pension_income = (pension_annual
                          if pension_annual > 0 and current_age >= int(claim_age)
                          else 0.0)
        rental_income  = (rental_annual
                          if rental_annual > 0 and current_age >= int(rental_start_age)
                          else 0.0)
        rm_income      = (rm_annual
                          if rm_annual > 0 and current_age >= int(rm_start_age)
                          else 0.0)
        passive_income = pension_income + rental_income + rm_income   # 合計被動收入

        # ── 1. 策略決定「總支出目標」 ──────────────────────────────
        if strategy == "fixed":
            total_spend = W_total

        elif strategy == "smile":
            # 線性平滑過渡（與蒙地卡羅引擎保持一致）
            if current_age < 73:
                mult = 1.0
            elif current_age <= 77:
                mult = 1.0 + (0.8 - 1.0) * (current_age - 73) / 4
            elif current_age < 83:
                mult = 0.8
            elif current_age <= 87:
                mult = 0.8 + (1.1 - 0.8) * (current_age - 83) / 4
            else:
                mult = 1.1
            total_spend = W_total * mult

        elif strategy == "gk":
            # GK 護欄四條完整規則（Guyton-Klinger 2006）：
            # 1. 繁榮法則：提領率 < IWR×0.8 → 加薪，上限 +6%（通膨調整上限）
            # 2. 保全法則：提領率 > IWR×1.2 → 減薪 10%（最後 15 年停用）
            remaining_yrs = n_years - i
            current_wr = gk_spend / A if A > 0 else 999.0
            if current_wr < IWR * 0.8:
                gk_spend *= 1.06   # 繁榮法則，上限 6%（原論文通膨調整上限）
            elif current_wr > IWR * 1.2 and remaining_yrs > 15:
                gk_spend *= 0.9    # 保全法則（最後 15 年停止減薪）
            total_spend = gk_spend

        else:
            total_spend = W_total

        # ── 2. 非線性醫療溢價（70 歲起指數遞增）─────────────────────
        if current_age >= 70 and med_rate > 0:
            med_base  = W_total * 0.15
            # 基點改為 70 歲：70 歲時溢出從 0 開始，之後指數複利遞增
            med_extra = med_base * ((1 + med_rate) ** (current_age - 70) - 1)
            total_spend += med_extra

        # ── 3. 從有價證券提領 = 總支出 − 全部被動收入 ───────────────
        spend_from_asset = max(0.0, total_spend - passive_income)

        # ── 4. 期初提領保守原則：先扣費再計息 ─────────────────────────
        A = (A - spend_from_asset) * (1 + r)

    return max(0, A)


@st.cache_data
def _solve_w0_to_zero_fixed(
    *,
    A0_eff: float,
    r_pct: float,
    n_years: int,
    age_start: int,
    med_premium_pct: float,
    pension_annual: float,
    claim_age: int,
    rental_annual: float,
    rental_start_age: int,
    rm_annual: float,
    rm_start_age: int,
    w0_guess: float,
) -> float:
    """
    求解「固定提領」下，W0（總支出目標）提高到哪裡會在目標年齡剛好歸零。
    - 使用 run_dynamic_projection（含被動收入/醫療溢價邏輯），以二分搜尋求解。
    - 回傳 W0_zero（NTD/年，實質購買力）。
    """
    if A0_eff <= 0 or n_years <= 0:
        return 0.0

    def f(w0_total: float) -> float:
        return float(run_dynamic_projection(
            A0_eff,
            float(w0_total),
            float(r_pct),
            int(n_years),
            int(age_start),
            strategy="fixed",
            med_premium_pct=float(med_premium_pct),
            pension_annual=float(pension_annual),
            claim_age=int(claim_age),
            rental_annual=float(rental_annual),
            rental_start_age=int(rental_start_age),
            rm_annual=float(rm_annual),
            rm_start_age=int(rm_start_age),
        ))

    # 先找上界 hi：使得 f(hi)=0（或足夠接近 0）
    lo = 0.0
    hi = max(float(w0_guess), 1.0)
    v_hi = f(hi)
    # 若 hi 太小，逐步放大
    for _ in range(40):
        if v_hi <= 0.0:
            break
        hi *= 1.5
        if hi > A0_eff * 20:  # 夠寬的上界，避免無限放大
            break
        v_hi = f(hi)

    # 若即使很大的 hi 仍不歸零，代表幾乎怎麼提都不會歸零（極罕見，通常是 W0=0 或 r 非常高）
    if f(hi) > 0.0:
        return hi

    # 二分搜尋
    for _ in range(60):
        mid = (lo + hi) / 2.0
        v_mid = f(mid)
        if v_mid > 0:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0)

st.set_page_config(
    page_title="退休規劃大師",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== 邊界條件：側邊欄 ==========
st.sidebar.header("🔧 邊界條件設定")
st.sidebar.caption("調整參數後，下方摘要與指標會即時更新。")

def _clamp_int(x: int, lo: int, hi: int) -> int:
    return int(min(hi, max(lo, int(x))))


def _clamp_float(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, float(x))))


def _estimate_cost_drag_pct(
    *,
    product_mode: str,
    expense_ratio_pct: float,
    friction_drag_pct: float,
    dividend_yield_pct: float,
    dividend_tax_pct: float,
) -> float:
    """與引擎相同的「年拖累%」合計（費用+摩擦+配息稅，簡化估算）。"""
    div_tax = (
        float(dividend_yield_pct) * (float(dividend_tax_pct) / 100.0)
        if str(product_mode).startswith("配息型")
        else 0.0
    )
    return float(expense_ratio_pct) + float(friction_drag_pct) + div_tax


# ── Wizard：一次只顯示一段（方向 3）────────────────────────────────────
wizard_mode = st.sidebar.toggle(
    "🧭 使用引導（Wizard）",
    value=True,
    help="一次只顯示一小段設定，3 步驟完成輸入；降低新手的操作負擔。",
)

# 統一預設值（Wizard/進階模式都會用到；避免 NameError）
A0_securities_wan = 3000
W0_wan = 120
inflation_pct = 2.0
medical_premium = 1.7
age_start = 40
age_end = 90
use_inferred_r = "依資產結構推論"
r_pct = None

product_mode = "累積型（不配息）"
expense_ratio_pct = 0.20
friction_drag_pct = 0.05
dividend_yield_pct = 2.0
dividend_tax_pct = 30.0

use_asset_alloc = True
use_re_inputs = False
use_pension_inputs = False
use_cost_drag = True

# 不動產/收入預設（避免 NameError）
include_re = False
re_home_wan = re_rental_wan = re_mortgage_wan = 0
re_net_wan = 0
re_liquidity_discount = 20
re_net_wan_eff = 0
rental_monthly_wan = 0.0
rental_start_age_input = 65
rm_start_age = 999
rm_monthly_wan = 0.0

pension_monthly_wan = 0.0
claim_age = 60


def _render_wizard() -> None:
    st.session_state.setdefault("wiz_step", 1)
    step = int(st.session_state["wiz_step"])
    step = 1 if step < 1 else 3 if step > 3 else step
    st.session_state["wiz_step"] = step

    st.sidebar.markdown(f"**Step {step} / 3**")
    st.sidebar.progress(step / 3)

    # Step 1 ───────────────────────────────────────────────────────────
    if step == 1:
        with st.sidebar.expander("Step 1｜生活費與年齡", expanded=True):
            st.session_state["age_start"] = st.number_input("起始年齡 (歲)", 25, 70, int(st.session_state.get("age_start", age_start)), 1, key="wiz_age_start")
            st.session_state["age_end"] = st.number_input(
                "目標年齡 (歲)",
                70,
                100,
                int(st.session_state.get("age_end", age_end)),
                1,
                key="wiz_age_end",
            )
            st.session_state["W0_wan"] = st.number_input(
                "年度生活費 W₀（實質，萬/年）",
                min_value=10.0,
                max_value=500.0,
                value=float(st.session_state.get("W0_wan", W0_wan)),
                step=5.0,
                key="wiz_w0",
            )
            st.session_state["inflation_pct"] = st.slider("通膨率 CPI (%)", 0.0, 8.0, float(st.session_state.get("inflation_pct", inflation_pct)), 0.5, key="wiz_cpi")
            st.session_state["medical_premium"] = st.slider(
                "醫療溢價 i_m (CPI + %)",
                0.0,
                4.0,
                float(st.session_state.get("medical_premium", medical_premium)),
                0.1,
                key="wiz_med",
            )

        c1, c2 = st.sidebar.columns(2)
        with c2:
            if st.button("下一步 ▶", key="wiz_next_1", use_container_width=True):
                st.session_state["wiz_step"] = 2
                st.rerun()

    # Step 2 ───────────────────────────────────────────────────────────
    elif step == 2:
        with st.sidebar.expander("Step 2｜資產負債與固定現金流", expanded=True):
            st.markdown("**資產負債（先算淨金融資產）**")
            assets = st.number_input(
                "金融資產（萬）",
                min_value=0,
                max_value=50_000,
                value=int(st.session_state.get("a0_assets_wan", 3000)),
                step=100,
                key="wiz_a_assets",
            )
            liab = st.number_input(
                "金融負債（信貸/卡債等，萬）",
                min_value=0,
                max_value=50_000,
                value=int(st.session_state.get("a0_liab_wan", 0)),
                step=50,
                key="wiz_a_liab",
            )
            st.session_state["a0_assets_wan"] = assets
            st.session_state["a0_liab_wan"] = liab
            net_fin = max(0, int(assets - liab))
            st.metric("淨金融資產（帶入引擎）", f"{net_fin:,} 萬")

            st.markdown("---")
            st.session_state["use_re_inputs"] = st.checkbox("我有不動產/房貸/租金/以房養老", value=bool(st.session_state.get("use_re_inputs", False)), key="wiz_use_re")
            if st.session_state["use_re_inputs"]:
                st.caption("小提醒：不動產屬於進階題，若你只是先看退休可行性，可先略過這段。")
                with st.expander("🏠 不動產（選填）", expanded=False):
                    st.markdown("**房產與房貸**")
                    st.session_state["re_home_wan"] = st.number_input("自用住宅市值 (萬)", 0, 20_000, int(st.session_state.get("re_home_wan", 0)), 100, key="wiz_re_home")
                    st.session_state["re_rental_wan"] = st.number_input("出租房產市值 (萬)", 0, 20_000, int(st.session_state.get("re_rental_wan", 0)), 100, key="wiz_re_rental")
                    st.session_state["guide_mortgage_wan"] = st.number_input("房貸餘額 (萬)", 0, 20_000, int(st.session_state.get("guide_mortgage_wan", 0)), 50, key="wiz_re_mort")
                    st.session_state["re_liquidity_discount"] = st.slider("流動性折扣 (%)", 0, 50, int(st.session_state.get("re_liquidity_discount", 20)), 1, key="wiz_re_disc")

                    st.markdown("**租金/以房養老（可選）**")
                    st.session_state["rental_monthly_wan"] = st.number_input(
                        "月租金淨收入 (萬/月)",
                        0.0,
                        100.0,
                        float(st.session_state.get("rental_monthly_wan", 0.0)),
                        0.5,
                        key="wiz_rent_m",
                        help=(
                            "請填「實際可拿來補生活費的到手金額」（淨現金流），而不是合約租金。"
                            "建議先用教育庫的『出租房產淨收益／租金折現』試算；若不確定，可先用『毛租金×0.75』作保守淨值。"
                        ),
                    )
                    st.caption(
                        "保守速算：淨租金（月）≈ 房產市值 × 年淨殖利率(1%～2.5%) ÷ 12。"
                        f"　→ 去做租金折現試算：[教育庫｜不動產收益護欄（Income Floor）]"
                        f"({_qs(nav='edu', edu_category='房產／資產負債表／心理帳戶', edu_topic='16｜不動產收益護欄：Income Floor 與折現風險調整')})"
                    )
                    st.session_state["rental_start_age_input"] = st.number_input(
                        "租金開始年齡 (歲)", 40, 85, int(st.session_state.get("rental_start_age_input", 65)), 1, key="wiz_rent_age"
                    )
                    use_rm = st.checkbox("啟用以房養老", value=bool(st.session_state.get("use_rm", False)), key="wiz_use_rm")
                    st.session_state["use_rm"] = use_rm
                    if use_rm:
                        st.session_state["rm_start_age"] = st.number_input("以房養老啟動年齡 (歲)", 60, 90, int(st.session_state.get("rm_start_age", 80)), 1, key="wiz_rm_age")
                        st.session_state["rm_monthly_wan"] = st.number_input("以房養老月領 (萬/月)", 0.0, 50.0, float(st.session_state.get("rm_monthly_wan", 3.0)), 0.5, key="wiz_rm_m")

            st.markdown("---")
            st.session_state["use_pension_inputs"] = st.checkbox("我有勞保/勞退月領", value=bool(st.session_state.get("use_pension_inputs", False)), key="wiz_use_pen")
            if st.session_state["use_pension_inputs"]:
                st.session_state["pension_monthly_wan"] = st.number_input(
                    "勞保＋勞退 月領 (萬/月)", 0.0, 50.0, float(st.session_state.get("pension_monthly_wan", 0.0)), 0.5, key="wiz_pen_m"
                )
                st.session_state["claim_age"] = st.number_input("請領年齡 (歲)", 55, 70, int(st.session_state.get("claim_age", 60)), 1, key="wiz_claim")

        c1, c2 = st.sidebar.columns(2)
        with c1:
            if st.button("◀ 上一步", key="wiz_back_2", use_container_width=True):
                st.session_state["wiz_step"] = 1
                st.rerun()
        with c2:
            if st.button("下一步 ▶", key="wiz_next_2", use_container_width=True):
                st.session_state["wiz_step"] = 3
                st.rerun()

    # Step 3 ───────────────────────────────────────────────────────────
    else:
        with st.sidebar.expander("Step 3｜報酬假設、成本拖累與策略", expanded=True):
            st.session_state["use_inferred_r"] = st.radio("實質報酬率 r 來源", ["依資產結構推論", "手動設定"], index=0, key="wiz_rsrc")
            if st.session_state["use_inferred_r"] == "手動設定":
                st.session_state["r_pct_manual"] = st.slider("實質報酬 r（毛，%）", 0.0, 15.0, float(st.session_state.get("r_pct_manual", 4.0)), 0.5, key="wiz_rmanual")
            else:
                st.markdown("**快速資產結構（用於推論 r）**")
                scenario = st.radio("報酬情境", ["保守", "中性", "積極"], index=0, horizontal=True, key="wiz_scn")
                r_table = {
                    "保守": (6.0, 5.0, 5.0, 4.0, 4.5),
                    "中性": (7.0, 6.5, 6.5, 5.5, 5.5),
                    "積極": (9.0, 8.5, 8.5, 7.0, 7.5),
                }
                r_us_stock, r_us_etf, r_tw_stock, r_tw_etf, r_global = r_table[scenario]
                pct_us_e = st.slider("美股ETF (%)", 0, 100, int(st.session_state.get("pct_us_e", 20)), 5, key="wiz_pct_us_e")
                pct_tw_e = st.slider("台股ETF (%)", 0, 100, int(st.session_state.get("pct_tw_e", 40)), 5, key="wiz_pct_tw_e")
                pct_glb = st.slider("全球ETF (%)", 0, 100, int(st.session_state.get("pct_glb", 10)), 5, key="wiz_pct_glb")
                pct_tw_s = st.slider("台股個股 (%)", 0, 100, int(st.session_state.get("pct_tw_s", 30)), 5, key="wiz_pct_tw_s")
                pct_us_s = max(0, 100 - (pct_us_e + pct_tw_e + pct_glb + pct_tw_s))
                st.caption(f"美股個股（自動補足）={pct_us_s}%（合計 100%）")
                inferred = (pct_us_s * r_us_stock + pct_us_e * r_us_etf + pct_tw_s * r_tw_stock + pct_tw_e * r_tw_etf + pct_glb * r_global) / 100
                st.session_state["r_pct_inferred"] = float(inferred)
                st.metric("推論實質報酬 r（毛）", f"{inferred:.2f}%")

            st.markdown("---")
            st.session_state["use_cost_drag"] = st.checkbox(
                "用「淨報酬」模擬（扣費用／摩擦／配息稅）",
                value=bool(st.session_state.get("use_cost_drag", True)),
                key="wiz_use_drag",
                help=(
                    "勾選時：從上方實質報酬（毛）再扣掉內扣成本與簡化的配息稅拖累，"
                    "較貼近長期實拿。取消勾選則毛報酬直接進引擎（偏樂觀）。"
                ),
            )
            if st.session_state["use_cost_drag"]:
                st.caption("多數人不必改細項；需要時展開「細項滑桿」即可。")
                with st.expander("📝 白話說明（可摺疊）", expanded=False):
                    st.markdown(
                        """
- **費用率**：ETF／基金內扣，約等於每年從報酬裡少掉這一段。
- **摩擦**：手續費、追蹤誤差、換匯與再平衡等，保守抓一點即可。
- **配息型＋稅**：配息會當所得課稅；這裡用「配息率 × 稅率」**粗略**估每年拖累（非報稅試算）。
- **累積型**：報酬留在淨值內、不配現金，此處**不**加配息稅拖累。
"""
                    )
                with st.expander("進階：細項滑桿（多數人用預設即可）", expanded=False):
                    st.session_state["product_mode"] = st.radio(
                        "配息模式",
                        ["累積型（不配息）", "配息型（配息會被課稅）"],
                        index=0 if not str(st.session_state.get("product_mode", "")).startswith("配息型") else 1,
                        key="wiz_pm",
                        help="累積型：不配息、不估算配息稅拖累。配息型：會依下方配息率與稅率加算拖累。",
                    )
                    st.session_state["expense_ratio_pct"] = st.slider(
                        "費用率拖累（%/年）",
                        0.0,
                        2.0,
                        float(st.session_state.get("expense_ratio_pct", 0.20)),
                        0.05,
                        key="wiz_er",
                        help="對照基金公開說明書「總費用率」；台股大盤 ETF 多為 0.1–0.5%。",
                    )
                    st.session_state["friction_drag_pct"] = st.slider(
                        "摩擦拖累（%/年）",
                        0.0,
                        1.0,
                        float(st.session_state.get("friction_drag_pct", 0.05)),
                        0.01,
                        key="wiz_fr",
                        help="交易、滑價、追蹤誤差等；被動長抱可設低一點。",
                    )
                    st.session_state["dividend_yield_pct"] = st.slider(
                        "配息率（%/年）",
                        0.0,
                        10.0,
                        float(st.session_state.get("dividend_yield_pct", 2.0)),
                        0.5,
                        key="wiz_dy",
                        help="僅在「配息型」時用來估算配息稅拖累；不是報酬加成。",
                    )
                    st.session_state["dividend_tax_pct"] = st.slider(
                        "配息稅負（%）",
                        0.0,
                        40.0,
                        float(st.session_state.get("dividend_tax_pct", 30.0)),
                        1.0,
                        key="wiz_dt",
                        help="綜所稅邊際稅率或預扣的簡化比例；實務依個案與年度所得而不同。",
                    )
                _pm_w = str(st.session_state.get("product_mode", product_mode))
                _er_w = float(st.session_state.get("expense_ratio_pct", expense_ratio_pct))
                _fr_w = float(st.session_state.get("friction_drag_pct", friction_drag_pct))
                _dy_w = float(st.session_state.get("dividend_yield_pct", dividend_yield_pct))
                _dt_w = float(st.session_state.get("dividend_tax_pct", dividend_tax_pct))
                _drag_w = _estimate_cost_drag_pct(
                    product_mode=_pm_w,
                    expense_ratio_pct=_er_w,
                    friction_drag_pct=_fr_w,
                    dividend_yield_pct=_dy_w,
                    dividend_tax_pct=_dt_w,
                )
                st.caption(
                    f"目前合計年拖累約 **{_drag_w:.2f}%**（費用 {_er_w:.2f}% + 摩擦 {_fr_w:.2f}%"
                    + (
                        f" + 配息稅拖累約 {_dy_w * (_dt_w / 100.0):.2f}%）"
                        if _pm_w.startswith("配息型")
                        else "；累積型不計配息稅）"
                    )
                )

            st.markdown("---")
            st.session_state["strategy_choice"] = st.radio("策略", ["固定提領", "消費微笑曲線", "GK 護欄"], index=2, horizontal=True, key="wiz_strat")

        c1, c2 = st.sidebar.columns(2)
        with c1:
            if st.button("◀ 上一步", key="wiz_back_3", use_container_width=True):
                st.session_state["wiz_step"] = 2
                st.rerun()
        with c2:
            if st.button("完成 ✅", key="wiz_done", use_container_width=True):
                # 注意：不可寫入與 widget key 同名的 session_state（會觸發 StreamlitAPIException）
                st.session_state["wiz_completed"] = True
                st.success("已完成 Wizard，右側結果已更新。")


if wizard_mode:
    _render_wizard()

    # Step 1
    age_start = int(st.session_state.get("age_start", age_start))
    age_end = int(st.session_state.get("age_end", age_end))
    W0_wan = float(st.session_state.get("W0_wan", W0_wan))
    inflation_pct = float(st.session_state.get("inflation_pct", inflation_pct))
    medical_premium = float(st.session_state.get("medical_premium", medical_premium))

    # Step 2
    A0_securities_wan = max(0, int(st.session_state.get("a0_assets_wan", 3000) - st.session_state.get("a0_liab_wan", 0)))
    use_re_inputs = bool(st.session_state.get("use_re_inputs", False))
    if use_re_inputs:
        include_re = True  # Wizard 勾選即視為納入，細部仍可在 Step2 補
        re_home_wan = int(st.session_state.get("re_home_wan", 0))
        re_rental_wan = int(st.session_state.get("re_rental_wan", 0))
        re_mortgage_wan = int(st.session_state.get("guide_mortgage_wan", 0))
        re_liquidity_discount = int(st.session_state.get("re_liquidity_discount", 20))
        rental_monthly_wan = float(st.session_state.get("rental_monthly_wan", 0.0))
        rental_start_age_input = int(st.session_state.get("rental_start_age_input", 65))
        if bool(st.session_state.get("use_rm", False)):
            rm_start_age = int(st.session_state.get("rm_start_age", 80))
            rm_monthly_wan = float(st.session_state.get("rm_monthly_wan", 3.0))

    use_pension_inputs = bool(st.session_state.get("use_pension_inputs", False))
    if use_pension_inputs:
        pension_monthly_wan = float(st.session_state.get("pension_monthly_wan", 0.0))
        claim_age = int(st.session_state.get("claim_age", 60))

    # Step 3
    use_inferred_r = str(st.session_state.get("use_inferred_r", use_inferred_r))
    if use_inferred_r == "手動設定":
        r_pct = float(st.session_state.get("r_pct_manual", 4.0))
    else:
        r_pct = float(st.session_state.get("r_pct_inferred", 4.0))
    use_cost_drag = bool(st.session_state.get("use_cost_drag", True))
    if use_cost_drag:
        product_mode = str(st.session_state.get("product_mode", product_mode))
        expense_ratio_pct = float(st.session_state.get("expense_ratio_pct", expense_ratio_pct))
        friction_drag_pct = float(st.session_state.get("friction_drag_pct", friction_drag_pct))
        dividend_yield_pct = float(st.session_state.get("dividend_yield_pct", dividend_yield_pct))
        dividend_tax_pct = float(st.session_state.get("dividend_tax_pct", dividend_tax_pct))
    else:
        # 取消「淨報酬」時應以毛報酬進引擎，勿沿用預設費用率
        product_mode = "累積型（不配息）"
        expense_ratio_pct = 0.0
        friction_drag_pct = 0.0
        dividend_yield_pct = 0.0
        dividend_tax_pct = 0.0

else:
    # 進階模式：關閉 Wizard 時顯示「全部選項」
    use_asset_alloc = True
    use_re_inputs = True
    use_pension_inputs = True

    with st.sidebar.expander("資產與提領（進階）", expanded=True):
        A0_securities_wan = st.number_input(
            "有價證券總額 (萬)",
            min_value=0,
            max_value=50_000,
            value=int(A0_securities_wan),
            step=100,
            help="股票、ETF、基金等流動性金融資產，以萬為單位",
        )
        W0_wan = st.number_input(
            "實質購買力 (萬/年)",
            min_value=10.0,
            max_value=500.0,
            value=float(W0_wan),
            step=5.0,
            help="起始年全年生活費目標（含所有支出），以萬/年為單位",
        )

    with st.sidebar.expander("報酬與通膨（進階）", expanded=True):
        inflation_pct = st.slider("預期通膨率 CPI (%)", 0.0, 8.0, float(inflation_pct), 0.5)
        use_inferred_r = st.radio("實質報酬率 r 來源", ["依資產結構推論", "手動設定"], index=0)
        if use_inferred_r == "手動設定":
            r_pct = st.slider("預期實質報酬率 r（毛，%）", 0.0, 15.0, 4.0, 0.5)
        else:
            r_pct = None
        medical_premium = st.slider("醫療溢價 i_m (CPI + %)", 0.0, 4.0, float(medical_premium), 0.1)

    with st.sidebar.expander("成本／稅務拖累（進階，落實到引擎）", expanded=False):
        use_cost_drag = st.checkbox(
            "用「淨報酬」模擬（扣費用／摩擦／配息稅）",
            value=True,
            key="adv_use_cost_drag",
            help="與 Wizard 相同：勾選才從實質報酬（毛）扣除下方拖累；取消則毛報酬直接進引擎。",
        )
        st.caption(
            "淨實質報酬 ＝ 毛實質報酬 −（費用率 + 摩擦 + 配息稅拖累）。配息稅拖累 ＝ 配息率 ×（稅率÷100），僅在「配息型」時列入。"
        )
        with st.expander("📝 白話說明", expanded=False):
            st.markdown(
                """
- **內扣費用**：每年從報酬中扣掉的百分比，可對照 ETF 總費用率。
- **摩擦**：交易、追蹤誤差等保守加總。
- **配息型**：用配息率與稅率**粗估**配息相關稅負對報酬的拖累（非報稅試算）。
- **累積型**：不配息現金流，此處不計配息稅拖累。
"""
            )
        if use_cost_drag:
            product_mode = st.radio(
                "ETF/基金配息模式",
                ["累積型（不配息）", "配息型（配息會被課稅）"],
                index=0 if not str(product_mode).startswith("配息型") else 1,
                key="adv_product_mode",
                help="決定是否把「配息×稅率」算進年拖累。",
            )
            expense_ratio_pct = st.slider(
                "內扣費用率拖累（%/年）",
                0.0,
                2.0,
                float(expense_ratio_pct),
                0.05,
                help="公開說明書之總費用率等。",
            )
            friction_drag_pct = st.slider(
                "交易/換股/追蹤誤差摩擦（%/年）",
                0.0,
                1.0,
                float(friction_drag_pct),
                0.01,
                help="手續費、滑價、再平衡頻率等。",
            )
            dividend_yield_pct = st.slider(
                "配息率（%/年，用於估算稅務拖累）",
                0.0,
                10.0,
                float(dividend_yield_pct),
                0.5,
                help="僅配息型會列入「配息率×稅率」；不是額外報酬。",
            )
            dividend_tax_pct = st.slider(
                "配息稅負/扣繳率（%）",
                0.0,
                40.0,
                float(dividend_tax_pct),
                1.0,
                help="簡化為單一有效稅率；實務依所得結構不同。",
            )
            _adv_drag = _estimate_cost_drag_pct(
                product_mode=product_mode,
                expense_ratio_pct=expense_ratio_pct,
                friction_drag_pct=friction_drag_pct,
                dividend_yield_pct=dividend_yield_pct,
                dividend_tax_pct=dividend_tax_pct,
            )
            st.caption(f"目前合計年拖累約 **{_adv_drag:.2f}%**。")
        else:
            product_mode = "累積型（不配息）"
            expense_ratio_pct = 0.0
            friction_drag_pct = 0.0
            dividend_yield_pct = 0.0
            dividend_tax_pct = 0.0

    with st.sidebar.expander("年齡區間（進階）", expanded=True):
        age_start = st.number_input("起始年齡 (歲)", 25, 70, int(age_start), 1)
        age_end = st.number_input("目標年齡 (歲)", 70, 100, int(age_end), 1)

# ── 不動產（選填）───────────────────────────────────────────────────────
if not wizard_mode:
    include_re = False
    re_home_wan = re_rental_wan = re_mortgage_wan = 0
    re_net_wan = 0
    re_liquidity_discount = 20
    re_net_wan_eff = 0
    rental_monthly_wan = 0.0
    rental_start_age_input = 65
    rm_start_age = 999
    rm_monthly_wan = 0.0

if (not wizard_mode) and use_re_inputs:
    with st.sidebar.expander("🏠 不動產（選填）", expanded=True):
        st.caption(
            "定位：房產淨值屬於**非流動性安全墊**，不會用投資報酬率 r 納入複利；"
            "只有『租金／以房養老』會在各自啟動年齡起，降低每年需要從有價證券提領的金額。"
        )
        include_re = st.toggle("將不動產納入計算", value=False,
                               help="開啟後：顯示房產折後淨值作為備援安全墊（不加入 A₀ 複利）；租金/以房養老會自動抵銷部分年度提領。")
        re_home_wan    = st.number_input(
            "自用住宅市值 (萬)",
            min_value=0, max_value=20_000, value=0, step=100,
            help="自住房屋目前市值，退休後居住成本已鎖定（不計入可提領現金流）",
        )
        re_rental_wan  = st.number_input(
            "出租房產市值 (萬)",
            min_value=0, max_value=20_000, value=0, step=100,
            help="出租物件的目前市值",
        )
        _mort_default = int(min(20_000, max(0, int(st.session_state.get("guide_mortgage_wan", 0)))))
        re_mortgage_wan = st.number_input(
            "未償房貸餘額 (萬)",
            min_value=0, max_value=20_000, value=_mort_default, step=50,
            help="所有房產尚未還清的貸款餘額，自動從淨資產中扣除",
        )
        re_net_wan     = max(0, re_home_wan + re_rental_wan - re_mortgage_wan)

        re_liquidity_discount = st.slider(
            "流動性折扣 (%)",
            min_value=0, max_value=50, value=20, step=1,
            help=(
                "這不是『房價看跌』，而是房產變現時的總摩擦成本估算，含：仲介費 2–6%、代書費、"
                "房地合一稅 15–45%（依持有年數）、搬遷費等。"
                "持有 10 年以上稅率 15%，建議最低折扣 18–20%；"
                "持有 2–5 年稅率 35%，建議折扣 40–45%。"
            ),
        )
        re_net_wan_eff = int(re_net_wan * (1 - re_liquidity_discount / 100))
        if re_net_wan > 0:
            st.caption(
                f"房產市值淨值（市值 − 貸款）：**{re_net_wan:,} 萬**　"
                f"→ 折後可用金額（−{re_liquidity_discount}%）：**{re_net_wan_eff:,} 萬**"
            )
        else:
            st.caption("房產淨值：**0 萬**")

        st.markdown("**租金收入**")
        rental_monthly_wan = st.number_input(
            "月租金淨收入 (萬/月)",
            min_value=0.0, max_value=100.0, value=0.0, step=0.5,
            help=(
                "請填「實際到手、可用來補生活費的淨現金流」：已扣除空置、維修、管理費、房屋稅/地價稅等成本後。"
                "若你手上只有合約租金（毛），建議先用教育庫的『出租房產淨收益計算機』或『租金可靠性折現（75% Income Floor）』估算後再填。"
                "不確定時可先用『毛租金×0.75』作保守淨值。"
            ),
        )
        st.caption(
            "保守速算：淨租金（月）≈ 房產市值 × 年淨殖利率(1%～2.5%) ÷ 12。"
            f"　→ 試算工具：[教育庫｜出租房產淨收益計算機]"
            f"({_qs(nav='edu', edu_category='房產／資產負債表／心理帳戶', edu_topic='14｜不動產收益：租金、殖利率與 REITs')})"
            f"｜[教育庫｜租金折現（Income Floor）]"
            f"({_qs(nav='edu', edu_category='房產／資產負債表／心理帳戶', edu_topic='16｜不動產收益護欄：Income Floor 與折現風險調整')})"
        )
        # 可信度提示：以「出租房產市值」推估淨殖利率（避免把毛租金誤填為淨租金）
        if re_rental_wan > 0 and rental_monthly_wan > 0:
            _net_yield_est = (rental_monthly_wan * 12) / float(re_rental_wan) * 100
            _yield_note = (
                "⚠️ 偏低（可能合理：台北精華區常見）" if _net_yield_est < 1.0 else
                "✅ 常見區間" if _net_yield_est <= 2.5 else
                "⚠️ 偏高，請確認是否已扣空置/維修/稅費" if _net_yield_est <= 3.5 else
                "❗ 異常偏高，極可能把毛租金當淨租金"
            )
            st.caption(f"淨殖利率（估）≈ **{_net_yield_est:.2f}%**　{_yield_note}")
        rental_start_age_input = st.number_input(
            "租金開始年齡 (歲)",
            min_value=40, max_value=85, value=int(65), step=1,
            help="若物件已出租可設等於起始年齡；尚未出租可設未來預計年齡",
        )

        st.markdown("**以房養老（選填）**")
        use_reverse_mortgage = st.toggle(
            "啟用以房養老",
            value=False,
            help="建議視為 80+ 或長照期的『末端流動性保險』：達到啟動年齡時將房屋抵押給銀行，換取每月固定收入直至身故（台灣各行 2025 年利率約 2.16–4%）",
        )
        if use_reverse_mortgage:
            rm_start_age   = st.number_input("以房養老啟動年齡 (歲)", min_value=60, max_value=90, value=80, step=1)
            rm_monthly_wan = st.number_input(
                "以房養老月領 (萬/月)",
                min_value=0.0, max_value=50.0, value=3.0, step=0.5,
                help="依房產市值與銀行方案估算，一般約為房產價值 × 0.2–0.3% / 月",
            )
        else:
            rm_start_age   = 999
            rm_monthly_wan = 0.0

# ── 將「萬」換算為引擎單位（NTD/年）────────────────────────────────────
A0 = float(A0_securities_wan) * 10_000  # 有價證券（淨金融資產）
W0 = float(W0_wan) * 10_000            # 年生活費目標（實質購買力）

# 計算不動產對 A₀ 與租金現金流的貢獻（以折後淨值 re_net_wan_eff 計算）
re_net_value   = re_net_wan_eff * 10_000   # 已套用流動性折扣
rental_annual  = rental_monthly_wan * 12 * 10_000   # NTD/年（租金，獨立啟動年齡）
rm_annual      = rm_monthly_wan     * 12 * 10_000   # NTD/年（以房養老，獨立啟動年齡）

# 租金與以房養老保持獨立，各自有不同啟動年齡，不再合併為單一參數
# rental_start_age_input：租金啟動年齡
# rm_start_age：以房養老啟動年齡（已在 expander 內定義）

# 若不納入計算，所有不動產數值歸零
if not include_re:
    re_net_value  = 0.0
    rental_annual = 0.0
    rm_annual     = 0.0

A0_re  = re_net_value if include_re else 0.0
# A0_eff 僅以有價證券計算，房產淨值不加入複利成長基礎
# 原因：房產為非流動性資產，其成長率與投資組合不同，
#       混入 A0 等同假設房產每年以相同 r 複利成長，高估資產規模。
# 房產折後淨值（re_net_wan_eff）獨立顯示為「不動產安全墊」，
# 供末端流動性備援參考，不納入蒙地卡羅引擎的報酬計算。
A0_eff = A0   # 有效初始資產 = 有價證券（不含房產）

if include_re and re_net_wan > 0:
    st.sidebar.caption(
        f"不動產折後淨值（安全墊，不加入 A₀）：**{re_net_wan_eff:,} 萬**"
        f"（原始 {re_net_wan:,} 萬，折扣 {re_liquidity_discount}%）　"
        f"| 有價證券 A₀ = **{(A0/1e4):,.0f} 萬**"
    )

# ── 有價證券資產配置（Section 1）────────────────────────────────────────
# Wizard 模式下僅在 Step 2 顯示（避免每一步都被大量參數干擾）
_wiz_step = int(st.session_state.get("wiz_step", 1)) if wizard_mode else 0
_show_asset_alloc = (not wizard_mode) or (_wiz_step == 2)
inferred_r = float(st.session_state.get("r_pct_inferred", 4.0))  # 若未顯示配置區，提供安全預設
if _show_asset_alloc:
    with st.sidebar.expander("📈 1. 有價證券配置", expanded=use_asset_alloc):
        asset_input_mode = st.radio(
            "輸入方式",
            ["填寫實際金額 (萬)", "填寫比例 (%)"],
            index=0,
            horizontal=True,
        )

        # 三段式報酬情境預設
        return_scenario = st.radio(
            "報酬情境假設",
            ["保守", "中性", "積極"],
            index=0,
            horizontal=True,
            help=(
                "保守：低於歷史均值，適合悲觀情境規劃\n"
                "中性：約歷史均值折半，平衡考量\n"
                "積極：接近歷史長期年化實質報酬\n\n"
                "歷史參考：0050 名目年化約11.6%（2003-2024）；"
                "VTI 名目年化約12.3%（近10年）；均扣除通膨後為實質報酬"
            ),
        )
        _r_table = {
            #            美股個股  美股ETF  台股個股  台股ETF  全球ETF
            "保守":   (   6.0,    5.0,    5.0,    4.0,    4.5),
            "中性":   (   7.0,    6.5,    6.5,    5.5,    5.5),
            "積極":   (   9.0,    8.5,    8.5,    7.0,    7.5),
        }
        r_us_stock, r_us_etf, r_tw_stock, r_tw_etf, r_global = _r_table[return_scenario]
        st.caption(
            f"各類實質報酬假設 — 美股個股 **{r_us_stock}%** · 美股ETF **{r_us_etf}%** · "
            f"台股個股 **{r_tw_stock}%** · 台股ETF **{r_tw_etf}%** · 全球ETF **{r_global}%**"
        )

        if asset_input_mode == "填寫實際金額 (萬)":
            amt_us_stock = st.number_input("美股個股 (萬)",        min_value=0, max_value=100_000, value=0,    step=50)
            amt_us_etf   = st.number_input("美股 ETF  (萬)",       min_value=0, max_value=100_000, value=600,  step=50)
            amt_tw_stock = st.number_input("台股個股 (萬)",        min_value=0, max_value=100_000, value=900,  step=50)
            amt_tw_etf   = st.number_input("台股 ETF  (萬)",       min_value=0, max_value=100_000, value=1500, step=50)
            amt_global   = st.number_input("全球 ETF (VWRA/VT) (萬)", min_value=0, max_value=100_000, value=0, step=50,
                                           help="VWRA（全球市場 ETF）或 VT 等全球分散型基金，涵蓋 50+ 國家，降低單一市場集中度風險")
            total_amt = amt_us_stock + amt_us_etf + amt_tw_stock + amt_tw_etf + amt_global
            if total_amt > 0:
                pct_us_s  = amt_us_stock / total_amt * 100
                pct_us_e  = amt_us_etf   / total_amt * 100
                pct_tw_s  = amt_tw_stock / total_amt * 100
                pct_tw_e  = amt_tw_etf   / total_amt * 100
                pct_glb   = amt_global   / total_amt * 100
            else:
                pct_us_s = pct_us_e = pct_tw_s = pct_tw_e = pct_glb = 20.0
            st.caption(f"合計：**{total_amt:,} 萬**（A₀ 以上方「初始資產」為準）")
        else:
            pct_us_stock = st.slider("美股個股 (%)",           0, 100, 0,  5)
            pct_us_etf   = st.slider("美股 ETF  (%)",          0, 100, 20, 5)
            pct_tw_stock = st.slider("台股個股 (%)",           0, 100, 30, 5)
            pct_tw_etf   = st.slider("台股 ETF  (%)",          0, 100, 40, 5)
            pct_global   = st.slider("全球 ETF (VWRA/VT) (%)", 0, 100, 10, 5,
                                     help="VWRA/VT 等全球分散型，涵蓋美國以外的已開發與新興市場")
            total_pct = pct_us_stock + pct_us_etf + pct_tw_stock + pct_tw_etf + pct_global
            if total_pct != 100:
                st.caption(f"⚠️ 目前加總 {total_pct}%，建議為 100%。以下推論依比例換算。")
            scale    = 100 / total_pct if total_pct > 0 else 1
            pct_us_s = pct_us_stock * scale
            pct_us_e = pct_us_etf   * scale
            pct_tw_s = pct_tw_stock * scale
            pct_tw_e = pct_tw_etf   * scale
            pct_glb  = pct_global   * scale

        inferred_r = (pct_us_s * r_us_stock + pct_us_e * r_us_etf +
                      pct_tw_s * r_tw_stock + pct_tw_e * r_tw_etf +
                      pct_glb  * r_global) / 100
        st.caption(f"推論實質報酬率：**{inferred_r:.1f}%**（加權平均）")

# 資產結構計算完成後，填入推論報酬率
if r_pct is None:
    r_pct = round(inferred_r, 1)

# ── 成本/稅務拖累：落實到引擎的「淨實質報酬」─────────────────────────
r_pct_gross = float(r_pct)
div_tax_drag_pct = (dividend_yield_pct * (dividend_tax_pct / 100.0)) if product_mode.startswith("配息型") else 0.0
r_drag_pct = float(expense_ratio_pct) + float(friction_drag_pct) + float(div_tax_drag_pct)
r_pct_net = r_pct_gross - r_drag_pct

# 下游一律使用「淨實質報酬」進入引擎
r_pct = r_pct_net

st.sidebar.caption(
    f"名目報酬（毛）≈ **{r_pct_gross + inflation_pct:.1f}%**（實質毛 {r_pct_gross:.2f}% + 通膨 {inflation_pct}%）"
)
st.sidebar.caption(
    f"淨實質報酬（引擎用）= **{r_pct:.2f}%** ＝ 毛 {r_pct_gross:.2f}% − 拖累 {r_drag_pct:.2f}%"
    + ("（含配息稅負）" if product_mode.startswith("配息型") else "（不含配息稅負）")
)
pension_monthly_wan = 0.0
claim_age = 60
if (not wizard_mode) and use_pension_inputs:
    with st.sidebar.expander("勞保／勞退（選填）", expanded=False):
        st.caption("填寫後，從請領年齡起「實質購買力」改由勞保＋勞退支應一部分，自有資產提領減少。")
        pension_monthly_wan = st.number_input(
            "勞保＋勞退 月領 (萬/月)",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=0.5,
            help="預期勞保老年給付＋勞退月領合計（選填）",
        )
        claim_age = st.number_input(
            "勞保/勞退 請領年齡 (歲)",
            min_value=55,
            max_value=70,
            value=60,
            step=1,
            help="勞退 60 歲可請領；勞保年金請領年齡逐步延後至 65 歲",
        )
pension_annual = pension_monthly_wan * 12 * 10_000  # NTD/年
has_pension    = pension_annual > 0
has_rental     = (include_re and (rental_annual > 0 or rm_annual > 0))

# 被動收入合計（從起始年齡已在請領者）
_passive_at_start = 0.0
if has_pension and age_start >= int(claim_age):
    _passive_at_start += pension_annual
if include_re and rental_annual > 0 and age_start >= int(rental_start_age_input):
    _passive_at_start += rental_annual
if include_re and rm_annual > 0 and age_start >= int(rm_start_age):
    _passive_at_start += rm_annual

W0_asset = max(0.0, W0 - _passive_at_start)   # 從有價證券提領的淨額

# 衍生：IWR（以「從有價證券提領的淨額」為準）
IWR      = (W0_asset / A0_eff) * 100 if A0_eff > 0 else 0
gk_lower = IWR * 0.8
gk_upper = IWR * 1.2

# ========== 主區：導覽（支援 URL 一鍵跳頁） ==========
_PAGES = [
    ("retire", "📊 退休規劃"),
    ("edu", "📚 教育資訊庫"),
    ("guide", "🛠️ 提領實務指南"),
    ("ins", "🛡️ 保險規劃（參考）"),
    ("changelog", "🗒️ 更新紀錄"),
]
_label_by_id = {pid: label for pid, label in _PAGES}
_id_by_label = {label: pid for pid, label in _PAGES}

_default_page_id = str(st.session_state.get("page_id", "retire"))
if _default_page_id not in _label_by_id:
    _default_page_id = "retire"
_default_label = _label_by_id[_default_page_id]
page_label = st.radio(
    "導覽",
    [label for _, label in _PAGES],
    index=[label for _, label in _PAGES].index(_default_label),
    horizontal=True,
    key="page_label",
    label_visibility="collapsed",
)
page_id = _id_by_label[page_label]
st.session_state["page_id"] = page_id

# ──────────────────────────────────────────────
# PAGE：退休規劃（原有內容）
# ──────────────────────────────────────────────
if page_id == "retire":
    st.title("退休規劃大師")
    st.caption(f"50 年長週期退休財務工程與資產動態管理 (2026–2076) · 金額以 2026 實質購買力計價，預設通膨 {inflation_pct}%")
    st.info(
        "**口徑統一（全站一致）**\n\n"
        "- **實質（Real）**：以 **2026 年購買力**計價（本工具所有核心計算/引擎與大多數指標的口徑）。\n"
        "- **名目（Nominal）**：未折算的未來金額；本工具僅在顯示時，用你設定的 **CPI 複利**把實質換算成名目作為直覺參考。\n"
        "- **報酬率 r**：除非特別標註，皆指 **實質報酬率**；若在蒙地卡羅啟用「通膨隨機化」，呼叫端會以**名目均值**傳入，並在引擎內每年扣除隨機 CPI 回到實質路徑。\n"
        "- **成功率（蒙地卡羅）**：定義為 P(A_end > 0)（在目標年齡時資產仍大於 0 的比例）。"
    )

    # ── 核心結果（最重要：成功率 / 終值 / 歸零臨界提領率）────────────
    n_years = max(1, age_end - age_start)
    _strat_label = str(st.session_state.get("strategy_choice", "GK 護欄"))
    _strat_map_ui = {"固定提領": "fixed", "消費微笑曲線": "smile", "GK 護欄": "gk"}
    _current_strat = _strat_map_ui.get(_strat_label, "gk")
    _kw_core = dict(
        pension_annual=float(pension_annual),
        claim_age=int(claim_age),
        rental_annual=float(rental_annual),
        rental_start_age=int(rental_start_age_input),
        rm_annual=float(rm_annual),
        rm_start_age=int(rm_start_age),
    )

    # 1) 到目標年齡剩餘資產（確定性基準路徑；含醫療溢價與被動收入）
    _final_base = float(run_dynamic_projection(
        A0_eff,
        W0,
        float(r_pct),
        int(n_years),
        int(age_start),
        strategy=_current_strat,
        med_premium_pct=float(medical_premium),
        **_kw_core,
    ))

    # 2) 成功率（蒙地卡羅：標準假設；用現行策略）
    _mc_sr, _mc_p10, _mc_p50, _mc_p90 = _run_monte_carlo(
        A0_eff,
        W0,
        float(r_pct),
        15.0,  # σ 固定為標準模式 15%
        int(n_years),
        _current_strat,
        float(pension_annual),
        int(claim_age),
        int(age_start),
        med_premium_pct=float(medical_premium),
        dist_mode="normal",
        rental_annual=float(rental_annual),
        rental_start_age=int(rental_start_age_input),
        rm_annual=float(rm_annual),
        rm_start_age=int(rm_start_age),
        inflation_randomize=False,
        inflation_mean_pct=float(inflation_pct),
        inflation_std_pct=0.8,
    )

    # 3) 固定提領下「剛好歸零」的臨界提領率（確定性上限）
    _w0_zero = _solve_w0_to_zero_fixed(
        A0_eff=float(A0_eff),
        r_pct=float(r_pct),
        n_years=int(n_years),
        age_start=int(age_start),
        med_premium_pct=float(medical_premium),
        pension_annual=float(pension_annual),
        claim_age=int(claim_age),
        rental_annual=float(rental_annual),
        rental_start_age=int(rental_start_age_input),
        rm_annual=float(rm_annual),
        rm_start_age=int(rm_start_age),
        w0_guess=float(W0),
    )
    _iwr_zero = (max(0.0, _w0_zero - float(_passive_at_start)) / float(A0_eff) * 100) if A0_eff > 0 else 0.0

    with st.container(border=True):
        st.subheader("核心結果（最重要）")
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("退休成功率（標準蒙地卡羅）", f"{_mc_sr:.1f}%", "用 1 萬次市場情境估算（標準假設）")
        with k2:
            _infl_factor = (1.0 + float(inflation_pct) / 100.0) ** float(n_years)
            _final_nominal = float(_final_base) * _infl_factor
            st.metric(
                f"{age_end} 歲剩餘資產（基準）",
                _fmt_asset(_final_nominal),
                f"折算回 2026 年購買力：約 {_fmt_asset(_final_base)}",
            )
        with k3:
            st.metric("固定提領：剛好歸零的臨界 IWR", f"{_iwr_zero:.2f}%", f"換成生活費：約每年 {_fmt_wan(_w0_zero)}")
        st.caption("提示：『臨界 IWR』是確定性上限（固定提領、固定報酬假設）；實務仍應看蒙地卡羅成功率與壓力測試。")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("初始提領率 IWR", f"{IWR:.2f}%", "安全邊界 4%" if IWR < 4 else "高於 4%")
    with c2:
        st.metric("繁榮護欄 (加薪觸發)", f"{gk_lower:.2f}%", "資產漲多時加薪 10%")
    with c3:
        st.metric("保全護欄 (減薪觸發)", f"{gk_upper:.2f}%", "資產跌多時減薪 10%")
    with c4:
        st.metric(
            "實質報酬 r（淨）",
            f"{r_pct:.2f}%",
            f"毛 {r_pct_gross:.2f}% − 拖累 {r_drag_pct:.2f}% · 通膨 {inflation_pct}% · 醫療+{medical_premium}%"
            + (" · 推論" if use_inferred_r == "依資產結構推論" else " · 手動"),
        )

    # ── 下一步建議（把指標轉成可行動的調整）────────────────────────────
    _next_steps: list[str] = []
    if IWR >= 4.0:
        _next_steps.append("IWR 偏高：優先降低 **W₀**、延後目標年齡/退休時點，或提高可投資資產 **A₀**。")
    else:
        _next_steps.append("IWR 在可控區：建議以 **GK 護欄**作為預設策略，讓市場好壞自動調節支出。")
    if not has_pension and not (include_re and (rental_annual > 0 or rm_annual > 0)):
        _next_steps.append("底層現金流（Income Floor）偏弱：可評估勞保/勞退、租金（保守折現後）或其他穩定收入，以降低有效提領率。")
    if include_re and rental_monthly_wan > 0:
        _next_steps.append("已填租金：請確認你填的是『淨租金』而非毛租金；不確定時先用『毛×0.75』保守化，或到教育庫做折現試算。")
    if _next_steps:
        st.info("**下一步建議（最少動作）**\n\n" + "\n".join([f"- {s}" for s in _next_steps[:3]]))

    cond_rows = [
        ("A₀ 初始資產", f"NTD {_fmt_currency(A0)}（{_fmt_wan(A0)}）"),
        ("實質購買力 W₀", f"NTD {_fmt_currency(W0)}/年（{_fmt_wan(W0)}）"),
    ]
    if has_pension:
        cond_rows.append(("勞保＋勞退 年領", f"{_fmt_wan(pension_annual)}（{pension_monthly_wan} 萬/月 × 12）"))
        cond_rows.append(("請領年齡起從資產提領", f"{_fmt_wan(max(0, W0 - pension_annual))}/年"))
    cond_rows += [
        ("IWR（從資產）", f"{IWR:.2f}%"),
        ("預期通膨 CPI", f"{inflation_pct}%"),
        ("實質報酬 r（毛→淨）", f"{r_pct_gross:.2f}% → {r_pct:.2f}%" + (" (推論)" if use_inferred_r == "依資產結構推論" else " (手動)")),
        ("醫療溢價", f"CPI+{medical_premium}%"),
        ("起始/目標年齡", f"{age_start} → {age_end} 歲"),
    ]
    cond_df = pd.DataFrame(cond_rows, columns=["參數", "數值"])
    st.dataframe(cond_df, use_container_width=True, hide_index=True)
    st.divider()

    # ── 執行摘要 ──
    st.subheader("執行摘要")
    if IWR < 2.0:
        safety_label = "遠低於"
        safety_note  = "系統具備**財務熱力學溢位**：核心威脅已由「生存風險」轉向「資本效率低下（無效資產堆積）」，規劃重心應移往消費優化。"
    elif IWR < 4.0:
        safety_label = "低於"
        safety_note  = "系統處於**安全區**：具備一定緩衝空間，建議以 GK 護欄動態管理提領，避免過度保守導致晚年資源閒置。"
    else:
        safety_label = "接近或超過"
        safety_note  = "系統處於**臨界區或危險區**：IWR 達到或超過 4% 安全閾值，退休初期若遇市場低迷（SORR），路徑崩潰風險顯著上升，強烈建議提高準備金或降低提領。"
    A_star = (W0 * (1 + r_pct / 100) / (r_pct / 100)) if r_pct > 0 else float("inf")
    survivability = "✅ 在目前報酬率下資產均衡點 **{:.0f} 萬** 高於初始資產，系統能長期維持（確定性路徑）。".format(A_star / 1e4) \
                    if A_star > A0 else \
                    "⚠️ 資產均衡點 **{:.0f} 萬** 低於初始資產，固定提領在目前報酬率下資產將逐漸耗盡；建議採用 GK 護欄或提高報酬率假設。".format(A_star / 1e4)
    st.markdown(f"""
- **系統屬性**：初始提領率 **{IWR:.2f}%**，{safety_label}學界公認的 4% 安全閾值。{safety_note}
- **核心建議**：採用 **蓋頓–克林格 (GK) 護欄策略**，以負回饋調節在資產增值時觸發繁榮法則、強制加薪，將複利轉化為高品質生命體驗與生物資本投資。
- **數學驗算**：{survivability}
- **風險**：建議未來每月儲蓄可全數配置全球分散型 ETF（如 VWRA），以「地理熵減」應對地緣風險。
    """.strip())
    st.info("**台灣情境提醒**：退休年齡越早、規劃年數越長，4% 法則風險越高。50 歲前退休建議提高準備金或降低提領率，並以蒙地卡羅評估「成功機率」。")
    st.divider()

    # ── 資產結構 ──
    # Wizard 模式下，資產結構的細節僅在 Step 2 顯示（避免 Step 1/3 因未渲染配置區而 NameError）
    if (not wizard_mode) or (int(st.session_state.get("wiz_step", 1)) == 2):
        st.subheader("資產結構現況")
        if asset_input_mode == "填寫實際金額 (萬)":
            _amts = [amt_us_stock, amt_us_etf, amt_tw_stock, amt_tw_etf]
            _amt_col = [f"{v:,} 萬" for v in _amts]
        else:
            _amt_col = [
                f"{A0 * pct_us_s / 100 / 10_000:,.1f} 萬",
                f"{A0 * pct_us_e / 100 / 10_000:,.1f} 萬",
                f"{A0 * pct_tw_s / 100 / 10_000:,.1f} 萬",
                f"{A0 * pct_tw_e / 100 / 10_000:,.1f} 萬",
            ]
        df_asset = pd.DataFrame({
            "類別": ["美股個股", "美股 ETF", "台股個股", "台股 ETF"],
            "實際金額": _amt_col,
            "佔比 (%)": [round(pct_us_s, 1), round(pct_us_e, 1), round(pct_tw_s, 1), round(pct_tw_e, 1)],
            "假設實質報酬": [f"{r_us_stock}%", f"{r_us_etf}%", f"{r_tw_stock}%", f"{r_tw_etf}%"],
        })
        st.dataframe(df_asset, use_container_width=True, hide_index=True)
        st.markdown(
            f"**推論實質報酬率**：**{inferred_r:.1f}%**（依上表占比與假設報酬加權平均）　"
            f"｜ 情境：**{return_scenario}**（可於左側切換）"
        )
        st.caption(
            "歷史參考：0050 名目年化約 11.6%（2003–2024）；VTI 名目年化約 9.5%（2005–2024，含2008危機）。"
            "（名目僅供直覺對照；本工具輸入/計算以**實質報酬**為主。）"
            "扣除通膨（約 2%）後，長期實質常見落在約 7–10%；保守情境可取歷史的一半，中性取 60–65%，積極取 80%。"
            "注意：算術加權平均略高於幾何平均（方差拖累約 0.5–1.1%），長期規劃建議偏向保守情境。"
        )
    else:
        st.subheader("資產結構現況")
        st.info("在 Wizard 模式下，資產結構細節會在 **Step 2** 顯示。")

    # 不動產貢獻摘要（若已啟用）
    _re_total_income = rental_annual + rm_annual
    if include_re and (re_net_wan > 0 or _re_total_income > 0):
        re_cols = st.columns(3)
        with re_cols[0]:
            st.metric("不動產安全墊（折後）", f"{re_net_wan_eff:,} 萬",
                      f"備援流動性（不加入投資 A₀）")
        with re_cols[1]:
            _income_desc = []
            if rental_annual > 0:
                _income_desc.append(f"租金 {_fmt_wan(rental_annual)}/年（{rental_start_age_input} 歲起）")
            if rm_annual > 0:
                _income_desc.append(f"以房養老 {_fmt_wan(rm_annual)}/年（{rm_start_age} 歲起）")
            st.metric("不動產年收入（合計）", _fmt_wan(_re_total_income) + "/年",
                      "；".join(_income_desc) if _income_desc else "尚無收入")
        with re_cols[2]:
            _cover_pct = _re_total_income / W0 * 100 if W0 > 0 else 0
            st.metric("生活費覆蓋率", f"{_cover_pct:.0f}%",
                      "不動產收入 ÷ 年生活費目標")
        st.caption("🏠 租金與以房養老各依獨立啟動年齡計入引擎，每年從有價證券的淨提領額將自動降低。")
    st.divider()

    # ── 壓力測試 ──
    st.subheader("壓力測試 (90 歲時剩餘資產)")
    n_years = max(1, age_end - age_start)
    pension_note  = f" · 勞保＋勞退 {_fmt_wan(pension_annual)}/年（{claim_age} 歲起）" if has_pension else ""
    rental_note   = f" · 租金 {_fmt_wan(rental_annual)}/年（{rental_start_age_input} 歲起）" if (include_re and rental_annual > 0) else ""
    rm_note       = f" · 以房養老 {_fmt_wan(rm_annual)}/年（{rm_start_age} 歲起）" if (include_re and rm_annual > 0) else ""
    _re_safety_note = f"（+ 不動產安全墊 {_fmt_wan(A0_re)}，備援用，未加入複利計算）" if A0_re > 0 else ""
    st.markdown(
        f"**有價證券 A₀ {_fmt_wan(A0)}{_re_safety_note}"
        f" · 年生活費目標 {_fmt_wan(W0)}{pension_note}{rental_note}{rm_note}"
        f" · 實質報酬 {r_pct}% · 規劃 {n_years} 年**"
    )
    # 情境矩陣以「淨實質報酬」為基礎做上下震盪
    r_low  = max(-5.0,  r_pct - 1.5)
    r_mid  = r_pct
    r_high = min(15.0, r_pct + 1.5)
    _kw = dict(pension_annual=pension_annual, claim_age=int(claim_age),
               rental_annual=rental_annual, rental_start_age=int(rental_start_age_input),
               rm_annual=rm_annual, rm_start_age=int(rm_start_age))
    col_low = [
        run_dynamic_projection(A0_eff, W0, r_low,  n_years, age_start, strategy="fixed", **_kw),
        run_dynamic_projection(A0_eff, W0, r_low,  n_years, age_start, strategy="smile", **_kw),
        run_dynamic_projection(A0_eff, W0, r_low,  n_years, age_start, strategy="gk",    **_kw),
    ]
    col_mid = [
        run_dynamic_projection(A0_eff, W0, r_mid,  n_years, age_start, strategy="fixed", **_kw),
        run_dynamic_projection(A0_eff, W0, r_mid,  n_years, age_start, strategy="smile", **_kw),
        run_dynamic_projection(A0_eff, W0, r_mid,  n_years, age_start, strategy="gk",    **_kw),
    ]
    col_high = [
        run_dynamic_projection(A0_eff, W0, r_high, n_years, age_start, strategy="fixed", **_kw),
        run_dynamic_projection(A0_eff, W0, r_high, n_years, age_start, strategy="smile", **_kw),
        run_dynamic_projection(A0_eff, W0, r_high, n_years, age_start, strategy="gk",    **_kw),
    ]
    _infl_factor_end = (1.0 + float(inflation_pct) / 100.0) ** float(n_years)

    def _fmt_cell(x):
        """合併顯示：90歲名目剩餘資產 + 折算回2026購買力"""
        if x is None or x <= 0:
            return "歸零"
        x_nom = float(x) * float(_infl_factor_end)
        return f"{_fmt_asset(x_nom)}（折算回2026：{_fmt_asset(x)}）"

    st.caption("格式說明：各格顯示「**90歲名目剩餘資產**（折算回2026：實質購買力）」；折算以你設定的 CPI 複利估算。")
    matrix_a = pd.DataFrame({
        "策略": ["固定提領 (實質購買力/年)", "消費微笑曲線", "GK 護欄"],
        f"悲觀｜實質 {r_low}%": [_fmt_cell(x) for x in col_low],
        f"基準｜實質 {r_mid}%": [_fmt_cell(x) for x in col_mid],
        f"樂觀｜實質 {r_high}%": [_fmt_cell(x) for x in col_high],
    })
    st.dataframe(matrix_a, use_container_width=True, hide_index=True)

    st.subheader("多重風險疊加情境")
    r_down = max(0, r_pct - 1)
    _kw_med = dict(strategy="gk", pension_annual=pension_annual, claim_age=int(claim_age),
                   rental_annual=rental_annual, rental_start_age=int(rental_start_age_input),
                   rm_annual=rm_annual, rm_start_age=int(rm_start_age))
    res_base  = run_dynamic_projection(A0_eff, W0, r_pct,  n_years, age_start, med_premium_pct=medical_premium,       **_kw_med)
    res_inf   = run_dynamic_projection(A0_eff, W0, r_down, n_years, age_start, med_premium_pct=medical_premium,       **_kw_med)
    res_med   = run_dynamic_projection(A0_eff, W0, r_pct,  n_years, age_start, med_premium_pct=medical_premium + 0.5, **_kw_med)
    res_multi = run_dynamic_projection(A0_eff, W0, r_down, n_years, age_start, med_premium_pct=medical_premium + 0.5, **_kw_med)
    scenarios = [
        ("基準",
         f"實質報酬 {r_pct}%、通膨 {inflation_pct}%、醫療溢價 CPI+{medical_premium}%",
         "依目前參數之基準路徑",
         _fmt_cell(res_base)),
        ("通膨/報酬下修",
         f"實質報酬 {r_pct}% → {r_down:.1f}%",
         "購買力侵蝕或市場低迷，複利減弱",
         _fmt_cell(res_inf)),
        ("醫療溢價激增",
         f"醫療溢價 CPI+{medical_premium}% → +{medical_premium + 0.5:.1f}%",
         "晚年護理支出非線性飆升（指數複利疊加）",
         _fmt_cell(res_med)),
        ("多重風險疊加",
         "報酬-1% 且 醫療溢價+0.5% 同時發生",
         "最壞情境，GK 護欄極限抗壓能力測試",
         _fmt_cell(res_multi)),
    ]
    df_risk = pd.DataFrame(scenarios, columns=["情境", "假設", "預估影響", "90歲剩餘資產（名目；括號=折算回2026）"])
    st.dataframe(df_risk, use_container_width=True, hide_index=True)
    st.caption("格式：90歲名目剩餘資產（折算回2026：實質購買力）。GK 護欄 + 指數複利醫療溢價；建議搭配流動性緩衝因應最壞情境。")
    st.divider()

    # ── 模型驗證 ──
    st.subheader("模型驗證")

    # 封閉公式驗算
    with st.expander("🔬 驗證 1：封閉公式對照（引擎精度檢查）", expanded=False):
        st.markdown("""
**固定提領（strategy=fixed）** 具備精確解析解，可驗算動態引擎的浮點誤差：

```
A_n = (A₀ − W) × (1+r)ⁿ − W × [(1+r)ⁿ − (1+r)] / r   （期初提領，r ≠ 0）
A_n = A₀ − W × n                                         （r = 0 時）
```
        """)
        # 驗算使用純有價證券 A0（不含房產），與封閉公式對象一致
        _sim_val = run_dynamic_projection(A0, W0, r_pct, n_years, age_start,
                                          strategy="fixed",
                                          med_premium_pct=0.0,
                                          pension_annual=0, claim_age=int(claim_age),
                                          rental_annual=0, rental_start_age=999)
        _ana_val = _analytical_fixed(A0, W0, r_pct / 100, n_years)
        _err_abs = abs(_sim_val - _ana_val)
        _err_pct = (_err_abs / _ana_val * 100) if _ana_val > 0 else 0.0

        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            st.metric("動態引擎輸出", _fmt_asset(_sim_val))
        with vc2:
            st.metric("封閉公式解", _fmt_asset(_ana_val))
        with vc3:
            st.metric("計算誤差", f"{_err_pct:.8f}%")

        if _err_pct < 0.001:
            st.success(f"✅ 驗算通過：誤差 {_err_pct:.2e}%，引擎計算邏輯正確。")
        else:
            st.error(f"⚠️ 誤差 {_err_pct:.4f}%，超過容許值，請檢查引擎邏輯。")
        st.caption(
            f"驗算對象：有價證券 A₀ = {_fmt_asset(A0)}（不含房產，確保與封閉公式一致）。"
            "不含勞保/勞退補貼（pension_annual=0）、不含醫療溢價（med=0）。"
            "消費微笑曲線與 GK 護欄為非線性動態模型，無封閉解，請用下方蒙地卡羅驗證。"
        )

    # 蒙地卡羅模擬
    with st.expander("🎲 驗證 2：蒙地卡羅模擬（成功機率 & SORR 量化）", expanded=False):
        st.markdown("""
**蒙地卡羅法**：將固定 r 改為「隨機報酬率」，模擬 **10,000 條**不同市場路徑，
統計「目標年齡時資產 > 0」的比例，並輸出 P10 / P50 / P90 三段分位數。

此法補充確定性模型無法量化的 **SORR（序列報酬風險）**，是業界最標準的退休規劃驗證方法。
        """)

        mc_mode = st.radio(
            "模擬模式",
            ["標準（建議）", "壓力測試（保守）", "進階（自行調參）"],
            index=0,
            horizontal=True,
            help="標準：少量參數即可得到成功率。壓力測試：自動採用肥尾/負偏態等保守設定。進階：展開全部參數。",
        )
        mc_row1_c1, mc_row1_c2, mc_row1_c3 = st.columns([2, 1, 1])
        with mc_row1_c1:
            if mc_mode == "進階（自行調參）":
                mc_std = st.slider(
                    "年報酬率標準差 σ (%)",
                    min_value=5.0, max_value=30.0, value=15.0, step=1.0,
                    help="股票型組合歷史波動率約 15–20%；保守組合可設 10–12%；0050/S&P500 約 18–20%",
                )
            elif mc_mode == "壓力測試（保守）":
                mc_std = 18.0
                st.caption("年波動率 σ：壓力測試固定為 **18%**（偏保守）。")
            else:
                mc_std = 15.0
                st.caption("年波動率 σ：標準模式固定為 **15%**。")
        with mc_row1_c2:
            if mc_mode == "進階（自行調參）":
                mc_dist = st.radio(
                    "報酬率分布",
                    ["常態分布", "t 分布（肥尾）", "歷史 Bootstrap"],
                    horizontal=True,
                    help=(
                        "常態分布：標準假設，計算快速\n"
                        "t 分布：模擬金融市場肥尾（極端漲跌比常態更頻繁），尾部風險通常較常態更高\n"
                        "歷史 Bootstrap：直接從台灣/美股 50 年歷史實質報酬重抽樣，"
                        "保留真實偏態與肥尾，σ 由歷史資料決定（忽略上方 σ 設定）"
                    ),
                )
            elif mc_mode == "壓力測試（保守）":
                mc_dist = "t 分布（肥尾）"
                st.caption("分布：壓力測試固定為 **t 分布（肥尾 + 負偏態）**。")
            else:
                mc_dist = "常態分布"
                st.caption("分布：標準模式固定為 **常態分布**。")
        with mc_row1_c3:
            mc_strategy = st.radio(
                "模擬策略",
                ["固定提領", "消費微笑曲線", "GK 護欄"],
                horizontal=True,
                help="與主引擎相同的三種策略，均包含醫療溢價及勞保補貼邏輯",
            )

        mc_use_t         = mc_dist == "t 分布（肥尾）"
        mc_use_bootstrap = mc_dist == "歷史 Bootstrap"
        mc_t_df   = 7    # 預設自由度
        mc_t_skew = 0.0  # 預設偏態（對稱）
        if mc_use_t and mc_mode == "壓力測試（保守）":
            mc_t_df = 7
            mc_t_skew = -0.3
        if mc_use_t and mc_mode == "進階（自行調參）":
            _t_col1, _t_col2 = st.columns(2)
            with _t_col1:
                mc_t_df = st.slider(
                    "t 分布自由度 ν",
                    min_value=3, max_value=30, value=7, step=1,
                    help=(
                        "ν 越小 → 尾部越肥（極端事件越多）。"
                        "金融研究常用 ν = 5–8；ν → ∞ 趨近常態分布。"
                        "ν < 3 時方差不存在，故限制最小值為 3。"
                    ),
                )
                _scale = np.sqrt(mc_t_df / (mc_t_df - 2))
                st.caption(
                    f"縮放因子 {_scale:.4f}　"
                    "（確保模擬 σ 與設定值一致）"
                )
            with _t_col2:
                mc_t_skew = st.slider(
                    "偏態係數 γ",
                    min_value=-1.0, max_value=0.0, value=-0.3, step=0.05,
                    help=(
                        "0 = 對稱 t 分布；負值 = 負偏態（左尾更重）。\n"
                        "實際金融報酬呈現負偏態：大跌比大漲更頻繁且幅度更大。\n"
                        "台股/美股歷史偏態係數約 -0.3 至 -0.6；建議預設 -0.3。\n"
                        "實作：負值樣本乘以 (1+|γ|)，再重新校正均值與標準差。"
                    ),
                )
                st.caption(
                    f"γ = {mc_t_skew:.2f}：負報酬幅度放大 {abs(mc_t_skew)*100:.0f}%，"
                    "均值與 σ 已重新校正。"
                )
        if mc_use_bootstrap:
            _hist_arr_disp = np.array(_HIST_REAL_RETURNS_PCT)
            st.caption(
                f"歷史資料：{len(_HIST_REAL_RETURNS_PCT)} 年（台灣加權 + 美股混合，1975–2024 實質報酬）　"
                f"歷史均值 {_hist_arr_disp.mean():.1f}%、標準差 {_hist_arr_disp.std():.1f}%、"
                f"最低 {_hist_arr_disp.min():.0f}%、最高 {_hist_arr_disp.max():.0f}%。"
                "Bootstrap 模式下 σ slider 不影響結果。"
            )

        # ── 通膨隨機化選項（非 Bootstrap 模式才有意義）──────────────────
        mc_infl_rand = False
        mc_infl_std  = 0.8
        if not mc_use_bootstrap and mc_mode == "壓力測試（保守）":
            mc_infl_rand = True
            mc_infl_std = 1.2
            st.caption("通膨：壓力測試開啟通膨隨機化（σ_CPI=1.2%）。")
        if (not mc_use_bootstrap) and (mc_mode == "進階（自行調參）"):
            mc_infl_rand = st.toggle(
                "通膨隨機化",
                value=False,
                help=(
                    "開啟後，每條路徑每年的通膨率從 N(CPI均值, σ_CPI²) 獨立抽樣，"
                    "報酬率輸入改視為名目報酬均值，實質報酬 = 名目報酬 − 當年隨機通膨。\n"
                    "此設計反映通膨本身的不確定性（2020–2023 年全球通膨超預期即為典型案例）。"
                ),
            )
            if mc_infl_rand:
                mc_infl_std = st.slider(
                    "通膨標準差 σ_CPI (%)",
                    min_value=0.2, max_value=3.0, value=0.8, step=0.1,
                    help=(
                        "台灣歷史通膨標準差約 0.8–1.2%；2020–2023 年波動期約 1.5–2.0%。"
                        "σ_CPI 越大代表通膨越難預測，實質報酬不確定性越高。"
                    ),
                )
                st.caption(
                    f"通膨均值：{inflation_pct}%（左側側欄設定）± {mc_infl_std:.1f}% 標準差。"
                    "名目報酬均值 ≈ 實質報酬 + 通膨均值。"
                )

        _strat_map = {"固定提領": "fixed", "消費微笑曲線": "smile", "GK 護欄": "gk"}
        mc_strat   = _strat_map[mc_strategy]
        if mc_use_bootstrap:
            mc_dist_mode = "bootstrap"
        elif mc_use_t:
            mc_dist_mode = "t"
        else:
            mc_dist_mode = "normal"

        _t0 = time.perf_counter()
        # 通膨隨機化時，傳入名目報酬均值（實質 + 通膨），引擎內部每年扣除隨機 CPI
        _mc_r_input = (r_pct + inflation_pct) if mc_infl_rand else r_pct
        mc_sr, mc_p10, mc_p50, mc_p90 = _run_monte_carlo(
            A0_eff, W0, _mc_r_input, mc_std,
            n_years, mc_strat,
            pension_annual, int(claim_age), age_start,
            med_premium_pct=medical_premium,
            dist_mode=mc_dist_mode,
            t_df=mc_t_df,
            t_skew=mc_t_skew,
            rental_annual=rental_annual,
            rental_start_age=int(rental_start_age_input),
            rm_annual=rm_annual,
            rm_start_age=int(rm_start_age),
            inflation_randomize=mc_infl_rand,
            inflation_mean_pct=inflation_pct,
            inflation_std_pct=mc_infl_std,
        )
        _elapsed_ms = (time.perf_counter() - _t0) * 1000
        _from_cache = _elapsed_ms < 5.0

        mm1, mm2, mm3, mm4 = st.columns(4)
        with mm1:
            st.metric("成功機率", f"{mc_sr:.1f}%",
                      "≥90% 為安全" if mc_sr >= 90 else ("75–90% 需注意" if mc_sr >= 75 else "< 75% 危險"))
        with mm2:
            st.metric("悲觀結果 P10", _fmt_asset(mc_p10), "最差10%情境")
        with mm3:
            st.metric("中位數 P50", _fmt_asset(mc_p50), "典型情境")
        with mm4:
            st.metric("樂觀結果 P90", _fmt_asset(mc_p90), "最佳10%情境")

        if mc_sr >= 90:
            st.success(f"✅ 高成功機率（{mc_sr:.1f}%）：10,000 條路徑中超過 90% 可支撐至 {age_end} 歲。")
        elif mc_sr >= 75:
            st.warning(f"⚠️ 中等成功機率（{mc_sr:.1f}%）：建議降低提領率、增加初始資產，或切換為 GK 護欄策略。")
        else:
            st.error(f"❌ 低成功機率（{mc_sr:.1f}%）：在大多數市場情境下資產將耗盡，強烈建議重新規劃。")

        if mc_use_bootstrap:
            _dist_label = f"歷史 Bootstrap（{len(_HIST_REAL_RETURNS_PCT)} 年）"
        elif mc_use_t:
            _skew_note = f"，γ={mc_t_skew:.2f}" if mc_t_skew != 0.0 else ""
            _dist_label = f"t 分布（ν={mc_t_df}{_skew_note}）"
        else:
            _dist_label = "常態分布"
        _cache_note = "⚡ 來自快取（參數未變動）" if _from_cache else f"🔄 即時計算完成（{_elapsed_ms:.0f} ms）"
        _infl_note = f"、通膨隨機化（σ_CPI={mc_infl_std:.1f}%）" if mc_infl_rand else ""
        st.caption(
            f"{_cache_note}　｜　"
            f"分布：{_dist_label}、σ = {mc_std}%、r̄ = {r_pct}%{_infl_note}、10,000 次模擬、{mc_strategy}、"
            f"醫療溢價 {medical_premium}%、起始年齡 {age_start} 歲、目標年齡 {age_end} 歲。"
            "　seed=42 固定，相同參數可重現。"
        )

        # ── 失敗路徑剖析（可視化 + CSV）─────────────────────────────────
        with st.expander("🧩 失敗路徑剖析（找出哪些情境會失敗）", expanded=False):
            st.caption("用途：把『成功率不是100%』拆解成：哪些路徑在幾歲歸零、早期報酬長什麼樣。")
            worst_n = st.slider("要畫出最差路徑數（線圖）", min_value=3, max_value=30, value=10, step=1)
            track_years = st.slider("早期觀察期（年）", min_value=3, max_value=10, value=5, step=1, help="用於輸出早期累積報酬等指標。")

            if st.button("產生剖析（可能需數秒）", key="btn_fail_path_analysis"):
                df_fail, A_path = _run_monte_carlo_failure_analysis(
                    A0_eff, W0, _mc_r_input, mc_std,
                    n_years, mc_strat,
                    pension_annual, int(claim_age), age_start,
                    med_premium_pct=medical_premium,
                    dist_mode=mc_dist_mode,
                    t_df=mc_t_df,
                    t_skew=mc_t_skew,
                    rental_annual=rental_annual,
                    rental_start_age=int(rental_start_age_input),
                    rm_annual=rm_annual,
                    rm_start_age=int(rm_start_age),
                    inflation_randomize=mc_infl_rand,
                    inflation_mean_pct=inflation_pct,
                    inflation_std_pct=mc_infl_std,
                    inflation_assumed_pct=inflation_pct,
                    n_sim=10_000,
                    track_years=int(track_years),
                )

                failures = df_fail[~df_fail["success"]].copy()
                st.metric("失敗路徑數", f"{len(failures):,}", f"佔比 {100 - mc_sr:.1f}%")

                if len(failures) == 0:
                    st.success("✅ 此設定下沒有失敗路徑（成功率=100%）。")
                else:
                    # 1) 歸零年齡分布
                    ruin_counts = failures["ruin_age"].value_counts().sort_index()
                    ruin_df = ruin_counts.reset_index()
                    ruin_df.columns = ["歸零年齡", "路徑數"]
                    st.subheader("失敗發生在幾歲？（歸零年齡分布）")
                    st.bar_chart(ruin_df.set_index("歸零年齡")["路徑數"], height=220)

                    # 2) 最差路徑資產曲線（實質購買力）
                    st.subheader("最差路徑長什麼樣？（資產曲線，實質購買力）")
                    worst_ids = (
                        failures.sort_values(["final_real_ntd", "ruin_age"], ascending=[True, True])
                        .head(int(worst_n))["sim_id"].astype(int).tolist()
                    )
                    ages = np.arange(age_start, age_start + n_years)
                    lines = []
                    for sid in worst_ids:
                        y = A_path[sid, :].astype(np.float64) / 10_000  # 萬
                        lines.append(pd.DataFrame({"年齡": ages, "資產（萬，實質）": y, "路徑": f"#{sid}"}))
                    df_lines = pd.concat(lines, ignore_index=True)
                    # 用 st.line_chart（不引入新依賴），以路徑作為顏色（以 pivot 方式）
                    pivot = df_lines.pivot_table(index="年齡", columns="路徑", values="資產（萬，實質）", aggfunc="first")
                    st.line_chart(pivot, height=260)

                    # 3) CSV：摘要（每條路徑一列）
                    st.subheader("下載：失敗路徑摘要（CSV）")
                    csv_bytes = df_fail.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                    st.download_button(
                        "下載 CSV（每條路徑一列）",
                        data=csv_bytes,
                        file_name="failure_paths_summary.csv",
                        mime="text/csv",
                    )

                    # 也把檔案落地到專案資料夾，回傳路徑
                    try:
                        base_dir = os.path.dirname(__file__)
                    except Exception:
                        base_dir = os.getcwd()
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = os.path.join(base_dir, f"failure_paths_summary_{ts}.csv")
                    with open(out_path, "wb") as f:
                        f.write(csv_bytes)
                    st.caption(f"已另存到：`{out_path}`")
    st.divider()

    # ── 學術背景說明 ──
    st.subheader("學術背景與模型說明")
    with st.expander("蓋頓–克林格 (GK) 護欄策略", expanded=False):
        trigger_raise = W0_asset * 1.1
        trigger_cut   = W0_asset * 0.9
        st.markdown("""
**理論**：基於控制論的動態提領協議，解決靜態提領法在 50 年長週期下必然產生的**資源錯配**——市場過熱時「強迫加薪」，市場低迷時「適度減薪」。

- **繁榮法則 (Prosperity Rule)**：當下提領率 < IWR × 0.8 時觸發（目前 **{:.2f}%**）。當資產膨脹至約 **{}** 時，強制將該年度「從資產」支出上調 10% → 約 **{}/年**。
- **資本保全法則 (Capital Preservation Rule)**：當下提領率 > IWR × 1.2 時觸發（目前 **{:.2f}%**）。當資產縮水至約 **{}** 時，「從資產」支出削減 10% → 約 **{}/年**，以保全本金。

**文獻**：Guyton & Klinger (2006).
        """.format(
            gk_lower,
            _fmt_wan(W0_asset / (gk_lower / 100)) if gk_lower > 0 else "—（IWR 為 0）",
            _fmt_wan(trigger_raise),
            gk_upper,
            _fmt_wan(W0_asset / (gk_upper / 100)) if gk_upper > 0 else "—（IWR 為 0）",
            _fmt_wan(trigger_cut),
        ))
    with st.expander("序列報酬風險 (SORR) 與提領風險", expanded=False):
        st.markdown("""
- **SORR**：退休初期若遇低報酬，固定提領會形成「逆向定期定額」——在低點變賣過多資產，易導致路徑崩潰。
- **低提領率優勢**：當 IWR 顯著低於 4% 時，系統具備「**自我修復能力**」——即便發生連續三年負回報，剩餘資本的利息覆蓋率仍可維持在健康狀態，提領行為不足以撼動複利結構。
        """)
    with st.expander("醫療溢價與生理投資 (Bio-Capital)", expanded=False):
        st.markdown("""
- 模型假設 70 歲起醫療支出以 **CPI + {:.1f}%** 成長（非對稱性增長的醫療成本）。
- **生理資本**：良好骨密度與肌肉適能可實質推遲「失能區（No-Go Phase）」到來，預計可**節省 15–20% 晚年護理支出**，形成對長照風險的自然對沖。
- **建議**：利用 GK 繁榮法則釋出的加薪額度，優先配置於高品質蛋白質與定期生理數據檢測。
        """.format(medical_premium))
    with st.expander("房產的戰略角色", expanded=False):
        st.markdown("""
- **居住成本對沖**：自有住宅無租金支出，鎖定生活成本，提領預算不含房租。
- **末端安全墊**：85 歲後若出現超出模型預測的醫療支出，可透過「以房養老」提供流動性。
        """)
    with st.expander("「歸零幻覺」除錯與實質/名目區分", expanded=False):
        st.markdown("""
- **實質 vs 名目**：若誤將「名目報酬率 4%」當成「實質報酬率 4%」，在 3% 通膨下實質僅約 1%，不足以抵銷提領，路徑會惡化。本模型一律採用**實質報酬**。
        """)
    with st.expander("地緣風險防禦與地理熵減", expanded=False):
        st.markdown("""
- 當財務生存率達 100% 時，主要殘餘威脅為「**單一板塊崩潰**」。若資產集中於單一國家或產業，風險較高。
- **建議**：將未來每月儲蓄全數買入**全球分散型 ETF（如 VWRA）**，以「**地理熵減**」應對地緣政治的隨機性。
        """)
    with st.expander("勞保／勞退與試算邏輯", expanded=False):
        st.markdown("""
- **勞退**：年滿 60 歲可請領，年資 15 年以上可選月領或一次領；雇主提撥 6%，勞工可自提 1%～6%。
- **勞保老年給付**：年資 15 年以上可領年金，請領年齡逐步延後至 65 歲；與勞退可同時請領。
- **試算邏輯**：左側填寫「勞保＋勞退 月領」與「請領年齡」後，從該年齡起「實質購買力」改由勞保勞退支應一部分，**從自有資產提領** = 實質購買力 − 勞保勞退年額；IWR 與 GK 護欄皆以「從資產提領」為準。
        """)
    with st.expander("封閉公式驗算：確定性引擎精度測試", expanded=False):
        st.markdown(r"""
### 方法論

固定提領（期初提領、年複利）在報酬率與提領額均為常數時，具備精確的解析解，可作為動態引擎的「黃金標準」對照。

#### 推導過程

設初始資產為 $A_0$，每期固定提領 $W$，實質報酬率為 $r$，第 $k$ 期結束時資產為 $A_k$。

每期遞推關係（期初提領）：

$$A_k = (A_{k-1} - W)(1+r)$$

展開至第 $n$ 期：

$$A_n = (A_0 - W)(1+r)^n - W\bigl[(1+r)^{n-1} + (1+r)^{n-2} + \cdots + (1+r)\bigr]$$

等比級數求和（$r \neq 0$）：

$$\boxed{A_n = (A_0 - W)(1+r)^n - W \cdot \frac{(1+r)^n - (1+r)}{r}}$$

當 $r = 0$ 時退化為：

$$A_n = A_0 - W \cdot n$$

#### 使用方式

將模型當前的 $A_0$、$W$、$r$、$n$ 代入公式，與動態引擎輸出比對，若誤差 $< 0.001\%$ 則確認引擎計算邏輯正確。

#### 適用範圍與限制

| 條件 | 說明 |
|---|---|
| ✅ 固定提領 (strategy=fixed) | 精確成立 |
| ❌ 消費微笑曲線 | 每期 W 動態改變，無封閉解 |
| ❌ GK 護欄策略 | 護欄觸發邏輯為條件式非線性，無封閉解 |
| ⚠️ 含勞保補貼 | 補貼使每期實際扣款 ≠ 常數，公式不適用，驗算時設 pension=0 |

此方法可驗證**引擎計算邏輯是否正確**，但無法評估市場假設是否合理，後者需蒙地卡羅法補充。
        """)

    with st.expander("蒙地卡羅模擬：隨機路徑與成功機率估算", expanded=False):
        st.markdown(r"""
### 方法論

確定性模型使用固定 $r$，隱含「每年報酬率完全可預測」的假設，此假設不符合真實市場。蒙地卡羅法以**隨機報酬率**替代固定值，模擬數千條可能的市場路徑，藉此量化不確定性。

---

#### 模式 A：常態分布

$$r_t \sim \mathcal{N}(\bar{r},\, \sigma^2)$$

標準假設，計算快速。已知限制：常態分布對極端尾部的機率估計偏低（低估金融危機頻率）。

---

#### 模式 B：Student-t 分布（肥尾修正）

金融報酬的實證分布比常態更「尖峰厚尾」（leptokurtic）：極端的大漲與大跌發生頻率遠高於常態預測。本模型使用 Student-t 分布模擬此特性：

$$r_t = \bar{r} + \sigma \cdot \frac{T_\nu}{\sqrt{\nu / (\nu - 2)}}, \quad T_\nu \sim t(\nu)$$

**縮放修正的必要性**：

標準 Student-t 分布 $T_\nu$ 的方差為 $\dfrac{\nu}{\nu-2}$（不等於 1）。若直接以 $r_t = \bar{r} + \sigma \cdot T_\nu$，模擬的實際標準差將被悄悄放大：

$$\text{Var}(r_t) = \sigma^2 \cdot \frac{\nu}{\nu-2} \neq \sigma^2$$

例如 $\nu = 7$：$\sqrt{7/5} \approx 1.183$，標準差被放大 **18.3%**，破產機率因此虛高。

正確做法是除以縮放因子 $\sqrt{\nu/(\nu-2)}$，使模擬標準差精確等於 σ，同時保留 t 分布的肥尾形狀：

$$\text{scale\_factor} = \sqrt{\frac{\nu}{\nu - 2}}$$

| $\nu$ | scale\_factor | 形狀描述 |
|---|---|---|
| 3 | 1.732 | 極度肥尾，尾部事件非常頻繁 |
| 5 | 1.291 | 重度肥尾，金融危機情境 |
| 7 | 1.183 | 中度肥尾（**本模型預設**，文獻常用值）|
| 15 | 1.036 | 輕度肥尾 |
| 30 | 1.017 | 接近常態分布 |
| ∞ | 1.000 | 退化為常態分布 |

**注意**：$\nu \geq 3$ 才能保證方差存在（$\nu < 3$ 時 $E[T^2] = \infty$），故自由度滑桿最小值設為 3。

---

#### 模擬步驟

1. 以固定隨機種子（seed=42）預生成 $10{,}000 \times n$ 個報酬率隨機數，shape = $(n_{sim},\, n_{years})$
2. 每條路徑獨立執行動態引擎，邏輯與確定性模型相同（含 GK 護欄、醫療溢價、勞保補貼）
3. 記錄每條路徑的終值 $A_n^{(i)}$
4. 統計輸出：

| 指標 | 定義 |
|---|---|
| **成功率** | $P(A_n > 0)$ |
| **P10** | $\text{Quantile}_{10\%}(A_n^{(i)})$ ── 悲觀情境 |
| **P50** | $\text{Quantile}_{50\%}(A_n^{(i)})$ ── 中位數 |
| **P90** | $\text{Quantile}_{90\%}(A_n^{(i)})$ ── 樂觀情境 |

---

#### 方差拖累（Variance Drag）

$$\text{幾何平均} \approx \bar{r} - \frac{\sigma^2}{2}$$

例如：$\bar{r} = 6\%$，$\sigma = 18\%$，方差拖累 $\approx 1.62\%$，實質幾何均值降至約 $4.38\%$。確定性模型使用算術平均 $\bar{r}$，蒙地卡羅 P50 因此略低於確定性終值。

#### 參考波動率設定

| 組合類型 | 建議 σ |
|---|---|
| 全股票（台股/美股 ETF） | 18–22% |
| 股六債四 | 12–15% |
| 保守（股四債六） | 8–12% |
| 台股 0050 歷史實測 | ≈ 20%（2003–2024） |

#### 剩餘限制

- 未模擬序列相關性（報酬率非嚴格 i.i.d.，景氣循環中存在動能效應）
- 若需更嚴謹的歷史分布，建議改用 Bootstrap 重抽樣法（以實際歷史年報酬序列抽樣）

**文獻**：Pfau, W. D. (2012). *An efficient frontier for retirement income.* Journal of Financial Planning.；Rachev, S. T. et al. (2005). *Fat-tailed and skewed asset return distributions.* Wiley.
        """)

    with st.expander("參考文獻與資料來源", expanded=False):
        st.markdown("""
- Guyton, J. T., & Klinger, W. J. (2006). *Decision rules and portfolio management for retirees.* Journal of Financial Planning.
- Blanchett, D. (2014). *Exploring the retirement consumption puzzle.* Journal of Financial Planning.
- Pfau, W. D. (2012). *An efficient frontier for retirement income.* Journal of Financial Planning.
- Morningstar: State of Retirement Income 2025 (2025/12/3).
- Bengen, W. P. (1994). *Determining withdrawal rates using historical data.* Journal of Financial Planning.
- NBLM 退休模型庫。
        """)

    st.divider()
    st.caption("退休規劃大師 · 嚴謹科學顧問級模組 · 信心水準：高")

# ──────────────────────────────────────────────
# PAGE：教育資訊庫
# ──────────────────────────────────────────────
elif page_id == "edu":
    st.title("📚 退休教育資訊庫")
    st.caption("整合：退休規劃彙整表 · 2026 台灣最新制度/公告重點 · 研究與公開資訊（台灣）")
    st.divider()

    # ── 分類：先選分類，再選主題（降低一次看到太多選項的負擔）
    _edu_categories = {
        "核心概念（先看這裡）": [
            "1｜退休三階段（消費微笑曲線）",
            "2｜安全提領率與 GK 護欄策略",
        ],
        "台灣制度／稅務": [
            "3｜114 年度台灣綜合所得稅（2026 報稅適用）",
            "4｜勞保老年年金計算",
            "5｜勞退新制試算",
        ],
        "醫療／長照／風險": [
            "6｜長照費用與風險",
            "18｜退休保險健檢（快速清單，教育用途）",
        ],
        "資產配置／市場觀察": [
            "7｜資產配置與 ETF 建議",
            "8｜台灣與全球經濟（2026 觀察）",
            "9｜退休金制度改革動態（2026 觀察）",
            "10｜配置典範革命：傳統「年齡=債券」vs 上升股票路徑",
        ],
        "房產／資產負債表／心理帳戶": [
            "11｜持有房產與心理帳戶陷阱",
            "12｜退休資產負債表：自住房產的防禦性與盲點",
            "13｜長照對沖與房產資金階梯",
            "14｜不動產收益：租金、殖利率與 REITs",
            "15｜出租物業的類年金效應（Buy-to-Let Quasi-Annuity）",
            "16｜不動產收益護欄：Income Floor 與折現風險調整",
            "17｜退休現金流全景：主動槓桿與房貸管理",
        ],
    }

    _default_cat = str(st.session_state.get("edu_category", "核心概念（先看這裡）"))
    if _default_cat not in _edu_categories:
        _default_cat = "核心概念（先看這裡）"
    edu_category = st.selectbox(
        "分類",
        list(_edu_categories.keys()),
        index=list(_edu_categories.keys()).index(_default_cat),
        key="edu_category",
        help="先用分類縮小範圍，再選主題閱讀。",
    )
    edu_topic = st.radio(
        "主題",
        _edu_categories[edu_category],
        horizontal=True,
        key="edu_topic",
    )
    st.divider()

    if _nav == "edu":
        st.success("已從「延伸閱讀」導覽過來：本頁已自動預選分類/主題。", icon="✅")

    if edu_topic.startswith("1｜"):
        st.subheader("退休三階段：消費微笑曲線 (Retirement Spending Smile)")
        st.markdown("> **學術來源**：David Blanchett (2014)「Exploring the retirement consumption puzzle」\n>\n> 退休支出並非線性遞減，而呈現「U型微笑曲線」：初期（活躍期）支出高，中期自然縮減，晚期（護理期）再度攀升。")
        phases = pd.DataFrame([
            ["Go-Go Years（活躍期）","65–75 歲","最高","體力充沛、旅遊頻繁；建議設定最高版本的生活費","相當於工作時期的 100%"],
            ["Slow-Go Years（緩速期）","75–85 歲","中等","行動力下降，長途旅遊減少；支出自然縮減約 15–20%","約為工作時期的 80%"],
            ["No-Go Years（護理期）","85 歲以上","醫療主導","行動幾乎停止；醫療/長照支出飆升；需備妥長照保障","醫療支出佔比大幅提升"],
        ], columns=["階段","年齡區間","活動力","特徵說明","相對支出水準"])
        st.dataframe(phases, use_container_width=True, hide_index=True)
        st.info("**模型實裝（V2.0引擎）**：消費微笑曲線採平滑過渡（<73：×1.0；73–77 線性降至 ×0.8；77–83：×0.8；83–87 線性升至 ×1.1；≥87：×1.1）；70 歲起疊加醫療溢價指數複利 `W₀×15% × ((1+rate)^(age-70)−1)`")
        st.markdown("""
#### 台灣 2025 實證：高齡家庭通膨
- **2025 年高齡家庭 CPI 年增 1.74%**，連續 7 年高於整體平均（整體 1.66%）
- 高齡家庭醫療保健權重約 **8%**（一般家庭約 5%）
- 主要推升：掛號費調漲、外籍看護費上漲、房租上升
- 資料來源：主計總處（自 2024 年起發布，並改為按月公布、追溯 2018 年以來各月指數；中央社 2026/01/10 引述）
        """)

    elif edu_topic.startswith("2"):
        st.subheader("安全提領率（SWR）與 Guyton-Klinger 護欄策略")
        st.markdown("""
### 4% 法則基礎
- 由 **William Bengen（1994）** 提出：每年從退休資產提領 4%，並按通膨逐年調整，理論上可支撐 **30 年以上**。
- **Morningstar 2025 年報**（2025/12/3）：2026 年最佳提領率下修為 **3.9%**。
- 若退休超過 40 年（50 歲前退休），建議降至 **3.0–3.5%**。
        """)
        ec1, ec2 = st.columns(2)
        with ec1:
            swr_df = pd.DataFrame([
                ["3.0%","100%","100%","100%"],
                ["3.5%","100%","100%","100%"],
                ["4.0%","92.65%","94.12%","95.59%"],
                ["4.5%","82.35%","85.00%","87.50%"],
                ["5.0%","66.18%","70.00%","73.50%"],
            ], columns=["提領率","股50%債50%","股60%債40%","股80%債20%"])
            st.markdown("#### 歷史回測成功率（30年）")
            st.dataframe(swr_df, use_container_width=True, hide_index=True)
            st.caption("資料來源：2025年台灣回測分析")
        with ec2:
            st.warning("**SORR**（序列報酬風險）：退休初期若遇市場大跌，影響遠比中期大跌嚴重。\n\n- 研究顯示：**70% 的計劃失敗來自前 5 年提領**\n- 機制：在低點被迫賣股 → 資產永久減少 → 後期複利失去基礎\n- 對策：退休初期降低股票比例；或採 GK 護欄動態調整")
        st.divider()
        st.markdown("### Guyton-Klinger (GK) 護欄策略詳解")
        gk_rules = pd.DataFrame([
            ["初始提領率（IWR）","W₀ ÷ A₀","退休第一年確定","建議 4%；GK研究顯示可設 5.2–5.6%（含65%股票）"],
            ["繁榮法則（加薪護欄）","當下提領率 < IWR × 0.80","觸發加薪 10%","資產漲太多 → 強制消費，避免過度節約"],
            ["保全法則（減薪護欄）","當下提領率 > IWR × 1.20","觸發減薪 10%","資產縮水 → 降支出保本，避免路徑崩潰"],
            ["通膨調整上限","年調幅上限 6%","不超過初始提領額 +6%/年","避免通膨年份過度提高支出"],
            ["最後防線原則","剩餘規劃年數 ≤ 15 年","暫停減薪護欄","避免晚年過度緊縮生活品質"],
        ], columns=["規則","觸發條件","執行動作","背後邏輯"])
        st.dataframe(gk_rules, use_container_width=True, hide_index=True)
        st.markdown("""
| 面向 | 固定提領 | 消費微笑曲線 | GK 護欄 |
|---|---|---|---|
| **核心邏輯** | 每年固定 W₀ | 依三段生命週期調整 | 依資產比例動態調整 |
| **市場應對** | 無 | 無 | 主動調整 |
| **破產風險** | 中等 | 略佳 | 最低 |
| **晚年遺產** | 最高 | 次高 | 最低（用得更多）|
        """)

    elif edu_topic.startswith("3"):
        st.subheader("114 年度台灣綜合所得稅率（115 年 5 月申報；2026 報稅適用）")
        ec1, ec2 = st.columns([3,2])
        with ec1:
            st.caption("財政部公告：114 年度各項免稅額、扣除額、課稅級距因 CPI 未達調整門檻（較上次調整年度上漲 2.29% < 3%），免予調整（公告日 2024/11/28）。")
            tax_df = pd.DataFrame([
                ["第 1 級","5%","NT$ 0 – 590,000","NT$ 0","應稅所得 × 5%"],
                ["第 2 級","12%","NT$ 590,001 – 1,330,000","NT$ 41,300","× 12% - 41,300"],
                ["第 3 級","20%","NT$ 1,330,001 – 2,660,000","NT$ 147,700","× 20% - 147,700"],
                ["第 4 級","30%","NT$ 2,660,001 – 4,980,000","NT$ 413,700","× 30% - 413,700"],
                ["第 5 級","40%","NT$ 4,980,001 以上","NT$ 911,700","× 40% - 911,700"],
            ], columns=["級別","稅率","綜合所得淨額範圍","累進差額","速算公式"])
            st.dataframe(tax_df, use_container_width=True, hide_index=True)
            st.caption("資料來源：財政部（114 年度公告；適用 115 年 5 月申報）")
        with ec2:
            deduct_df = pd.DataFrame([
                ["免稅額（一般）","NT$ 97,000"],
                ["免稅額（70 歲以上）","NT$ 145,500"],
                ["標準扣除額（單身）","NT$ 131,000"],
                ["標準扣除額（已婚）","NT$ 262,000"],
                ["薪資所得特別扣除額","NT$ 218,000"],
                ["基本生活費","NT$ 210,000"],
                ["房屋租金支出扣除額","NT$ 180,000（租屋族）"],
            ], columns=["項目","金額"])
            st.dataframe(deduct_df, use_container_width=True, hide_index=True)
        st.divider()
        ec3, ec4 = st.columns(2)
        with ec3:
            st.info("**退職所得定額免稅額（114 年度）**\n- 一次領取：每年資 **NT$ 198,000** 以內免稅；超過 198,000 未達 398,000 元：50% 課稅；超過 398,000：全數課稅\n- 分期領取（月領）：每年減除 **NT$ 859,000**\n\n（實際適用仍以財政部公告附表為準）")
        with ec4:
            st.success("**退休後節稅策略**\n- 善用 **70 歲以上免稅額加成**（依當年度公告）\n- 勞保勞退月領 + 適度從自有資產提領，控制在低稅級\n- 股利：分離課稅（28%）vs 合併申報（擇優）")
        st.markdown("#### 稅後實質購買力試算（單身，年領 120 萬，標準扣除）\n- 應稅所得 = 120萬 - 9.7萬 - 13.1萬 = **97.2 萬**\n- 稅額 = 97.2萬 × 5% = **NT$ 48,600**（稅率約 4%）\n- **稅後實質購買力 ≈ NT$ 1,151,400 / 年**")

    elif edu_topic.startswith("4"):
        st.subheader("勞保老年年金計算（2026 重點整理）")
        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown("#### 公式一\n```\n平均月投保薪資 × 保險年資 × 1.55%\n```\n#### 公式二（加計基本年金）\n```\n平均月投保薪資 × 保險年資 × 0.775% + 3,000 元\n```\n> 兩式**擇優發給**，勞保局自動選有利者。\n> **平均月投保薪資**：最高 60 個月平均計算")
        with ec2:
            st.code("範例：年資 35.5 年、月投保 NT$ 45,800\n\n公式一：45,800 × 35.5 × 1.55% = NT$ 25,213/月\n公式二：45,800 × 35.5 × 0.775% + 3,000 = NT$ 15,607/月\n\n→ 擇優：NT$ 25,213/月（年領約 30.3 萬）")
        st.divider()
        age_df = pd.DataFrame([
            ["提前 5 年（最多）","60 歲","-20%","財務充裕才考慮"],
            ["提前 1 年","64 歲","-4%",""],
            ["標準請領","65 歲","±0%","法定年齡（逐步延後中）"],
            ["展延 1 年","66 歲","+4%",""],
            ["展延 5 年（最多）","70 歲","+20%","有其他收入可考慮展延"],
        ], columns=["方式","請領年齡","增減給比例","備註"])
        st.dataframe(age_df, use_container_width=True, hide_index=True)
        st.info("- 勞保老年給付**完全免稅**\n- 申請後**不可撤回**，請先至勞保局網站試算再申請\n- 可同時領勞退 + 勞保，形成雙層月領收入")
        st.caption("資料來源（官方）：勞動部勞工保險局（勞工保險-老年給付-給付標準；FAQ：老年年金給付請領資格及給付標準；含 115 年後法定請領年齡 65 歲之說明）。")

    elif edu_topic.startswith("5"):
        st.subheader("勞退新制個人專戶（2026 更新：績效參考）")
        st.markdown("- **雇主強制提繳**：月薪的 **6%** 存入個人專戶\n- **勞工自願提繳**：最高 **6%**（享薪資所得扣除優惠）\n- 帳戶歸個人所有，不受雇主倒閉影響")
        st.code("月退休金 = 個人專戶結算金額 ÷ 期初年金現值因子 ÷ 12")
        sample_df = pd.DataFrame([
            ["100 萬","NT$ 3,896","NT$ 46,752 / 年"],
            ["200 萬","NT$ 7,792","NT$ 93,504 / 年"],
            ["500 萬","NT$ 19,480","NT$ 233,760 / 年"],
            ["1,000 萬","NT$ 38,960","NT$ 467,520 / 年"],
            ["1,500 萬","NT$ 58,440","NT$ 701,280 / 年"],
        ], columns=["專戶累積金額","估計月領","估計年領"])
        st.dataframe(sample_df, use_container_width=True, hide_index=True)
        st.caption("以 60 歲、平均餘命 24 年估算；實際依申請當月公告利率計算")
        st.warning(
            "**勞退基金績效參考（以勞動基金運用局公告/文件為準）**\n"
            "- 近一年（113 年）新制勞退基金收益率：**16.16%**\n"
            "- 114 年截至 9 月底新制勞退基金收益率：**7.69%**\n"
            "- ⚠️ 歷史績效不代表未來；做退休試算建議仍用保守的長期實質報酬假設（例如 3–5% 區間）。"
        )
        st.markdown("|  | 勞退新制 | 勞保老年年金 |\n|---|---|---|\n| **性質** | 個人儲蓄帳戶 | 社會保險 |\n| **請領年齡** | 60 歲（年資 15 年） | 65 歲（可提前/展延）|\n| **稅務** | 一次領部分須申報 | 完全免稅 |\n| **可合併** | ✅ 可同時領 | ✅ 同上 |")

    elif edu_topic.startswith("6"):
        st.subheader("長照費用與風險（台灣 2026：長照 3.0 更新）")
        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown(
                "- **申請**：撥打 **1966 長照專線** → 照管專員到府評估（CMS 2–8 級）\n"
                "- **照顧及專業服務額度（每月）**：\n"
                "  - CMS2：10,020 元；CMS3：15,460 元；CMS4：18,580 元；CMS5：24,100 元\n"
                "  - CMS6：28,070 元；CMS7：32,090 元；CMS8：36,180 元\n"
                "- **部分負擔**：第一類政府全額補助；第二類自付 5～10%；第三類自付 16～30%（依經濟狀況）\n"
                "- **提醒**：住宿型機構/24H 看護常呈現「高自費、長期現金流」特性，仍是退休規劃最大尾部風險之一。"
            )
            st.caption("來源（官方）：衛福部長照專區（1966）「申請長照服務」與相關給付/支付資訊。")
        with ec2:
            ltc_df = pd.DataFrame([
                ["70 歲起醫療溢價","CPI + 1.7%","主計總處高齡家庭 CPI 實證"],
                ["醫療相關支出佔比","生活費 15%","V2.0 引擎預設參數"],
                ["No-Go 期支出調整","+10%","Blanchett 消費微笑曲線"],
                ["可節省護理支出","15–20%","具備良好體能者（Bio-Capital）"],
                ["長照保障建議","以房養老 / 房產安全墊","末端流動性補充"],
            ], columns=["參數","數值","說明"])
            st.dataframe(ltc_df, use_container_width=True, hide_index=True)
        st.divider()
        st.markdown("### 生理資本（Bio-Capital）的財務價值\n- 良好的體能可**延後進入 No-Go Phase 3–5 年**，節省護理費用 **NT$ 100–300 萬**\n- 肌力追蹤（硬舉/深蹲）、有氧能力（VO₂max）、骨密度（DEXA）\n- 建議：GK 繁榮法則加薪額度，優先配置於**預防醫學**，而非奢侈消費")

    elif edu_topic.startswith("7"):
        st.subheader("退休資產配置與 ETF 建議（台灣 2025 年）")
        alloc_df = pd.DataFrame([
            ["美股個股","6.0%","4.0–8.0%","個股風險集中；需具備選股能力"],
            ["美股 ETF (VTI/VWRA)","5.0%","3.5–7.0%","全球分散；費用率低；被動投資首選"],
            ["台股個股","5.0%","3.0–7.0%","本地資訊優勢；集中科技板塊"],
            ["台股 ETF (0050/006208)","4.0%","3.0–6.0%","追蹤台灣50；費用率低、流動性佳"],
        ], columns=["類別","預設實質報酬","合理區間","說明"])
        st.dataframe(alloc_df, use_container_width=True, hide_index=True)
        st.divider()
        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown("#### 退休前逐步調整\n| 距退休年數 | 股票 | 債券/防禦 |\n|---|---|---|\n| 10+ 年 | 80–90% | 10–20% |\n| 5–10 年 | 70–80% | 20–30% |\n| 退休時 | 60–70% | 30–40% |")
        with ec2:
            st.markdown("#### 退休後維持成長\n| 階段 | 股票 | 防禦/現金 |\n|---|---|---|\n| Go-Go（65–75） | 60–70% | 30–40% |\n| Slow-Go（75–85） | 50–60% | 40–50% |\n| No-Go（85+） | 40–50% | 50–60% |")
        st.divider()
        etf_df = pd.DataFrame([
            ["006208","富邦台50","台股","0.07%","追蹤台灣50，費用率最低"],
            ["0050","元大台灣50","台股","0.46%","台灣最知名指數ETF"],
            ["0056","元大高股息","台股高息","0.34%","高股息策略，成長性有限"],
            ["VTI","Vanguard全美ETF","美股","0.03%","覆蓋全美股票"],
            ["VWRA","Vanguard全球ETF","全球","0.22%","全球分散，對沖地緣政治風險"],
            ["00761B","國泰A級公司債","債券","0.30%","降低組合波動"],
        ], columns=["代號","名稱","類型","費用率","說明"])
        st.dataframe(etf_df, use_container_width=True, hide_index=True)
        st.markdown("#### 台灣投資人常見錯誤\n1. **過度集中台灣科技**：台股 IT 佔大盤 60%+，加上個人持股形成雙重暴露\n2. **過度仰賴高股息**：成長性不足，長期難以跑贏通膨\n3. **忽略全球分散**：台海風險是最大系統性風險，VWRA 是「地理熵減」工具\n4. **退休後全轉固定收益**：過度保守造成「長壽風險」——購買力被通膨侵蝕")

    elif edu_topic.startswith("18｜"):
        st.subheader("退休保險健檢（快速清單）")
        st.caption("用途：把保險當「風險轉嫁／避免燒穿退休金」的工具，用清單盤點缺口；非投保建議。")
        st.warning("此主題僅教育用途；不替代保單條款、也不提供商品推薦。")
        st.info("若要看更完整的保險閱讀地圖，請到分頁 **🛡️ 保險規劃（參考）**。")
        st.markdown("""
#### 1) 先盤點：你想防哪一種「退休尾部風險」？
- **短期大額醫療**：住院、手術、自費醫材、標靶藥物等
- **長期照護**：失能、失智、看護／機構照護（常是長年現金流）
- **重大傷病**：治療期長、收入中斷、復健費用
- **長壽**：活得更久，現金流不夠用（需要「終身／長期」現金流來源）

#### 2) 用三個問題定位缺口（不談商品）
- **如果明年就需要每月 3–6 萬照護費，你能撐幾年？**
- **如果突然需要一筆 100–300 萬的自費醫療支出，你會從哪裡出？**（現金？賣股？借錢？）
- **你的退休計畫裡，有沒有「醫療／照護專款」？沒有就先建立概念。**

#### 3) 保單健檢常見四個盲點
- **有效性**：主約／附約是否仍有效（停效、繳費期、續保條件）
- **理賠門檻**：長照／失能的認定方式、等待期／免責期、是否需要定期重新認定
- **通膨**：固定給付未必跟通膨連動（照護成本常上升）
- **現金流壓力**：保費會擠壓可投資金與生活費緩衝；先以年度預算試算可承受度

#### 4) 什麼時候最該做年度健檢？
- 轉職／結婚／生子／買房（責任變大）
- 50 歲前後（體況與保費結構轉折）
- 退休前 3 年（把保障與退休金流一起校準）
        """)

    elif edu_topic.startswith("8"):
        st.subheader("台灣與全球經濟（2026 觀察）")
        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown("#### 台灣（官方：中央銀行）")
            tw_df = pd.DataFrame([
                ["台灣經濟成長率（本年）", "7.28%", "中央銀行理監事會決議新聞稿（2026-03-19）"],
                ["CPI 年增率（本年）", "1.80%", "同上"],
                ["核心 CPI 年增率（本年）", "1.75%", "同上"],
                ["重貼現率", "2.0%", "同上（維持不變）"],
            ], columns=["指標", "數值", "官方來源"])
            st.dataframe(tw_df, use_container_width=True, hide_index=True)
        with ec2:
            st.markdown("#### 全球（官方：IMF）")
            st.markdown("""
- **IMF World Economic Outlook (WEO), April 2026**：提供全球與主要經濟體的成長與通膨預測（資料更新至 2026-04-01）。
- 本專案建議將「宏觀情境」視為**假設輸入**：你可用 IMF WEO 作為基準，再用「保守/中性/積極」對報酬率與通膨進行情境化。
            """)
        st.caption("資料來源（官方）：中央銀行理監事聯席會議決議新聞稿（2026-03-19）；IMF World Economic Outlook, April 2026。")
        st.markdown("| 情境 | 對投資組合影響 | 對提領策略影響 |\n|---|---|---|\n| 低通膨（< 2%）| 債券表現回穩 | 可維持較高提領率 |\n| 高通膨（> 3%）| 股票抗通膨、固定收益受損 | 應降低固定提領，採 GK 護欄 |\n| GDP 放緩 | 股市波動加大，SORR 風險上升 | 退休初期配置應更保守 |\n| 地緣政治 | 台灣科技板塊特定風險 | 全球分散 ETF 為對沖工具 |")

    elif edu_topic.startswith("9"):
        st.subheader("退休金制度改革動態（2026 觀察｜官方摘要）")
        ec1, ec2 = st.columns(2)
        with ec1:
            st.info(
                "**勞保財務精算（官方：勞動部）**\n"
                "- 勞動部依《勞工保險條例》完成每三年一次之費率與財務精算作業（2025-01-21 公告）。\n"
                "- 勞動部指出：政府已連續 6 年撥補勞保基金合計 **3,870 億元**，並搭配多元投資，有助穩定基金流量。\n"
                "- 立場：勞保為國家辦的社會保險，政府負最後支付責任。\n"
            )
        with ec2:
            st.info(
                "**勞動基金績效（官方：勞動基金運用局）**\n"
                "- 勞動基金運用局新聞稿（115-02-02）公告：整體勞動基金 114 年度收益率 **16.06%**。\n"
                "- 新制勞退基金 114 年度收益率 **15.60%**（同新聞稿）。\n"
            )
        st.caption("資料來源（官方）：勞動部〈最新勞保精算報告已完成〉（2025-01-21）；勞動基金運用局〈勞動基金114年度投資收益為歷年新高〉（115-02-02）。")
        st.divider()
        system_df = pd.DataFrame([
            ["勞保老年年金","社會保險","不計入個人帳戶","65 歲（延後中）","月投保薪資 × 年資 × 1.55%","月領終身 / 免稅"],
            ["勞退新制","個人帳戶制","歸個人所有","60 歲（年資 15 年）","專戶金額 ÷ 年金現值因子 ÷12","月領 / 一次領"],
            ["國民年金","社會保險","不計入個人帳戶","65 歲","月投保金額 × 年資 × 1.3%","未加入勞保者適用"],
        ], columns=["制度","性質","帳戶歸屬","請領年齡","計算公式","特點"])
        st.dataframe(system_df, use_container_width=True, hide_index=True)
        perf_df = pd.DataFrame([
            ["2003 年","虧損 -40%","全球股市崩潰（網路泡沫、SARS）","序列報酬風險典型案例"],
            ["2017 年","虧損 -11.33%","新興市場修正","可透過 GK 減薪護欄緩解"],
            ["2019 年","虧損 -16.28%","中美貿易戰","政治風險難以預測"],
            ["2024 年","盈餘 +16.16%","AI 驅動科技股大漲","集中科技的高波動特性"],
        ], columns=["年份","績效","主因","退休規劃啟示"])
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
        st.caption("資料來源（官方）：勞動基金運用局（績效公告/新聞稿）。歷史績效不代表未來。")
        st.info("- 勞保月領是「基礎保障」，不應視為唯一退休收入\n- 勞退個人專戶 + 自行投資 = 三層退休安全網\n- 了解並善用 GK 護欄策略，讓自有資產成為最靈活的第三層防線")

    elif edu_topic.startswith("10"):
        st.subheader("配置典範革命：傳統「年齡 = 債券比例」vs 現代上升股票路徑")
        st.markdown("""
> **核心文獻**：Pfau & Kitces (2014)「Reducing Retirement Risk with a Rising Equity Glide Path」；
> DMS Global Investment Returns Yearbook 2024；Vanguard Life-Cycle Investing Model 2025
        """)

        st.markdown("### 一、傳統規則為何正在失效")
        st.markdown("""
**「年齡等於債券比例」（Age-in-Bonds Rule）** 是 20 世紀最廣為流傳的退休配置口訣：
60 歲持 60% 債券、70 歲持 70% 債券、80 歲持 80% 債券……

這條規則源於 1950–1990 年代的環境假設：
- 債券實質報酬率 3–5%，可對抗通膨
- 平均壽命約 70–75 歲，退休期不超過 15 年
- 醫療費用佔退休支出比例相對低

**但 2025 年的環境已根本改變：**
        """)

        paradigm_df = pd.DataFrame([
            ["平均壽命（台灣）", "70–75 歲（1980 年代）", "83.7 歲（2024）；部分可達 95–100 歲",
             "退休期從 15 年延長至 30–40 年，需要更強的長期複利"],
            ["債券實質報酬率", "3–5%（高利率時代）", "0–1.5%（2015–2025）",
             "低利率環境下，重押債券等同慢速侵蝕購買力"],
            ["通膨壓力", "相對可預測", "高齡家庭 CPI 系統性高於整體 CPI",
             "醫療、長照、房租通膨加速，需要更強的資產成長對沖"],
            ["醫療費用", "退休支出的 5% 左右", "高齡期佔 20–30%，且以複利成長",
             "被低估的超支來源，債券報酬完全不足以覆蓋"],
        ], columns=["面向", "舊假設（20世紀）", "新現實（21世紀）", "對配置的影響"])
        st.dataframe(paradigm_df, use_container_width=True, hide_index=True)

        st.markdown("### 二、配置風險的核心：長壽期下的通膨與 SORR")
        st.markdown("""
這裡不引用單一「破產機率百分比」作結論（不同資料期間、資產定義、通膨口徑、再平衡規則都會大幅改變結果），而以官方可驗證的制度資料＋學術研究作為原則：

- **通膨**：長期且具累積性，對退休現金流具確定性侵蝕。
- **SORR**（序列報酬風險）：退休初期遇到大跌會放大耗盡風險；可透過現金桶、動態提領（GK 護欄）與資產配置路徑緩解。
        """)
        st.caption("研究參考：Blanchett (2014)；Guyton & Klinger (2006)；Pfau & Kitces (2014)；Kitces（Bond Tent 系列文章）。")

        st.info("**直覺反轉**：在 30 年以上的長退休期中，「越保守的配置」反而帶來越高的資產耗盡風險——"
                "因為通膨是慢而確定的毀滅，市場波動是快而可恢復的衝擊。")

        st.markdown("### 三、Pfau–Kitces 上升股票比例路徑（Rising Equity Glidepath）")
        st.markdown("""
2014 年，Wade Pfau 與 Michael Kitces 發表研究挑戰傳統智慧，提出**退休後反向增加股票比例**的策略。

**核心邏輯：**

退休後的最大風險不是「永久持有高股票」，而是「在最脆弱的時間點（退休初期）持有高股票」。
序列報酬風險（SORR）集中在退休後第 1–10 年：這段時間若遭遇熊市，
每賣一元股票以供生活費，損失的複利效應最大。

**解決方案：退休時降低股票（降低 SORR 暴露），然後隨著時間推進再緩慢拉回股票比例**
（因為後期的 SORR 敏感度已大幅降低）。
        """)

        rising_df = pd.DataFrame([
            ["退休時（65 歲）", "30–40%", "70–60%", "SORR 最高風險期；先以債券/現金作緩衝（「債券帳篷」）"],
            ["退休 5–10 年（70–75 歲）", "40–50%", "60–50%", "危險地帶度過後，逐步回補股票部位"],
            ["退休 15 年（80 歲）", "50–60%", "50–40%", "長壽風險成為主要威脅；股票成長力必須拉高"],
            ["退休 20 年以上（85+）", "50–70%", "50–30%", "對抗通膨+長照費用；股票比例可高於傳統建議"],
        ], columns=["退休年資", "股票比例（上升路徑）", "債券/現金比例（下降）", "邏輯"])
        st.dataframe(rising_df, use_container_width=True, hide_index=True)

        st.markdown("### 四、債券帳篷（Bond Tent）策略")
        st.markdown("""
**Bond Tent** 是上升股票路徑的具體實作工具，由 Kitces (2017) 正式命名：

```
股票比例（概念圖）
         ▲
  80% ───┤                          ╱──── 上升路徑（後半段）
  60% ───┤      ╲               ╱
  50% ───┤        ╲           ╱
  40% ───┤          ╲       ╱
  30% ───┤            ╲   ╱ ← 帳篷頂點（退休當天最低股票）
         │──────────────────────────────►
        55歲        65歲         80歲       年齡
              退休前逐步降低  退休後逐步拉高
```

**操作步驟：**
1. **退休前 5–7 年（55–60 歲）**：開始把股票比例從 70% 降至 40–50%，建立「帳篷前側坡」
2. **退休當天（65 歲）**：股票比例在最低點（30–40%），現金/短期債大量持有
3. **退休後逐年**：每年將股票比例提高 1–2%，建立「帳篷後側坡」
4. **80–85 歲**：股票比例回升至 50–60%，維持對通膨+長壽的保護

**Bond Tent 的關鍵洞見**：
- 低點股票不是因為老了就要保守，而是因為 **SORR 敏感度在退休初期最高**，保護的是最危險的那幾年
- 後段拉回股票不是因為變積極，而是因為 **長壽風險超越了 SORR 風險**，通膨才是晚年的頭號殺手
        """)

        st.markdown("### 五、CAPE 估值調整（進階）")
        st.markdown("""
Pfau 與 Kitces 的延伸研究（2014）發現：**當退休時股市 CAPE 比率處於歷史高位時，應加速「Bond Tent」的防禦深度**。

| 退休當天 CAPE | 退休起點建議股票比例 | 邏輯 |
|---|---|---|
| CAPE < 15（低估）| 50–60% | 未來預期報酬高，可承受更多股票 |
| CAPE 15–25（正常）| 40–50% | 標準 Bond Tent 配置 |
| CAPE 25–35（偏高）| 30–40% | 未來報酬下修預期，加深防禦 |
| CAPE > 35（高估）| 20–30% | 最大防禦；類 2000 年科技泡沫前夕 |

2025 年美股 CAPE 約 34–36，處於歷史高位，**依此標準應採取較深的 Bond Tent（退休起點股票 25–35%）**，
然後隨著時間推進逐步回升至 55–65%。
        """)

        st.markdown("### 六、台灣本土化應用")
        tw_apply_df = pd.DataFrame([
            ["55–60 歲", "債券帳篷前坡", "0050 → 50%，0056/00878 → 20%，短債/定存 → 30%",
             "退休前開始建立緩衝"],
            ["65 歲（退休當天）", "帳篷頂點（最低股票）", "0050 → 25%，0056/00878 → 20%，現金/短債 → 55%",
             "SORR 最高風險；三桶金現金桶充足"],
            ["70–75 歲", "後坡開始爬升", "0050 → 30%，0056/00878 → 25%，中期債 → 45%",
             "SORR 敏感度降低；緩步回補股票"],
            ["80–85 歲", "長壽防禦期", "0050 → 35%，0056/00878 → 30%，現金/債 → 35%",
             "長壽+通膨雙重壓力；股票比例回升超越傳統建議"],
            ["85 歲以上", "通膨對抗期", "0050 → 40%，配息 ETF → 30%，現金 → 30%",
             "傳統規則在此年齡會建議 85% 債券；現代研究反對此做法"],
        ], columns=["年齡", "Bond Tent 階段", "台灣 ETF 配置建議", "邏輯"])
        st.dataframe(tw_apply_df, use_container_width=True, hide_index=True)

        col_warn1, col_warn2 = st.columns(2)
        with col_warn1:
            st.error("**傳統錯誤**\n\n退休後持續降低股票 → 80 歲持 80% 債券\n\n在超低利率 + 高通膨環境下，這等同於「慢性破產計畫」")
        with col_warn2:
            st.success("**現代共識**\n\n退休初期用 Bond Tent 保護 SORR，之後隨年齡「逆傳統」地回升股票比例，以對抗長壽+通膨的雙重壓力")

        st.caption(
            "文獻：① Pfau & Kitces (2014) 'Reducing Retirement Risk with a Rising Equity Glide Path' "
            "② Kitces (2017) 'The Bond Tent: Managing Portfolio Size Effect in the Retirement Red Zone' "
            "③ Vanguard VLCM Life-Cycle Model (2025)"
        )

    # ── 11. 持有房產與心理帳戶陷阱 ────────────────────────────────────────
    elif edu_topic.startswith("11"):
        st.subheader("持有房產與心理帳戶陷阱 (Holding Real Estate & Mental Accounting)")
        st.markdown("""
> **行為財務學依據**：Richard Thaler (1985, 1999)「Mental Accounting Matters」；
> Shefrin & Statman (1985)「The Disposition Effect」；
> 台灣政大財金系 2024 退休資產調查。
        """)

        st.markdown("### 什麼是心理帳戶（Mental Accounting）？")
        st.markdown("""
心理帳戶是行為財務學最核心的偏誤之一：人們會把相同的「錢」放進不同的「心理帳戶」，
給予不同的心理重量與花費規則——即使在財務上完全等值。

**對退休族的典型表現**：
- 「這棟房子是我家，不是投資工具，我不能動它」
- 「股票的錢可以花，但房子的錢不算錢」
- 「孩子以後可以繼承，不能賣」
- 「我買時才 300 萬，現在漲到 2000 萬——怎麼捨得賣？」
        """)

        st.markdown("### 五大房產心理陷阱")
        traps_df = pd.DataFrame([
            ["① 房屋不算資產的幻覺",
             "將自住房排除在退休計算外，低估實際淨資產",
             "台灣家庭金融調查：65 歲以上家庭，房產佔總資產平均 **72%**，但大多數人在計算退休金時完全忽略這部分",
             "在本 App 啟用「🏠 不動產」模組，強迫自己計入房產淨值"],
            ["② 稟賦效應（Endowment Effect）",
             "高估自有房產的價值；即使市場已充分定價，仍覺得「賣掉很虧」",
             "Kahneman (1990) 實驗顯示：人們願意出售自有資產的最低價格，平均比願意購買同資產的最高價格高出 **2.2 倍**",
             "強制以「市場估值 × 0.90」的保守角度計算（折讓仲介費、稅、搬家成本）"],
            ["③ 機會成本盲點",
             "房屋閒置或低效使用，卻不計算其「應該創造的現金流」",
             "一棟市值 2000 萬的台北市公寓，若能出租月租 3.5 萬，年現金流 42 萬；等效資本報酬率 2.1%。但若閒置，機會成本每年超過 **60 萬**（按無風險利率 3%）",
             "將房產的「機會成本」加入退休計算，即使自住也要計算「放棄的收入」"],
            ["④ 傳承偏誤（Legacy Bias）",
             "為了留給子女而犧牲自身退休品質",
             "研究顯示：台灣 65 歲以上退休族中，約 **61%** 表示「不打算動用房產因為要留給孩子」，但同族群中有 **34%** 表示退休金不夠用",
             "先確保自身退休現金流充足；遺產規劃應在「我的退休品質有保障後」才考慮"],
            ["⑤ 槓桿低估（Leverage Underestimation）",
             "忽略房貸槓桿對資產負債表的實質影響",
             "房市值 2500 萬、房貸 1000 萬 → 淨資產 1500 萬，但槓桿倍數 1.67x；若房市跌 20%，淨資產縮水 **33%**（從 1500 萬跌至 1000 萬）",
             "本 App「未償房貸餘額」欄位會自動計算淨值，確保你看到的是真實淨資產"],
        ], columns=["陷阱", "偏誤描述", "數據佐證", "應對方法"])
        st.dataframe(traps_df, use_container_width=True, hide_index=True)

        st.markdown("### 房產的「流動性折價」：被鎖定的財富")
        st.markdown("""
相較於股票（T+2 可變現），不動產的變現需要：

| 步驟 | 所需時間 | 成本 |
|---|---|---|
| 委託仲介刊登 | 1–4 週 | 0 |
| 看屋協商議價 | 1–6 個月 | 精力成本 |
| 簽約至交屋 | 1–3 個月 | 仲介費 2–6%、代書費 |
| 資本利得稅（房地合一 2.0） | 持有 < 2 年：45%；2–5 年：35%；5–10 年：20%；> 10 年：15% | 視持有年限 |

**退休流動性危機情境**：
- 突發醫療費用 200 萬，但現金只有 50 萬
- 房產市值 2000 萬但短期無法變現
- 被迫以高利率信用貸款應急 → 反而支付更多利息

> **解方**：現金（三桶金第一桶）≥ 2 年生活費；以「以房養老」機制預先安排長期流動性。
        """)

        st.markdown("### 台灣不動產的特殊性：情感資本（Emotional Capital）")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.info(
                "**台灣文化脈絡**\n\n"
                "- 「有土斯有財」的農業社會遺緒\n"
                "- 房產代表「成就感」與「社會地位」\n"
                "- 家族共同記憶（子女從小住的家）\n"
                "- 鄰居、社區關係網絡的錨定\n\n"
                "→ 這些是真實的「情感報酬」，不應完全忽視，\n"
                "但不應成為阻礙退休財務優化的唯一理由"
            )
        with col_e2:
            st.warning(
                "**情感資本的財務代價**\n\n"
                "- 持有 2000 萬自住宅（無房貸）\n"
                "- 若出租月租 3.5 萬：年現金流 **+42 萬**\n"
                "- 若以房養老（月領 4 萬）：年現金流 **+48 萬**\n"
                "- 若縮小換屋（換 1000 萬小宅）：釋出 **1000 萬** 可投資資金\n\n"
                "每年「選擇不行動」的隱性代價：42–80 萬元"
            )

        st.markdown("### 如何正確將房產整合進退休計畫")
        framework_df = pd.DataFrame([
            ["步驟 1", "計算房產淨值",
             "市值 × 0.90（扣仲介+稅） - 未償房貸 = 真實淨資產",
             "本 App 側欄「🏠 不動產」模組自動計算"],
            ["步驟 2", "評估流動性需求",
             "三桶金第一桶 = 2 年生活費（現金、貨幣基金）",
             "若三桶金不足，考慮部分房產變現或以房養老"],
            ["步驟 3", "計算租金現金流機會",
             "市值 × 2–2.5% = 估算年租金（台北市毛租金收益率）",
             "扣除管理費、空置率（約 10–15%）= 淨收益"],
            ["步驟 4", "決定房產策略",
             "自住不動 / 出租 / 縮小換屋 / 以房養老（擇一或組合）",
             "本 App 四個欄位：出租月收入、以房養老月領、啟動年齡"],
            ["步驟 5", "定期重估",
             "每 2–3 年重新評估市值與策略適合性",
             "房市大漲時考慮縮小換屋釋出資金；房市回落時暫緩"],
        ], columns=["步驟", "內容", "計算方法", "本 App 操作"])
        st.dataframe(framework_df, use_container_width=True, hide_index=True)

        st.markdown("### 「以房養老」vs「賣房投資」：哪個更划算？")
        st.markdown("""
假設：**台北自住宅市值 2000 萬，無房貸，65 歲退休**
        """)
        compare_df = pd.DataFrame([
            ["維持自住（不動）",
             "0", "0",
             "繼承完整；熟悉環境", "機會成本最高；完全依靠金融資產"],
            ["以房養老（月領 4 萬）",
             "4 萬/月 = 48 萬/年", "2000 萬（直到身故）",
             "不用搬家；月現金流改善", "身故後房產歸銀行；需繼續負擔房屋稅/管理費"],
            ["出租 + 租屋（縮小）",
             "月租收 4.5 萬 - 月租付 1.5 萬 = 3 萬淨", "2000 萬（仍持有）",
             "現金流改善；資產保留", "需面對搬遷、租屋管理麻煩"],
            ["賣房換 1000 萬小宅 + 投資 1000 萬",
             "投資年化 5% → 50 萬/年", "1000 萬（小宅）+ 投資組合",
             "現金流最高；靈活性最大", "搬家；失去原社區連結"],
        ], columns=["策略", "年現金流改善", "房產殘值", "優點", "缺點"])
        st.dataframe(compare_df, use_container_width=True, hide_index=True)

        st.success(
            "**核心結論**：房產是退休財務的「沉默的大象」——它體積龐大、情感複雜、難以移動。"
            "心理帳戶偏誤讓我們把它排除在退休計算外，但它往往是最大的單一資產。"
            "「看見它、計入它、規劃它」，才能讓退休計畫建立在真實的財務版圖上。"
        )
        st.caption(
            "文獻：① Thaler (1985, 1999) 'Mental Accounting Matters' "
            "② Kahneman, Knetsch & Thaler (1990) 'Endowment Effect' "
            "③ 台灣政大財金系家庭金融調查 2024 "
            "④ 內政部不動產資訊平台 2025 "
            "⑤ 台灣金融研訓院「以房養老方案比較」2025"
        )

    # ── 12. 退休資產負債表 ────────────────────────────────────────────────
    elif edu_topic.startswith("12"):
        st.subheader("退休資產負債表：自住房產的防禦性與行為盲點")
        st.markdown("""
> **理論依據**：Franco Modigliani (1954) 生命週期理論；
> Shiller (2005)「Irrational Exuberance」；
> Vanguard Research「Home Bias in Retirement Portfolio」(2024)
        """)

        st.markdown("""
在退休資產負債表中，**自住房產往往是佔比最大的非流動性資產**——
它同時具備兩種截然對立的屬性，這正是多數人退休規劃失準的核心原因之一：

| 屬性 | 說明 |
|---|---|
| **防禦性** | 通膨對沖、強迫儲蓄、居住成本鎖定、心理安全感 |
| **行為盲點** | 流動性不足、機會成本隱形、估值高估、傳承偏誤 |
        """)

        # ── 互動式退休資產負債表 ──────────────────────────────────────────
        st.markdown("### 你的退休資產負債表（即時試算）")
        st.caption("數值自動讀取左側「資產與提領」及「🏠 不動產」設定，也可在下方另行輸入。")

        bs_c1, bs_c2 = st.columns(2)
        with bs_c1:
            bs_sec_wan     = st.number_input("有價證券總額 (萬)",  min_value=0, max_value=100_000,
                                             value=int(A0_securities_wan), step=100,
                                             key="bs_sec")
            bs_home_wan    = st.number_input("自用住宅市值 (萬)", min_value=0, max_value=50_000,
                                             value=int(re_home_wan), step=100,
                                             key="bs_home")
            bs_rental_wan  = st.number_input("出租房產市值 (萬)", min_value=0, max_value=50_000,
                                             value=int(re_rental_wan), step=100,
                                             key="bs_rental")
        with bs_c2:
            bs_cash_wan    = st.number_input("現金/定存/貨幣基金 (萬)", min_value=0, max_value=10_000,
                                             value=0, step=50,
                                             key="bs_cash",
                                             help="三桶金第一桶；通常為 1–2 年生活費")
            bs_pension_wan = st.number_input("勞保/勞退 現值估算 (萬)", min_value=0, max_value=5_000,
                                             value=0, step=50,
                                             key="bs_pension",
                                             help="月領 × 12 ÷ 4% 可估算等值資本（年金現值法）；選填")
            bs_mortgage_wan= st.number_input("未償房貸餘額 (萬)", min_value=0, max_value=20_000,
                                             value=int(re_mortgage_wan), step=50,
                                             key="bs_mort")

        # 計算資產負債表各項目
        bs_total_liquid   = bs_sec_wan + bs_cash_wan          # 流動性資產
        bs_total_illiquid = bs_home_wan + bs_rental_wan        # 非流動性資產
        bs_pension_pv     = bs_pension_wan                     # 年金現值（補充）
        bs_total_assets   = bs_total_liquid + bs_total_illiquid + bs_pension_pv
        bs_total_liab     = bs_mortgage_wan
        bs_net_worth      = bs_total_assets - bs_total_liab

        # 流動性比率
        liquidity_ratio   = bs_total_liquid / bs_net_worth * 100 if bs_net_worth > 0 else 0
        illiquidity_ratio = bs_total_illiquid / bs_net_worth * 100 if bs_net_worth > 0 else 0
        home_ratio        = bs_home_wan / bs_net_worth * 100 if bs_net_worth > 0 else 0

        # 顯示資產負債表
        st.markdown("#### 資產負債表")
        bs_asset_rows = [
            ["【流動性資產】", "", ""],
            ["　有價證券（股票/ETF/基金）", f"{bs_sec_wan:,} 萬", "高流動性｜T+2 可變現"],
            ["　現金 / 定存 / 貨幣基金",    f"{bs_cash_wan:,} 萬", "最高流動性｜隨時可用"],
            ["【非流動性資產】", "", ""],
            ["　自用住宅市值",               f"{bs_home_wan:,} 萬", "低流動性｜變現需 3–9 個月"],
            ["　出租房產市值",               f"{bs_rental_wan:,} 萬", "低流動性｜但能產生現金流"],
            ["【年金現值（補充）】", "", ""],
            ["　勞保/勞退 估算現值",         f"{bs_pension_wan:,} 萬", "隱性資產｜不可轉讓但有現金流"],
            ["", "", ""],
            ["**資產合計**",                f"**{bs_total_assets:,} 萬**", ""],
            ["【負債】", "", ""],
            ["　未償房貸餘額",               f"{bs_mortgage_wan:,} 萬", ""],
            ["**負債合計**",                f"**{bs_total_liab:,} 萬**", ""],
            ["", "", ""],
            ["**淨資產（總資產 − 負債）**", f"**{bs_net_worth:,} 萬**", "**你的真實退休財富**"],
        ]
        bs_df = pd.DataFrame(bs_asset_rows, columns=["項目", "金額", "備註"])
        st.dataframe(bs_df, use_container_width=True, hide_index=True)

        # 流動性評估
        st.markdown("#### 資產流動性診斷")
        diag_c1, diag_c2, diag_c3 = st.columns(3)
        with diag_c1:
            st.metric("流動性資產佔淨資產", f"{liquidity_ratio:.1f}%",
                      help="建議退休組合中流動性資產 ≥ 40%")
        with diag_c2:
            st.metric("非流動性（房產）佔淨資產", f"{illiquidity_ratio:.1f}%",
                      help="台灣平均 65+ 家庭約 72%；超過 60% 需特別注意流動性風險")
        with diag_c3:
            st.metric("自住宅佔淨資產", f"{home_ratio:.1f}%",
                      help="自住宅無法直接產生現金流；超過 50% 代表「富有但現金貧乏」")

        # 流動性警示
        if liquidity_ratio < 20:
            st.error(
                f"⚠️ **流動性嚴重不足**：流動資產僅佔淨資產 {liquidity_ratio:.1f}%。"
                "退休初期若遭遇熊市或醫療支出，將被迫以不利條件處分房產。"
                "建議：增加有價證券配置，或規劃以房養老提升流動性。"
            )
        elif liquidity_ratio < 40:
            st.warning(
                f"⚠️ **流動性偏低**：流動資產佔淨資產 {liquidity_ratio:.1f}%，"
                "低於建議的 40%。當市場下跌時，可能因缺乏流動性而被迫在低點賣出。"
            )
        else:
            st.success(
                f"✅ **流動性良好**：流動資產佔淨資產 {liquidity_ratio:.1f}%，"
                "能夠應對市場波動與突發支出需求。"
            )

        # 「富有但現金貧乏」陷阱
        st.markdown("---")
        st.markdown("### 自住房產的防禦性：三大保護機制")
        defense_df = pd.DataFrame([
            ["通膨對沖（Inflation Hedge）",
             "房價長期與通膨高度正相關",
             "台灣六都房價 2015–2025 年均漲幅約 7–12%；遠超 CPI（1.8%）",
             "持有房產是對抗「購買力侵蝕」的天然工具"],
            ["強迫儲蓄（Forced Saving）",
             "每月繳房貸 = 強制積累房產淨值",
             "相比租屋族，自有住宅者退休時淨資產平均高出 40–60%（主計總處 2024）",
             "心理上不易動用，形成退休資產的「硬核」"],
            ["居住成本鎖定（Housing Cost Lock）",
             "無房貸後，每月最大固定支出消失",
             "若退休後無房貸，每月省下 2–5 萬（視地區），等效每年 24–60 萬的「被動節省」",
             "降低所需提領金額，讓退休規劃更從容"],
        ], columns=["防禦機制", "原理", "台灣數據", "退休意義"])
        st.dataframe(defense_df, use_container_width=True, hide_index=True)

        st.markdown("### 自住房產的行為盲點：四大認知偏誤")
        blindspot_df = pd.DataFrame([
            ["「不算錢」幻覺\n（心理帳戶偏誤）",
             "把房產放入獨立帳戶，從退休計算中排除",
             f"你的自住宅市值 **{bs_home_wan:,} 萬**，若從退休計算中排除，等同低估總資產 {bs_home_wan/(bs_total_assets+0.001)*100:.0f}%"],
            ["估值高估\n（稟賦效應）",
             "認為自己的房子比市場定價更值錢",
             "建議以「市值 × 0.90」計算（含仲介 2–4%、稅務成本、時間成本）；本 App 已預設保守估算"],
            ["「留給子女」凍結\n（傳承偏誤）",
             "退休金不足，但房產因遺產考量而凍結不動",
             "先確保自身 30 年退休現金流充足；子女可繼承「更多有價證券」而非「一棟難以分割的房子」"],
            ["非流動性低估\n（可得性偏誤）",
             "因房價長期上漲而忽略短期無法變現的現實",
             f"你的非流動性資產 **{bs_total_illiquid:,} 萬** 佔淨資產 {illiquidity_ratio:.0f}%；"
             "緊急時無法在一週內變現，需預留充足現金緩衝"],
        ], columns=["行為偏誤", "描述", "你的情境（試算）"])
        st.dataframe(blindspot_df, use_container_width=True, hide_index=True)

        st.markdown("### 最佳化建議：「讓房產發揮應有功能」")
        st.markdown(f"""
根據你目前的資產負債表，以下是對應的優化方向：

| 現況 | 建議 | 本 App 操作 |
|---|---|---|
| 流動資產 {liquidity_ratio:.0f}%（{"充足" if liquidity_ratio >= 40 else "不足"}） | {"維持現有配置" if liquidity_ratio >= 40 else "增加有價證券或現金配置至 40%+"} | 左側「有價證券總額」調整 |
| 自住宅佔 {home_ratio:.0f}% | {"注意流動性" if home_ratio > 50 else "比例合理"} | 考慮以房養老釋放月現金流 |
| 房貸餘額 {bs_mortgage_wan:,} 萬 | {"退休前優先還清以降低固定支出" if bs_mortgage_wan > 0 else "無房貸，居住成本已鎖定"} | 左側「未償房貸」更新 |
| 年金現值 {bs_pension_wan:,} 萬 | {"強大的底層保障，已計入" if bs_pension_wan > 0 else "填入勞保/勞退可更精準估算"} | 左側「勞保/勞退月領」輸入 |
        """)

        st.success(
            "**核心洞見**：退休資產負債表的目標，不是「最大化總資產」，"
            "而是「確保足夠的流動性資產支撐 30 年提領，同時讓非流動性房產發揮通膨對沖的防禦功能」。"
            "兩者缺一不可。"
        )
        st.caption(
            "文獻：① Modigliani & Brumberg (1954) 生命週期假說 "
            "② Shiller (2005) 'Irrational Exuberance' Ch.2 "
            "③ Vanguard 'Home Bias and Retirement Portfolios' (2024) "
            "④ 主計總處「家庭收支調查」2024 "
            "⑤ 台灣內政部不動產資訊平台 2025"
        )

    # ── 13. 長照對沖與房產資金階梯 ──────────────────────────────────────
    elif edu_topic.startswith("13"):
        st.subheader("長照對沖與房產資金階梯（LTC Hedge & Asset Ladder）")
        st.markdown("""
> **依據**：衛福部「長期照顧十年計畫 2.0」(2025)；
> Genworth Cost of Care Survey (2024)；
> Pfeiffer et al. (2019)「The Glidepath Illusion」；
> 台灣長照保險籌備報告 (2024)
        """)

        st.markdown("""
退休最難對沖的風險，不是市場崩盤，而是**你還活著但已無法照顧自己**。
長照費用在「No-Go Years（停滯期，85歲+）」將急遽攀升，而此時金融資產可能已大量耗盡。
**自住房產，正是為這個階段而存在的終極對沖工具**——
前提是：你願意打破心理帳戶的封印，讓它在對的時間點發揮作用。
        """)

        # 長照成本時間軸
        st.markdown("### 退休三階段的長照費用軌跡")
        ltc_phase_df = pd.DataFrame([
            ["Go-Go Years\n活躍期", "65–75 歲",
             "低至中等", "0–5 萬/年",
             "偶爾門診、健檢；長照需求幾乎為零",
             "金融資產提領期；應積累三桶金緩衝"],
            ["Slow-Go Years\n緩速期", "75–85 歲",
             "中至高", "12–36 萬/年",
             "居家照服員 1–2 次/週；輔具需求；部分交通協助",
             "消費微笑曲線上揚；本 App 醫療溢價開始顯著"],
            ["No-Go Years\n停滯期", "85 歲以上",
             "高至極高", "60–200 萬/年",
             "住宿式機構（安養院 3–10 萬/月）；24 小時居家看護（6–18 萬/月）；失智症照護費用更高",
             "⚠️ 此時若金融資產耗盡，房產是最後防線"],
        ], columns=["階段", "年齡", "長照需求程度", "估算年費用", "主要需求項目", "財務規劃意涵"])
        st.dataframe(ltc_phase_df, use_container_width=True, hide_index=True)
        st.caption("台灣 2025 數據：住宿式機構每月自費約 3–8 萬；外籍看護月薪約 2.5–3 萬（不含加班費）；本國看護月薪約 4–6 萬。")

        # 長照資金缺口互動試算
        st.markdown("### 長照資金缺口試算")
        ltc_c1, ltc_c2 = st.columns(2)
        with ltc_c1:
            ltc_onset_age  = st.number_input("預估長照需求開始年齡 (歲)", min_value=70, max_value=95,  value=82, step=1,
                                             help="台灣平均失能年齡約 80–85 歲")
            ltc_end_age_default = int(min(105, max(int(ltc_onset_age) + 1, 95)))
            ltc_end_age    = st.number_input("規劃至年齡 (歲)",           min_value=ltc_onset_age+1, max_value=105, value=ltc_end_age_default, step=1)
        with ltc_c2:
            ltc_monthly_wan = st.number_input("月長照費用估算 (萬/月)", min_value=0.5, max_value=20.0, value=5.0, step=0.5,
                                              help="住宿機構約 4–8 萬/月；居家 24H 看護約 5–12 萬/月")
            ltc_coverage_wan= st.number_input("長照險/長照補助可抵用 (萬/月)", min_value=0.0, max_value=10.0, value=1.5, step=0.5,
                                              help="勞保失能給付 + 長照補助（依當年度制度）+ 長照險理賠合計")

        ltc_years      = max(0, ltc_end_age - ltc_onset_age)
        ltc_net_monthly = max(0.0, ltc_monthly_wan - ltc_coverage_wan)
        ltc_annual_gap  = ltc_net_monthly * 12
        ltc_total_gap   = ltc_annual_gap * ltc_years
        # 現值折算（以 3% 折現，退休初期至 ltc_onset_age 之間的年數）
        yrs_to_onset    = max(0, ltc_onset_age - age_start)
        ltc_pv          = ltc_total_gap / ((1.03) ** yrs_to_onset) if yrs_to_onset > 0 else ltc_total_gap

        col_ltc1, col_ltc2, col_ltc3 = st.columns(3)
        with col_ltc1:
            st.metric("長照期間", f"{ltc_years} 年", f"{ltc_onset_age}–{ltc_end_age} 歲")
        with col_ltc2:
            st.metric("每年自付缺口", f"{ltc_annual_gap:.0f} 萬/年",
                      f"月淨缺口 {ltc_net_monthly:.1f} 萬")
        with col_ltc3:
            st.metric("長照總資金需求（名目）", f"{ltc_total_gap:.0f} 萬",
                      f"現值約 {ltc_pv:.0f} 萬（3% 折現）")

        if ltc_total_gap > 0:
            st.info(
                f"**你需要在 {ltc_onset_age} 歲前，額外準備 {ltc_pv:.0f} 萬（今日購買力）專用於長照**。\n\n"
                f"若金融資產屆時不足，**市值 {re_home_wan:,} 萬的自住宅**是最直接的備援來源，"
                f"透過 Downsizing 或以房養老均可挹注此缺口。"
            )

        # 資金階梯（Ladder）概念
        st.markdown("### 房產資金階梯（Asset Ladder）：三段式釋放策略")
        st.markdown("""
資金階梯不是一次性出售，而是**依照生命週期分階段解鎖房產價值**，
在正確的時間點提供正確的資金，同時避免過早犧牲居住品質。
        """)

        ladder_df = pd.DataFrame([
            ["第一梯\n65–75 歲（活躍期）",
             "維持自住，不動房產",
             "金融資產（有價證券+現金）",
             "享受居住品質；房產靜態升值；心理安全感維持",
             "此時動用房產往往太早，機會成本高"],
            ["第二梯\n75–85 歲（緩速期）",
             "出租部分空間 / 考慮 Downsizing",
             "金融資產 + 少量房產現金流",
             "子女可能已離家；大房變成負擔（管理、稅、維護費）；Downsizing 釋出 500–1500 萬",
             "⭐ 最佳 Downsizing 視窗：體力尚可搬遷，認知功能完整"],
            ["第三梯\n85 歲以上（停滯期）",
             "以房養老 / 出售房產支應機構費用",
             "房產為主要現金流來源",
             "長照費用急升；金融資產可能耗盡；以房養老月領可直接補足缺口",
             "此時才啟動是萬不得已的安全網，但仍優於毫無準備"],
        ], columns=["階梯", "房產策略", "主要資金來源", "說明", "注意事項"])
        st.dataframe(ladder_df, use_container_width=True, hide_index=True)

        # Downsizing 試算
        st.markdown("### Downsizing（大屋換小屋）釋出資金試算")
        ds_c1, ds_c2 = st.columns(2)
        with ds_c1:
            ds_current_wan = st.number_input("目前住宅市值 (萬)",   min_value=0, max_value=20_000,
                                             value=int(re_home_wan), step=100, key="ds_curr")
            ds_target_wan  = st.number_input("換屋目標市值 (萬)",   min_value=0, max_value=10_000,
                                             value=int(re_home_wan * 0.5) if re_home_wan > 0 else 500,
                                             step=100, key="ds_tgt",
                                             help="通常換至 1–2 房小宅，約現宅的 40–60%")
            ds_acquisition_wan = st.number_input(
                "原始取得成本 (萬)",
                min_value=0, max_value=20_000,
                value=int(re_home_wan * 0.4) if re_home_wan > 0 else 0,
                step=100, key="ds_acq",
                help=(
                    "當初購買此房產支付的總成本（含房價、裝修、代書費等），"
                    "房地合一稅以「出售價 − 取得成本 − 交易費用」為稅基，"
                    "填入後稅額計算更精確。若不確定可保留預設估值（現值 × 40%）。"
                ),
            )
        with ds_c2:
            ds_hold_yrs    = st.number_input("持有年數 (年，用於計算房地合一稅)", min_value=0, max_value=50, value=15, step=1)
            ds_cost_pct    = st.number_input("交易成本（仲介+代書+搬家，%）", min_value=0.0, max_value=10.0, value=4.0, step=0.5)

        # 房地合一稅率（依持有年數）
        if ds_hold_yrs < 2:
            ltx_rate = 0.45
        elif ds_hold_yrs < 5:
            ltx_rate = 0.35
        elif ds_hold_yrs < 10:
            ltx_rate = 0.20
        else:
            ltx_rate = 0.15

        # 正確稅基 = 出售價 − 取得成本 − 交易費用（非「差價」）
        ds_cost           = ds_current_wan * ds_cost_pct / 100
        ds_taxable_gain   = max(0.0, ds_current_wan - ds_acquisition_wan - ds_cost)
        ds_tax            = ds_taxable_gain * ltx_rate
        # 換屋淨釋出 = 現宅 − 小宅 − 稅 − 交易成本
        ds_gross_gain     = max(0, ds_current_wan - ds_target_wan)
        ds_net_release    = ds_gross_gain - ds_tax - ds_cost
        _tax_basis_note   = f"（稅基 = 市值{ds_current_wan:,} − 取得成本{ds_acquisition_wan:,} − 交易費{ds_cost:.0f} = {ds_taxable_gain:.0f} 萬）"

        col_ds1, col_ds2, col_ds3 = st.columns(3)
        with col_ds1:
            st.metric("換屋毛釋出金額（現宅 − 小宅）", f"{ds_gross_gain:.0f} 萬")
        with col_ds2:
            st.metric(
                f"房地合一稅（{ltx_rate*100:.0f}%，持有 {ds_hold_yrs} 年）",
                f"−{ds_tax:.0f} 萬",
                _tax_basis_note,
            )
        with col_ds3:
            st.metric("淨釋出資金",      f"{ds_net_release:.0f} 萬",
                      f"可覆蓋 {ds_net_release/ltc_annual_gap:.1f} 年長照" if ltc_annual_gap > 0 else "")

        if ds_net_release > 0 and ltc_total_gap > 0:
            cover_pct = ds_net_release / ltc_total_gap * 100
            if cover_pct >= 100:
                st.success(f"✅ Downsizing 淨釋出 **{ds_net_release:.0f} 萬**，可完整覆蓋長照總缺口 {ltc_total_gap:.0f} 萬（覆蓋率 {cover_pct:.0f}%）")
            elif cover_pct >= 50:
                st.warning(f"⚠️ Downsizing 可覆蓋長照缺口的 **{cover_pct:.0f}%**（{ds_net_release:.0f}/{ltc_total_gap:.0f} 萬）；差額需由金融資產或以房養老補足")
            else:
                st.error(f"❗ Downsizing 僅能覆蓋長照缺口的 **{cover_pct:.0f}%**；建議同步規劃長照險與以房養老")

        # 心理帳戶的封印
        st.markdown("---")
        st.markdown("### 心理帳戶的封印：為何這個計畫在現實中常常失敗？")
        st.markdown("""
即使數字清楚顯示 Downsizing 或以房養老是最優解，
許多退休者仍會選擇「縮衣節食」而非「動用房產」。這是行為財務學最典型的**心理帳戶固化（Mental Account Freezing）**：
        """)

        col_ma1, col_ma2 = st.columns(2)
        with col_ma1:
            st.error(
                "**心理帳戶封印的五個聲音**\n\n"
                "1. 「這是我一輩子打拼的家，不能賣」\n"
                "2. 「孩子還會回來住，不能換小的」\n"
                "3. 「現在不是好時機，等房價更高再說」\n"
                "4. 「賣掉就什麼都沒了，心裡不踏實」\n"
                "5. 「我撐得過去，不需要靠房子」"
            )
        with col_ma2:
            st.success(
                "**打破封印的認知重框**\n\n"
                "1. 「家是記憶，不是磚頭；記憶住在心裡」\n"
                "2. 「孩子希望你過得好，而不是住在大房子裡受苦」\n"
                "3. 「最好的時機是你還有體力與認知能力決策的時候」\n"
                "4. 「釋出的資金讓你有能力選擇更好的長照品質」\n"
                "5. 「預先規劃是主動選擇，不是失敗的退場」"
            )

        st.markdown("""
**行為財務學洞見**：心理帳戶偏誤在長照決策上的殺傷力，
往往不是讓人「破產」，而是讓人「有錢有房卻活得很苦」——
被困在維護成本高昂的大房子裡，卻無法負擔應有的照護品質。
        """)

        st.markdown("### 最佳執行時機：在認知功能完整時做決定")
        st.warning(
            "**關鍵洞見**：Downsizing 的最佳視窗是 **75–80 歲**（體力尚可、認知功能完整、家庭溝通有效）。"
            "若等到 85 歲以後才決策，可能面臨：認知退化無法簽約、身體虛弱無法搬遷、"
            "子女意見分歧導致決策癱瘓。\n\n"
            "**建議**：在 65–70 歲（退休初期）就與家人明確討論並記錄「長照備援計畫」，"
            "包含 Downsizing 觸發條件、以房養老的啟動標準，以及長照資金的優先使用順序。"
        )
        st.caption(
            "文獻：① 衛福部「長期照顧十年計畫 2.0」2025 "
            "② Genworth Cost of Care Survey 2024 "
            "③ Pfeiffer, Finke & Blanchett (2019) 'The Glidepath Illusion' "
            "④ Thaler (1999) 'Mental Accounting Matters' "
            "⑤ 台灣長照保險籌備報告 2024 "
            "⑥ 內政部不動產資訊平台 2025"
        )

    # ── 14. 不動產收益：租金、殖利率與 REITs ────────────────────────────
    elif edu_topic.startswith("14"):
        st.subheader("不動產收益：租金殖利率、現金流計算與 REITs 替代方案")
        st.markdown("""
> **依據**：內政部不動產資訊平台（2025）；
> 信義房屋「2025 台灣租金殖利率調查」；
> NAREIT Global Real Estate Index Series (2024)；
> 台灣證交所 REITs 資料庫 (2025)
        """)

        st.markdown("""
退休後，不動產收益可扮演「底層現金流錨」的角色——
與勞保年金並列，在市場下跌時提供不受股市波動影響的穩定收入。
但實際淨收益往往比表面租金低得多，正確計算至關重要。
        """)

        # ── 台灣租金殖利率現況 ──────────────────────────────────────────
        st.markdown("### 台灣各地區租金殖利率（2025 實證）")
        yield_df = pd.DataFrame([
            ["台北市（大安/信義）",  "1,800–3,500", "1.0–1.5%", "0.6–1.0%",
             "房價過高導致殖利率極低；持有成本（地價稅+房屋稅）侵蝕淨收益"],
            ["台北市（文山/士林）",  "900–1,500",   "1.5–2.0%", "1.0–1.4%",
             "相對合理；捷運沿線出租率高"],
            ["新北市（板橋/新莊）",  "600–1,200",   "2.0–2.5%", "1.4–1.8%",
             "性價比較高；通勤需求穩定"],
            ["桃園市",              "400–700",     "2.5–3.5%", "1.8–2.6%",
             "近年工業區帶動，殖利率相對全台最高"],
            ["台中市",              "500–900",     "2.0–3.0%", "1.5–2.2%",
             "台積電效應帶動；七期豪宅殖利率偏低"],
            ["高雄市",              "400–700",     "2.5–3.5%", "1.8–2.6%",
             "近年大幅上漲後殖利率下降；亞灣區新興"],
        ], columns=["地區", "每坪均價（萬）", "毛租金殖利率", "淨租金殖利率（估）", "備註"])
        st.dataframe(yield_df, use_container_width=True, hide_index=True)
        st.caption("淨殖利率 = 毛殖利率扣除：空置損失（5–10%）、維修費（1–2%）、管理費（0.3%）、地價稅+房屋稅（0.3–0.8%）、所得稅（視申報方式）")

        # ── 互動式租金淨收益計算機 ──────────────────────────────────────
        st.markdown("### 出租房產淨收益計算機（互動式）")
        ri_c1, ri_c2 = st.columns(2)
        with ri_c1:
            ri_property_wan  = st.number_input("房產市值 (萬)",          min_value=0,   max_value=20_000,
                                               value=int(re_rental_wan) if re_rental_wan > 0 else 1000,
                                               step=100, key="ri_prop")
            ri_monthly_rent_default = float(min(30.0, max(0.0, float(rental_monthly_wan)))) if rental_monthly_wan > 0 else 2.0
            ri_monthly_rent  = st.number_input("月租金 (萬/月)",         min_value=0.0, max_value=30.0,
                                               value=ri_monthly_rent_default,
                                               step=0.5, key="ri_rent",
                                               help="含管理費由房東收取的全部租金收入")
            ri_mortgage_wan  = st.number_input("本房產房貸餘額 (萬)",    min_value=0,   max_value=10_000,
                                               value=0, step=50, key="ri_mort")
            ri_mortgage_rate = st.number_input("房貸利率 (%)",           min_value=0.0, max_value=8.0,
                                               value=2.3, step=0.1, key="ri_mrate",
                                               help="2025 台灣房貸平均約 2.1–2.5%")
        with ri_c2:
            ri_vacancy_pct   = st.slider("空置率 (%)",                   0, 30,  8, 1, key="ri_vac",
                                         help="台灣平均約 5–12%；電梯大樓管理良好者約 5%")
            ri_maintain_pct  = st.slider("維修/翻新費 (年租金的 %)",     0, 20, 10, 1, key="ri_maint",
                                         help="老屋較高；每年約年租金的 5–15%")
            ri_tax_mode      = st.radio("租金所得申報方式", ["43% 必要費用扣除（房東常用）", "實際費用列舉"], horizontal=True, key="ri_tax")
            ri_income_rate   = st.slider("適用所得稅率 (%)", 5, 40, 20, 5, key="ri_itax",
                                         help="依全年綜合所得額適用稅率；詳見 Tab2 主題 3")

        # 計算
        ri_annual_gross   = ri_monthly_rent * 12               # 年毛租金
        ri_vacancy_loss   = ri_annual_gross * ri_vacancy_pct / 100
        ri_effective_rent = ri_annual_gross - ri_vacancy_loss  # 有效租金
        ri_maintain_cost  = ri_annual_gross * ri_maintain_pct / 100
        ri_prop_tax       = ri_property_wan * 0.005            # 估算地價稅+房屋稅 0.5%
        ri_mortgage_int   = ri_mortgage_wan * ri_mortgage_rate / 100  # 年利息

        # 稅後收益
        if ri_tax_mode.startswith("43%"):
            ri_taxable        = ri_effective_rent * 0.57        # 43% 必要費用扣除後
        else:
            ri_taxable        = max(0, ri_effective_rent - ri_maintain_cost - ri_prop_tax - ri_mortgage_int)
        ri_income_tax     = ri_taxable * ri_income_rate / 100
        ri_net_income     = ri_effective_rent - ri_maintain_cost - ri_prop_tax - ri_mortgage_int - ri_income_tax

        # 殖利率計算
        ri_gross_yield    = ri_annual_gross / ri_property_wan * 100 if ri_property_wan > 0 else 0
        ri_net_yield      = ri_net_income   / ri_property_wan * 100 if ri_property_wan > 0 else 0
        ri_cash_on_cash   = ri_net_income   / max(ri_property_wan - ri_mortgage_wan, 1) * 100  # 自有資金報酬率

        st.markdown("#### 試算結果")
        col_ri1, col_ri2, col_ri3, col_ri4 = st.columns(4)
        with col_ri1:
            st.metric("年毛租金",    f"{ri_annual_gross:.1f} 萬")
        with col_ri2:
            st.metric("年淨收益",    f"{ri_net_income:.1f} 萬",
                      f"月均 {ri_net_income/12:.2f} 萬")
        with col_ri3:
            st.metric("毛殖利率",    f"{ri_gross_yield:.2f}%")
        with col_ri4:
            st.metric("淨殖利率",    f"{ri_net_yield:.2f}%",
                      help="淨收益 ÷ 房產市值")

        # 成本明細
        cost_detail = pd.DataFrame([
            ["年毛租金",             f"+{ri_annual_gross:.2f} 萬", ""],
            ["　空置損失",           f"−{ri_vacancy_loss:.2f} 萬",  f"({ri_vacancy_pct}%)"],
            ["　維修/翻新費",        f"−{ri_maintain_cost:.2f} 萬", f"(年租金 {ri_maintain_pct}%)"],
            ["　地價稅+房屋稅（估）",f"−{ri_prop_tax:.2f} 萬",     "(市值×0.5%)"],
            ["　房貸利息",           f"−{ri_mortgage_int:.2f} 萬",  f"({ri_mortgage_wan}萬×{ri_mortgage_rate}%)"],
            ["　所得稅",             f"−{ri_income_tax:.2f} 萬",    f"(適用稅率 {ri_income_rate}%)"],
            ["**年淨收益**",         f"**{ri_net_income:.2f} 萬**", f"淨殖利率 **{ri_net_yield:.2f}%**"],
        ], columns=["項目", "金額", "說明"])
        st.dataframe(cost_detail, use_container_width=True, hide_index=True)

        if ri_net_yield < 1.0:
            st.error(f"⚠️ 淨殖利率 {ri_net_yield:.2f}% 極低，低於定存利率（約 1.7%），持有成本很可能超過收益。")
        elif ri_net_yield < 2.0:
            st.warning(f"⚠️ 淨殖利率 {ri_net_yield:.2f}% 偏低。考慮 REITs 或增加有價證券配置，可能提供更高的流動性調整後報酬。")
        else:
            st.success(f"✅ 淨殖利率 {ri_net_yield:.2f}% 合理，租金現金流具備退休收入補充價值。")

        # ── REITs 替代方案 ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### REITs 替代方案：流動性 + 不動產曝險的最佳平衡")
        st.markdown("""
如果直接持有出租房產的管理負擔過重（尤其退休後），
**REITs（不動產投資信託）**提供了絕佳的替代路徑：
以股票流動性享受不動產收益，且自動分散至數十至數百個物件。
        """)

        reit_df = pd.DataFrame([
            ["富邦 R1（01001T）",    "台灣", "辦公室/商業", "3.5–5.5%", "台幣計價；流動性一般；台灣最早成立 REITs",
             "適合保守型投資人；配息穩定"],
            ["新光 R1（01003T）",    "台灣", "辦公室/商業", "3.0–4.5%", "台灣六檔 REITs 中流動性較佳",
             "小規模，分散性有限"],
            ["VNQ（Vanguard REIT）", "美國", "多元（辦公/住宅/工業/醫療）", "3.5–4.5%", "美元計價；全球最大 REIT ETF；費用率 0.12%",
             "退休組合核心 REITs 持倉；最高分散性"],
            ["REET（iShares全球REIT）","全球", "多元+跨國分散", "3.0–4.0%", "美元計價；涵蓋 30+ 國家 REITs；費用率 0.14%",
             "降低單一市場集中度；全球不動產曝險"],
            ["00712（復華富時不動產）","台灣", "全球 REIT（追蹤 FTSE NAREIT）", "3.0–4.5%", "台幣計價；台灣交易所掛牌；省去外幣兌換",
             "台灣投資人的全球 REITs 入門選擇"],
        ], columns=["商品", "市場", "物件類型", "配息殖利率（估）", "特性", "退休用途"])
        st.dataframe(reit_df, use_container_width=True, hide_index=True)

        st.markdown("### 直接持有 vs REITs：退休角度的全面比較")
        compare_re_df = pd.DataFrame([
            ["流動性",        "低（3–9 個月變現）",       "高（T+2 可變現）"],
            ["最小投資金額",  "高（通常 500 萬起）",      "低（幾千元即可）"],
            ["管理負擔",      "高（租客、維修、稅申報）", "無（被動投資）"],
            ["分散性",        "低（集中單一物件）",        "高（VNQ 涵蓋 150+ 物件）"],
            ["槓桿",          "可用房貸放大（雙刃劍）",   "基金內部有槓桿，投資人無法調整"],
            ["稅務",          "複雜（43%必要費用/實列）", "簡單（海外股利 28% 預扣/合併申報）"],
            ["通膨對沖",      "強（房價+租金均可隨通膨調整）", "中等（REITs 含不動產，間接對沖）"],
            ["情感/心理價值", "高（自有房產的安全感）",   "低（僅數字，無形體感）"],
            ["淨殖利率",      "1–3%（扣除所有成本後）",  "3–4.5%（費用率極低）"],
        ], columns=["比較維度", "直接持有出租房", "REITs"])
        st.dataframe(compare_re_df, use_container_width=True, hide_index=True)

        st.markdown("### 整合進退休規劃的建議比例")
        st.info(
            "**不動產在退休組合的建議比重（以流動性為前提）**：\n\n"
            "- 自住房（無法產生現金流）：不計入「退休投資組合」，但計入「總資產負債表」\n"
            "- 出租房（直接持有）：建議不超過總金融資產的 **20–30%**，"
            "因管理負擔隨年齡增加而上升\n"
            "- REITs（VNQ/REET/00712）：建議 **5–15%** 作為不動產曝險的流動性替代品，"
            "放入三桶金第二桶（中期資產）\n"
            "- **隨年齡增加，逐步將直接持有出租房轉換為 REITs**，降低管理負擔，保留不動產收益"
        )
        st.caption(
            "文獻：① 內政部不動產資訊平台 2025 "
            "② 信義房屋「2025 台灣租金殖利率調查」"
            "③ NAREIT Global Real Estate Index Series 2024 "
            "④ 台灣證交所 REITs 資料庫 2025 "
            "⑤ Vanguard 'REITs in a Retirement Portfolio' (2024)"
        )

    # ── 15. 出租物業的類年金效應 ─────────────────────────────────────────
    elif edu_topic.startswith("15"):
        st.subheader("出租物業的類年金效應（Buy-to-Let Quasi-Annuity）")
        st.markdown("""
> **依據**：Pfau (2012)「Evaluating Investments for Retirees」；
> Milevsky & Huang (2011)「Spending Retirement on Planet Vulcan」；
> Bengen (1994) 4% Rule Extension；
> 本 App 引擎邏輯（`rental_annual` 參數說明）
        """)

        st.markdown("""
對於持有出租物業（住宅或商用不動產）的退休者，
租金收益在提領模型中扮演著**類年金（Quasi-Annuity）**的核心角色——
它與勞保年金並列，構成退休收入的「固定底層」，
讓金融資產只需負擔**超出底層的部分**，大幅降低提領壓力與序列報酬風險。
        """)

        # ── 類年金的數學等效性 ──────────────────────────────────────────
        st.markdown("### 一、類年金的數學等效性")
        st.markdown(r"""
設定符號：
- $W_0$：年度目標生活費（實質購買力）
- $R_{rental}$：年租金淨收益
- $A_0$：金融資產（有價證券）
- $r$：金融資產實質報酬率

**無租金收入**：金融資產每年須提領 $W_0$，有效提領率 $= W_0 / A_0$

**有租金收入**：金融資產每年僅需提領 $W_0 - R_{rental}$，有效提領率 $= (W_0 - R_{rental}) / A_0$

> 這與勞保年金完全相同的數學結構：
> 凡是「不依賴金融資產、固定到帳的現金流」，
> 都能等比例**降低金融資產的提領率**，延長組合存活年限。
        """)

        # 互動式類年金效應試算
        st.markdown("### 二、你的類年金效應試算（即時）")
        qa_c1, qa_c2 = st.columns(2)
        with qa_c1:
            qa_A0       = st.number_input("金融資產 A₀ (萬)",    min_value=0, max_value=100_000,
                                          value=int(A0_securities_wan), step=100, key="qa_a0")
            qa_W0       = st.number_input("年度生活費 W₀ (萬/年)", min_value=0.0, max_value=1000.0,
                                          value=float(W0 / 10_000), step=10.0, key="qa_w0")
        with qa_c2:
            qa_rental_default = float(min(500.0, max(0.0, float((rental_annual + rm_annual) / 10_000))))
            qa_rental   = st.number_input("年租金淨收益 (萬/年)", min_value=0.0, max_value=500.0,
                                          value=qa_rental_default, step=5.0,
                                          key="qa_rent",
                                          help="租金淨收入＋以房養老年收入合計（已各自依啟動年齡分離計算）")
            qa_pension_default = float(min(200.0, max(0.0, float(pension_annual / 10_000))))
            qa_pension  = st.number_input("年勞保/勞退收入 (萬/年)", min_value=0.0, max_value=200.0,
                                          value=qa_pension_default, step=5.0, key="qa_pen")

        qa_passive     = qa_rental + qa_pension
        qa_net_draw    = max(0.0, qa_W0 - qa_passive)
        qa_raw_wdr     = qa_W0     / qa_A0 * 100 if qa_A0 > 0 else 0
        qa_eff_wdr     = qa_net_draw / qa_A0 * 100 if qa_A0 > 0 else 0
        qa_coverage    = qa_passive / qa_W0 * 100 if qa_W0 > 0 else 0

        # 顯示指標
        col_qa1, col_qa2, col_qa3, col_qa4 = st.columns(4)
        with col_qa1:
            st.metric("總被動收入", f"{qa_passive:.1f} 萬/年",
                      f"租金 {qa_rental:.1f} + 年金 {qa_pension:.1f}")
        with col_qa2:
            st.metric("生活費覆蓋率", f"{qa_coverage:.1f}%",
                      help="被動收入 ÷ 年度生活費；越高越安全")
        with col_qa3:
            st.metric("名目提領率", f"{qa_raw_wdr:.2f}%",
                      "不含被動收入時")
        with col_qa4:
            st.metric("有效提領率", f"{qa_eff_wdr:.2f}%",
                      f"扣除被動收入後 ↓{qa_raw_wdr - qa_eff_wdr:.2f}%",
                      delta_color="inverse")

        # 等效資本
        if qa_rental > 0:
            equiv_capital = qa_rental / 0.04  # 以 4% 提領率反推等效資本
            st.info(
                f"**等效資本概念**：年租金淨收益 **{qa_rental:.1f} 萬**，"
                f"以 4% 提領率反推，相當於額外持有 **{equiv_capital:.0f} 萬** 的金融資產組合。\n\n"
                f"換言之，你的出租物業在退休提領模型中，等效為一個市值 **{equiv_capital:.0f} 萬**、"
                f"每年穩定配息 4% 的「隱形資產」。"
            )

        # 提領率安全區間警示
        if qa_eff_wdr == 0:
            st.success("✅ 被動收入已完全覆蓋生活費，金融資產無需提領，持續複利增長。")
        elif qa_eff_wdr <= 3.0:
            st.success(f"✅ 有效提領率 {qa_eff_wdr:.2f}% ≤ 3%：極度安全，組合幾乎不會在 40 年內耗盡。")
        elif qa_eff_wdr <= 4.0:
            st.success(f"✅ 有效提領率 {qa_eff_wdr:.2f}% 在 3–4% 之間：安全區間（Bengen 4% Rule），歷史成功率 > 95%。")
        elif qa_eff_wdr <= 5.0:
            st.warning(f"⚠️ 有效提領率 {qa_eff_wdr:.2f}% 在 4–5% 之間：尚可接受，但建議搭配 GK 護欄策略動態調整。")
        else:
            st.error(f"❗ 有效提領率 {qa_eff_wdr:.2f}% > 5%：偏高，長期耗盡風險顯著，建議增加被動收入來源或降低生活費。")

        # ── 類年金 vs 真年金 vs 股息 ────────────────────────────────────
        st.markdown("---")
        st.markdown("### 三、租金收益 vs 年金 vs 股息：作為底層現金流的比較")
        annuity_compare = pd.DataFrame([
            ["勞保老年年金",
             "終身固定（政府保證）",
             "與 CPI 連動（部分）",
             "極低（政府擔保）",
             "不可提前贖回",
             "⭐⭐⭐⭐⭐"],
            ["即期年金（壽險公司）",
             "終身固定（保險公司保證）",
             "通常無通膨調整",
             "低（保險業監理）",
             "不可贖回，身故即終止",
             "⭐⭐⭐⭐"],
            ["出租房產租金",
             "持續至出售（非終身）",
             "可隨市場租金調漲（通膨對沖）",
             "中（空置、租客風險、維修）",
             "可出售變現（3–9 個月）",
             "⭐⭐⭐"],
            ["REITs 配息",
             "持續至出售（非終身）",
             "與租金市場連動",
             "中（市場波動影響配息）",
             "高（T+2）",
             "⭐⭐⭐"],
            ["高股息 ETF（00878）",
             "持續至出售（非終身）",
             "無保證；依公司盈利",
             "中高（股市波動）",
             "高（T+2）",
             "⭐⭐"],
            ["定存利息",
             "持續至解約",
             "無通膨對沖（固定利率）",
             "極低",
             "高（解約可取回）",
             "⭐⭐⭐（但購買力侵蝕）"],
        ], columns=["收入來源", "持續性", "通膨對沖能力", "風險", "流動性", "退休底層評分"])
        st.dataframe(annuity_compare, use_container_width=True, hide_index=True)

        # ── 本 App 引擎如何處理租金 ──────────────────────────────────────
        st.markdown("---")
        st.markdown("### 四、本 App 引擎的類年金處理邏輯")
        st.markdown("""
在本 App 的 `run_dynamic_projection` 引擎中，每一年的計算邏輯如下：

```
每年淨提領 = max(0, 生活費支出 − 被動收入)
被動收入   = 勞保/勞退年收入 + 租金淨年收入
（達到各自請領/啟動年齡後才開始計入）
```

這意味著：
- **租金的每一元，都是一對一地減少金融資產的提領壓力**
- 在蒙地卡羅模擬中，有租金的組合路徑，壞年份（熊市）的提領量更小，
  直接降低了序列報酬風險（SORR）的破壞力
- 若 `租金 ≥ 生活費`，金融資產完全不需提領，得以在最長的時間內複利增長
        """)

        st.markdown("### 五、商用不動產 vs 住宅出租：退休角度的差異")
        commercial_df = pd.DataFrame([
            ["殖利率", "通常 4–7%（較高）", "通常 1.5–3%（較低）"],
            ["租約穩定性", "長期（3–10 年合約）；違約成本高", "短期（1 年為主）；換租頻繁"],
            ["空置風險", "空置期更長（半年–2 年），但頻率低", "空置期短（1–3 個月），但可能頻繁"],
            ["管理負擔", "較低（商業租戶自行維護）", "較高（居住損耗、設備維修）"],
            ["景氣敏感度", "高（景氣衰退時租金/空置率惡化）", "中（民生需求支撐租金底部）"],
            ["類年金穩定性", "中（合約期間穩定，到期後有不確定性）", "中低（但供需較穩定）"],
            ["適合退休者", "擁有大型商業物業者；建議委由專業管理", "管理能力強者；建議限 1–2 間"],
        ], columns=["比較維度", "商用不動產", "住宅出租"])
        st.dataframe(commercial_df, use_container_width=True, hide_index=True)

        st.success(
            "**核心洞見**：租金收益最大的退休價值，不在於殖利率的高低，"
            "而在於它**降低了有效提領率**——這個效果會在複利的時間軸下，"
            "非線性地擴大金融資產的存活年限。"
            "一個有效提領率從 5% 降至 3% 的組合，其 40 年存活機率可從約 75% 提升至 98% 以上。"
        )
        st.caption(
            "文獻：① Pfau (2012) 'Evaluating Investments for Retirees' "
            "② Milevsky & Huang (2011) 'Spending Retirement on Planet Vulcan' "
            "③ Bengen (1994) '4% Rule' "
            "④ 台灣內政部不動產資訊平台 2025 "
            "⑤ 信義房屋商用不動產白皮書 2025"
        )

    # ── 16. Income Floor 與折現風險調整 ──────────────────────────────────
    elif edu_topic.startswith("16"):
        st.subheader("不動產收益護欄：Income Floor 與折現風險調整")
        st.markdown("""
> **依據**：Pfau (2013)「Safety-First vs. Probability-Based」；
> Milevsky (2012)「Pensionize Your Nest Egg」；
> Ameriks et al. (2011)「The Joy of Giving or Assisted Living?」；
> 本 App `run_dynamic_projection` 引擎設計說明
        """)

        # ── Income Floor 概念 ─────────────────────────────────────────────
        st.markdown("### 一、什麼是 Income Floor（底層收入護欄）？")
        st.markdown("""
**Income Floor** 是 Wade Pfau「Safety-First（安全第一）」退休框架的核心概念：

> 先用**不可耗盡的固定收入**覆蓋基本生活費（Floor），
> 再用**投資組合**應對彈性支出（Upside）。

當收入來源具備「**確定性**」與「**持續性**」時，它的財務功能等同於**確定給付制（Defined Benefit, DB）退休金**——
即使市場崩盤、金融資產大跌，底層現金流仍然穩定到帳，讓退休者不必在最壞的時機出售資產。
        """)

        # Floor vs. DB 對比
        floor_db_df = pd.DataFrame([
            ["確定給付制（DB）退休金", "政府/雇主保證終身給付", "極高", "無法轉讓", "名義上"],
            ["勞保老年年金",           "勞保局終身給付",         "極高", "無法轉讓", "部分連動 CPI"],
            ["不動產租金（自有出租）", "市場租金，非終身保證",   "中等", "可出售變現", "可隨市場調漲"],
            ["REITs 配息",             "依基金收益，無保證",     "中低", "高流動性",  "部分連動通膨"],
            ["股息（高股息 ETF）",     "依公司盈利，無保證",     "低",  "高流動性",  "不確定"],
        ], columns=["收入來源", "保證程度", "確定性", "流動性", "通膨連動"])
        st.dataframe(floor_db_df, use_container_width=True, hide_index=True)

        st.markdown("""
**不動產租金**介於「勞保年金」與「股息」之間——
它不像勞保那樣由政府保證，但比股息更穩定、更不受市場波動影響，
因此在 Income Floor 架構中，應以**折現後的金額**而非全額計入。
        """)

        # ── 折現風險調整 ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 二、收益折現：如何對不確定的租金收入打折？")
        st.markdown("""
**核心原則**：確定性越低的收入，在規劃中越應保守估算。
租金收入面臨以下不確定性，每項都應給予一定的「**可靠性折扣**」：
        """)

        haircut_df = pd.DataFrame([
            ["空置風險",       "租客更換空窗期；景氣下滑空屋率上升",              "5–15%",
             "以歷史空置率上限（悲觀估計）設定"],
            ["租金下跌風險",   "區域供給增加、景氣循環導致租金下跌",              "0–20%",
             "近 10 年台灣部分區域租金已停滯；商辦更明顯"],
            ["維修/資本支出",  "屋齡增加後大型修繕（管線、外牆、設備）",          "5–10%",
             "屋齡 > 20 年應預留更高比例"],
            ["管理負擔轉化成本","退休後體力下降，委外管理費用增加",               "3–8%",
             "委託管理公司通常收月租的 5–10%"],
            ["稅務風險",       "政策變動（房地合一 3.0、租金課稅加重）",          "3–5%",
             "台灣 2026 後租金課稅政策仍有不確定性"],
            ["租霸風險 ⚠️",   "惡意欠租、拒不退房；強制執行訴訟曠日廢時",       "2–8%",
             "台灣法院強制執行平均耗時 6–18 個月；損失含期間零收入＋訴訟費"],
        ], columns=["風險來源", "說明", "折扣幅度（毛租金的 %）", "台灣具體建議"])
        st.dataframe(haircut_df, use_container_width=True, hide_index=True)

        st.info(
            "**精算安全邊際原則（Actuarial Safety Margin）**\n\n"
            "儘管租金收益在結構上類似年金，但它**並非 100% 保證**。"
            "面臨空窗期、修繕成本與租霸風險，國際精算學界的共識建議：\n\n"
            "> **在精算提領率時，僅將預期租金的 75% 計入確定現金流（Income Floor），**\n"
            "> **保留 25% 作為安全邊際，以對沖上述不可控風險的複合效應。**\n\n"
            "例如：預期年租金 40 萬 → 精算 Floor 計入 **30 萬**（×75%）；\n"
            "剩餘 10 萬視為「機率性收入」，不納入 Floor，但在蒙地卡羅中以期望值體現。"
        )

        # 互動式折現計算機
        st.markdown("#### 你的租金可靠性折現試算（含 75% 安全邊際）")
        hc_c1, hc_c2 = st.columns(2)
        with hc_c1:
            hc_gross_default = float(min(500.0, max(0.0, float(rental_monthly_wan * 12))))
            hc_gross_wan   = st.number_input("年毛租金 (萬/年)",    min_value=0.0, max_value=500.0,
                                             value=hc_gross_default, step=5.0, key="hc_gross")
            hc_vacancy_pct = st.slider("空置折扣 (%)",              0, 20, 8,  1, key="hc_vac")
            hc_rent_drop   = st.slider("租金下跌風險折扣 (%)",      0, 20, 5,  1, key="hc_rdrop")
            hc_bad_tenant  = st.slider("租霸風險折扣 (%)",          0, 10, 3,  1, key="hc_bad",
                                       help="惡意欠租、拒不退房的機率損失；台灣建議 2–5%（委外管理可降低）")
        with hc_c2:
            hc_maintain    = st.slider("維修/資本支出折扣 (%)",     0, 15, 7,  1, key="hc_maint2")
            hc_mgmt        = st.slider("管理負擔折扣 (%)",          0, 12, 5,  1, key="hc_mgmt")
            hc_tax         = st.slider("稅務風險折扣 (%)",          0, 10, 3,  1, key="hc_tax2")

        total_haircut   = hc_vacancy_pct + hc_rent_drop + hc_bad_tenant + hc_maintain + hc_mgmt + hc_tax
        hc_modeled      = hc_gross_wan * (1 - total_haircut / 100)
        # 75% 安全邊際上限：計入 Floor 的最大值為毛租金 × 75%
        hc_floor_cap    = hc_gross_wan * 0.75
        hc_reliable     = min(hc_modeled, hc_floor_cap)       # 精算 Floor 值（取保守值）
        hc_aggressive   = hc_gross_wan * 0.90                 # 樂觀（僅扣空置）
        hc_conserv      = hc_gross_wan * max(0, 1 - (total_haircut + 10) / 100)

        col_hc1, col_hc2, col_hc3 = st.columns(3)
        with col_hc1:
            st.metric("樂觀估計（Floor 上限）",         f"{hc_aggressive:.1f} 萬/年", "僅扣空置損失")
        with col_hc2:
            st.metric("精算 Floor（建議計入 Income Floor）", f"{hc_reliable:.1f} 萬/年",
                      f"{'75% 上限封頂' if hc_modeled > hc_floor_cap else f'總折扣 {total_haircut}%'}",
                      delta_color="off")
        with col_hc3:
            st.metric("悲觀估計（壓力測試用值）",        f"{hc_conserv:.1f} 萬/年",  "再加 10% 緩衝")

        if hc_modeled > hc_floor_cap:
            st.warning(
                f"你的折扣合計僅 {total_haircut}%，推算值 {hc_modeled:.1f} 萬 **超過 75% 安全邊際上限**（{hc_floor_cap:.1f} 萬）。"
                f"依精算原則，建議 Income Floor 僅計入 **{hc_floor_cap:.1f} 萬/年**，"
                f"超出部分（{hc_modeled - hc_floor_cap:.1f} 萬）視為機率性收入，不納入確定底層。"
            )
        st.caption(
            f"精算建議：**Income Floor 計入 {hc_reliable:.1f} 萬/年**（毛租金 {hc_gross_wan:.1f} 萬 × "
            f"{hc_reliable/hc_gross_wan*100:.0f}%）。"
            f"請將此值填入本 App 左側「月租金淨收入」欄位（÷12 = {hc_reliable/12:.2f} 萬/月）；"
            f"蒙地卡羅壓力測試可改用悲觀估計 {hc_conserv:.1f} 萬/年。"
        )

        # ── 具體數值示例 ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 三、具體數值示例：租金如何降低證券提領壓力")

        ex_c1, ex_c2 = st.columns(2)
        with ex_c1:
            ex_living  = st.number_input("年度生活費目標 (萬/年)", min_value=0.0, max_value=500.0,
                                         value=float(W0/10_000), step=10.0, key="ex_live")
            ex_pension_default = float(min(200.0, max(0.0, float(pension_annual/10_000))))
            ex_pension = st.number_input("勞保/勞退年收入 (萬/年)", min_value=0.0, max_value=200.0,
                                         value=ex_pension_default, step=5.0, key="ex_pen2")
        with ex_c2:
            ex_rental_default = float(min(300.0, max(0.0, round(hc_reliable, 1))))
            ex_rental  = st.number_input("租金可靠淨收益 (萬/年)", min_value=0.0, max_value=300.0,
                                         value=ex_rental_default, step=5.0, key="ex_rent2",
                                         help="建議填入上方「中性估計」值")
            ex_A0      = st.number_input("金融資產 A₀ (萬)", min_value=0, max_value=100_000,
                                         value=int(A0_securities_wan), step=100, key="ex_a0")

        ex_floor       = ex_pension + ex_rental
        ex_gap         = max(0.0, ex_living - ex_floor)
        ex_raw_wr      = ex_living / ex_A0 * 100 if ex_A0 > 0 else 0
        ex_eff_wr      = ex_gap    / ex_A0 * 100 if ex_A0 > 0 else 0
        ex_floor_cover = ex_floor  / ex_living * 100 if ex_living > 0 else 0

        # 圖示化的收入堆疊
        st.markdown("#### 退休收入堆疊（Income Stack）")
        stack_rows = [
            ["第一層：勞保/勞退年金",       f"{ex_pension:.1f} 萬/年", "政府保證；不可耗盡", "底層護欄"],
            ["第二層：租金淨收益（折現後）", f"{ex_rental:.1f} 萬/年",  "市場收益；中等確定性", "底層護欄"],
            ["──── Income Floor 合計 ────", f"{ex_floor:.1f} 萬/年",   f"覆蓋生活費 {ex_floor_cover:.1f}%", "≥ 70% 為理想"],
            ["第三層：金融資產提領",         f"{ex_gap:.1f} 萬/年",     "需從證券組合提領的部分", "彈性支出層"],
            ["──── 生活費目標 ────",        f"{ex_living:.1f} 萬/年",  f"有效提領率 {ex_eff_wr:.2f}%（名目 {ex_raw_wr:.2f}%）", ""],
        ]
        stack_df = pd.DataFrame(stack_rows, columns=["收入層次", "年金額", "說明", "備註"])
        st.dataframe(stack_df, use_container_width=True, hide_index=True)

        # 視覺化降幅
        wr_delta = ex_raw_wr - ex_eff_wr
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        with col_ex1:
            st.metric("Income Floor 覆蓋率", f"{ex_floor_cover:.1f}%",
                      "建議 ≥ 60–70%" if ex_floor_cover < 60 else "理想範圍 ✅")
        with col_ex2:
            st.metric("名目提領率", f"{ex_raw_wr:.2f}%", "不含任何被動收入")
        with col_ex3:
            st.metric("有效提領率", f"{ex_eff_wr:.2f}%",
                      f"−{wr_delta:.2f}% ↓ 因 Floor 護欄", delta_color="inverse")

        if wr_delta > 0:
            st.success(
                f"**Floor 護欄效果**：租金 + 年金合計 {ex_floor:.1f} 萬/年，"
                f"將金融資產有效提領率從 **{ex_raw_wr:.2f}%** 壓低至 **{ex_eff_wr:.2f}%**，"
                f"降幅 **{wr_delta:.2f} 個百分點**。"
                f"這使整體退休系統在熊市期間更具韌性——"
                f"因為最壞的年份，金融資產只需提領 {ex_gap:.1f} 萬，而非 {ex_living:.1f} 萬。"
            )

        # ── Floor 充足性框架 ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 四、Floor 充足性框架：你的護欄是否夠強？")
        adequacy_df = pd.DataFrame([
            ["Floor ≥ 90% 生活費", "極強護欄",
             "金融資產幾乎無需提領，僅需管理通膨調整；蒙地卡羅成功率接近 99%",
             "有多間出租物業 + 勞保年金者"],
            ["Floor 70–90% 生活費", "強護欄",
             "金融資產僅需應對小額彈性支出；有效提領率 < 2%；極度安全",
             "一間出租物業（中型）+ 勞保年金"],
            ["Floor 40–70% 生活費", "中等護欄（本 App 常見情境）",
             "仍需從金融資產提領，但有效提領率明顯降低；建議搭配 GK 護欄策略",
             "小型出租物業或僅有勞保年金"],
            ["Floor < 40% 生活費", "弱護欄",
             "大部分依賴金融資產；完全暴露於 SORR；應優先補強被動收入",
             "無不動產、勞保年金偏低者"],
            ["Floor = 0% 生活費", "無護欄（最高風險）",
             "100% 依賴金融資產；退休頭 5–10 年的熊市可能造成不可逆損害",
             "純金融資產退休者；需嚴格執行 GK 護欄或降低提領率至 3.5% 以下"],
        ], columns=["Floor 覆蓋率", "護欄強度", "財務意涵", "典型情境"])
        st.dataframe(adequacy_df, use_container_width=True, hide_index=True)

        st.info(
            "**設計原則（Pfau Safety-First）**：\n\n"
            "1. 先計算「基本生活費 Floor」（不包含旅遊、奢侈品等彈性支出）\n"
            "2. 用**確定性高的收入**（勞保 + 租金折現）覆蓋 Floor，目標 ≥ 70%\n"
            "3. 金融資產只負責覆蓋剩餘缺口 + 彈性支出 + 長照備援\n"
            "4. 這樣的系統在任何市場環境下都能「**活下去**」，\n"
            "   市場好的時候才享受「**活得好**」"
        )
        st.caption(
            "文獻：① Pfau (2013) 'Safety-First vs. Probability-Based Retirement' "
            "② Milevsky (2012) 'Pensionize Your Nest Egg' "
            "③ Ameriks et al. (2011) 'The Joy of Giving or Assisted Living?' "
            "④ Bengen (1994) '4% Rule' "
            "⑤ 本 App 引擎設計文件（run_dynamic_projection 邏輯說明）"
        )

    # ── 17. 退休現金流全景 ────────────────────────────────────────────────
    elif edu_topic.startswith("17"):
        st.subheader("退休現金流全景：主動槓桿、房貸與多元資金流管理")
        st.markdown("""
> **依據**：Pfau (2017)「Reverse Mortgages」；
> Kitces (2013)「Should Retirees Pay Off Their Mortgage?」；
> Vanguard Research「The Role of Debt in Retirement」(2024)；
> 台灣金管會「退休族借貸行為調查」2025
        """)
        st.markdown("""
退休現金流管理的核心挑戰，是在**收入結構根本改變**（從薪資→提領）後，
同時駕馭多條異質現金流——各有不同的開始時間、確定性、通膨連動與流動性屬性。
房貸等**主動負債**則在收支方程式中引入固定的資金外流，
它既可能是拖垮現金流的包袱，也可能是利用槓桿放大資產效率的工具，
關鍵在於：**利率環境、資產報酬率與個人行為傾向的三方平衡**。
        """)

        # ── 一、退休現金流全景地圖 ─────────────────────────────────────
        st.markdown("### 一、退休現金流全景地圖")
        cf_c1, cf_c2 = st.columns(2)
        with cf_c1:
            st.markdown("#### 收入端（Income Side）")
            income_map = pd.DataFrame([
                ["勞保老年年金",       "請領年齡起（終身）",  "極高", "政府保證；與 CPI 部分連動"],
                ["勞退帳戶月領",       "60 歲起（可選）",     "高",   "個人帳戶；可一次或月領"],
                ["金融資產提領",       "退休起（彈性）",      "中",   "依策略（Fixed/GK/Smile）動態調整"],
                ["租金淨收益",         "出租期間",            "中",   "依 75% 精算折現計入 Floor"],
                ["以房養老月領",       "啟動年齡起（終身）",  "高",   "房屋抵押換現金流；不可逆"],
                ["兼職/半退休收入",    "退休初期（彈性）",    "低",   "體力允許時的補充；通常 65 歲前"],
                ["資產出售（Downsizing）","一次性事件",        "中",   "大屋換小屋釋出資金"],
            ], columns=["來源", "時間軸", "確定性", "說明"])
            st.dataframe(income_map, use_container_width=True, hide_index=True)
        with cf_c2:
            st.markdown("#### 支出端（Expense Side）")
            expense_map = pd.DataFrame([
                ["基本生活費",         "終身",         "高（穩定）", "消費微笑曲線（引擎採平滑過渡）：<73 ×1.0；73–77 線性至 ×0.8；77–83 ×0.8；83–87 線性至 ×1.1；≥87 ×1.1"],
                ["醫療溢價",           "70 歲起遞增",  "中",        "本 App 引擎：以 70 歲為基點，指數複利遞增"],
                ["長照費用",           "約 82–95 歲",  "高",        "最大單一風險；60–200 萬/年"],
                ["房貸本息攤還",       "至還清日止",   "高（固定）","資金外流的硬性義務；利率風險"],
                ["地價稅/房屋稅",      "持有期間",     "高（固定）","每年固定；持有多房者更高"],
                ["子女/家庭支援",      "不定期",       "低",        "行為偏誤：常被低估"],
                ["通膨調整後消費成長", "每年複利",     "中",        "高齡 CPI 系統性高於整體 CPI"],
            ], columns=["支出項目", "時間軸", "確定性", "說明"])
            st.dataframe(expense_map, use_container_width=True, hide_index=True)

        # ── 二、退休房貸分析 ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 二、退休時仍有房貸：全面影響分析")
        st.markdown("""
台灣 2025 年調查：**約 28% 的 60–65 歲族群仍有未償房貸**，
平均餘額約 450 萬，月均攤還約 2.2 萬。
這筆固定支出在退休後成為**現金流最大的剛性壓力源**之一。
        """)

        ml_c1, ml_c2, ml_c3 = st.columns(3)
        with ml_c1:
            ml_balance_default = int(min(5_000, max(0, int(re_mortgage_wan))))
            ml_balance   = st.number_input("房貸餘額 (萬)",    min_value=0,   max_value=5_000, value=ml_balance_default, step=50, key="ml_bal")
            ml_rate      = st.number_input("房貸利率 (%/年)",  min_value=0.5, max_value=8.0,   value=2.3,  step=0.1,  key="ml_rate")
        with ml_c2:
            ml_years_left= st.number_input("剩餘還款年數 (年)",min_value=1,   max_value=30,    value=15,   step=1,    key="ml_yrs")
            ml_retire_age_default = int(min(75, max(50, age_start)))
            ml_retire_age= st.number_input("退休年齡 (歲)",    min_value=50,  max_value=75,    value=ml_retire_age_default, step=1, key="ml_retage")
        with ml_c3:
            ml_invest_r_default = float(min(12.0, max(0.0, float(r_pct))))
            ml_invest_r  = st.number_input("金融資產實質報酬 (%)", min_value=0.0, max_value=12.0, value=ml_invest_r_default, step=0.5, key="ml_ir",
                                           help="若提前還清房貸，這是那筆錢「本可獲得」的投資報酬率")
            ml_inflation = st.number_input("通膨率 (%)",       min_value=0.0, max_value=6.0,   value=float(inflation_pct), step=0.5, key="ml_inf2")

        # 計算月付額（年金公式）
        ml_monthly_rate = ml_rate / 100 / 12
        ml_n_months     = ml_years_left * 12
        if ml_monthly_rate > 0 and ml_balance > 0:
            ml_monthly_pay = (ml_balance * 10_000 * ml_monthly_rate *
                              (1 + ml_monthly_rate) ** ml_n_months /
                              ((1 + ml_monthly_rate) ** ml_n_months - 1)) / 10_000
        else:
            ml_monthly_pay = ml_balance / max(ml_years_left * 12, 1)
        ml_annual_pay   = ml_monthly_pay * 12
        ml_total_pay    = ml_annual_pay * ml_years_left
        ml_total_int    = ml_total_pay - ml_balance if ml_balance > 0 else 0

        col_ml1, col_ml2, col_ml3, col_ml4 = st.columns(4)
        with col_ml1:
            st.metric("月還款額",       f"{ml_monthly_pay:.2f} 萬/月")
        with col_ml2:
            st.metric("年還款額",       f"{ml_annual_pay:.1f} 萬/年")
        with col_ml3:
            st.metric("剩餘總利息支出", f"{ml_total_int:.0f} 萬",
                      help="還款期間累計支付利息")
        with col_ml4:
            st.metric("佔年生活費比例", f"{ml_annual_pay / (W0/10_000) * 100:.1f}%" if W0 > 0 else "—",
                      help="房貸年攤還佔年度生活費目標的比例")

        # 還款時間軸
        st.markdown("#### 房貸對現金流的逐年衝擊")
        payoff_age = ml_retire_age + ml_years_left
        timeline_rows = []
        bal = ml_balance
        for yr in range(min(ml_years_left + 1, 31)):
            age = ml_retire_age + yr
            annual_int  = bal * ml_rate / 100
            annual_prin = min(ml_annual_pay - annual_int, bal) if bal > 0 else 0
            net_cf_hit  = ml_annual_pay if bal > 0 else 0
            bal         = max(0.0, bal - annual_prin)
            timeline_rows.append({
                "年齡": age,
                "當年還款": f"{net_cf_hit:.1f} 萬" if net_cf_hit > 0 else "—（還清）",
                "  其中利息": f"{annual_int:.1f} 萬" if net_cf_hit > 0 else "—",
                "  其中本金": f"{annual_prin:.1f} 萬" if net_cf_hit > 0 else "—",
                "剩餘餘額": f"{bal:.0f} 萬",
                "現金流狀態": "⚠️ 有房貸負擔" if net_cf_hit > 0 else "✅ 房貸已還清",
            })
        st.dataframe(pd.DataFrame(timeline_rows), use_container_width=True, hide_index=True, height=280)
        if ml_balance > 0:
            st.caption(f"預計在 **{payoff_age} 歲**還清房貸，之後每年現金流釋放 **{ml_annual_pay:.1f} 萬**。")

        # ── 三、主動槓桿分析：還是不還？ ───────────────────────────────
        st.markdown("---")
        st.markdown("### 三、主動槓桿決策：提前還清 vs 持續持貸投資")
        st.markdown("""
當你持有低利率房貸（如 2–2.5%），同時金融資產能賺取 5–8% 實質報酬，
數學上**保留房貸並投資差額**是更有效率的選擇——這就是「**主動槓桿**（Active Leverage）」策略。
但它在行為面與現金流面帶來真實的風險，必須全面評估。
        """)

        lev_df = pd.DataFrame([
            ["利率套利空間",
             f"房貸 {ml_rate:.1f}% vs 投資 {ml_invest_r:.1f}% → 淨套利 **{ml_invest_r - ml_rate:.1f}%**",
             f"{ml_balance:.0f} 萬 × {ml_invest_r - ml_rate:.1f}% = **{ml_balance * (ml_invest_r - ml_rate) / 100:.1f} 萬/年**（理論增益）",
             f"{'✅ 保留房貸有數學優勢' if ml_invest_r > ml_rate + 1 else '⚠️ 套利空間不足，建議優先還款'}"],
            ["現金流壓力",
             f"每年固定流出 {ml_annual_pay:.1f} 萬（佔生活費 {ml_annual_pay/(W0/10_000)*100:.0f}%）",
             "熊市時仍須償還房貸，等同強迫在低點賣出資產",
             "SORR 放大器：房貸使退休初期的序列報酬風險更具破壞力"],
            ["稅後成本計算",
             f"房貸實際成本（稅後）≈ {ml_rate:.1f}% × (1 − 20%) = {ml_rate*0.8:.2f}%（若可列舉扣除）",
             "台灣自用住宅房貸利息可列舉扣除（上限 30 萬/年）",
             "扣除後實際成本降低，保留房貸數學優勢更明顯"],
            ["心理與行為風險",
             "持有債務對部分退休者造成持續焦慮",
             "焦慮影響決策品質；熊市恐慌賣出的風險顯著上升",
             "行為成本是真實的財務成本；若無法承受，還清房貸有其心理價值"],
            ["利率上升風險",
             f"若房貸為浮動利率，利率上升 1% → 年還款增加約 {ml_balance * 0.01:.1f} 萬",
             "2022–2023 美國升息週期：30 年固定房貸從 3% 升至 7.5%",
             "台灣房貸多為浮動利率，退休後應優先鎖定固定利率或提前還清"],
        ], columns=["因素", "計算", "數據", "退休建議"])
        st.dataframe(lev_df, use_container_width=True, hide_index=True)

        # 決策框架
        st.markdown("#### 主動槓桿決策矩陣")
        decision_df = pd.DataFrame([
            ["低利率（< 2%）+ 高報酬環境（> 6%）", "高",
             "保留房貸，差額全額投資；每年套利空間大",
             "需確保三桶金第一桶 ≥ 3 年生活費 + 一年房貸"],
            ["中利率（2–3%）+ 中報酬環境（4–6%）", "中",
             "視行為偏好決定；數學上略有優勢但不顯著",
             "半還半投；或用租金收入對沖房貸支出"],
            ["高利率（> 3%）+ 低報酬環境（< 4%）", "低",
             "優先還清房貸；負債成本高於投資報酬",
             "退休前清償；釋出的現金流大幅改善退休品質"],
            ["任何利率 + 心理壓力大",              "—",
             "行為成本優先：還清房貸消除焦慮",
             "無債一身輕的心理安全感有真實財務價值"],
            ["有租金收入且能覆蓋房貸",              "高",
             "租金支付房貸 → 資產同時保留兩個效益",
             "最佳情境：槓桿免費，因租金負擔了成本"],
        ], columns=["情境", "槓桿優勢", "建議策略", "執行提示"])
        st.dataframe(decision_df, use_container_width=True, hide_index=True)

        # 綜合評估
        net_arb = ml_invest_r - ml_rate
        if ml_balance == 0:
            st.success("✅ 目前無房貸，現金流最大化，退休序列報酬風險最低。")
        elif net_arb > 2:
            st.success(
                f"📊 **主動槓桿有優勢**：投資報酬率（{ml_invest_r:.1f}%）比房貸利率（{ml_rate:.1f}%）"
                f"高 {net_arb:.1f} 個百分點，每年理論增益約 **{ml_balance * net_arb / 100:.1f} 萬**。\n\n"
                "建議：保留房貸並繼續投資，但確保三桶金第一桶覆蓋至少一年房貸支出作為緩衝。"
            )
        elif net_arb > 0:
            st.warning(
                f"⚠️ **槓桿空間有限**：套利僅 {net_arb:.1f}%，扣除行為風險與現金流壓力後優勢不明顯。"
                "建議依個人風險承受度決定，有租金收入者可考慮以租金覆蓋房貸。"
            )
        else:
            st.error(
                f"❗ **負槓桿**：房貸利率（{ml_rate:.1f}%）超過投資報酬率（{ml_invest_r:.1f}%），"
                "持有房貸是淨成本。建議優先還清，釋放現金流。"
            )

        # ── 四、淨現金流壓力測試 ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 四、退休淨現金流壓力測試（多情境）")
        st.markdown("以下假設退休第一年的靜態現金流結構，比較三種情境下的淨現金流。")

        base_living  = W0 / 10_000
        base_pension = pension_annual / 10_000
        base_rental  = (rental_annual + rm_annual) / 10_000
        base_draw    = max(0, base_living - base_pension - base_rental)

        stress_df = pd.DataFrame([
            ["基準情境",
             f"{base_living:.1f}",
             f"{base_pension:.1f}",
             f"{base_rental:.1f}",
             f"{ml_annual_pay:.1f}" if ml_balance > 0 else "0",
             f"{base_draw + (ml_annual_pay if ml_balance > 0 else 0):.1f}",
             "—"],
            ["熊市衝擊（第一年跌 30%）",
             f"{base_living:.1f}",
             f"{base_pension:.1f}",
             f"{base_rental * 0.85:.1f}（空置上升）",
             f"{ml_annual_pay:.1f}" if ml_balance > 0 else "0",
             f"{base_draw * 1.15 + (ml_annual_pay if ml_balance > 0 else 0):.1f}",
             "⚠️ 被迫在低點賣出"],
            ["通膨衝擊（通膨 5%）",
             f"{base_living * 1.05:.1f}",
             f"{base_pension:.1f}",
             f"{base_rental * 1.03:.1f}（租金部分跟漲）",
             f"{ml_annual_pay:.1f}（浮動利率+0.5%）" if ml_balance > 0 else "0",
             f"{(base_living * 1.05 - base_pension - base_rental * 1.03) + (ml_annual_pay * 1.02 if ml_balance > 0 else 0):.1f}",
             "⚠️ 購買力侵蝕"],
            ["租金中斷（空置 6 個月）",
             f"{base_living:.1f}",
             f"{base_pension:.1f}",
             f"{base_rental * 0.5:.1f}",
             f"{ml_annual_pay:.1f}" if ml_balance > 0 else "0",
             f"{base_living - base_pension - base_rental * 0.5 + (ml_annual_pay if ml_balance > 0 else 0):.1f}",
             "⚠️ 被動收入腰斬"],
            ["最壞複合情境",
             f"{base_living * 1.05:.1f}",
             f"{base_pension:.1f}",
             "0（完全空置）",
             f"{ml_annual_pay * 1.05:.1f}（利率上升）" if ml_balance > 0 else "0",
             f"{base_living * 1.05 - base_pension + (ml_annual_pay * 1.05 if ml_balance > 0 else 0):.1f}",
             "❗ 需從資產大量提領"],
        ], columns=["情境", "生活費 (萬)", "年金收入 (萬)", "租金收入 (萬)", "房貸支出 (萬)", "需從資產提領 (萬)", "風險提示"])
        st.dataframe(stress_df, use_container_width=True, hide_index=True)

        # ── 五、最適現金流架構建議 ──────────────────────────────────────
        st.markdown("---")
        st.markdown("### 五、最適退休現金流架構：七大設計原則")
        principles = [
            ("① Floor 優先", "先用確定性收入（年金+折現後租金）覆蓋基本生活費，目標覆蓋率 ≥ 70%"),
            ("② 房貸在退休前清償", "退休時有房貸等同在收入下降後新增固定義務；若無法退休前還清，確保租金可覆蓋房貸"),
            ("③ 主動槓桿需有緩衝", "保留房貸+投資的策略，三桶金第一桶必須額外保留 ≥ 1 年房貸支出的現金"),
            ("④ 浮動利率鎖定或提前清償", "退休後無法承受月付額波動，利率上升 2% 相當於每百萬貸款增加月付約 1,000 元"),
            ("⑤ 現金流波動 > 報酬率波動", "退休規劃中，現金流是否足夠比報酬率高低更關鍵；優先保障現金流的可預測性"),
            ("⑥ 以租金覆蓋房貸是最佳槓桿形式", "租金收入抵銷房貸支出 → 槓桿的成本轉嫁給租客，同時保留資產升值潛力"),
            ("⑦ 定期重估現金流結構", "每 2–3 年重新評估：利率環境、健康狀況、家庭結構變化都會影響最適策略"),
        ]
        for title, desc in principles:
            st.markdown(f"**{title}**：{desc}")

        st.success(
            "**核心洞見**：退休現金流管理的本質，是在「確定性」與「效率」之間取得動態平衡。"
            "房貸槓桿提升資產效率，但降低現金流確定性；"
            "還清房貸犧牲部分數學效益，但大幅提升心理安全感與抗壓韌性。"
            "沒有放諸四海皆準的答案——**最好的策略，是你能在熊市中仍能安然執行的策略**。"
        )
        st.caption(
            "文獻：① Pfau (2017) 'Reverse Mortgages: How to Use Reverse Mortgages to Secure Your Retirement' "
            "② Kitces (2013) 'Should Retirees Pay Off Their Mortgage?' "
            "③ Vanguard 'The Role of Debt in Retirement' (2024) "
            "④ 台灣金管會「退休族借貸行為調查」2025 "
            "⑤ Bengen (1994) 4% Rule with Mortgage Extension"
        )

    # ── 延伸閱讀（跨分頁導覽）：集中在頁尾，維持版面整潔 ────────────────
    _edu_id = str(edu_topic).split("｜", 1)[0]
    _edu_related = {
        "1": [
            ("🛠️ 提領實務指南", "核心框架（先看這裡）", "1｜三桶金策略（Bucket Strategy）"),
            ("🛠️ 提領實務指南", "核心框架（先看這裡）", "3｜滑行路徑：激進→保守的轉換節奏"),
        ],
        "2": [
            ("🛠️ 提領實務指南", "賣出／提領規則", "2｜何時賣股票：五大觸發條件"),
            ("🛠️ 提領實務指南", "核心框架（先看這裡）", "4｜再平衡規則：時間 vs 門檻觸發"),
            ("🛠️ 提領實務指南", "熊市／風險管理", "6｜熊市/股災應對手冊"),
        ],
        "3": [
            ("🛠️ 提領實務指南", "賣出／提領規則", "5｜提領順序與稅務效率"),
        ],
        "6": [
            ("🛠️ 提領實務指南", "熊市／風險管理", "6｜熊市/股災應對手冊"),
            ("🛡️ 保險規劃（參考）", "（此分頁）", "2｜退休常見保障面向（對照生活事件）"),
        ],
        "7": [
            ("🛠️ 提領實務指南", "台灣 ETF 實務案例", "7｜台灣 ETF 實務（0050/0056/00878）"),
            ("🛠️ 提領實務指南", "熊市／風險管理", "9｜全球分散化：降低非系統性風險"),
            ("🛠️ 提領實務指南", "台灣 ETF 實務案例", "10｜50% 006208 + 50% VT 雙基金提領策略"),
        ],
        "18": [
            ("🛡️ 保險規劃（參考）", "（此分頁）", "1｜為什麼要分開：投資報酬 vs 保險"),
            ("🛡️ 保險規劃（參考）", "（此分頁）", "2｜退休常見保障面向（對照生活事件）"),
        ],
    }
    _rels = _edu_related.get(_edu_id, [])
    if _rels:
        st.divider()
        with st.expander("延伸閱讀（推薦下一步）", expanded=False):
            st.caption("可點擊超連結直接預選目標頁的分類/主題（不影響引擎計算）。")
            for page, cat, topic in _rels[:3]:
                if page.startswith("🛠️"):
                    href = _qs(nav="guide", guide_category=cat, guide_topic=topic)
                elif page.startswith("🛡️"):
                    href = _qs(nav="ins", ins_topic=topic)
                else:
                    href = _qs(nav="edu", edu_category=cat, edu_topic=topic)
                st.markdown(f"- [{page} → 分類 **{cat}** → 主題 **{topic}**]({href})")

    st.divider()
    st.caption("資料來源：① 退休規劃彙整表 ② 2026 退休安全策略手冊/研究整理 ③ Morningstar State of Retirement Income 2025 ④ 財政部（114年度）綜所稅相關公告與附表 ⑤ 勞動部勞保局（年金規定）⑥ 主計總處高齡家庭CPI（2024 起月資料）⑦ 主要研究機構公開預測（如 SPF 等）")

# ──────────────────────────────────────────────
# PAGE：提領實務指南
# ──────────────────────────────────────────────
elif page_id == "guide":
    st.title("🛠️ 退休提領實務指南")
    st.caption("整合：Morningstar、Kitces、Christine Benz、T. Rowe Price 等公開研究與實務框架（並以台灣ETF情境示例）。")
    st.info(
        "本指南回答退休最實際的三個問題：**「何時賣股票」「何時由攻轉守」「遇到股災怎麼辦」**。"
        "內容以台灣市場（0050、0056、00878）為核心，並整合國際最新研究。"
    )
    if _nav == "guide":
        st.success("已從「延伸閱讀」導覽過來：本頁已自動預選分類/主題。", icon="✅")
    with st.expander("台灣 KOL/作者共識（2025–2026，已排除黑名單）", expanded=False):
        st.markdown(_tw_kol_consensus_md())
        st.caption("用途：把『口語建議』落地成可操作規則，並對齊本 App 已實裝功能。")
    st.divider()

    # ── 分類：先選分類，再選主題（避免單排太長）
    _guide_categories = {
        "核心框架（先看這裡）": [
            "1｜三桶金策略（Bucket Strategy）",
            "4｜再平衡規則：時間 vs 門檻觸發",
            "3｜滑行路徑：激進→保守的轉換節奏",
        ],
        "賣出／提領規則": [
            "2｜何時賣股票：五大觸發條件",
            "5｜提領順序與稅務效率",
        ],
        "熊市／風險管理": [
            "6｜熊市/股災應對手冊",
            "9｜全球分散化：降低非系統性風險",
        ],
        "台灣 ETF 實務案例": [
            "7｜台灣 ETF 實務（0050/0056/00878）",
            "10｜50% 006208 + 50% VT 雙基金提領策略",
        ],
        "海外 ETF 實務": [
            "11｜海外市值型 ETF：海外券商（FirstTrade）vs 複委託",
        ],
        "工具／互動式": [
            "8｜Bond Tent 規劃工具（互動式）",
        ],
    }

    _g_default = str(st.session_state.get("guide_category", "核心框架（先看這裡）"))
    if _g_default not in _guide_categories:
        _g_default = "核心框架（先看這裡）"
    guide_category = st.selectbox(
        "分類",
        list(_guide_categories.keys()),
        index=list(_guide_categories.keys()).index(_g_default),
        key="guide_category",
        help="先用分類縮小範圍，再選主題閱讀。",
    )
    guide_topic = st.radio(
        "主題",
        _guide_categories[guide_category],
        horizontal=True,
        key="guide_topic",
    )
    st.divider()

    # ── 1. 三桶金策略 ──────────────────────────────────────────────────
    if guide_topic.startswith("1｜"):
        st.subheader("三桶金策略（Bucket Strategy）")
        st.markdown("> **提出者**：Harold Evensky（1985）；現代化整理：Christine Benz, Morningstar（2010–2025）")
        st.markdown("""
三桶金策略的核心思想是：**把「何時需要用錢」轉化為資產配置的決策依據**，
讓你在股市下跌時永遠有現金可用，不必賤價賣股，從而解決 SORR（序列報酬風險）。
        """)

        buckets = pd.DataFrame([
            ["🟢 短期桶（Bucket 1）", "1–2 年生活費", "現金、定存、貨幣市場基金",
             "不投資；隨時可動用", "SORR 防火牆，股災時絕不動用此桶以外的資產"],
            ["🟡 中期桶（Bucket 2）", "3–10 年生活費", "債券、平衡型基金、高股息 ETF（0056/00878）",
             "每年再平衡時補充短期桶", "穩定配息補短期桶；波動較低"],
            ["🔴 長期桶（Bucket 3）", "11 年以上", "全市場指數 ETF（0050、VTI、VWRA）",
             "市場強時獲利了結補充中期桶", "對抗通膨、實現長期複利"],
        ], columns=["桶別", "用途時程", "建議資產類型", "補充邏輯", "設計目的"])
        st.dataframe(buckets, use_container_width=True, hide_index=True)

        st.subheader("桶的補充流程（年度操作）")
        st.markdown("""
```
每年底 / 需要用錢時 → 按以下順序補充：

Step 1 │ 短期桶不足 1 年生活費？
        │  YES → 從中期桶轉入
        │  NO  → 進入 Step 2

Step 2 │ 中期桶 < 3 年生活費？
        │  YES → 長期桶有獲利？賣出長期桶補充
        │         長期桶下跌中？暫時只動用短期桶，等待市場回升
        │  NO  → 不做動作，讓各桶繼續成長
```
        """)

        st.subheader("Christine Benz 2025 最新建議")
        st.markdown("""
- **不必嚴格「順序消費」每個桶**，應每年評估市場狀況，從**表現最好的資產類別**賣出供生活費，這同時完成了再平衡。
- 短期桶現金持有量：**保守型持 2 年；進取型持 1 年**，過多現金等於拖累長期報酬。
- 2025 年環境：高殖利率債券提供罕見機會，中期桶可適度拉長存續期間（Duration）。
        """)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.success("**優點**\n- 心理安全感高（看得見現金）\n- 天然的 SORR 保護機制\n- 操作直覺易理解")
        with col_b2:
            st.warning("**缺點**\n- 過多現金拖累長期報酬\n- 桶間補充需主動管理\n- 稅務效率不及「總報酬法」")

    # ── 2. 何時賣股票 ───────────────────────────────────────────────────
    elif guide_topic.startswith("2"):
        st.subheader("何時賣股票：五大觸發條件")
        st.markdown("""
退休後賣股票的核心原則是：**永遠賣「表現最好的」，而不是「最容易賣的」**。
這條規則同時完成兩件事：補充生活費 + 自動再平衡。
        """)

        st.markdown("### 觸發條件 1：股票已超過目標配置比例")
        st.markdown("""
| 情況 | 操作 |
|---|---|
| 股票實際佔比 > 目標 + 5% | 賣出多出的部分，補充現金桶或再平衡至目標 |
| 例：目標 60% 股票，現在漲到 68% | 賣出約 8% 的股票，轉入債券/現金 |
- 這是**最重要的賣股時機**：此時賣股票同時鎖住了漲幅，又不用犧牲任何資產
- 工具：每年底或每半年檢視一次配置比例
        """)

        st.markdown("### 觸發條件 2：短期桶現金不足（1 年以內）")
        st.markdown("""
| 優先賣哪一種？ | 理由 |
|---|---|
| 1️⃣ 股票（若市場在高點或平盤） | 賣出後補現金，同時降低股票比例至目標 |
| 2️⃣ 長期債券（若股票在低點） | 避免「低點賣股」；債券波動小，適合低市場時補現金 |
| 3️⃣ 高股息 ETF 配息自然流入 | 0056/00878 配息直接補充短期桶，無需主動賣出 |
- **嚴格禁止**：股市下跌超過 20% 時賣股票補生活費（破壞複利基礎）
        """)

        st.markdown("### 觸發條件 3：股票出現重大基本面變化")
        st.markdown("""
- 持有個股（非 ETF）時，公司出現：倒閉風險、長期競爭優勢消失、配息持續削減
- ETF 本身不存在個別公司倒閉風險；但若指數追蹤錯誤（tracking error > 2%）需考慮換基金
- 台灣案例：持有金融股個股時，若銀行面臨監管重大調整，可考慮轉換至 00878
        """)

        st.markdown("### 觸發條件 4：達到年度提領目標（4% 法則執行）")
        st.markdown("""
| 情境 | 賣股方式 |
|---|---|
| 年初一次性提領 | 元月第一週，賣出達到全年生活費的股票或債券 |
| 每季分批提領 | 每季賣出 1/4 年度預算，平滑市場時機風險 |
| 依配息補貼 | 先收 0050/0056 配息（每年約 2–3 次），不足部分再賣出本金 |
- **分批賣出優於單次**：避免單一高點或低點的市場時機風險（類似「逆向定期定額」）
        """)

        st.markdown("### 觸發條件 5：稅務規劃（年底損失收割）")
        st.markdown("""
- **台灣現況（2025）**：台股證交稅 0.3%（賣出時課），沖銷機制有限
- **海外股票/ETF**：若持有美股 ETF（VTI、VWRA）且有帳面虧損，可在年底賣出認列損失，隔年再買回（wash-sale rule 在台灣無法規限制）
- **股利稅務**：綜所稅率 ≤ 20% 者應選「合併計稅」享 8.5% 抵減；≥ 30% 者選「分離課稅」28%
        """)

        st.error("**絕對禁止的賣股情境**\n"
                 "- 股市下跌 > 20% 時為「停損」而賣出（退休後沒有理由停損）\n"
                 "- 因恐慌、新聞、市場情緒賣出（情緒性交易是退休破產第一大原因）\n"
                 "- 賣出全部股票轉換為現金/定存（通膨在 20 年內會侵蝕 40% 以上購買力）")

    # ── 3. 滑行路徑：激進→保守 ─────────────────────────────────────────
    elif guide_topic.startswith("3"):
        st.subheader("滑行路徑（Glide Path）：激進資產轉向保守資產的節奏")
        st.markdown("> **概念來源**：目標日期基金（TDF）設計原理，Vanguard、Fidelity Target Date Fund")

        st.markdown("""
滑行路徑的核心邏輯：**你的「人力資本」（未來薪資的現值）是隱形的無風險資產**。
年輕時人力資本龐大，投資組合可以激進；退休後人力資本歸零，金融資產必須自己承擔穩定性。
        """)

        glide_df = pd.DataFrame([
            ["25–40 歲", "累積期", "90%", "10%", "最大化長期複利；人力資本是最大保護墊"],
            ["40–55 歲", "增長期", "70–80%", "20–30%", "逐步降低波動；開始建立債券部位"],
            ["55–60 歲", "退休前 5 年", "60%", "40%", "進入「脆弱地帶」；SORR 風險開始上升"],
            ["60–65 歲", "退休過渡", "50%", "50%", "建立 1–2 年現金桶；完成三桶金設置"],
            ["65–75 歲", "退休初期", "50–60%", "40–50%", "維持股票對抗通膨；開始消費微笑曲線"],
            ["75–85 歲", "退休中期", "40–50%", "50–60%", "緩速期支出縮減；適度降低股票波動"],
            ["85 歲以上", "護理期", "30–40%", "60–70%", "醫療支出主導；流動性優先；遺產規劃"],
        ], columns=["年齡區間", "人生階段", "股票比例", "債券/現金比例", "調整邏輯"])
        st.dataframe(glide_df, use_container_width=True, hide_index=True)

        st.subheader("兩種主流滑行路徑設計")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("#### 傳統遞減型（To Retirement）")
            st.markdown("""
- 退休當天達到最低股票比例（如 30%）
- **Vanguard 目標退休基金** 的設計
- 優點：退休時風險最低
- 缺點：早期低股票比例可能讓資產跑輸通膨
            """)
        with col_g2:
            st.markdown("#### 通過退休型（Through Retirement）")
            st.markdown("""
- 退休後繼續持有 50%+ 股票，直到 75–80 歲才達低點
- **Fidelity Freedom Fund** 的設計（Morningstar 2025 年建議此方向）
- 優點：長壽風險保護更強
- 缺點：退休初期波動較大，心理承受壓力高
            """)

        st.subheader("台灣情境的滑行路徑實作")
        st.markdown("""
| 年齡 | 0050（指數成長）| 00878/0056（高股息）| 債券/現金 |
|---|---|---|---|
| 55–60 歲 | 40% | 20% | 40% |
| 60–65 歲 | 30% | 20% | 50% |
| 65–75 歲 | 25% | 25% | 50% |
| 75–85 歲 | 15% | 25% | 60% |
| 85 歲以上 | 10% | 20% | 70% |

> 高股息 ETF（00878/0056）扮演「類債券」的角色：配息穩定但仍有股票成長性，
> 是台灣特有的中間過渡資產，填補純股票與純債券之間的缺口。
        """)

        st.subheader("何時應該加速由攻轉守？")
        trigger_df = pd.DataFrame([
            ["市值跌破初始資產 70%", "加速降低股票比例 10–15%", "高"],
            ["IWR（實際提領率）超過 5%", "降低股票比例，增加固定收益資產", "高"],
            ["健康狀況惡化、確診重大疾病", "縮短時間視野，提高現金比例", "高"],
            ["退休後遭遇熊市（第 1–5 年）", "暫停再平衡，優先從現金桶/債券桶提領", "中"],
            ["遺產規劃意圖改變（想多留）", "可適度增加股票以提高長期報酬", "低"],
        ], columns=["觸發條件", "建議操作", "優先等級"])
        st.dataframe(trigger_df, use_container_width=True, hide_index=True)

        st.warning("**常見錯誤**：退休時把所有股票換成定存或高股息 ETF → 通膨在 20 年內侵蝕 40% 購買力。即使 85 歲，仍建議維持至少 20–30% 的指數型 ETF。")

        st.divider()
        st.subheader("進階：Pfau–Kitces 上升股票比例路徑與債券帳篷")
        st.markdown("""
> **Pfau & Kitces (2014)**「Reducing Retirement Risk with a Rising Equity Glide-Path」——
> 這篇研究顛覆了「越老越保守」的傳統教條。

上述「傳統遞減型 / 通過退休型」的本質差異，只是下降速度快慢的問題。
Pfau–Kitces 的研究更進一步問：**退休後，股票比例可不可以不只「降得慢」，而是實際「往上升」？**

答案是：**在特定條件下，這樣做反而能提高長期存活率。**
        """)

        st.markdown("#### 為什麼逆向做反而更安全？")
        st.markdown("""
**序列報酬風險（SORR）的時間非對稱性**：

同樣遭遇 -40% 的熊市，對退休第 1 年的打擊，是退休第 20 年的 **5 倍以上**。
原因是退休初期每賣一元股票以供生活費，損失的是那筆錢未來 20–30 年的複利潛力。

| 遭遇熊市的時間 | 資產回復所需年數（估算）| 說明 |
|---|---|---|
| 退休第 1 年 | 14–18 年 | 本金已大幅縮水，複利基礎永久受損 |
| 退休第 5 年 | 8–12 年 | 有部分複利緩衝，但仍嚴重 |
| 退休第 15 年 | 3–5 年 | 消費已自然縮減，資產基礎相對穩固 |
| 退休第 20 年 | 1–2 年 | 時間視野縮短，SORR 敏感度大幅降低 |

因此，**退休初期用較多債券/現金作 SORR 的緩衝盾牌，是有科學依據的**；
但之後當 SORR 風險降低、長壽風險升高，就應該把盾牌換回成長的矛。
        """)

        st.markdown("#### 債券帳篷（Bond Tent）——具體實作方式")
        st.markdown("""
```
股票配置比例（示意圖）

 80% ─────────────────────────────────────╮  ←（退休後逐年拉高股票）
 70% ──────────────────────────────────╮  │
 60% ───────────────────────────────╮  │  │
 50% ──────────────────────────╮    │  │  │
 40% ─────────────────────╮    │    │  │  │
 30% ──────────────────╮  │    │    │  │  │     帳篷頂點
     ─────────────────╮│  │    │    │  │  │  ← 退休當天最低股票
 30%               ╲  ││  │    │    │  │  │
                    ╲ ││  ▼    ▼    ▼  ▼  ▼
                     ╲╯│ 67  70  73  77 80 歲
                       ╰─── 退休後逐年回升 ───→
      55  58  61  65
        退休前逐年下降
```

**前半段（退休前 5–7 年）**：主動把股票比例從 70% 降至 30–40%，蓄積現金與短期債作緩衝。
**帳篷頂點（退休當天）**：股票比例最低，SORR 防護最強。
**後半段（退休後每 1–3 年）**：每年補回 1–2% 股票，直到 80 歲後穩定在 50–60%。
        """)

        st.markdown("#### 此策略的適用條件與限制")
        bt_df = pd.DataFrame([
            ["退休時 CAPE < 20（市場低估）", "帳篷頂點可提高到 40–50%；SORR 風險相對低",
             "Pfau-Kitces 研究支持"],
            ["退休時 CAPE > 30（市場高估）", "帳篷頂點降至 20–30%；後段回升應更積極",
             "2025 年現況（CAPE ≈ 35），應採較深防禦"],
            ["有穩定勞保/勞退月領", "帳篷可淺一點（股票比例高 5–10%），因為有固定現金流抵銷提領壓力",
             "月領越多，SORR 緩衝需求越低"],
            ["持有不動產租金", "效果類似穩定月領；可支撐稍高的退休初期股票比例",
             "與本 App 的租金現金流模組結合"],
            ["資產遠超提領需求（IWR < 2%）", "Bond Tent 效益相對低；靜態高股票反而更優",
             "IWR 越低，SORR 抵抗力天然越強"],
        ], columns=["情境", "調整建議", "依據"])
        st.dataframe(bt_df, use_container_width=True, hide_index=True)

        st.success(
            "**Bond Tent 的直覺總結**：\n\n"
            "退休時主動「躲進帳篷」（降股票）→ 安全度過最脆弱的前 10 年 → "
            "再「走出帳篷」（升股票）→ 以成長對抗餘下 20–30 年的通膨與長壽威脅。\n\n"
            "這不是「保守」，而是在對的時間用對的武器。"
        )
        st.caption(
            "文獻：Pfau & Kitces (2014) 'Reducing Retirement Risk with a Rising Equity Glide Path', "
            "Journal of Financial Planning；"
            "Kitces (2017) 'The Bond Tent: Managing Portfolio Size Effect in the Retirement Red Zone'"
        )

    # ── 4. 再平衡規則 ──────────────────────────────────────────────────
    elif guide_topic.startswith("4"):
        st.subheader("再平衡規則：時間驅動 vs 門檻觸發")
        st.markdown("> **文獻**：Kitces (2015)「An In-Depth Look at Rebalancing Strategies」；Morningstar 2025")

        st.markdown("""
再平衡是退休後**唯一有科學依據的「賣高買低」機制**，不依賴任何對市場的預測。
        """)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("#### 方法 A：時間驅動（Calendar Rebalancing）")
            st.markdown("""
- **執行時機**：每年固定一次（建議 12 月底或 1 月初）
- **優點**：簡單、可預期、適合投資新手
- **缺點**：市場極端行情時反應遲鈍
- **適合族群**：懶人投資者、不想頻繁操作者
            """)
        with col_r2:
            st.markdown("#### 方法 B：門檻觸發（Threshold Rebalancing）")
            st.markdown("""
- **執行時機**：任何資產類別偏離目標超過 **±5%** 時
- **優點**：貼近市場、自動化「賣高買低」
- **缺點**：需持續監控；交易成本略高
- **適合族群**：有定期追蹤習慣者
            """)

        st.markdown("#### 方法 C（建議）：混合模式")
        st.markdown("""
| 規則 | 內容 |
|---|---|
| **主要檢視點** | 每年 12 月底，無論市場如何都強制檢視 |
| **即時警戒門檻** | 任何資產類別偏離 ±10%（寬鬆門檻，避免過度交易）|
| **提領再平衡** | 需要賣股補生活費時，**優先賣漲最多的資產**，天然完成再平衡 |
| **熊市期間** | 股票下跌 > 20% 時，**暫停再平衡，等待反彈後再執行** |
        """)

        st.subheader("再平衡的執行步驟（逐步教學）")
        st.markdown("""
**Step 1｜設定目標配置**
例：股票 50%、債券 30%、現金 20%

**Step 2｜每年底計算實際配置**
```
股票市值 / 總資產 = 實際股票比例
```

**Step 3｜判斷是否需要再平衡**
```
偏差 = 實際比例 - 目標比例
若 |偏差| > 5% → 執行再平衡
若 |偏差| ≤ 5% → 不動作
```

**Step 4｜執行方式（優先順序）**
1. **用新增現金/配息** 買入不足的資產（零成本再平衡）
2. **提領時從漲最多的資產賣出**（同時補生活費）
3. 上述不足時，**賣出超配資產、買入低配資產**
        """)

        st.subheader("再平衡的稅務考量（台灣）")
        rebal_tax = pd.DataFrame([
            ["台股 ETF（0050/0056/00878）", "賣出時課 0.3% 證交稅", "低頻再平衡以降低交易成本"],
            ["海外 ETF（VTI/VWRA）", "無台灣證交稅，需申報海外所得", "年底損失收割可抵減其他所得"],
            ["高股息配息再投入", "不賣出即無證交稅；配息算利息所得", "用配息買入不足資產 = 零稅務再平衡"],
        ], columns=["資產類型", "稅務說明", "最佳實踐"])
        st.dataframe(rebal_tax, use_container_width=True, hide_index=True)

    # ── 5. 提領順序與稅務效率 ──────────────────────────────────────────
    elif guide_topic.startswith("5"):
        st.subheader("提領順序與稅務效率")
        st.markdown("> **文獻**：Christine Benz, Morningstar (2025)；Kitces Tax-Efficient Withdrawal Sequencing")

        st.markdown("### 國際通用提領順序（稅務效率最高）")
        order_df = pd.DataFrame([
            ["第 1 順位", "應稅帳戶（台灣：一般證券帳戶）",
             "台灣股票無資本利得稅，提領成本最低。優先賣出已實現利潤的部位。"],
            ["第 2 順位", "稅務遞延帳戶（勞退個人專戶、IRA類）",
             "提領時計入綜合所得稅，但長期複利效益強。退休後所得較低時提領最有利。"],
            ["第 3 順位", "免稅帳戶（勞保老年年金、Roth IRA類）",
             "勞保月領完全免稅，應盡量延後動用，讓其繼續累計。"],
        ], columns=["順序", "帳戶類型", "理由"])
        st.dataframe(order_df, use_container_width=True, hide_index=True)

        st.markdown("### 台灣退休後的實際提領流程")
        st.markdown("""
```
月度收入來源優先序：

1. 勞保老年年金（完全免稅）→ 每月固定入帳，直接支應生活費
2. 勞退新制月領 or 定期存款利息 → 補充勞保不足的部分
3. ETF 配息（0056/00878 每季/半年配息）→ 補充季度性費用
4. 出售 ETF 本金（0050 或 VTI）→ 最後才動，且只在股市高點或年度再平衡時操作
```
        """)

        st.markdown("### 降低稅負的實務技巧")
        tax_tips = pd.DataFrame([
            ["控制年度實現所得", "台股 ETF 賣出無資本利得稅；但股利計入綜所稅，需控制單一年度股利總額在最低稅率級距內",
             "全年股利 < 54 萬（免稅額 + 標準扣除額）→ 實際零稅"],
            ["合併 vs 分離課稅", "綜所稅率 ≤ 20% 選合併計稅（享 8.5% 抵減）；≥ 30% 選分離課稅 28%",
             "70 歲後免稅額提升（依當年度公告），更容易達到低稅或零稅"],
            ["二代健保補充保費", "單筆股利 ≥ 2 萬扣 2.11%。可分散持股或選季配/月配 ETF 降低單筆金額",
             "00878 月配息設計可自然規避大額單筆"],
            ["夫妻合併申報", "退休後若其中一方無所得，合併申報可多 1 份免稅額與扣除額",
             "每年可多節省約 4–8 萬稅負"],
        ], columns=["技巧", "說明", "效果估算"])
        st.dataframe(tax_tips, use_container_width=True, hide_index=True)

        st.subheader("Roth 轉換策略（若持有海外帳戶）")
        st.markdown("""
對持有美國帳戶的投資人：
- 退休初期所得較低的年份，可將傳統 IRA 轉換為 Roth IRA
- 繳一次所得稅後，未來所有提領免稅
- **最佳執行視窗**：退休後至開始請領社會安全年金（SS）前的 5–10 年
        """)

    # ── 6. 熊市/股災應對手冊 ────────────────────────────────────────────
    elif guide_topic.startswith("6"):
        st.subheader("熊市/股災應對手冊")
        st.markdown("""
退休後遇到熊市，**最大的敵人不是市場，而是自己的行為**。
Morningstar 2025 研究顯示：退休後不當停損所造成的損失，遠大於熊市本身。
        """)

        st.markdown("### 歷史股災對退休提領的實際衝擊")
        crash_df = pd.DataFrame([
            ["2000–2002 科技泡沫", "-49%（S&P500）", "退休第 1 年遇到此波段，4% 法則失敗率 > 50%",
             "三桶金短期桶 2 年現金可完全度過，無需賣股"],
            ["2008–2009 金融海嘯", "-57%（S&P500）；0050 跌 43%", "退休 5 年內遇到此波段，資產可能腰斬",
             "GK 護欄自動啟動「減薪 10%」可延長存活年限 5–8 年"],
            ["2020 新冠崩盤", "-34%（S&P500）；但 40 天反彈", "短暫但劇烈；反彈速度超預期",
             "不賣股者 6 個月後資產完整恢復"],
            ["2022 升息熊市", "-25%（S&P500）；債券同跌 -17%", "股債同跌，三桶金中期桶也受損",
             "現金桶重要性大增；高通膨情境須謹慎調整提領"],
        ], columns=["事件", "最大跌幅", "對退休的衝擊", "三桶金/GK護欄的應對"])
        st.dataframe(crash_df, use_container_width=True, hide_index=True)

        st.markdown("### 股災期間的逐步應對 SOP")
        st.markdown("""
**階段一：股市下跌 10–20%（修正）**
- ✅ 維持原計畫，繼續從短期桶提領生活費
- ✅ 暫停再平衡（等待反彈後執行）
- ❌ 禁止賣出任何長期桶股票

**階段二：股市下跌 20–40%（熊市）**
- ✅ 完全停止從長期桶賣出，全面切換至短期桶 + 中期桶（債券/高股息配息）
- ✅ 觸發 GK 護欄「資本保全法則」：主動降低提領額 10%
- ✅ 評估可否從其他來源補充（兼職、租金、勞保月領）
- ❌ 禁止「情緒性轉換為全現金/全定存」

**階段三：股市下跌 > 40%（系統性危機）**
- ✅ 優先確保 2 年現金桶充足（若不足，賣出債券補充，不賣股票）
- ✅ 提領率臨時下調至 2.5–3%（GK 護欄若觸發會自動執行）
- ✅ 考慮提前申請勞保老年年金（若未達最佳請領年齡，評估利弊）
- ✅ 維持長期桶股票不動，等待歷史驗證的均值回歸
        """)

        st.markdown("### 不同退休年資的熊市脆弱度")
        vuln_df = pd.DataFrame([
            ["退休第 1–5 年", "極高", "SORR 最高風險期，每賣一元股票損失的複利效應最大",
             "短期桶放足 2 年；GK 護欄一定要開啟"],
            ["退休第 6–15 年", "中等", "資產基礎已建立，短期衝擊可恢復",
             "1 年短期桶即足；每年正常再平衡"],
            ["退休第 16–25 年", "較低", "75–85 歲；消費微笑曲線使支出下降，緩解壓力",
             "三桶金中期桶可縮小；股票比例已自然下降"],
            ["退休第 25 年以上", "低", "85 歲以上；遺產規劃優先；長壽風險取代市場風險",
             "流動性與醫療費用準備優先於股票報酬"],
        ], columns=["退休年資", "熊市脆弱度", "原因", "應對重點"])
        st.dataframe(vuln_df, use_container_width=True, hide_index=True)

        st.error("""
**熊市中絕對禁止的五件事**
1. 把全部資產換成現金（通膨會在 5–10 年內讓購買力大幅縮水）
2. 因新聞恐慌賣出指數型 ETF（0050 / VTI 從未歸零）
3. 停止三桶金流程（讓系統自動保護你）
4. 把退休金拿去「低點加碼」單一個股（退休資金不適合集中押注）
5. 比較今日帳戶價值與退休時最高點（帳面虧損 ≠ 實際損失，未賣出前不成立）
        """)

    # ── 7. 台灣 ETF 實務 ────────────────────────────────────────────────
    elif guide_topic.startswith("7"):
        st.subheader("台灣 ETF 退休提領實務（0050 / 0056 / 00878）")
        st.markdown("""
台灣三大退休 ETF 在三桶金策略中各有明確定位，了解其特性才能正確分配。
        """)

        etf_df = pd.DataFrame([
            ["0050 元大台灣 50",
             "追蹤台灣前 50 大市值公司；每年 2 次配息（約 3–4%）",
             "長期桶核心持股",
             "高波動（σ ≈ 20%）；大跌時心理壓力大；配息率低於高股息",
             "退休後持 30–40%；熊市時絕不賣；以指數成長對抗通膨"],
            ["0056 元大高股息",
             "追蹤 50 支高股息公司；每年 1 次配息（殖利率約 4–6%）",
             "中期桶 / 配息補現金桶",
             "選股條件限制成長性；2022 年起改季配，單次金額降低",
             "配息直接補充生活費；佔整體 15–20%"],
            ["00878 國泰永續高股息",
             "ESG 選股 + 高股息；每季配息（殖利率約 5–7%）",
             "中期桶 / 月現金流",
             "成立較晚（2020）；歷史數據較短",
             "季配息解決現金流問題；佔整體 15–20%；與 0056 互補"],
        ], columns=["ETF", "特性", "三桶金定位", "風險", "退休實務操作"])
        st.dataframe(etf_df, use_container_width=True, hide_index=True)

        st.subheader("配息再投入 vs 直接消費：何時該轉換？")
        st.markdown("""
| 人生階段 | 建議策略 | 理由 |
|---|---|---|
| 退休前（仍工作） | 配息全部再投入 | 複利效應最大化；稅務一樣要繳，不如讓本金繼續滾 |
| 剛退休（65–70 歲） | 配息部分消費、部分再投入 | 短期桶補充為主；過剩配息仍應再投入長期桶 |
| 穩定退休（70–80 歲） | 配息大部分直接消費 | 消費微笑曲線：支出趨於穩定；不再需要大幅本金成長 |
| 晚年（80 歲以上） | 配息完全消費；停止再投入 | 遺產規劃優先；流動性需求升高 |
        """)

        st.subheader("0050 在台灣的實測 SORR 分析（2008 金融海嘯情境）")
        sorr_df = pd.DataFrame([
            ["退休時資產", "2,000 萬", "2,000 萬"],
            ["退休年份遇到", "2007 年（高點前一年）", "2009 年（低點後一年）"],
            ["第 1 年報酬", "-43%（0050 大跌）", "+73%（0050 大反彈）"],
            ["固定提領 4%（80 萬/年）後", "資產剩 1,060 萬", "資產長至 3,320 萬"],
            ["第 10 年資產估算", "約 800–1,000 萬（瀕臨危險）", "約 4,000–5,000 萬"],
        ], columns=["項目", "2007 退休（最壞時機）", "2009 退休（最佳時機）"])
        st.dataframe(sorr_df, use_container_width=True, hide_index=True)
        st.caption("以上為簡化估算（示意用）；實際結果受配息再投入、手續費、稅務與追蹤指數口徑影響。")

        st.info("""
**對台灣投資人的具體建議（2026 版）**
- 退休組合建議：**0050（40%）+ 00878（20%）+ 台債/美債 ETF（20%）+ 現金（20%）**
- 每季收取 00878 配息補充生活費，減少主動賣股需求
- 0050 每年只在「超過目標配置 5%」或「年底再平衡」時才賣出
- 海外 ETF（VTI/VWRA）持有者：配息用複委託帳戶申報，適時選擇合併 vs 分離課稅
        """)

    # ── 8. Bond Tent 互動規劃工具 ──────────────────────────────────────
    elif guide_topic.startswith("8"):
        st.subheader("Bond Tent 動態滑行路徑規劃工具")
        st.markdown("""
根據你的年齡、退休時間與風險偏好，自動生成**最適 Bond Tent 滑行路徑**，
並與傳統遞減路徑逐年對比，讓你清楚看到每個年齡應持有的股票與債券比例。
        """)

        bt_c1, bt_c2, bt_c3 = st.columns(3)
        with bt_c1:
            bt_now_age_default = int(min(75, max(30, int(age_start))))
            bt_now_age    = st.number_input("目前年齡 (歲)",       min_value=30, max_value=75, value=bt_now_age_default, step=1)
            bt_retire_age_default = int(min(75, max(int(bt_now_age), 65)))
            bt_retire_age = st.number_input("預計退休年齡 (歲)",   min_value=bt_now_age, max_value=75, value=bt_retire_age_default, step=1)
        with bt_c2:
            bt_curr_eq    = st.slider("目前股票比例 (%)",          10, 100, 70, 5,
                                      help="含台股+美股+全球ETF的合計比例")
            bt_min_eq     = st.slider("退休當天最低股票 (%)",      10,  60, 35, 5,
                                      help="Bond Tent 頂點（帳篷最低點）；CAPE 高時建議 25–35%，低時可 40–50%")
        with bt_c3:
            bt_peak_eq    = st.slider("80 歲目標股票 (%)",         30,  80, 60, 5,
                                      help="退休後逐步回升的目標；長壽風險高者建議 55–65%")
            bt_rise_yrs   = st.slider("退休後爬升年數 (年)",        5,  20, 15, 1,
                                      help="從退休當天最低點爬升至目標所需年數；建議 12–18 年")

        # 傳統路徑：從現在線性降至退休後持續降低（每年 -1%），最低 20%
        # Bond Tent 路徑：退休前降，退休後升
        path_rows = []
        for age in range(bt_now_age, 91):
            years_to_retire  = bt_retire_age - bt_now_age
            # Bond Tent
            if age <= bt_retire_age:
                frac = (age - bt_now_age) / max(years_to_retire, 1)
                bt_eq = bt_curr_eq + (bt_min_eq - bt_curr_eq) * frac
            else:
                frac = min((age - bt_retire_age) / bt_rise_yrs, 1.0)
                bt_eq = bt_min_eq + (bt_peak_eq - bt_min_eq) * frac
            # 傳統路徑：退休前降至 bt_min_eq，退休後繼續每年降 1%，最低 15%
            if age <= bt_retire_age:
                frac_t = (age - bt_now_age) / max(years_to_retire, 1)
                trad_eq = bt_curr_eq + (bt_min_eq - bt_curr_eq) * frac_t
            else:
                trad_eq = max(15, bt_min_eq - (age - bt_retire_age) * 1.0)

            phase = ("退休前 ▼" if age < bt_retire_age
                     else ("退休當天 ⬟" if age == bt_retire_age
                           else ("爬升期 ▲" if age <= bt_retire_age + bt_rise_yrs
                                 else "穩定期 ─")))
            path_rows.append({
                "年齡": age,
                "Bond Tent 股票": f"{bt_eq:.0f}%",
                "Bond Tent 債券/現金": f"{100 - bt_eq:.0f}%",
                "傳統遞減 股票": f"{trad_eq:.0f}%",
                "傳統遞減 債券/現金": f"{100 - trad_eq:.0f}%",
                "階段": phase,
            })

        path_df = pd.DataFrame(path_rows)
        st.dataframe(path_df, use_container_width=True, hide_index=True, height=300)

        # 關鍵年齡摘要
        st.subheader("關鍵年齡節點摘要")
        key_ages = [bt_now_age, bt_retire_age,
                    min(bt_retire_age + 5, 90),
                    min(bt_retire_age + bt_rise_yrs, 90), 80, 90]
        key_ages = sorted(set(a for a in key_ages if bt_now_age <= a <= 90))
        key_rows = [r for r in path_rows if r["年齡"] in key_ages]
        st.dataframe(pd.DataFrame(key_rows), use_container_width=True, hide_index=True)

        st.subheader("與確定性退休模型的連結")
        bt_retire_eq = bt_min_eq
        bt_80_eq     = min(bt_peak_eq, bt_min_eq + (bt_peak_eq - bt_min_eq))
        st.info(
            f"**你的 Bond Tent 建議**：\n\n"
            f"- 退休當天起始股票比例：**{bt_retire_eq}%**（r 的推論請對應保守情境）\n"
            f"- 80 歲時目標股票比例：**{bt_peak_eq}%**（對應中性情境）\n"
            f"- 爬升期間（{bt_retire_age}–{min(bt_retire_age + bt_rise_yrs, 90)} 歲）："
            f"每年約回補 **{(bt_peak_eq - bt_retire_eq) / bt_rise_yrs:.1f}%** 股票部位"
        )

        st.markdown("#### 對應本 App 左側設定建議")
        bond_tent_table = pd.DataFrame([
            [f"退休初期（{bt_retire_age}–{bt_retire_age+5} 歲）",
             f"股票 {bt_retire_eq}% / 債券現金 {100-bt_retire_eq}%",
             "報酬情境選「保守」；r 偏低反映低股票比例"],
            [f"退休中期（{bt_retire_age+5}–{bt_retire_age+bt_rise_yrs} 歲）",
             f"股票 {int(bt_retire_eq + (bt_peak_eq-bt_retire_eq)*0.5)}% / 其餘現金債券",
             "報酬情境選「中性」；逐漸調高股票 ETF 比例"],
            [f"退休後期（{bt_retire_age+bt_rise_yrs}–90 歲）",
             f"股票 {bt_peak_eq}% / 其餘 {100-bt_peak_eq}%",
             "報酬情境選「中性/積極」；以全球 ETF 降低集中度"],
        ], columns=["退休階段", "建議股票比例", "本 App 對應操作"])
        st.dataframe(bond_tent_table, use_container_width=True, hide_index=True)

        st.caption(
            "文獻：Pfau & Kitces (2014)；Kitces (2017) 'Bond Tent'；Vanguard VLCM (2025)。"
            "本工具為概念規劃輔助，非投資建議。"
        )

    # ── 9. 全球分散化 ──────────────────────────────────────────────────
    elif guide_topic.startswith("9"):
        st.subheader("全球分散化：以降低非系統性風險提升成功率")
        st.markdown("""
> **學術依據**：現代投資組合理論（Markowitz, 1952）；
> DMS Global Investment Returns Yearbook (2024)；
> Vanguard Life-Cycle Investing Model (2025)
        """)

        st.markdown("### 為何單一市場集中是退休組合的隱性風險？")
        st.markdown("""
退休組合最需要的特性不是「最高報酬」，而是**「穩定性」與「可持續性」**。
單押台股（0050）或美股（VTI/SPY）雖然歷史報酬亮眼，但埋藏了兩種退休組合最忌諱的風險：
        """)

        conc_df = pd.DataFrame([
            ["台股 0050 集中度",
             "前十大成分股佔比 > 65%；台積電單一持股 > 33%（2025）",
             "半導體週期崩潰時，0050 可能跌幅遠超全球市場；2022 年 0050 跌 30%，VWRA 跌 18%"],
            ["美股集中度",
             "S&P500 前十大股票佔比 > 35%（科技 MAG7 主導）；CAPE ≈ 34–36（歷史高位）",
             "科技泡沫重演（2000年 -49%）時，退休初期持有美股的 SORR 破壞力極高"],
            ["貨幣集中度",
             "全部持有台幣計價資產，面臨台幣長期貶值風險",
             "台幣若對美元貶值 20%，購買力自動縮水；全球 ETF 天然持有多元貨幣"],
            ["地緣政治集中度",
             "台海風險、兩岸關係不確定性",
             "分散至全球市場可對沖地緣政治的黑天鵝事件"],
        ], columns=["集中度風險", "具體數據", "對退休的影響"])
        st.dataframe(conc_df, use_container_width=True, hide_index=True)

        st.markdown("### 全球分散化的實證效果")
        diversify_df = pd.DataFrame([
            ["100% 台股 0050", "11.6%", "~20%", "半導體週期、地緣政治、台幣", "高"],
            ["100% 美股 VTI", "9.5%（20年）", "~18%", "科技估值泡沫、美元", "高"],
            ["60% VTI + 40% VXUS（美+非美）", "8.0–9.0%（估）", "~15%", "全球系統性風險", "中"],
            ["VWRA/VT 全球一籃子", "7.5–8.5%（估）", "~14%", "全球系統性風險", "最低"],
            ["40% 台股 + 30% 美股 + 30% 全球", "8.0–9.5%（加權估）", "~15–16%", "組合後降低", "中低"],
        ], columns=["組合", "估算名目年化", "年波動率 σ", "主要風險", "集中度"])
        st.dataframe(diversify_df, use_container_width=True, hide_index=True)
        st.caption("以上為歷史估算，未來報酬不保證。本工具以「實質報酬／2026 購買力」為主要口徑；表中的名目年化僅供直覺對照。")

        st.markdown("### 全球 ETF 的實務選擇（台灣投資人）")
        etf_global_df = pd.DataFrame([
            ["VWRA", "Vanguard FTSE All-World", "全球 50+ 國家；3,700+ 股票",
             "美元計價；複委託購買；費用率 0.22%", "最廣泛的全球分散，適合退休核心持倉"],
            ["VT", "Vanguard Total World Stock", "全球 9,000+ 股票（含小型股）",
             "美元計價；費用率 0.07%（最低）", "比 VWRA 更完整；適合追求最大分散"],
            ["00881 國泰全球品牌 50", "追蹤全球 50 大品牌企業", "台灣交易所掛牌",
             "台幣計價；無外匯兌換成本", "分散但集中大型品牌；非全市場"],
            ["ACWI", "iShares MSCI All Country World", "全球已開發+新興市場",
             "美元計價；知名度高", "與 VWRA 高度重疊；可擇一"],
        ], columns=["ETF", "追蹤指數", "涵蓋範圍", "特性", "退休用途建議"])
        st.dataframe(etf_global_df, use_container_width=True, hide_index=True)

        st.markdown("### 全球分散與 Bond Tent 的協同效果")
        st.markdown("""
全球分散化與 Bond Tent 結合，能同時解決退休組合的兩大威脅：

| 威脅 | Bond Tent 的應對 | 全球分散的應對 | 協同效果 |
|---|---|---|---|
| **SORR（序列報酬風險）** | 退休初期低股票，降低熊市衝擊 | 分散市場，避免單一市場熊市全吃 | 雙重保護退休頭 10 年 |
| **通膨（購買力侵蝕）** | 後期回升股票，保持成長力 | 全球資產對沖單一貨幣貶值 | 30年購買力維護 |
| **長壽風險** | 老後高股票比例維持成長 | 全球成長動能（非僅台/美）| 50 年長週期存活率提升 |
| **地緣政治** | 無直接應對 | ✅ 直接分散至 50+ 國家 | 退休組合的最後防線 |

**建議配置方向（退休期間）**：
- 台股 ETF（0050）：20–30%（本土熟悉資產）
- 美股 ETF（VTI）：15–25%（全球最大市場）
- 全球 ETF（VWRA/VT）：20–30%（非美國際市場補充）
- 高股息 ETF（00878）：10–20%（現金流來源）
- 現金 / 短期債：10–20%（三桶金短期桶）
        """)

        st.success(
            "**核心洞見**：全球分散化不是為了賺更多報酬，"
            "而是在相同期望報酬下，顯著降低組合波動率（σ），"
            "這在蒙地卡羅模擬中直接轉化為更高的成功機率。"
            "方差拖累公式 `幾何均值 ≈ r̄ - σ²/2` 說明：σ 從 20% 降至 15%，"
            "相當於每年白白多賺約 **0.875%** 的實質報酬。"
        )
        st.caption(
            "文獻：① Markowitz (1952) 'Portfolio Selection' ② DMS Global Investment Returns Yearbook 2024 "
            "③ Vanguard VLCM 2025"
        )

    # ── 10. 50% 006208 + 50% VT 雙基金提領策略 ────────────────────────────
    elif guide_topic.startswith("10"):
        st.subheader("50% 006208 + 50% VT 雙基金退休提領策略")
        st.markdown("""
> **策略核心**：以富邦台灣50（006208）錨定台灣市場低成本敞口，
> 搭配 Vanguard 全世界股票 ETF（VT）取得全球9,000＋股票分散，
> 形成「台灣 + 全球」的兩基金極簡組合，透過年度再平衡提供退休提領所需的穩定性與成長力。
        """)

        st.markdown("### 兩檔 ETF 基本特性對照")
        etf_comp = pd.DataFrame([
            ["006208 富邦台灣50",
             "台灣50指數（FTSE TWSE Taiwan 50 Index）",
             "台股前50大市值公司；台積電約33–35%",
             "台幣（TWD）",
             "0.03%（全台最低之一）",
             "11–13%（估；含股息）",
             "~20–22%",
             "季配息（3/6/9/12月）"],
            ["VT Vanguard Total World",
             "FTSE Global All Cap Index",
             "全球47個市場、9,800＋檔股票（含小型股）",
             "美元（USD）",
             "0.07%",
             "7–9%（估；20年歷史）",
             "~14–16%",
             "季配息（3/6/9/12月）"],
        ], columns=["ETF", "追蹤指數", "涵蓋範圍", "計價幣別", "費用率", "估算名目年化報酬", "年波動率σ", "配息頻率"])
        st.dataframe(etf_comp, use_container_width=True, hide_index=True)
        st.caption("報酬率為歷史估算，未來不保證。本工具以「實質報酬／2026 購買力」為主要口徑；若引用名目年化，請先扣通膨換算成實質再帶入模型。")

        st.markdown("### 為什麼選擇這個組合？")
        col_a, col_b = st.columns(2)
        with col_a:
            st.success("""
**選擇 006208（非 0050）的理由**
- 費用率 **0.03%** vs 0050 的 **0.43%**，30年複利效果差距顯著
- 追蹤完全相同的台灣50指數，報酬幾乎無差異
- 台積電敞口相同，但持有成本大幅降低
- 配息頻率相同（季配息），現金流規劃一致
            """)
        with col_b:
            st.info("""
**選擇 VT（非 VWRA）的理由**
- 費用率 **0.07%** vs VWRA 的 **0.22%**，長期累積差異可觀
- 涵蓋小型股（9,800＋股票），分散更完整
- 美股最大市場（NYSE）流動性最佳
- 等比例持有已開發＋新興市場，不需另買 VXUS
            """)

        st.markdown("### 50/50 組合的風險報酬特性（估算）")
        st.markdown("""
依據現代投資組合理論，若006208與VT的相關係數 ρ ≈ 0.55（台股與全球股市歷史相關性），
組合的年波動率 σ_portfolio 可估算為：

$$\\sigma_p = \\sqrt{0.5^2 \\cdot \\sigma_{006208}^2 + 0.5^2 \\cdot \\sigma_{VT}^2 + 2 \\cdot 0.5 \\cdot 0.5 \\cdot \\rho \\cdot \\sigma_{006208} \\cdot \\sigma_{VT}}$$

代入 σ₁ ≈ 21%、σ₂ ≈ 15%、ρ ≈ 0.55：**σ_p ≈ 16.5%**（低於單持006208的21%）
        """)

        portfolio_stats = pd.DataFrame([
            ["100% 006208", "11–13%", "9–11%", "~21%", "台股集中、半導體週期"],
            ["100% VT", "7–9%", "5–7%", "~15%", "估值偏高時全球同步下跌"],
            ["50% 006208 + 50% VT", "9–11%（加權估）", "7–9%", "~16.5%", "分散降低，但仍受全球股市牽連"],
            ["傳統 60股/40債", "6–8%", "4–6%", "~11%", "通膨侵蝕債券實質報酬"],
        ], columns=["組合", "估算名目年化", "估算實質年化", "年波動率σ", "主要殘餘風險"])
        st.dataframe(portfolio_stats, use_container_width=True, hide_index=True)
        st.caption("此組合為100%股票，適合有長期視野且能承受市場波動的退休人士。")

        st.markdown("### 退休提領操作細節")
        st.markdown("""
#### 核心提領原則

本策略採用「**護城河現金池（Cash Moat）＋年度再平衡提領**」的混合機制：

| 步驟 | 操作 | 說明 |
|---|---|---|
| **步驟 1** | 維持 **1–2 年生活費現金池** | 年初從ETF賣出補滿；股災時從現金池支應，避免低點賣股 |
| **步驟 2** | 年底評估兩者偏離程度 | 若任一ETF偏離目標50%超過±5%，執行再平衡 |
| **步驟 3** | 賣出漲多的ETF補充現金池 | 同步完成提領 + 再平衡，減少交易次數 |
| **步驟 4** | 配息直接收取不再投入 | 006208＋VT季配息可作為現金池自然補充來源 |
        """)

        st.markdown("#### 年化提領率建議")
        wr_df = pd.DataFrame([
            ["保守型", "3.0–3.5%", "現金池2年＋股市下跌時不提額外款項", "適合預期壽命35年以上或市場CAPE偏高時"],
            ["穩健型", "3.5–4.0%", "現金池1.5年；每3年重新評估提領率", "適合預期壽命25–35年；主流退休者"],
            ["積極型", "4.0–4.5%", "現金池1年；搭配護欄規則（Guardrail）", "預期壽命20–25年；接受彈性削減提領"],
        ], columns=["類型", "年提領率", "對應操作", "適用情境"])
        st.dataframe(wr_df, use_container_width=True, hide_index=True)
        st.caption("全股票組合（無債券緩衝）在退休初期遭遇熊市時，SORR較高，建議提領率採偏保守值（3.5%以下）。")

        st.markdown("### 再平衡規則（門檻觸發）")
        st.markdown("""
本策略採用「**5% 門檻觸發再平衡**」而非固定日曆再平衡，以避免過度交易稅負：

```
每季末或提領前 → 檢查持倉比例

006208 佔比 > 55%？ → 賣出 006208，買入 VT（或補充現金池）
VT    佔比 > 55%？ → 賣出 VT，買入 006208（或補充現金池）
兩者均在 45–55%？ → 不動，節省交易成本與稅負
```

**交易順序建議**（台灣稅務最優化）：
1. 優先用「**配息收益**」補現金池（免稅效果最佳）
2. 其次賣出**虧損部位**（可申報損失，台灣目前股利所得計入綜所稅）
3. 最後才賣出獲利部位（注意二代健保補充保費2.11%在單次獲利超過2萬時觸發）
        """)

        st.markdown("### 熊市/股災應對（全股票組合特別注意）")
        st.warning("""
⚠️ **全股票組合在熊市的脆弱性**：006208＋VT 均為股票型 ETF，
股災時兩者可能同步下跌30–50%，無債券作為緩衝。
必須做好以下準備，才能堅守不在低點賣出。
        """)
        bear_df = pd.DataFrame([
            ["現金池保護", "維持1–2年生活費現金", "熊市期間完全不碰ETF，全靠現金池支應生活", "最關鍵的防線"],
            ["心理預演", "事先接受帳面虧損30–50%", "每次入帳前做最壞情境演練，避免恐慌賣出", "行為財務學核心"],
            ["停止定期賣出", "熊市期間暫停賣ETF補現金池", "若現金池尚足1年，股災期間不補充，靜待回升", "SORR防護"],
            ["VT配息不中斷", "VT股災期間可能小幅減配息", "仍有部分現金流入；006208台股配息亦有支撐", "現金流輔助"],
            ["不增加非計畫性支出", "股災期間嚴控非必要開銷", "避免低點被迫賣股應急", "支出紀律"],
        ], columns=["防護機制", "操作", "邏輯", "重要性"])
        st.dataframe(bear_df, use_container_width=True, hide_index=True)

        st.markdown("### 稅務與成本分析（台灣投資人）")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("""
**006208（台灣ETF）**
- 股票交易稅：0.1%（ETF優惠，非0.3%）
- 資本利得稅：目前**免徵**（台灣）
- 配息所得：計入綜合所得稅
- 二代健保補充保費：單次配息 > 2萬元觸發（2.11%）
- 費用率：**0.03%**（極低）
            """)
        with col_t2:
            st.markdown("""
**VT（美國ETF，海外所得）**
- 美國股息預扣稅：**30%**（台灣未簽租稅協定）
- 海外所得：超過 **100萬台幣** 需申報最低稅負制（20%）
- 資本利得（台灣）：目前免徵
- 費用率：**0.07%**
- 建議透過複委託或海外帳戶購買
            """)
        st.info("""
💡 **稅務最優化建議**：
若海外所得（VT配息＋資本利得）接近100萬門檻，可考慮將VT部位換成台灣掛牌的 **00856（元大MSCI世界）** 或 **00923（群益台美龍頭）** 等，
雖費用率略高，但可規避美國預扣稅與海外最低稅負申報複雜度。
        """)

        st.markdown("### 策略優缺點總結")
        pros_cons = pd.DataFrame([
            ["✅ 優點", "極低費用率（兩者均在0.10%以內），長期省下大量成本"],
            ["✅ 優點", "操作極簡：只有兩檔ETF，每季檢視一次即可"],
            ["✅ 優點", "台灣＋全球雙向分散，降低單一市場集中度"],
            ["✅ 優點", "006208季配息＋VT季配息，每季都有自然現金流"],
            ["✅ 優點", "全股票長期實質報酬高（7–9%），對抗通膨能力強"],
            ["⚠️ 缺點", "無債券緩衝，熊市帳面最大回撤可達-40%至-55%"],
            ["⚠️ 缺點", "VT美元計價，面臨台幣升值時帳面縮水風險"],
            ["⚠️ 缺點", "VT美國預扣稅30%，股息效率低於台灣ETF"],
            ["⚠️ 缺點", "台積電單一股票佔006208達33%，半導體週期集中風險仍存在"],
            ["⚠️ 缺點", "需要2年現金池作為額外資金準備，初期流動性門檻較高"],
        ], columns=["項目", "說明"])
        st.dataframe(pros_cons, use_container_width=True, hide_index=True)

        st.markdown("### 適合對象")
        st.markdown("""
此策略最適合以下特徵的退休人士：

| 特徵 | 說明 |
|---|---|
| **風險承受度高** | 能接受帳面淨值在熊市縮水30–50%，不恐慌賣出 |
| **退休初期資產充裕** | 總資產足夠維持3–5%提領率，有2年現金池餘裕 |
| **投資知識中等以上** | 了解再平衡邏輯，不會因波動改變策略 |
| **費用敏感型投資人** | 厭惡高費率產品，追求總成本最小化 |
| **台灣本土生活為主** | 生活費以台幣計算，006208提供台幣現金流 |
| **預期壽命20年以上** | 全股票組合的優勢在長期，需要時間等待複利發揮 |
        """)

        st.success("""
**核心洞見**：006208＋VT 的雙基金組合，在「費用最小化」與「全球分散化」兩個維度同時達到接近最優解。
相對於持有0050的台灣退休者，此組合將年化費用從0.43%降至約0.05%（加權平均），
30年複利下等同於「免費多享有」約1–1.5%的年化超額報酬，
同時透過VT的全球敞口，在台股半導體週期下行時提供有效緩衝。
        """)

        st.caption(
            "文獻：① Vanguard 'Total World Stock ETF (VT) Factsheet' 2025 "
            "② 富邦投信「006208富邦台灣50」公開說明書 2025 "
            "③ Markowitz (1952) Portfolio Selection "
            "④ Bengen (1994) 4% Rule & Pfau (2021) Safety-First Retirement "
            "⑤ 台灣財政部「海外所得最低稅負制說明」2025 "
            "⑥ Kitces 'Two-Fund Portfolio for Retirees' 2024"
        )

    # ── 11. 海外市值型 ETF 實務（海外券商 vs 複委託）──────────────────────
    elif guide_topic.startswith("11"):
        st.subheader("海外市值型 ETF：海外券商（FirstTrade 等）vs 複委託（台灣券商）")
        st.caption("定位：以市值型 ETF 做全球分散核心，並在退休期間可穩定提領。以下為操作框架與風險提醒，非投資/稅務/法律建議。")

        st.info(
            "本章以 **市值型 ETF** 為主（例：VTI / VXUS / VT / VWRA / ACWI）。"
            "若你是『靠配息過生活』的路線，仍建議回到總報酬觀念：**配息是現金流形式，不是額外報酬**。"
        )

        st.markdown("### 先選路徑：海外券商 vs 複委託")
        route_df = pd.DataFrame(
            [
                [
                    "海外券商（例：FirstTrade / IBKR 等）",
                    "商品選擇多、可能更容易做到低成本的全球分散；長期成本結構通常更透明",
                    "開戶/KYC、匯款與換匯、稅務文件與出金流程較多；介面/客服多為英文",
                    "想把長期成本壓到最低、願意自己處理流程者",
                ],
                [
                    "複委託（台灣券商）",
                    "介面與金流更貼近台灣投資人；中文操作、常見文件較好取得",
                    "手續費/匯差可能較高；可交易標的、下單規則依券商差異較大",
                    "想省事、希望流程穩定、用成本換取低摩擦者",
                ],
            ],
            columns=["路徑", "優點（你得到什麼）", "代價/摩擦（你要付什麼）", "更適合誰"],
        )
        st.dataframe(route_df, use_container_width=True, hide_index=True)
        st.caption("選擇原則：**成本 vs 摩擦**。退休時要把『出金穩定』也算進決策。")

        st.markdown("### 退休提領視角：你要的是「可控的現金流」")
        st.markdown("""
- 海外市值型 ETF 多為季配或配息不固定；拿它來「精準對齊每月生活費」通常不理想。
- 因此常見做法是：**現金桶/短債桶** 負責 6–24 個月支出，海外 ETF 負責長期成長。
- 需要現金時用「總報酬法」：**賣出一小部分份額**，而非等待配息。
        """)

        st.markdown("### 稅務/申報：用『年度清單』管理（概念）")
        st.warning("下表是你需要意識到的面向；細節以當年度法規、券商文件與專業人士為準。")
        tax_df = pd.DataFrame(
            [
                ["股息預扣稅", "持有美國註冊 ETF（如 VT/VTI）常見有股息預扣稅；會影響股息效率。"],
                ["海外所得/最低稅負（AMT）", "台灣投資人要留意海外所得與最低稅負制；複委託也不代表自動免除申報義務。"],
                ["文件與對帳", "海外券商通常需要自己整理年度對帳/股息紀錄；複委託多半較容易取得中文對帳。"],
                ["身故承接", "海外帳戶要思考身故後承接流程與文件；不同法域可能有不同稅務規則。"],
            ],
            columns=["面向", "你要知道的事（概念）"],
        )
        st.dataframe(tax_df, use_container_width=True, hide_index=True)

        st.markdown("### 成本拆解：不要只看手續費")
        cost_df = pd.DataFrame(
            [
                ["一次性成本", "匯款費、中轉費、開戶時間成本"],
                ["每次交易成本", "手續費、最低手續費、交易所費用"],
                ["隱性成本", "換匯匯差、點差、追蹤誤差、再平衡頻率造成的摩擦"],
                ["持有成本", "ETF 費用率（Expense Ratio）"],
            ],
            columns=["成本類型", "常見項目"],
        )
        st.dataframe(cost_df, use_container_width=True, hide_index=True)
        st.caption("你在本 App 看到的「費用率/摩擦/配息稅負」就是把這些因素轉成『每年拖累%』的概念化輸入。")

        st.markdown("### 市值型 ETF 組合（示例）：一到兩檔買到位")
        etf_list = pd.DataFrame(
            [
                ["VT", "全球（美+非美，含小型股）", "一檔到底、再平衡最省事", "極簡核心"],
                ["VTI + VXUS", "美國 + 非美國", "可自行控制美國比重；需要明確再平衡規則", "可控性高"],
                ["VWRA / ACWI", "全球（已開發+新興）", "指數與註冊地不同；可作 VT 的替代閱讀地圖", "替代方案"],
            ],
            columns=["代表 ETF（示例）", "涵蓋", "特性提醒", "定位"],
        )
        st.dataframe(etf_list, use_container_width=True, hide_index=True)
        st.caption("本表為閱讀地圖，非推薦名單；請以各 ETF 公開說明書與券商可交易性為準。")

        st.markdown("### 一套簡單、可長期執行的退休操作流程")
        st.markdown("""
**Step 1｜建立現金桶**：至少 6–24 個月生活費（依風險承受度調整）  
**Step 2｜固定頻率再平衡**：每年或每半年一次，規則勝過臨場判斷  
**Step 3｜提領時機**：需要補現金桶時，優先賣「漲多」的資產，順便再平衡  
**Step 4｜年度文件整理**：年底把股息、交易損益、海外所得相關資料整理成一包（減少隔年痛苦）
        """)

    # ── 延伸閱讀（跨分頁導覽）：集中在頁尾，維持版面整潔 ────────────────
    _gid = str(guide_topic).split("｜", 1)[0]
    _guide_related = {
        "1": [
            ("📚 教育資訊庫", "核心概念（先看這裡）", "1｜退休三階段（消費微笑曲線）"),
            ("📚 教育資訊庫", "核心概念（先看這裡）", "2｜安全提領率與 GK 護欄策略"),
        ],
        "2": [
            ("📚 教育資訊庫", "核心概念（先看這裡）", "2｜安全提領率與 GK 護欄策略"),
            ("📚 教育資訊庫", "台灣制度／稅務", "3｜114 年度台灣綜合所得稅（2026 報稅適用）"),
        ],
        "5": [
            ("📚 教育資訊庫", "台灣制度／稅務", "3｜114 年度台灣綜合所得稅（2026 報稅適用）"),
        ],
        "6": [
            ("📚 教育資訊庫", "醫療／長照／風險", "6｜長照費用與風險"),
            ("🛡️ 保險規劃（參考）", "（此分頁）", "2｜退休常見保障面向（對照生活事件）"),
        ],
        "7": [
            ("📚 教育資訊庫", "資產配置／市場觀察", "7｜資產配置與 ETF 建議"),
        ],
        "10": [
            ("📚 教育資訊庫", "資產配置／市場觀察", "7｜資產配置與 ETF 建議"),
        ],
        "11": [
            ("📚 教育資訊庫", "資產配置／市場觀察", "7｜資產配置與 ETF 建議"),
            ("📚 教育資訊庫", "核心概念（先看這裡）", "2｜安全提領率與 GK 護欄策略"),
        ],
    }
    _g_rels = _guide_related.get(_gid, [])
    if _g_rels:
        st.divider()
        with st.expander("延伸閱讀（推薦下一步）", expanded=False):
            st.caption("可點擊超連結直接預選目標頁的分類/主題（不影響引擎計算）。")
            for page, cat, topic in _g_rels[:3]:
                if page.startswith("📚"):
                    href = _qs(nav="edu", edu_category=cat, edu_topic=topic)
                elif page.startswith("🛡️"):
                    href = _qs(nav="ins", ins_topic=topic)
                else:
                    href = _qs(nav="guide", guide_category=cat, guide_topic=topic)
                st.markdown(f"- [{page} → 分類 **{cat}** → 主題 **{topic}**]({href})")

    st.divider()
    st.caption(
        "資料來源：① Morningstar State of Retirement Income 2025 "
        "② Christine Benz, Morningstar Bucket Portfolio 2025 "
        "③ Kitces.com Rebalancing Strategies (2015) "
        "④ T. Rowe Price Dynamic Withdrawal (2025)"
    )

# ──────────────────────────────────────────────
# PAGE：保險規劃（僅展示參考資訊，不連動退休引擎）
# ──────────────────────────────────────────────
elif page_id == "ins":
    st.title("🛡️ 保險規劃資訊（參考）")
    st.caption(
        "整理自公開媒體專訪、學者觀點與產業報導；供你對照「退休金模擬」與「風險轉嫁」分工，**不取代保單條款或個人化規劃**。"
    )
    st.warning(
        "**本頁不連動「📊 退休規劃」引擎**：側欄參數、提領率、資產模擬皆未納入保費／理賠／年金給付。"
        "若日後要接軌模擬，需另行定義現金流與情境假設。"
    )
    st.divider()

    ins_topic = st.radio(
        "選擇主題",
        [
            "1｜為什麼要分開：投資報酬 vs 保險",
            "2｜退休常見保障面向（對照生活事件）",
            "3｜公開論述摘要（專家／媒體，第二手整理）",
            "4｜日後若接軌引擎：可思考的參數（仍不實作）",
        ],
        horizontal=True,
        key="ins_topic",
    )
    st.divider()
    if _nav == "ins":
        st.success("已從「延伸閱讀」導覽過來：本頁已自動預選主題。", icon="✅")

    if ins_topic.startswith("1｜"):
        st.subheader("投資報酬與保險：角色不同、不要混在同一個旋鈕")
        st.markdown("""
多數退休財務論述會把問題拆成兩類：

| 類型 | 典型工具 | 在規劃裡回答的問題 |
|------|----------|-------------------|
| **累積與提領（流量／存量）** | ETF、股債配置、勞保勞退、現金桶 | 退休後每年要從資產拿多少錢？能否對抗通膨與長壽？ |
| **不確定的大額支出** | 醫療險、長照險、部分一次金／年金商品 | 若發生重病、長照、失能，誰來付「可能一次燒掉幾百萬」的帳？ |

**重點**：保險在這裡主要是 **風險轉嫁／財務緩衝**，不是用來「拉高投資報酬率 r」的替代品。
把保單當成和股票同一個「報酬率」去估，容易與現實脫鉤（保費、給付結構、解約、通膨連動等都很難用單一數字代表）。
        """)
        st.info(
            "**與本 App 的關係**：「📊 退休規劃」分頁模擬的是投資報酬、通膨、提領策略與（可選）勞退／房租等現金流；"
            "保險細節請在本分頁自行對照，或諮詢具資格之顧問。"
        )

    elif ins_topic.startswith("2｜"):
        st.subheader("常見保障面向與退休事件的對照（概念表）")
        evt = pd.DataFrame(
            [
                ["住院／手術／自費醫材", "實支實付醫療險、手術／住院日額（若仍有效）", "避免「為了付醫藥費被迫在低點賣股」"],
                ["癌症／重大傷病", "癌症險、重大傷病險（一次金或分期）", "治療期長、自費項目多；媒體常舉百萬級距為討論起點"],
                ["失能／長期照顧", "長照險、失能險（依商品定義）", "照護月費×年數可非常可觀；與「生活費」分開思考較清楚"],
                ["活太久、怕錢不夠花", "即期年金、生存還本、部分年金險設計", "補「可預測現金流」；須注意給付多為名目金額、未必跟通膨連動"],
                ["身故／家庭責任尾端", "定期壽險、終壽（視家庭狀況）", "退休後需求通常下降，但若有債務／配偶依賴仍要檢視"],
            ],
            columns=["生活事件（舉例）", "常見對應險種（概稱）", "規劃上想防的是什麼"],
        )
        st.dataframe(evt, use_container_width=True, hide_index=True)
        st.caption(
            "險種名稱與理賠定義以各公司條款、主管機關公告為準；上表僅為閱讀地圖，非商品推薦。"
        )

    elif ins_topic.startswith("3｜"):
        st.subheader("公開論述摘要（第二手整理，非投保建議）")
        st.markdown("""
以下為媒體／專欄中常見的「方向性」說法，**不代表單一正解**，僅供你與顧問討論時當 checklist：

- **「三桶金」式敘事**：除了退休金本金與投資，另把 **醫療費、長照費** 當獨立關注點；並強調用保險轉嫁部分不確定支出（例如今周刊等對理財顧問／CFP 的專訪整理）。
- **醫療 vs 長照先後**：有受訪者明確寫出 **醫療險優先、長照險次之** 的順序（仍須看預算與體況）；長照部分亦常討論理賠定義（生理／認知功能評估）。
- **學者角度（風險管理）**：退休準備常見四大壓力——**壽命延長、醫療健康、通膨、投資**；並提醒 **保單健檢**、不健康期間醫療負擔可能極重，應避免「全用退休金硬扛」。
- **實務建議（電視／網路節目常見說法）**：除「退休金帳戶」外，另闢 **健康帳戶**；實支實付可思考 **分散兩家保險公司** 等操作層提醒；另會討論 **重大傷病** 與癌症治療費用的量級意識。
- **精算／雜誌觀點**：區分 **必然發生的生活費** 與 **不確定的大額風險**；指出國人 **長照險覆蓋率偏低** 的結構問題；並提醒 **年金／固定給付未必有通膨連動**。
- **對「儲蓄險當退休主體」的警語**：部分理財專欄強調 **報酬、流動性、通膨** 與 **解約損失**，認為不宜只靠儲蓄險取代長期資產配置（與「保障」目的分開看較清楚）。

**延伸閱讀（自行檢索原文）**：今周刊、風傳媒（證基會系列）、健康2.0、經濟日報保險線、《現代保險》雜誌等對退休與保險之專題。
        """)
        st.caption("引用為閱讀整理；數字、商品與費率以最新法規與契約為準。")

    else:
        st.subheader("日後若要把「保險」接進模擬引擎：可思考的參數（本版未實作）")
        st.markdown("""
若未來要在試算裡量化保險，通常會拆成 **現金流** 與 **情境**，而不是把「多買一張險」直接加成 **實質報酬 r**：

| 參數類型 | 例子 | 說明 |
|----------|------|------|
| **固定流出** | 年繳／月繳保費合計 | 會減少可投資金與生活費緩衝，與「提領需求」一併列帳較合理 |
| **固定流入** | 生存金、年金給付、還本 | 類似「另一條勞退／租金」的穩定入帳（仍要注意名目 vs 實質購買力） |
| **情境式流入** | 一次金理賠、長照分期金 | 常以「若某年齡後發生事件則注入現金」做壓力測試；機率模型會明顯變複雜 |
| **自費上限／共保** | 醫療自費假設上限 | 簡化時可用「每年醫療自費不超過 X 萬」降低對資產提領的衝擊 |

**本 App 現況**：以上皆 **未** 寫入「📊 退休規劃」計算邏輯；本頁僅供你記錄思路與和顧問溝通用。
        """)
        st.success(
            "若你希望下一階段「只加側欄輸入、仍不接條款細節」或「只做保費年流出的簡化版」，可在對話裡指定優先順序與假設範圍。"
        )

    st.divider()
    st.caption(
        "免責：本頁為一般性教育整理，非個人化投保、投資或節稅建議；商品適合度與理賠以保險公司與主管機關資訊為準。"
    )

# ──────────────────────────────────────────────
# PAGE：更新紀錄
# ──────────────────────────────────────────────
elif page_id == "changelog":
    st.title("🗒️ 更新紀錄")
    st.caption("本頁以「可回溯、可驗證」為原則，記錄近期針對 UX、顯示口徑與風險剖析所做的改動。")

    st.subheader("摘要（給快速瀏覽）")
    summary_df = pd.DataFrame(
        [
            ["2026-04", "口徑/顯示", "實質/名目口徑全站統一", "所有金額主要以 2026 購買力理解；同時看到名目換算", "📊 退休規劃（頁首口徑宣告、核心結果、壓力測試）"],
            ["2026-04", "核心指標", "核心結果置頂 + 框選", "一進頁面先看到最關鍵 3 指標", "📊 退休規劃（核心結果容器）"],
            ["2026-04", "風險剖析", "失敗路徑剖析（直方圖/最差曲線/CSV）", "知道「失敗發生在幾歲、最差路徑長怎樣」並可下載分析", "📊 退休規劃 → 蒙地卡羅驗證區"],
            ["2026-04", "壓力測試", "表格顯示改為名目 + 括號實質", "避免把 90 歲資產誤解為 2026 金額", "📊 退休規劃（壓力測試）"],
            ["2026-04", "不動產 UX", "租金淨收入/流動性折扣/以房養老文案校正", "更容易正確填參數，不改任何演算法", "側欄不動產（Wizard/Advanced）"],
        ],
        columns=["月份", "類別", "變更", "使用者可見影響", "位置"],
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.divider()

    st.subheader("詳細紀錄（每項可展開，含驗證方式）")

    with st.expander("A｜口徑/顯示一致性（實質 vs 名目）", expanded=False):
        st.markdown("""
**目的**  
避免使用者把「未來名目金額」誤認為「2026 購買力」，或把「名目報酬」誤當「實質報酬」導致規劃偏差。

**方法論（固定口徑）**  
- 核心引擎以 **實質購買力（2026）**計算。  
- 名目金額僅做顯示：以使用者設定的 CPI 複利把實質換算成名目作為直覺對照。  

**變更內容**  
- 在 `📊 退休規劃` 頁首新增「口徑統一（全站一致）」提示。  
- 將散落在教育/表格中的口徑說明統一為同一套用語（實質/名目/成功率定義）。  

**如何驗證（你可以自己檢查）**  
- 將通膨設定為 0%：名目與實質應趨於相同（折算差異消失）。  
- 將通膨設定為 3%：90 歲名目資產應明顯大於折算回 2026 的實質購買力。  
        """)

    with st.expander("B｜核心結果置頂（3 大指標）", expanded=False):
        st.markdown("""
**新增指標（置頂框選）**  
- 退休成功率（標準蒙地卡羅）  
- 目標年齡剩餘資產（主顯示：名目；輔助：折算回 2026）  
- 固定提領：剛好歸零的臨界 IWR（確定性上限）  

**使用者價值**  
讓使用者不必先閱讀長篇教育內容，就能先抓到「成不成功、剩多少、提領是不是逼近極限」。

**如何驗證**  
- 提高 `W₀`：成功率下降、臨界 IWR 逼近/低於現況 IWR。  
- 提高 `A₀`：成功率上升、終值上升、臨界 IWR 上升。  
        """)

    with st.expander("C｜壓力測試顯示口徑對齊", expanded=False):
        st.markdown("""
**變更**  
- 壓力測試矩陣與多重風險表格：統一顯示為  
  - **名目剩餘資產（括號＝折算回2026：實質購買力）**

**使用者價值**  
避免「90 歲 2 億卻顯示仍是 2 億（2026）」這類直覺衝突。

**如何驗證**  
- 通膨提高時（例如 3%）：同一格的名目數字上升，括號內實質不會同幅上升。  
        """)

    with st.expander("D｜失敗路徑剖析（蒙地卡羅）+ CSV", expanded=False):
        st.markdown("""
**新增**  
- 歸零年齡分布（直方圖）  
- 最差 N 條資產曲線（以實質購買力）  
- CSV 下載（每條路徑一列，含 success/ruin_age/final_real/final_nominal/min_real/early_return）  

**使用者價值**  
把「不是 100% 成功」具體化：失敗通常發生在哪些年齡區間、最壞路徑的資產跌落型態，以及可外部分析的資料出口。

**如何驗證**  
- 把提領拉高/報酬拉低：失敗路徑數上升，直方圖往更早年齡移動。  
- 下載 CSV：應能在 Excel/PowerBI 直接開啟（UTF-8-SIG）。  
        """)

    with st.expander("E｜不動產欄位 UX（不改引擎）", expanded=False):
        st.markdown("""
**變更**  
- 租金欄位明確定義「淨租金現金流」，提供保守估算提示（例如毛×0.75）。  
- 流動性折扣說明校正（變現摩擦，不是房價下跌假設）。  
- 以房養老定位（晚年流動性保險/長照備援）更清楚。  

**使用者價值**  
降低填錯參數的機率，提升結果可信度與 UX；不觸碰計算引擎行為。
        """)

    st.divider()
    st.caption("如需把本頁輸出成 Markdown/CSV 做版本留存，我可以再加『下載更新紀錄』按鈕。")
