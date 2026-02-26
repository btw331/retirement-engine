# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
退休規劃大師 — 50年長週期資產動態管理
Streamlit 版：精簡排版、學術說明、可調邊界條件
"""

import streamlit as st
import pandas as pd
import numpy as np
import time

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

# ── 蒙地卡羅核心（NumPy 全向量化，無 Python 內層迴圈）────────────────
@st.cache_data
def _run_monte_carlo(A0, W0, r_mean_pct, r_std_pct,
                     n_years, strategy, pension_annual,
                     claim_age, age_start,
                     med_premium_pct=0.0,
                     dist_mode="normal",
                     t_df=7,
                     rental_annual=0.0,
                     rental_start_age=65,
                     n_sim=10_000):
    """
    隨機報酬率模擬 n_sim 條路徑（NumPy 全向量化）。
    dist_mode="normal" → 常態分布 N(r̄, σ²)
    dist_mode="t"      → Student-t(df)，縮放後保留相同 r̄ 與 σ，但有肥尾
    包含：固定提領 / 消費微笑曲線 / GK 護欄、醫療溢價、勞保補貼、租金收入。
    固定隨機種子（seed=42）確保相同參數可重現。
    """
    rng      = np.random.default_rng(seed=42)
    r_mean   = r_mean_pct / 100.0
    r_std    = r_std_pct  / 100.0
    med_rate = med_premium_pct / 100.0

    # ── 生成報酬率矩陣：shape (n_sim, n_years) ──────────────────────────
    if dist_mode == "t":
        # Student-t 原始樣本的方差 = df/(df-2)，需縮放使 Var = σ²
        # 縮放因子 = sqrt(df/(df-2))，除以此值後 Var → σ²、均值維持 r_mean
        scale_factor = np.sqrt(t_df / (t_df - 2))
        raw_t = rng.standard_t(df=t_df, size=(n_sim, n_years))
        returns = r_mean + r_std * raw_t / scale_factor
    else:
        returns = rng.normal(r_mean, r_std, (n_sim, n_years))

    A         = np.full(n_sim, float(A0))       # 每條路徑的資產餘額
    alive     = np.ones(n_sim, dtype=bool)       # 尚未歸零的路徑遮罩
    IWR       = W0 / A0 if A0 > 0 else 0.0
    gk_spend  = np.full(n_sim, float(W0))        # GK 每路徑追蹤的支出目標

    for j in range(n_years):
        current_age = age_start + j

        # ── 被動收入：勞保/勞退 + 租金 ──────────────────────────────
        pension_income = (pension_annual
                          if pension_annual > 0 and current_age >= claim_age
                          else 0.0)
        rental_income  = (rental_annual
                          if rental_annual > 0 and current_age >= rental_start_age
                          else 0.0)
        passive_income = pension_income + rental_income

        # ── 策略決定本期總目標支出 ────────────────────────────────────
        if strategy == "fixed":
            spend = np.full(n_sim, float(W0))

        elif strategy == "smile":
            if 75 <= current_age <= 84:
                mult = 0.8
            elif current_age >= 85:
                mult = 1.1
            else:
                mult = 1.0
            spend = np.full(n_sim, float(W0) * mult)

        elif strategy == "gk":
            # GK 護欄：向量化版本，每條路徑獨立調整提領目標
            safe  = A > 0
            cur_wr = np.where(safe, gk_spend / np.where(safe, A, 1.0), 999.0)
            gk_spend = np.where(cur_wr < IWR * 0.8, gk_spend * 1.1, gk_spend)
            gk_spend = np.where(cur_wr > IWR * 1.2, gk_spend * 0.9, gk_spend)
            spend = gk_spend.copy()

        else:
            spend = np.full(n_sim, float(W0))

        # ── 醫療溢價指數複利（70 歲起，與主引擎邏輯一致）─────────────
        if current_age >= 70 and med_rate > 0:
            med_base  = W0 * 0.15
            med_extra = med_base * ((1 + med_rate) ** (current_age - 65) - 1)
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
):
    """
    動態提領模擬器 V2.1 — 四項科學修正：
      A. 消費微笑曲線（75-84 歲縮減 20%、85 歲後增加 10%）
      B. 醫療溢價指數複利（70 歲起 (1+rate)^(age-65) 疊加於總需求 15%）
      C. 期初提領保守原則：(A - spend) × (1+r)，先扣費再計息
      D. 不動產租金現金流：rental_start_age 起每年減少從有價證券提領的金額
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

        # 本年度固定收入：勞保/勞退 + 租金（分別依起領年齡判斷）
        pension_income = (pension_annual
                          if pension_annual > 0 and current_age >= int(claim_age)
                          else 0.0)
        rental_income  = (rental_annual
                          if rental_annual > 0 and current_age >= int(rental_start_age)
                          else 0.0)
        passive_income = pension_income + rental_income   # 合計被動收入

        # ── 1. 策略決定「總支出目標」 ──────────────────────────────
        if strategy == "fixed":
            total_spend = W_total

        elif strategy == "smile":
            total_spend = W_total
            if 75 <= current_age <= 84:
                total_spend *= 0.8
            elif current_age >= 85:
                total_spend *= 1.1

        elif strategy == "gk":
            current_wr = gk_spend / A if A > 0 else 999.0
            if current_wr < IWR * 0.8:
                gk_spend *= 1.1
            elif current_wr > IWR * 1.2:
                gk_spend *= 0.9
            total_spend = gk_spend

        else:
            total_spend = W_total

        # ── 2. 非線性醫療溢價（70 歲起指數遞增）─────────────────────
        if current_age >= 70 and med_rate > 0:
            med_base  = W_total * 0.15
            med_extra = med_base * ((1 + med_rate) ** (current_age - 65) - 1)
            total_spend += med_extra

        # ── 3. 從有價證券提領 = 總支出 − 全部被動收入 ───────────────
        spend_from_asset = max(0.0, total_spend - passive_income)

        # ── 4. 期初提領保守原則：先扣費再計息 ─────────────────────────
        A = (A - spend_from_asset) * (1 + r)

    return max(0, A)

st.set_page_config(
    page_title="退休規劃大師",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== 邊界條件：側邊欄 ==========
st.sidebar.header("🔧 邊界條件設定")
st.sidebar.caption("調整參數後，下方摘要與指標會即時更新。")

with st.sidebar.expander("資產與提領", expanded=True):
    A0_securities_wan = st.number_input(
        "有價證券總額 (萬)",
        min_value=0,
        max_value=50_000,
        value=3_000,
        step=100,
        help="股票、ETF、基金等流動性金融資產，以萬為單位",
    )
    W0_wan = st.number_input(
        "實質購買力 (萬/年)",
        min_value=10,
        max_value=500,
        value=120,
        step=5,
        help="起始年全年生活費目標（含所有支出），以萬/年為單位",
    )
    A0 = A0_securities_wan * 10_000   # 基礎值；不動產淨值另行加總
    W0 = W0_wan * 10_000

with st.sidebar.expander("報酬與通膨", expanded=True):
    inflation_pct = st.slider(
        "預期通膨率 CPI (%)",
        min_value=0.0,
        max_value=8.0,
        value=2.0,
        step=0.5,
        help="預期年通膨率，可參考主計總處與高齡家庭 CPI（台灣近年約 1.5%～2%）",
    )
    use_inferred_r = st.radio(
        "實質報酬率 r 來源",
        ["依資產結構推論", "手動設定"],
        index=0,
        help="推論值由下方的資產結構加權計算",
    )
    if use_inferred_r == "手動設定":
        r_pct = st.slider(
            "預期實質報酬率 r (%)",
            min_value=0.0,
            max_value=15.0,
            value=4.0,
            step=0.5,
            help="扣除通膨後的年化回報",
        )
    else:
        r_pct = None  # 待資產結構計算後填入
    medical_premium = st.slider(
        "醫療溢價 i_m (CPI + %)",
        min_value=0.0,
        max_value=4.0,
        value=1.7,
        step=0.1,
        help="台灣高齡家庭 CPI 長期高於整體，醫療保健權重較高，建議 1.5%～2%；預設 1.7% 符合實證",
    )

with st.sidebar.expander("年齡區間", expanded=True):
    age_start = st.number_input("起始年齡 (歲)", 25, 70, 40, 1)
    age_end = st.number_input("目標年齡 (歲)", 70, 100, 90, 1)

# ── 不動產（選填）───────────────────────────────────────────────────────
with st.sidebar.expander("🏠 不動產（選填）", expanded=False):
    st.caption("填寫後可將房產淨值計入初始資產，並以租金收入減少每年從有價證券的提領。")
    include_re = st.toggle("將不動產納入計算", value=False,
                           help="開啟後，房產淨值加入 A₀；租金收入自動抵銷部分年度提領")

    re_home_wan    = st.number_input("自用住宅市值 (萬)",   min_value=0, max_value=20_000, value=0,   step=100,
                                     help="自住房屋目前市值，退休後居住成本已鎖定（不計入可提領現金流）")
    re_rental_wan  = st.number_input("出租房產市值 (萬)",   min_value=0, max_value=20_000, value=0,   step=100,
                                     help="出租物件的目前市值")
    re_mortgage_wan= st.number_input("未償房貸餘額 (萬)",   min_value=0, max_value=20_000, value=0,   step=50,
                                     help="所有房產尚未還清的貸款餘額，自動從淨資產中扣除")
    re_net_wan     = max(0, re_home_wan + re_rental_wan - re_mortgage_wan)
    st.caption(f"房產淨值（市值 − 貸款）：**{re_net_wan:,} 萬**")

    st.markdown("**租金收入**")
    rental_monthly_wan = st.number_input(
        "月租金淨收入 (萬/月)",
        min_value=0.0, max_value=100.0, value=0.0, step=0.5,
        help="已扣除房屋稅、管理費、維修費、空置率後的實際到手月租金（台灣各區淨報酬率約 1.5–3.5%）",
    )
    rental_start_age_input = st.number_input(
        "租金開始年齡 (歲)",
        min_value=40, max_value=85, value=int(65), step=1,
        help="若物件已出租可設等於起始年齡；尚未出租可設未來預計年齡",
    )

    st.markdown("**以房養老（選填）**")
    use_reverse_mortgage = st.toggle("啟用以房養老", value=False,
                                     help="達到啟動年齡時將房屋抵押給銀行，換取每月固定收入直至身故（台灣各行 2025 年利率約 2.16–4%）")
    if use_reverse_mortgage:
        rm_start_age   = st.number_input("以房養老啟動年齡 (歲)", min_value=60, max_value=90, value=80, step=1)
        rm_monthly_wan = st.number_input("以房養老月領 (萬/月)", min_value=0.0, max_value=50.0, value=3.0, step=0.5,
                                         help="依房產市值與銀行方案估算，一般約為房產價值 × 0.2–0.3% / 月")
    else:
        rm_start_age   = 999
        rm_monthly_wan = 0.0

# 計算不動產對 A₀ 與租金現金流的貢獻
re_net_value   = re_net_wan * 10_000
rental_annual  = rental_monthly_wan * 12 * 10_000   # NTD/年
rm_annual      = rm_monthly_wan     * 12 * 10_000   # NTD/年

# 合併租金 + 以房養老為「不動產被動收入」（兩者啟動年齡可各異）
# 引擎呼叫時以 rental_annual 代表「固定年齡後全部不動產收入」
# → 若需要精確分離兩個啟動年齡，可拆成兩個 passive_income 欄位
# 這裡採簡化：取較早啟動的那一組為代表，另一組視作加碼
rental_combined_annual = rental_annual + rm_annual
rental_combined_start  = min(rental_start_age_input,
                             rm_start_age if use_reverse_mortgage else 999)

# 若不納入計算，所有不動產數值歸零
if not include_re:
    re_net_value           = 0.0
    rental_combined_annual = 0.0
    rental_combined_start  = 999

A0_re  = re_net_value if include_re else 0.0
A0_eff = A0 + A0_re   # 有效初始資產（有價證券 + 房產淨值）

if include_re and re_net_wan > 0:
    st.sidebar.caption(
        f"不動產淨值已納入：**+{re_net_wan:,} 萬**　"
        f"| 有效 A₀ = **{(A0_eff/1e4):,.0f} 萬**"
    )

# ── 有價證券資產配置（Section 1）────────────────────────────────────────
with st.sidebar.expander("📈 1. 有價證券配置", expanded=True):
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
st.sidebar.caption(f"名目報酬 ≈ **{r_pct + inflation_pct:.1f}%**（實質 {r_pct}% + 通膨 {inflation_pct}%）")
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
has_rental     = (include_re and rental_combined_annual > 0)

# 被動收入合計（從起始年齡已在請領者）
_passive_at_start = 0.0
if has_pension and age_start >= int(claim_age):
    _passive_at_start += pension_annual
if has_rental and age_start >= rental_combined_start:
    _passive_at_start += rental_combined_annual

W0_asset = max(0.0, W0 - _passive_at_start)   # 從有價證券提領的淨額

# 衍生：IWR（以「從有價證券提領的淨額」為準）
IWR      = (W0_asset / A0_eff) * 100 if A0_eff > 0 else 0
gk_lower = IWR * 0.8
gk_upper = IWR * 1.2

# ========== 主區：分頁 ==========
tab1, tab2, tab3 = st.tabs(["📊 退休規劃", "📚 教育資訊庫", "🛠️ 提領實務指南"])

# ──────────────────────────────────────────────
# TAB 1：退休規劃（原有內容）
# ──────────────────────────────────────────────
with tab1:
    st.title("退休規劃大師")
    st.caption(f"50 年長週期退休財務工程與資產動態管理 (2026–2076) · 金額以 2026 實質購買力計價，預設通膨 {inflation_pct}%")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("初始提領率 IWR", f"{IWR:.2f}%", "安全邊界 4%" if IWR < 4 else "高於 4%")
    with c2:
        st.metric("繁榮護欄 (加薪觸發)", f"{gk_lower:.2f}%", "資產漲多時加薪 10%")
    with c3:
        st.metric("保全護欄 (減薪觸發)", f"{gk_upper:.2f}%", "資產跌多時減薪 10%")
    with c4:
        st.metric("實質報酬 r", f"{r_pct}%", f"通膨 {inflation_pct}% · 醫療+{medical_premium}%" + (" · 推論" if use_inferred_r == "依資產結構推論" else " · 手動"))

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
        ("實質報酬 r", f"{r_pct}%" + (" (推論)" if use_inferred_r == "依資產結構推論" else " (手動)")),
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
        "扣除通膨（2%）後實質約 7–10%，保守情境約為歷史值的一半，中性約 60–65%，積極約 80%。"
        "注意：算術加權平均略高於幾何平均（方差拖累約 0.5–1.1%），長期規劃建議偏向保守情境。"
    )

    # 不動產貢獻摘要（若已啟用）
    if include_re and (re_net_wan > 0 or rental_combined_annual > 0):
        re_cols = st.columns(3)
        with re_cols[0]:
            st.metric("房產淨值納入 A₀", f"+{re_net_wan:,} 萬",
                      f"有效 A₀ = {A0_eff/1e4:,.0f} 萬")
        with re_cols[1]:
            st.metric("不動產年收入", _fmt_wan(rental_combined_annual) + "/年",
                      f"{rental_combined_start} 歲起生效")
        with re_cols[2]:
            _cover_pct = rental_combined_annual / W0 * 100 if W0 > 0 else 0
            st.metric("生活費覆蓋率", f"{_cover_pct:.0f}%",
                      "不動產收入 ÷ 年生活費目標")
        st.caption("🏠 不動產收入已整合進引擎，每年從有價證券的淨提領額將自動降低。")
    st.divider()

    # ── 壓力測試 ──
    st.subheader("壓力測試 (90 歲時剩餘資產)")
    n_years = max(1, age_end - age_start)
    pension_note  = f" · 勞保＋勞退 {_fmt_wan(pension_annual)}/年（{claim_age} 歲起）" if has_pension else ""
    rental_note   = f" · 租金 {_fmt_wan(rental_combined_annual)}/年（{rental_combined_start} 歲起）" if has_rental else ""
    st.markdown(
        f"**有效起始資產 {_fmt_wan(A0_eff)}（有價證券 {_fmt_wan(A0)} + 房產淨值 {_fmt_wan(A0_re)}）"
        f" · 年生活費目標 {_fmt_wan(W0)}{pension_note}{rental_note}"
        f" · 實質報酬 {r_pct}% · 規劃 {n_years} 年**"
    )
    r_low  = max(0,  r_pct - 1.5)
    r_mid  = r_pct
    r_high = min(15, r_pct + 1.5)
    _kw = dict(pension_annual=pension_annual, claim_age=int(claim_age),
               rental_annual=rental_combined_annual, rental_start_age=int(rental_combined_start))
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
    def _fmt_pwr(x):
        """剩餘資產 × 4% → 年可提領額（2026 實質購買力）"""
        if x is None or x <= 0:
            return "歸零"
        return f"{x * 0.04 / 1e4:,.0f} 萬/年"

    def _fmt_cell(x):
        """合併顯示：剩餘資產 + 約當年購買力"""
        if x is None or x <= 0:
            return "歸零"
        return f"{_fmt_asset(x)}  (→ {_fmt_pwr(x)})"

    st.caption("格式說明：各格顯示「**90歲剩餘資產**（→ 約當2026年可提領額/年）」，括號內 = 剩餘資產 × 4%，代表彼時每年可持續支出的生活費（2026實質購買力）。")
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
                   rental_annual=rental_combined_annual, rental_start_age=int(rental_combined_start))
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
    df_risk = pd.DataFrame(scenarios, columns=["情境", "假設", "預估影響", "90歲剩餘資產（括號=約當2026年可提領/年）"])
    st.dataframe(df_risk, use_container_width=True, hide_index=True)
    st.caption("格式：剩餘資產 (→ 約當2026年可提領額)。GK 護欄 + 指數複利醫療溢價；建議搭配流動性緩衝因應最壞情境。")
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
        _sim_val = run_dynamic_projection(A0_eff, W0, r_pct, n_years, age_start,
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
            "注意：此驗算不含勞保/勞退補貼（pension_annual=0）以確保公式成立。"
            "消費微笑曲線與 GK 護欄為非線性動態模型，無封閉解，請用下方蒙地卡羅驗證。"
        )

    # 蒙地卡羅模擬
    with st.expander("🎲 驗證 2：蒙地卡羅模擬（成功機率 & SORR 量化）", expanded=False):
        st.markdown("""
**蒙地卡羅法**：將固定 r 改為「隨機報酬率」，模擬 **10,000 條**不同市場路徑，
統計「目標年齡時資產 > 0」的比例，並輸出 P10 / P50 / P90 三段分位數。

此法補充確定性模型無法量化的 **SORR（序列報酬風險）**，是業界最標準的退休規劃驗證方法。
        """)

        mc_row1_c1, mc_row1_c2, mc_row1_c3 = st.columns([2, 1, 1])
        with mc_row1_c1:
            mc_std = st.slider(
                "年報酬率標準差 σ (%)",
                min_value=5.0, max_value=30.0, value=15.0, step=1.0,
                help="股票型組合歷史波動率約 15–20%；保守組合可設 10–12%；0050/S&P500 約 18–20%",
            )
        with mc_row1_c2:
            mc_dist = st.radio(
                "報酬率分布",
                ["常態分布", "t 分布（肥尾）"],
                horizontal=True,
                help=(
                    "常態分布：標準假設，計算快速\n"
                    "t 分布：模擬金融市場肥尾（極端漲跌比常態更頻繁），"
                    "破產機率通常比常態高 2–5%"
                ),
            )
        with mc_row1_c3:
            mc_strategy = st.radio(
                "模擬策略",
                ["固定提領", "消費微笑曲線", "GK 護欄"],
                horizontal=True,
                help="與主引擎相同的三種策略，均包含醫療溢價及勞保補貼邏輯",
            )

        mc_use_t = mc_dist == "t 分布（肥尾）"
        mc_t_df = 7   # 預設自由度
        if mc_use_t:
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
                f"縮放因子 √(ν/(ν−2)) = √({mc_t_df}/{mc_t_df-2}) = **{_scale:.4f}**　"
                "（除以此值確保模擬的 σ 與設定值一致，均值維持 r̄）"
            )

        _strat_map = {"固定提領": "fixed", "消費微笑曲線": "smile", "GK 護欄": "gk"}
        mc_strat   = _strat_map[mc_strategy]
        mc_dist_mode = "t" if mc_use_t else "normal"

        _t0 = time.perf_counter()
        mc_sr, mc_p10, mc_p50, mc_p90 = _run_monte_carlo(
            A0_eff, W0, r_pct, mc_std,
            n_years, mc_strat,
            pension_annual, int(claim_age), age_start,
            med_premium_pct=medical_premium,
            dist_mode=mc_dist_mode,
            t_df=mc_t_df,
            rental_annual=rental_combined_annual,
            rental_start_age=int(rental_combined_start),
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

        _dist_label = f"t 分布（ν={mc_t_df}）" if mc_use_t else "常態分布"
        _cache_note = "⚡ 來自快取（參數未變動）" if _from_cache else f"🔄 即時計算完成（{_elapsed_ms:.0f} ms）"
        st.caption(
            f"{_cache_note}　｜　"
            f"分布：{_dist_label}、σ = {mc_std}%、r̄ = {r_pct}%、10,000 次模擬、{mc_strategy}、"
            f"醫療溢價 {medical_premium}%、起始年齡 {age_start} 歲、目標年齡 {age_end} 歲。"
            "　seed=42 固定，相同參數可重現。"
        )
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
# TAB 2：教育資訊庫
# ──────────────────────────────────────────────
with tab2:
    st.title("📚 退休教育資訊庫")
    st.caption("整合：2025年退休規劃彙整表 · 2026全球與台灣退休安全策略手冊 · 最新網路資料（台灣）")
    st.divider()

    edu_topic = st.radio(
        "選擇主題",
        [
            "1｜退休三階段（消費微笑曲線）",
            "2｜安全提領率與 GK 護欄策略",
            "3｜2025 台灣綜合所得稅率",
            "4｜勞保老年年金計算",
            "5｜勞退新制試算",
            "6｜長照費用與風險",
            "7｜資產配置與 ETF 建議",
            "8｜2025–2026 台灣與全球經濟預測",
            "9｜退休金制度改革動態",
            "10｜配置典範革命：傳統「年齡=債券」vs 上升股票路徑",
            "11｜持有房產與心理帳戶陷阱",
            "12｜退休資產負債表：自住房產的防禦性與盲點",
            "13｜長照對沖與房產資金階梯",
            "14｜不動產收益：租金、殖利率與 REITs",
            "15｜出租物業的類年金效應（Buy-to-Let Quasi-Annuity）",
            "16｜不動產收益護欄：Income Floor 與折現風險調整",
            "17｜退休現金流全景：主動槓桿與房貸管理",
        ],
        horizontal=True,
    )
    st.divider()

    if edu_topic.startswith("1"):
        st.subheader("退休三階段：消費微笑曲線 (Retirement Spending Smile)")
        st.markdown("> **學術來源**：David Blanchett (2014)「Exploring the retirement consumption puzzle」\n>\n> 退休支出並非線性遞減，而呈現「U型微笑曲線」：初期（活躍期）支出高，中期自然縮減，晚期（護理期）再度攀升。")
        phases = pd.DataFrame([
            ["Go-Go Years（活躍期）","65–75 歲","最高","體力充沛、旅遊頻繁；建議設定最高版本的生活費","相當於工作時期的 100%"],
            ["Slow-Go Years（緩速期）","75–85 歲","中等","行動力下降，長途旅遊減少；支出自然縮減約 15–20%","約為工作時期的 80%"],
            ["No-Go Years（護理期）","85 歲以上","醫療主導","行動幾乎停止；醫療/長照支出飆升；需備妥長照保障","醫療支出佔比大幅提升"],
        ], columns=["階段","年齡區間","活動力","特徵說明","相對支出水準"])
        st.dataframe(phases, use_container_width=True, hide_index=True)
        st.info("**模型實裝（V2.0引擎）**：75–84 歲 `spend × 0.80`；85 歲以上 `spend × 1.10`；70 歲起疊加醫療溢價指數複利 `W₀×15% × ((1+rate)^(age-65)−1)`")
        st.markdown("""
#### 台灣 2025 實證：高齡家庭通膨
- **2025 年高齡家庭 CPI 年增 1.74%**，連續 7 年高於整體平均（整體 1.66%）
- 高齡家庭醫療保健權重約 **8%**（一般家庭約 5%）
- 主要推升：掛號費調漲、外籍看護費上漲、房租上升
- 資料來源：主計總處，2025年1月起按月公布高齡家庭 CPI
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
        st.subheader("2025 年台灣綜合所得稅率（113 年度，114 年 5 月申報）")
        ec1, ec2 = st.columns([3,2])
        with ec1:
            tax_df = pd.DataFrame([
                ["第 1 級","5%","NT$ 0 – 590,000","NT$ 0","應稅所得 × 5%"],
                ["第 2 級","12%","NT$ 590,001 – 1,330,000","NT$ 41,300","× 12% - 41,300"],
                ["第 3 級","20%","NT$ 1,330,001 – 2,660,000","NT$ 147,700","× 20% - 147,700"],
                ["第 4 級","30%","NT$ 2,660,001 – 4,980,000","NT$ 413,700","× 30% - 413,700"],
                ["第 5 級","40%","NT$ 4,980,001 以上","NT$ 911,700","× 40% - 911,700"],
            ], columns=["級別","稅率","綜合所得淨額範圍","累進差額","速算公式"])
            st.dataframe(tax_df, use_container_width=True, hide_index=True)
            st.caption("資料來源：財政部 2025 年度綜合所得稅稅率表")
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
            st.info("**退職所得免稅額（2025 年）**\n- 一次領取：每年資 **NT$ 206,000** 以內免稅；超過 206,000 未達 414,000 元：50% 課稅；超過 414,000：全數課稅\n- 分期領取（月領）：每年減除 **NT$ 894,000**")
        with ec4:
            st.success("**退休後節稅策略**\n- 善用 **70 歲以上免稅額加成**（145,500 vs 97,000）\n- 勞保勞退月領 + 適度從自有資產提領，控制在低稅級\n- 股利：分離課稅（28%）vs 合併申報（擇優）")
        st.markdown("#### 稅後實質購買力試算（單身，年領 120 萬，標準扣除）\n- 應稅所得 = 120萬 - 9.7萬 - 13.1萬 = **97.2 萬**\n- 稅額 = 97.2萬 × 5% = **NT$ 48,600**（稅率約 4%）\n- **稅後實質購買力 ≈ NT$ 1,151,400 / 年**")

    elif edu_topic.startswith("4"):
        st.subheader("勞保老年年金計算（2025 年最新）")
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

    elif edu_topic.startswith("5"):
        st.subheader("勞退新制個人專戶（2025 年）")
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
        st.warning("**勞退基金績效參考**\n- 近 10 年平均名目報酬率約 **6.7%**\n- 2024 年：年度收益率 **16.16%**（AI 帶動科技股）\n- ⚠️ 歷史績效不代表未來，個人規劃建議用保守假設（實質 4–5%）")
        st.markdown("|  | 勞退新制 | 勞保老年年金 |\n|---|---|---|\n| **性質** | 個人儲蓄帳戶 | 社會保險 |\n| **請領年齡** | 60 歲（年資 15 年） | 65 歲（可提前/展延）|\n| **稅務** | 一次領部分須申報 | 完全免稅 |\n| **可合併** | ✅ 可同時領 | ✅ 同上 |")

    elif edu_topic.startswith("6"):
        st.subheader("長照費用與風險（台灣 2025 年）")
        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown("- **日照中心**：約 NT$ 1,000/天（政府補助 84%，自付約 160 元）\n- **住宿型機構**：月補助約 NT$ 10,000，實際開銷遠超\n- **外籍看護**：費用持續上漲，推升高齡家庭 CPI\n- **長照 2.0 預算**（114 年度）：**NT$ 879 億**（YoY +6%）")
            st.error("⚠️ **長照基金永續危機**\n\n保守估計，長照基金可能於 **2028 年用罄**，政府正研議「長照保險」制度。目前台灣約有 **86 萬個**失能、失智或身心障礙照顧家庭。")
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

    elif edu_topic.startswith("8"):
        st.subheader("2025–2026 台灣與全球經濟預測")
        ec1, ec2 = st.columns(2)
        with ec1:
            eco25 = pd.DataFrame([
                ["全球 GDP","2.1%","IMF / 費城聯準會 SPF"],
                ["美國 GDP","1.9%","前次預測上修 0.2 個百分點"],
                ["美國失業率","4.2%","與前次預測大致相同"],
                ["美國 CPI","2.8% → 趨降","Fed 目標 2%"],
            ], columns=["指標","2025 預測","資料來源"])
            st.markdown("#### 2025 年預測")
            st.dataframe(eco25, use_container_width=True, hide_index=True)
        with ec2:
            eco26 = pd.DataFrame([
                ["全球 GDP","2.2%","Aberdeen / J.P. Morgan"],
                ["美國 GDP","1.8%","AI 投資帶動"],
                ["美國失業率","4.5%","2025年後上升"],
                ["歐元區通膨","1.9–2.4%","趨近目標值"],
                ["日本","利率上行、通膨上升","日圓兌美元變數大"],
            ], columns=["指標","2026 預測","資料來源"])
            st.markdown("#### 2026 年預測")
            st.dataframe(eco26, use_container_width=True, hide_index=True)
        st.info("**2026 策略手冊核心建議**\n- 2026 年最佳提領率：**3.9%**（Morningstar 2025 年報，下修自 4%）\n- 美國 CPI 穩定在 2.8%，歐洲 1.9–2.4%\n- AI 帶動台灣科技股短期有利，仍需防集中度風險\n- 建議 Atlas 策略：FIA 年金消除 SORR + 投資組合捕捉成長")
        st.markdown("| 情境 | 對投資組合影響 | 對提領策略影響 |\n|---|---|---|\n| 低通膨（< 2%）| 債券表現回穩 | 可維持較高提領率 |\n| 高通膨（> 3%）| 股票抗通膨、固定收益受損 | 應降低固定提領，採 GK 護欄 |\n| GDP 放緩 | 股市波動加大，SORR 風險上升 | 退休初期配置應更保守 |\n| 地緣政治 | 台灣科技板塊特定風險 | 全球分散 ETF 為對沖工具 |")

    elif edu_topic.startswith("9"):
        st.subheader("退休金制度改革動態（台灣 2025–2026）")
        ec1, ec2 = st.columns(2)
        with ec1:
            st.error("**勞保基金現況（2025年）**\n- 基金規模：**NT$ 10.29 兆**\n- 每年虧損：約 NT$ **2,000 億**\n- 2028 年：改以精算評估，預計確認財務缺口\n- 核心問題：繳費率約 12–13%，遠低於平衡所需的 27%+")
        with ec2:
            st.warning("**改革討論方向**\n1. **提高繳費率**：需達 27%+ 才能長期平衡\n2. **延後請領年齡**：向 OECD 靠攏（67–69 歲）\n3. **年金積分制**：引入德國「積分制」\n4. **個人帳戶強化**：降低隨收隨付風險")
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
        st.caption("資料來源：勞動部勞動基金運用局。歷史績效不代表未來。")
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

        st.markdown("### 二、數據的衝擊：各種配置的破產機率")
        failure_df = pd.DataFrame([
            ["100% 債券", "39%", "不含任何股票；通膨慢速蠶食本金", "❌ 危險"],
            ["股六債四（60/40）", "19%", "傳統「平衡」組合；長壽情境下仍不足", "⚠️ 中等"],
            ["股八債二（80/20）", "12%", "偏積極；短期波動大但長壽保護強", "✅ 較好"],
            ["100% 全球股票", "7%", "最低破產率；但需承受極端短期波動", "✅✅ 最優（長壽）"],
        ], columns=["資產配置", "30年破產機率", "說明", "評估"])
        st.dataframe(failure_df, use_container_width=True, hide_index=True)
        st.caption("資料來源：台灣聲音媒體研究整理；20年回測（台股年化12.7%、全球股票9.4%、債券2.8–6.6%）")

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
            "③ Vanguard VLCM Life-Cycle Model (2025) "
            "④ 台灣聲音媒體/今周刊 2025 年退休研究整理"
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
            ltc_end_age    = st.number_input("規劃至年齡 (歲)",           min_value=ltc_onset_age+1, max_value=105, value=95, step=1)
        with ltc_c2:
            ltc_monthly_wan = st.number_input("月長照費用估算 (萬/月)", min_value=0.5, max_value=20.0, value=5.0, step=0.5,
                                              help="住宿機構約 4–8 萬/月；居家 24H 看護約 5–12 萬/月")
            ltc_coverage_wan= st.number_input("長照險/長照補助可抵用 (萬/月)", min_value=0.0, max_value=10.0, value=1.5, step=0.5,
                                              help="勞保失能給付 + 長照 2.0 補助 + 長照險理賠合計")

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
        with ds_c2:
            ds_hold_yrs    = st.number_input("持有年數 (年，用於計算房地合一稅)", min_value=0, max_value=50, value=15, step=1)
            ds_cost_pct    = st.number_input("交易成本（仲介+代書+搬家，%）", min_value=0.0, max_value=10.0, value=4.0, step=0.5)

        # 房地合一稅率
        if ds_hold_yrs < 2:
            ltx_rate = 0.45
        elif ds_hold_yrs < 5:
            ltx_rate = 0.35
        elif ds_hold_yrs < 10:
            ltx_rate = 0.20
        else:
            ltx_rate = 0.15

        ds_gross_gain  = max(0, ds_current_wan - ds_target_wan)
        ds_tax         = ds_gross_gain * ltx_rate
        ds_cost        = ds_current_wan * ds_cost_pct / 100
        ds_net_release = ds_gross_gain - ds_tax - ds_cost

        col_ds1, col_ds2, col_ds3 = st.columns(3)
        with col_ds1:
            st.metric("毛釋出金額",      f"{ds_gross_gain:.0f} 萬")
        with col_ds2:
            st.metric(f"房地合一稅（{ltx_rate*100:.0f}%，持有{ds_hold_yrs}年）", f"−{ds_tax:.0f} 萬")
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
            ri_monthly_rent  = st.number_input("月租金 (萬/月)",         min_value=0.0, max_value=30.0,
                                               value=float(rental_monthly_wan) if rental_monthly_wan > 0 else 2.0,
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
            qa_rental   = st.number_input("年租金淨收益 (萬/年)", min_value=0.0, max_value=500.0,
                                          value=float(rental_combined_annual / 10_000), step=5.0,
                                          key="qa_rent",
                                          help="已扣除空置、維修、稅務的淨收益；對應左側「月租金淨收入」×12")
            qa_pension  = st.number_input("年勞保/勞退收入 (萬/年)", min_value=0.0, max_value=200.0,
                                          value=float(pension_annual / 10_000), step=5.0, key="qa_pen")

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
            hc_gross_wan   = st.number_input("年毛租金 (萬/年)",    min_value=0.0, max_value=500.0,
                                             value=float(rental_monthly_wan * 12), step=5.0, key="hc_gross")
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
            ex_pension = st.number_input("勞保/勞退年收入 (萬/年)", min_value=0.0, max_value=200.0,
                                         value=float(pension_annual/10_000), step=5.0, key="ex_pen2")
        with ex_c2:
            ex_rental  = st.number_input("租金可靠淨收益 (萬/年)", min_value=0.0, max_value=300.0,
                                         value=round(hc_reliable, 1), step=5.0, key="ex_rent2",
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
                ["基本生活費",         "終身",         "高（穩定）", "消費微笑曲線：75–84 歲 ×0.8，85+ ×1.1"],
                ["醫療溢價",           "65 歲起遞增",  "中",        "本 App 引擎：每年複利遞增"],
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
            ml_balance   = st.number_input("房貸餘額 (萬)",    min_value=0,   max_value=5_000, value=int(re_mortgage_wan), step=50, key="ml_bal")
            ml_rate      = st.number_input("房貸利率 (%/年)",  min_value=0.5, max_value=8.0,   value=2.3,  step=0.1,  key="ml_rate")
        with ml_c2:
            ml_years_left= st.number_input("剩餘還款年數 (年)",min_value=1,   max_value=30,    value=15,   step=1,    key="ml_yrs")
            ml_retire_age= st.number_input("退休年齡 (歲)",    min_value=50,  max_value=75,    value=age_start, step=1, key="ml_retage")
        with ml_c3:
            ml_invest_r  = st.number_input("金融資產實質報酬 (%)", min_value=0.0, max_value=12.0, value=float(r_pct), step=0.5, key="ml_ir",
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
        base_rental  = rental_combined_annual / 10_000
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

    st.divider()
    st.caption("資料來源：① 2025年退休規劃彙整表 ② 2026全球與台灣退休安全策略手冊 ③ Morningstar State of Retirement Income 2025 ④ 財政部2025綜所稅率表 ⑤ 勞動部勞保局 ⑥ 主計總處高齡家庭CPI ⑦ 費城聯準會SPF 2025Q4")

# ──────────────────────────────────────────────
# TAB 3：提領實務指南
# ──────────────────────────────────────────────
with tab3:
    st.title("🛠️ 退休提領實務指南")
    st.caption("整合：Morningstar 2025、今周刊、HBR Taiwan、Kitces、Christine Benz、T. Rowe Price 最新研究")
    st.info(
        "本指南回答退休最實際的三個問題：**「何時賣股票」「何時由攻轉守」「遇到股災怎麼辦」**。"
        "內容以台灣市場（0050、0056、00878）為核心，並整合國際最新研究。"
    )
    st.divider()

    guide_topic = st.radio(
        "選擇主題",
        [
            "1｜三桶金策略（Bucket Strategy）",
            "2｜何時賣股票：五大觸發條件",
            "3｜滑行路徑：激進→保守的轉換節奏",
            "4｜再平衡規則：時間 vs 門檻觸發",
            "5｜提領順序與稅務效率",
            "6｜熊市/股災應對手冊",
            "7｜台灣 ETF 實務（0050/0056/00878）",
            "8｜Bond Tent 規劃工具（互動式）",
            "9｜全球分散化：降低非系統性風險",
        ],
        horizontal=True,
    )
    st.divider()

    # ── 1. 三桶金策略 ──────────────────────────────────────────────────
    if guide_topic.startswith("1"):
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
             "70 歲後免稅額提升至 145,500，更容易達到低稅或零稅"],
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
        st.caption("以上為簡化估算，實際結果受配息再投入、手續費、稅務影響。資料參考：今周刊 2025/12 分析")

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
            bt_now_age    = st.number_input("目前年齡 (歲)",       min_value=30, max_value=75, value=age_start, step=1)
            bt_retire_age = st.number_input("預計退休年齡 (歲)",   min_value=bt_now_age, max_value=75, value=min(65, max(bt_now_age, 65)), step=1)
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
        st.caption("以上為歷史估算，未來報酬不保證。名目報酬需扣通膨（約2%）得實質報酬。")

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
            "③ Vanguard VLCM 2025 ④ 台灣聲音媒體退休研究 2025"
        )

    st.divider()
    st.caption(
        "資料來源：① Morningstar State of Retirement Income 2025 "
        "② Christine Benz, Morningstar Bucket Portfolio 2025 "
        "③ Kitces.com Rebalancing Strategies (2015) "
        "④ 今周刊「0050+0056退休最強攻略」(2025/12) "
        "⑤ HBR Taiwan「三桶金策略」 "
        "⑥ 商業周刊「退休前何時調整投資規畫」 "
        "⑦ T. Rowe Price Dynamic Withdrawal (2025)"
    )
