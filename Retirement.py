# -*- coding: utf-8 -*-
"""
退休規劃大師 — B 版（iPhone 設定頁風格）

目標：
- 保留核心計算邏輯（確定性投影 + 蒙地卡羅）
- 大幅簡化主頁資訊層級：先圖/卡，再表，文字最後且可收合
- 不依賴 import Retirement.py（避免匯入即執行 Classic UI）
"""

from __future__ import annotations

import time
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import streamlit as st


# ──────────────────────────────────────────────────────────────────────────────
# URL helpers（保留日後擴充導覽彈性）
# ──────────────────────────────────────────────────────────────────────────────
def _qp_first(v, default: str | None = None) -> str | None:
    if v is None:
        return default
    if isinstance(v, (list, tuple)):
        return str(v[0]) if v else default
    return str(v)


def _qs(**kwargs) -> str:
    clean = {k: v for k, v in kwargs.items() if v is not None and v != ""}
    return "?" + urlencode(clean, doseq=False)


# ──────────────────────────────────────────────────────────────────────────────
# Formatting
# ──────────────────────────────────────────────────────────────────────────────
def _fmt_asset(x: float) -> str:
    if x is None or x <= 0:
        return "歸零"
    if x >= 100_000_000:
        return f"{x/1e8:,.2f} 億"
    return f"{x/1e4:,.0f} 萬"


def _fmt_wan(n: float) -> str:
    if n is None or n < 0:
        return "0"
    return f"{n/1e4:,.0f} 萬"


def _fmt_currency(n: float) -> str:
    if n is None or n < 0:
        return "0"
    return f"{int(n):,}"


# ──────────────────────────────────────────────────────────────────────────────
# Historical returns (real, pct) for bootstrap mode (1975–2024)
# ──────────────────────────────────────────────────────────────────────────────
_HIST_REAL_RETURNS_PCT: list[float] = [
    -13.0,  22.5,  14.2,  -8.3,  31.0,  18.7,   5.4, -12.0,  42.8,  38.5,
     21.3, -55.0,  65.2,  14.0,  -4.5,  26.3,  18.9,  -8.0, -21.0, -16.0,
     30.5,  25.0,  12.3,  -6.2,  38.0,   4.5,  -3.0,  22.0,  17.5,  31.2,
    -38.0,  45.0,  16.8,  -5.5,  12.0,  24.5,   8.3,  -9.0,  35.0,  18.2,
     -3.5,  28.7,  11.5,   6.0,  -2.0,  19.0,  26.5, -12.5,  33.0,  16.2,
]


# ──────────────────────────────────────────────────────────────────────────────
# Core engines (same logic as Classic)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def _run_monte_carlo(
    A0: float,
    W0: float,
    r_mean_pct: float,
    r_std_pct: float,
    n_years: int,
    strategy: str,
    pension_annual: float,
    claim_age: int,
    age_start: int,
    *,
    med_premium_pct: float = 0.0,
    dist_mode: str = "normal",
    t_df: int = 7,
    t_skew: float = 0.0,
    rental_annual: float = 0.0,
    rental_start_age: int = 65,
    rm_annual: float = 0.0,
    rm_start_age: int = 999,
    inflation_randomize: bool = False,
    inflation_mean_pct: float = 2.0,
    inflation_std_pct: float = 0.8,
    n_sim: int = 10_000,
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed=42)
    r_mean = r_mean_pct / 100.0
    r_std = r_std_pct / 100.0
    med_rate = med_premium_pct / 100.0

    if dist_mode == "t":
        scale_factor = np.sqrt(t_df / (t_df - 2))
        raw_t = rng.standard_t(df=t_df, size=(n_sim, n_years))
        if t_skew != 0.0:
            skew_amp = abs(t_skew)
            raw_t = np.where(raw_t < 0, raw_t * (1.0 + skew_amp), raw_t)
            _t_mean = raw_t.mean()
            _t_std = raw_t.std()
            raw_t = (raw_t - _t_mean) / (_t_std if _t_std > 0 else 1.0)
        returns = r_mean + r_std * raw_t / scale_factor
    elif dist_mode == "bootstrap":
        hist_arr = np.array(_HIST_REAL_RETURNS_PCT) / 100.0
        idx = rng.integers(0, len(hist_arr), size=(n_sim, n_years))
        returns = hist_arr[idx]
    else:
        returns = rng.normal(r_mean, r_std, (n_sim, n_years))

    if inflation_randomize and dist_mode != "bootstrap":
        infl_mean = inflation_mean_pct / 100.0
        infl_std = inflation_std_pct / 100.0
        cpi_draws = rng.normal(infl_mean, infl_std, (n_sim, n_years))
        returns = returns - cpi_draws

    A = np.full(n_sim, float(A0))
    alive = np.ones(n_sim, dtype=bool)
    IWR = W0 / A0 if A0 > 0 else 0.0
    gk_spend = np.full(n_sim, float(W0))

    for j in range(n_years):
        current_age = age_start + j

        pension_income = pension_annual if pension_annual > 0 and current_age >= claim_age else 0.0
        rental_income = rental_annual if rental_annual > 0 and current_age >= rental_start_age else 0.0
        rm_income = rm_annual if rm_annual > 0 and current_age >= rm_start_age else 0.0
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
            med_base = W0 * 0.15
            med_extra = med_base * ((1 + med_rate) ** (current_age - 70) - 1)
            spend = spend + med_extra

        spend_from_asset = np.where(alive, np.maximum(0.0, spend - passive_income), 0.0)
        A = (A - spend_from_asset) * (1.0 + returns[:, j])
        alive = alive & (A > 0)

    finals = np.maximum(0.0, A)
    success_rate = float(alive.mean() * 100)
    p10 = float(np.percentile(finals, 10))
    p50 = float(np.percentile(finals, 50))
    p90 = float(np.percentile(finals, 90))
    return success_rate, p10, p50, p90


def run_dynamic_projection(
    A0: float,
    W0: float,
    r_pct: float,
    n_years: int,
    start_age: int,
    *,
    strategy: str = "fixed",
    med_premium_pct: float = 0.0,
    pension_annual: float = 0.0,
    claim_age: int = 60,
    rental_annual: float = 0.0,
    rental_start_age: int = 65,
    rm_annual: float = 0.0,
    rm_start_age: int = 999,
) -> float:
    A = float(A0)
    W_total = float(W0)
    r = r_pct / 100
    med_rate = med_premium_pct / 100
    IWR = W_total / A if A > 0 else 0.0
    gk_spend = W_total

    for i in range(n_years):
        if A <= 0:
            return 0.0
        current_age = start_age + i

        pension_income = pension_annual if pension_annual > 0 and current_age >= int(claim_age) else 0.0
        rental_income = rental_annual if rental_annual > 0 and current_age >= int(rental_start_age) else 0.0
        rm_income = rm_annual if rm_annual > 0 and current_age >= int(rm_start_age) else 0.0
        passive_income = pension_income + rental_income + rm_income

        if strategy == "fixed":
            total_spend = W_total
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
            total_spend = W_total * mult
        elif strategy == "gk":
            remaining_yrs = n_years - i
            current_wr = gk_spend / A if A > 0 else 999.0
            if current_wr < IWR * 0.8:
                gk_spend *= 1.06
            elif current_wr > IWR * 1.2 and remaining_yrs > 15:
                gk_spend *= 0.9
            total_spend = gk_spend
        else:
            total_spend = W_total

        if current_age >= 70 and med_rate > 0:
            med_base = W_total * 0.15
            med_extra = med_base * ((1 + med_rate) ** (current_age - 70) - 1)
            total_spend += med_extra

        spend_from_asset = max(0.0, total_spend - passive_income)
        A = (A - spend_from_asset) * (1 + r)

    return max(0.0, A)


# ──────────────────────────────────────────────────────────────────────────────
# UI (B style)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="退休規劃大師（B 版）",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("退休規劃大師（B 版）")
st.caption("iPhone 設定頁風格：先圖/卡片 → 再表格 → 文字最後且可收合。金額以今日實質購買力估算。")


# Sidebar — Wizard only (B版主打新手流程)
st.sidebar.header("🧭 快速設定")
st.sidebar.caption("3 步完成輸入；選填項都可先略過。")

st.session_state.setdefault("b_step", 1)
step = int(st.session_state.get("b_step", 1))
step = 1 if step < 1 else 3 if step > 3 else step
st.session_state["b_step"] = step

st.sidebar.markdown(f"**Step {step} / 3**")
st.sidebar.progress(step / 3)

# Defaults (保守但可用)
age_start = int(st.session_state.get("b_age_start", 40))
age_end = int(st.session_state.get("b_age_end", 90))
W0_wan = float(st.session_state.get("b_W0_wan", 120.0))
inflation_pct = float(st.session_state.get("b_inflation", 2.0))
medical_premium = float(st.session_state.get("b_med_premium", 1.7))

A0_assets_wan = int(st.session_state.get("b_assets_wan", 3000))
A0_liab_wan = int(st.session_state.get("b_liab_wan", 0))

pension_monthly_wan = float(st.session_state.get("b_pension_m", 0.0))
claim_age = int(st.session_state.get("b_claim_age", 60))

use_re = bool(st.session_state.get("b_use_re", False))
re_rental_wan = int(st.session_state.get("b_re_rental_wan", 0))
rental_monthly_wan = float(st.session_state.get("b_rental_m", 0.0))
rental_start_age = int(st.session_state.get("b_rental_age", 65))
rm_use = bool(st.session_state.get("b_rm_use", False))
rm_start_age = int(st.session_state.get("b_rm_age", 80))
rm_monthly_wan = float(st.session_state.get("b_rm_m", 3.0))

r_mode = str(st.session_state.get("b_r_mode", "平衡（建議）"))
strategy_mode = str(st.session_state.get("b_strategy", "GK 護欄（建議）"))
use_cost_drag = bool(st.session_state.get("b_use_drag", True))

# Step 1
if step == 1:
    with st.sidebar.expander("生活費與年齡", expanded=True):
        age_start = st.number_input("目前年齡", 25, 75, age_start, 1, key="b_age_start")
        age_end = st.number_input("規劃到幾歲", 70, 105, age_end, 1, key="b_age_end")
        W0_wan = st.number_input("年度生活費（萬/年）", 10.0, 500.0, W0_wan, 5.0, key="b_W0_wan")
        inflation_pct = st.slider("通膨（CPI，%）", 0.0, 8.0, inflation_pct, 0.5, key="b_inflation")
        medical_premium = st.slider("醫療溢價（CPI + %）", 0.0, 4.0, medical_premium, 0.1, key="b_med_premium")

    c1, c2 = st.sidebar.columns(2)
    with c2:
        if st.button("下一步 ▶", use_container_width=True):
            st.session_state["b_step"] = 2
            st.rerun()

# Step 2
elif step == 2:
    with st.sidebar.expander("資產與底層現金流（Income Floor）", expanded=True):
        A0_assets_wan = st.number_input("金融資產（萬）", 0, 50_000, A0_assets_wan, 100, key="b_assets_wan")
        A0_liab_wan = st.number_input("金融負債（萬）", 0, 50_000, A0_liab_wan, 50, key="b_liab_wan")
        net_fin = max(0, int(A0_assets_wan - A0_liab_wan))
        st.metric("淨金融資產", f"{net_fin:,} 萬")

        st.divider()
        pension_monthly_wan = st.number_input("勞保＋勞退（月領，萬/月）", 0.0, 50.0, pension_monthly_wan, 0.5, key="b_pension_m")
        claim_age = st.number_input("請領年齡", 55, 70, claim_age, 1, key="b_claim_age")

    st.sidebar.divider()
    use_re = st.sidebar.toggle("🏠 我想納入不動產/租金/以房養老（選填）", value=use_re, key="b_use_re")
    if use_re:
        with st.sidebar.expander("🏠 不動產（選填）", expanded=False):
            re_rental_wan = st.number_input("出租房產市值（萬）", 0, 20_000, re_rental_wan, 100, key="b_re_rental_wan")
            rental_monthly_wan = st.number_input(
                "月租金淨收入（萬/月）",
                0.0,
                100.0,
                rental_monthly_wan,
                0.5,
                key="b_rental_m",
                help="請填到手淨現金流；不確定時可先用『毛租金×0.75』保守化。",
            )
            rental_start_age = st.number_input("租金開始年齡", 40, 90, rental_start_age, 1, key="b_rental_age")
            if re_rental_wan > 0 and rental_monthly_wan > 0:
                net_yield = (rental_monthly_wan * 12) / float(re_rental_wan) * 100
                st.caption(f"淨殖利率（估）≈ **{net_yield:.2f}%**（常見約 1%～2.5%）")
                st.caption(
                    "租金折現（Income Floor）工具："
                    f"[點我]({_qs(nav='edu', edu_category='房產／資產負債表／心理帳戶', edu_topic='16｜不動產收益護欄：Income Floor 與折現風險調整')})"
                )

            rm_use = st.toggle("啟用以房養老（選填）", value=rm_use, key="b_rm_use",
                               help="建議視為 80+ / 長照期的末端流動性保險。")
            if rm_use:
                rm_start_age = st.number_input("以房養老啟動年齡", 60, 95, rm_start_age, 1, key="b_rm_age")
                rm_monthly_wan = st.number_input("以房養老月領（萬/月）", 0.0, 50.0, rm_monthly_wan, 0.5, key="b_rm_m")

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("◀ 上一步", use_container_width=True):
            st.session_state["b_step"] = 1
            st.rerun()
    with c2:
        if st.button("下一步 ▶", use_container_width=True):
            st.session_state["b_step"] = 3
            st.rerun()

# Step 3
else:
    with st.sidebar.expander("投資假設與策略", expanded=True):
        r_mode = st.radio("投資風格", ["保守", "平衡（建議）", "積極"], index=1, horizontal=True, key="b_r_mode")
        use_cost_drag = st.toggle("用『淨報酬』模擬（建議）", value=use_cost_drag, key="b_use_drag")
        strategy_mode = st.radio("策略", ["GK 護欄（建議）", "固定提領", "消費微笑曲線"], index=0, horizontal=True, key="b_strategy")

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("◀ 上一步", use_container_width=True):
            st.session_state["b_step"] = 2
            st.rerun()
    with c2:
        if st.button("完成 ✅", use_container_width=True):
            st.session_state["b_step"] = 1
            st.toast("已更新設定", icon="✅")


# ──────────────────────────────────────────────────────────────────────────────
# Derive inputs
# ──────────────────────────────────────────────────────────────────────────────
n_years = max(1, int(age_end - age_start))
A0 = float(max(0, int(A0_assets_wan - A0_liab_wan))) * 10_000
W0 = float(W0_wan) * 10_000

pension_annual = float(pension_monthly_wan) * 12 * 10_000
rental_annual = float(rental_monthly_wan) * 12 * 10_000 if use_re else 0.0
rm_annual = float(rm_monthly_wan) * 12 * 10_000 if (use_re and rm_use) else 0.0
rm_start_age_eff = int(rm_start_age) if (use_re and rm_use) else 999

# r mapping (real, net)
if r_mode == "保守":
    r_gross = 4.0
    sigma_std = 12.0
elif r_mode.startswith("平衡"):
    r_gross = 5.0
    sigma_std = 15.0
else:
    r_gross = 6.5
    sigma_std = 18.0

# cost drag (simple preset, no extra sliders in B)
drag = 0.25 if use_cost_drag else 0.0
r_net = float(max(-5.0, r_gross - drag))

strategy = "gk" if strategy_mode.startswith("GK") else ("fixed" if strategy_mode.startswith("固定") else "smile")

# Floor coverage
floor_annual = 0.0
if pension_annual > 0 and age_start >= int(claim_age):
    floor_annual += pension_annual
if rental_annual > 0 and age_start >= int(rental_start_age):
    floor_annual += rental_annual
if rm_annual > 0 and age_start >= int(rm_start_age_eff):
    floor_annual += rm_annual
floor_cover = (floor_annual / W0 * 100) if W0 > 0 else 0.0

W0_asset = max(0.0, W0 - floor_annual)
IWR = (W0_asset / A0 * 100) if A0 > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Main area — iPhone setting style: Overview / Actions / Risk
# ──────────────────────────────────────────────────────────────────────────────
view = st.radio("檢視", ["總覽", "調整", "風險"], index=0, horizontal=True, label_visibility="collapsed")


def _status_bucket(sr: float | None, iwr: float) -> tuple[str, str]:
    """
    回傳（狀態字、簡短理由）
    sr=None 代表未跑蒙地卡羅
    """
    if sr is None:
        if iwr >= 4.0:
            return "注意", "IWR 偏高，建議做風險測試"
        return "安全", "IWR 低於 4%"
    if sr >= 90:
        return "安全", "成功率 ≥ 90%"
    if sr >= 75:
        return "注意", "成功率 75–90%"
    return "危險", "成功率 < 75%"


if "b_mc_run" not in st.session_state:
    st.session_state["b_mc_run"] = False


def _run_mc_if_needed(mode: str) -> tuple[float | None, float | None, float | None, float | None, dict]:
    if not st.session_state.get("b_mc_run", False):
        return None, None, None, None, {}

    if mode == "壓力（保守）":
        dist_mode = "t"
        sigma = 18.0
        t_df = 7
        t_skew = -0.3
        infl_rand = True
        infl_std = 1.2
    else:
        dist_mode = "normal"
        sigma = float(sigma_std)
        t_df = 7
        t_skew = 0.0
        infl_rand = False
        infl_std = 0.8

    _t0 = time.perf_counter()
    sr, p10, p50, p90 = _run_monte_carlo(
        A0, W0, r_net, sigma,
        n_years, strategy,
        pension_annual, int(claim_age), int(age_start),
        med_premium_pct=float(medical_premium),
        dist_mode=dist_mode,
        t_df=t_df,
        t_skew=t_skew,
        rental_annual=float(rental_annual),
        rental_start_age=int(rental_start_age),
        rm_annual=float(rm_annual),
        rm_start_age=int(rm_start_age_eff),
        inflation_randomize=infl_rand,
        inflation_mean_pct=float(inflation_pct),
        inflation_std_pct=float(infl_std),
    )
    elapsed_ms = (time.perf_counter() - _t0) * 1000
    meta = {
        "dist_mode": dist_mode,
        "sigma": sigma,
        "t_df": t_df,
        "t_skew": t_skew,
        "infl_rand": infl_rand,
        "infl_std": infl_std,
        "elapsed_ms": elapsed_ms,
    }
    return sr, p10, p50, p90, meta


if view == "總覽":
    # Monte Carlo summary (optional)
    top_left, top_right = st.columns([2, 1])
    with top_right:
        st.toggle("顯示成功率（蒙地卡羅）", value=bool(st.session_state["b_mc_run"]), key="b_mc_run",
                  help="建議開啟；不想面對太多參數就用預設即可。")
        mc_mode = st.radio("模式", ["標準", "壓力（保守）"], index=0, horizontal=True, label_visibility="collapsed")

    sr, p10, p50, p90, meta = _run_mc_if_needed(mc_mode)
    status, why = _status_bucket(sr, IWR)

    with top_left:
        st.subheader("退休可行性")
        st.metric("狀態", status, why)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("有效提領率 IWR（從資產）", f"{IWR:.2f}%")
    m2.metric("Income Floor 覆蓋率", f"{floor_cover:.0f}%")
    m3.metric("淨實質報酬 r", f"{r_net:.2f}%", f"毛 {r_gross:.1f}% − 拖累 {drag:.2f}%")
    m4.metric("規劃年數", f"{n_years} 年", f"{age_start}→{age_end} 歲")

    st.divider()

    # 圖 1：Income Stack（用條圖取代表格）
    floor_items = []
    if pension_annual > 0:
        floor_items.append(("年金", pension_annual))
    if rental_annual > 0:
        floor_items.append(("租金", rental_annual))
    if rm_annual > 0:
        floor_items.append(("以房養老", rm_annual))
    floor_total = sum(v for _, v in floor_items)
    draw_need = max(0.0, W0 - floor_total)
    stack = [("年金/租金等（Floor）", floor_total), ("需從資產提領", draw_need)]
    df_stack = pd.DataFrame(stack, columns=["來源", "金額"])
    df_stack["金額（萬/年）"] = df_stack["金額"] / 10_000
    st.subheader("退休現金流堆疊（年）")
    st.bar_chart(df_stack.set_index("來源")["金額（萬/年）"], height=220)

    # 圖 2：蒙地卡羅終值分位（若有）
    if sr is not None:
        st.subheader(f"風險摘要（終值分位）— {mc_mode}")
        df_q = pd.DataFrame(
            {
                "分位": ["P10", "P50", "P90"],
                "終值（萬）": [p10 / 10_000, p50 / 10_000, p90 / 10_000],
            }
        )
        st.bar_chart(df_q.set_index("分位")["終值（萬）"], height=220)
        st.caption(f"成功率 {sr:.1f}%｜計算 {meta.get('elapsed_ms', 0):.0f} ms｜seed=42")

    with st.expander("了解更多（假設與細節）", expanded=False):
        st.markdown(
            f"- **A₀**：NTD {_fmt_currency(A0)}（{_fmt_wan(A0)}）\n"
            f"- **W₀**：NTD {_fmt_currency(W0)}/年（{_fmt_wan(W0)}）\n"
            f"- **Income Floor（起始年齡可用）**：{_fmt_wan(floor_annual)}/年\n"
            f"- **策略**：{strategy_mode}\n"
            f"- **實質報酬（淨）**：{r_net:.2f}%（模式：{r_mode}）\n"
        )


elif view == "調整":
    st.subheader("你最該先動哪個？")

    # Choose one primary knob
    if st.session_state.get("b_mc_run", False):
        # Run quick standard MC to decide (cheap enough; cached)
        sr, _, _, _, _ = _run_mc_if_needed("標準")
    else:
        sr = None

    if IWR >= 4.0:
        primary = ("先調整：年度生活費 W₀", "降低 W₀ 可直接降低 IWR、提高成功率。")
    elif (pension_annual <= 0) and (rental_annual <= 0) and (rm_annual <= 0):
        primary = ("先補強：Income Floor", "提高底層現金流可降低 SORR 破壞力。")
    else:
        primary = ("先確認：投資假設與策略", "用 GK 護欄 + 壓力測試，確保熊市也能執行。")

    st.info(f"**{primary[0]}**\n\n{primary[1]}")

    st.divider()
    st.subheader("建議清單（點開才看細節）")

    items: list[tuple[str, str]] = []
    if IWR >= 4.0:
        items.append(("降低 W₀", "先把生活費調到你願意在熊市也能執行的水準。"))
        items.append(("延後目標年齡/退休時點", "縮短規劃年數通常能明顯改善成功率。"))
        items.append(("啟用壓力測試", "用肥尾/負偏態與通膨不確定性檢視韌性。"))
    else:
        items.append(("用 GK 護欄（建議）", "市場好自動加薪、市場差自動減薪，減少行為錯誤。"))

    if use_re and rental_monthly_wan > 0 and re_rental_wan > 0:
        items.append(("確認租金『淨』", "若你填的是毛租金，請先用毛×0.75 保守化。"))

    if not items:
        items = [("維持現況", "目前設定已足以運行；建議到『風險』頁做壓力測試。")]

    for title, detail in items[:5]:
        with st.expander(title, expanded=False):
            st.write(detail)


else:
    st.subheader("風險測試")

    mode = st.radio("模式", ["標準", "壓力（保守）"], index=0, horizontal=True)
    st.toggle("顯示成功率（蒙地卡羅）", value=True, key="b_mc_run_force", label_visibility="collapsed")
    st.session_state["b_mc_run"] = True

    sr, p10, p50, p90, meta = _run_mc_if_needed(mode)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("成功率", f"{sr:.1f}%", "≥90% 安全" if sr >= 90 else ("75–90% 注意" if sr >= 75 else "<75% 危險"))
    c2.metric("P10", _fmt_asset(p10))
    c3.metric("P50", _fmt_asset(p50))
    c4.metric("P90", _fmt_asset(p90))

    st.divider()
    st.subheader("壓力矩陣（精簡版）")
    # 仍用你現有 deterministic engine，先用 heat-like emoji（最少文字）
    r_low = max(-5.0, r_net - 1.5)
    r_mid = r_net
    r_high = min(15.0, r_net + 1.5)
    _kw = dict(
        pension_annual=float(pension_annual),
        claim_age=int(claim_age),
        rental_annual=float(rental_annual),
        rental_start_age=int(rental_start_age),
        rm_annual=float(rm_annual),
        rm_start_age=int(rm_start_age_eff),
        med_premium_pct=float(medical_premium),
    )
    grid = []
    for strat in ["fixed", "smile", "gk"]:
        row = []
        for rr in [r_low, r_mid, r_high]:
            v = run_dynamic_projection(A0, W0, rr, n_years, int(age_start), strategy=strat, **_kw)
            row.append(v)
        grid.append(row)

    def _tile(v: float) -> str:
        if v <= 0:
            return "❗"
        if v < A0 * 0.5:
            return "⚠️"
        return "✅"

    df_heat = pd.DataFrame(
        {
            "策略": ["固定", "微笑", "GK"],
            f"悲觀 r={r_low:.1f}%": [_tile(x) for x in [grid[0][0], grid[1][0], grid[2][0]]],
            f"基準 r={r_mid:.1f}%": [_tile(x) for x in [grid[0][1], grid[1][1], grid[2][1]]],
            f"樂觀 r={r_high:.1f}%": [_tile(x) for x in [grid[0][2], grid[1][2], grid[2][2]]],
        }
    )
    st.dataframe(df_heat, use_container_width=True, hide_index=True, height=160)
    st.caption("✅ 充足｜⚠️ 可能偏緊｜❗ 可能歸零（此為快速判讀；細節可回 Classic 版看完整表格）")
    st.caption(f"分布 {meta.get('dist_mode')}｜σ={meta.get('sigma')}｜通膨隨機化={meta.get('infl_rand')}｜seed=42")

