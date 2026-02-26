# -*- coding: utf-8 -*-
"""
退休規劃大師 — 50年長週期資產動態管理
Streamlit 版：精簡排版、學術說明、可調邊界條件
"""

import streamlit as st
import pandas as pd

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
):
    """
    動態提領模擬器 V2.0 — 三項科學修正：
      A. 消費微笑曲線（75-84 歲縮減 20%、85 歲後增加 10%）
      B. 醫療溢價指數複利（70 歲起 (1+rate)^(age-65) 疊加於總需求 15%）
      C. 期初提領保守原則：(A - spend) × (1+r)，先扣費再計息
    支援台灣勞保/勞退：claim_age 起從資產提領 = W0 − pension_annual
    """
    A            = float(A0)
    W_total      = float(W0)
    r            = r_pct / 100
    med_rate     = med_premium_pct / 100
    IWR          = W_total / A if A > 0 else 0.0   # 基於總實質購買力
    gk_spend     = W_total                          # GK 追蹤總支出目標（絕對值）

    for i in range(n_years):
        if A <= 0:
            return 0
        current_age = start_age + i

        # 本年度勞保/勞退收入
        pension_income = (pension_annual
                          if pension_annual > 0 and current_age >= int(claim_age)
                          else 0.0)

        # ── 1. 策略決定「總支出目標」 ──────────────────────────────
        if strategy == "fixed":
            total_spend = W_total

        elif strategy == "smile":
            total_spend = W_total
            if 75 <= current_age <= 84:
                total_spend *= 0.8          # Go-Go → Slow-Go 縮減 20%
            elif current_age >= 85:
                total_spend *= 1.1          # No-Go 醫療密集期增加 10%

        elif strategy == "gk":
            current_wr = gk_spend / A if A > 0 else 999.0
            if current_wr < IWR * 0.8:
                gk_spend *= 1.1             # 繁榮法則：強制加薪 10%
            elif current_wr > IWR * 1.2:
                gk_spend *= 0.9             # 保全法則：強制減薪 10%
            total_spend = gk_spend

        else:
            total_spend = W_total

        # ── 2. 非線性醫療溢價（70 歲起指數遞增）─────────────────────
        if current_age >= 70 and med_rate > 0:
            med_base  = W_total * 0.15      # 生活費中約 15% 屬醫療相關
            med_extra = med_base * ((1 + med_rate) ** (current_age - 65) - 1)
            total_spend += med_extra

        # ── 3. 從資產提領 = 總支出 − 勞保勞退收入 ─────────────────────
        spend_from_asset = max(0.0, total_spend - pension_income)

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
    A0_wan = st.number_input(
        "初始資產 A₀ (萬)",
        min_value=100,
        max_value=50_000,
        value=3_000,
        step=100,
        help="系統的起始能量，以萬為單位（預設 3,000 萬）",
    )
    W0_wan = st.number_input(
        "實質購買力 (萬/年)",
        min_value=10,
        max_value=500,
        value=120,
        step=5,
        help="起始年提領 W₀，以萬/年為單位（預設 120 萬）",
    )
    A0 = A0_wan * 10_000
    W0 = W0_wan * 10_000

# 資產結構：支援「金額」或「百分比」兩種輸入模式
with st.sidebar.expander("資產結構配置", expanded=True):
    asset_input_mode = st.radio(
        "輸入方式",
        ["填寫實際金額 (萬)", "填寫比例 (%)"],
        index=0,
        horizontal=True,
    )
    r_us_stock, r_us_etf, r_tw_stock, r_tw_etf = 6.0, 5.0, 5.0, 4.0

    if asset_input_mode == "填寫實際金額 (萬)":
        amt_us_stock = st.number_input("美股個股 (萬)", min_value=0, max_value=100_000, value=0,    step=50)
        amt_us_etf   = st.number_input("美股 ETF  (萬)", min_value=0, max_value=100_000, value=600,  step=50)
        amt_tw_stock = st.number_input("台股個股 (萬)", min_value=0, max_value=100_000, value=900,  step=50)
        amt_tw_etf   = st.number_input("台股 ETF  (萬)", min_value=0, max_value=100_000, value=1500, step=50)
        total_amt = amt_us_stock + amt_us_etf + amt_tw_stock + amt_tw_etf
        if total_amt > 0:
            pct_us_s = amt_us_stock / total_amt * 100
            pct_us_e = amt_us_etf   / total_amt * 100
            pct_tw_s = amt_tw_stock / total_amt * 100
            pct_tw_e = amt_tw_etf   / total_amt * 100
        else:
            pct_us_s = pct_us_e = pct_tw_s = pct_tw_e = 25.0
        st.caption(f"合計：**{total_amt:,} 萬**（A₀ 以上方「初始資產」為準）")
    else:
        pct_us_stock = st.slider("美股個股 (%)", 0, 100, 0,  5)
        pct_us_etf   = st.slider("美股 ETF  (%)", 0, 100, 20, 5)
        pct_tw_stock = st.slider("台股個股 (%)", 0, 100, 30, 5)
        pct_tw_etf   = st.slider("台股 ETF  (%)", 0, 100, 50, 5)
        total_pct = pct_us_stock + pct_us_etf + pct_tw_stock + pct_tw_etf
        if total_pct != 100:
            st.caption(f"⚠️ 目前加總 {total_pct}%，建議為 100%。以下推論依比例換算。")
        scale = 100 / total_pct if total_pct > 0 else 1
        pct_us_s = pct_us_stock * scale
        pct_us_e = pct_us_etf   * scale
        pct_tw_s = pct_tw_stock * scale
        pct_tw_e = pct_tw_etf   * scale

    inferred_r = (pct_us_s * r_us_stock + pct_us_e * r_us_etf +
                  pct_tw_s * r_tw_stock + pct_tw_e * r_tw_etf) / 100
    st.caption(f"推論實質報酬率：**{inferred_r:.1f}%**（加權平均）")

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
        help="推論值由上方的資產結構加權計算",
    )
    if use_inferred_r == "依資產結構推論":
        r_pct = round(inferred_r, 1)
    else:
        r_pct = st.slider(
            "預期實質報酬率 r (%)",
            min_value=0.0,
            max_value=15.0,
            value=4.0,
            step=0.5,
            help="扣除通膨後的年化回報",
        )
    st.caption(f"名目報酬 ≈ **{r_pct + inflation_pct:.1f}%**（實質 {r_pct}% + 通膨 {inflation_pct}%）")
    medical_premium = st.slider(
        "醫療溢價 i_m (CPI + %)",
        min_value=0.0,
        max_value=4.0,
        value=1.7,
        step=0.1,
        help="台灣高齡家庭 CPI 長期高於整體，醫療保健權重較高，建議 1.5%～2%；預設 1.7% 符合實證",
    )

with st.sidebar.expander("年齡與集中度", expanded=True):
    age_start = st.number_input("起始年齡 (歲)", 25, 70, 40, 1)
    age_end = st.number_input("目標年齡 (歲)", 70, 100, 90, 1)
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
has_pension = pension_annual > 0
W0_asset = (max(0.0, W0 - pension_annual)
            if has_pension and age_start >= int(claim_age)
            else W0)

# 衍生：IWR（以「從自有資產提領」為準）
IWR = (W0_asset / A0) * 100 if A0 > 0 else 0
gk_lower = IWR * 0.8   # 繁榮護欄
gk_upper = IWR * 1.2   # 保全護欄

# ========== 主區：分頁 ==========
tab1, tab2 = st.tabs(["📊 退休規劃", "📚 教育資訊庫"])

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
    st.markdown(f"**推論實質報酬率**：**{inferred_r:.1f}%**（依上表占比與假設報酬加權平均。可於左側改為「手動設定」覆寫。）")
    st.caption("台灣參照：勞動基金近 10 年名目報酬約 6%～7%，扣除通膨後實質約 4%～5%，可作保守假設參考。")
    st.divider()

    # ── 壓力測試 ──
    st.subheader("壓力測試 (90 歲時剩餘資產)")
    n_years = max(1, age_end - age_start)
    pension_note = f" · 勞保＋勞退 {_fmt_wan(pension_annual)}/年（{claim_age} 歲起）" if has_pension else ""
    st.markdown(f"**您的設定：起始資產 {_fmt_wan(A0)} · 實質購買力 {_fmt_wan(W0)}/年{pension_note} · 實質報酬 {r_pct}% · 規劃 {n_years} 年**")
    r_low  = max(0,  r_pct - 1.5)
    r_mid  = r_pct
    r_high = min(15, r_pct + 1.5)
    _kw = dict(pension_annual=pension_annual, claim_age=int(claim_age))
    col_low = [
        run_dynamic_projection(A0, W0, r_low,  n_years, age_start, strategy="fixed", **_kw),
        run_dynamic_projection(A0, W0, r_low,  n_years, age_start, strategy="smile", **_kw),
        run_dynamic_projection(A0, W0, r_low,  n_years, age_start, strategy="gk",    **_kw),
    ]
    col_mid = [
        run_dynamic_projection(A0, W0, r_mid,  n_years, age_start, strategy="fixed", **_kw),
        run_dynamic_projection(A0, W0, r_mid,  n_years, age_start, strategy="smile", **_kw),
        run_dynamic_projection(A0, W0, r_mid,  n_years, age_start, strategy="gk",    **_kw),
    ]
    col_high = [
        run_dynamic_projection(A0, W0, r_high, n_years, age_start, strategy="fixed", **_kw),
        run_dynamic_projection(A0, W0, r_high, n_years, age_start, strategy="smile", **_kw),
        run_dynamic_projection(A0, W0, r_high, n_years, age_start, strategy="gk",    **_kw),
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
    _kw_med = dict(strategy="gk", pension_annual=pension_annual, claim_age=int(claim_age))
    res_base  = run_dynamic_projection(A0, W0, r_pct,  n_years, age_start, med_premium_pct=medical_premium,       **_kw_med)
    res_inf   = run_dynamic_projection(A0, W0, r_down, n_years, age_start, med_premium_pct=medical_premium,       **_kw_med)
    res_med   = run_dynamic_projection(A0, W0, r_pct,  n_years, age_start, med_premium_pct=medical_premium + 0.5, **_kw_med)
    res_multi = run_dynamic_projection(A0, W0, r_down, n_years, age_start, med_premium_pct=medical_premium + 0.5, **_kw_med)
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
    with st.expander("參考文獻與資料來源", expanded=False):
        st.markdown("""
- Guyton, J. T., & Klinger, W. J. (2006). *Decision rules and portfolio management for retirees.* Journal of Financial Planning.
- Blanchett, D. (2014). *Exploring the retirement consumption puzzle.* Journal of Financial Planning.
- Morningstar: State of Retirement Income 2025 (2025/12/3).
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

    st.divider()
    st.caption("資料來源：① 2025年退休規劃彙整表 ② 2026全球與台灣退休安全策略手冊 ③ Morningstar State of Retirement Income 2025 ④ 財政部2025綜所稅率表 ⑤ 勞動部勞保局 ⑥ 主計總處高齡家庭CPI ⑦ 費城聯準會SPF 2025Q4")
