# app.py
# Streamlit Mortgage + SDLT + Affordability + Scenario Compare + JSON Save/Load
#
# Run:
#   pip install streamlit pandas numpy
#   streamlit run app.py
#
# Notes:
# - “Client-side profile” is implemented as:
#   (1) Download JSON (user keeps file) + (2) Upload JSON to restore.
# - UK PAYE take-home: uses configurable 2025/26-style parameters + £100k personal allowance taper.
#   Update the constants below to match the exact tax year you want.

from __future__ import annotations

import json
import copy
import string
from dataclasses import dataclass, asdict
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# --------------------- Input helpers ---------------------
def _format_commas(value: float, decimals: int = 0) -> str:
    try:
        if decimals <= 0:
            return f"{float(value):,.0f}"
        return f"{float(value):,.{decimals}f}".replace(",", "_TMP_").replace("_TMP_", ",")
    except Exception:
        return str(value)

def _parse_commad_number(s: str) -> Optional[float]:
    """Parse numbers that may include commas, spaces, or common currency symbols."""
    if s is None:
        return None
    t = str(s).strip()
    if t == "":
        return None
    # allow things like £, commas, spaces, underscores
    for ch in ["£", "$", "€"]:
        t = t.replace(ch, "")
    t = t.replace(",", "").replace(" ", "").replace("_", "")
    # allow trailing/leading plus/minus
    try:
        return float(t)
    except Exception:
        return None

def comma_number_input(label: str, current_value: float, key: str, decimals: int = 0, help: str | None = None) -> float:
    """
    Text input that:
      - shows the number with comma separators
      - accepts commas / currency symbols in user input
      - returns the parsed float (or keeps current_value if invalid)

    Note: we *do not* write back to st.session_state[key] after the widget is created,
    because Streamlit forbids mutating a widget's key in the same run after instantiation.
    """
    if key not in st.session_state:
        st.session_state[key] = _format_commas(current_value, decimals=decimals)

    s = st.text_input(label, key=key, help=help)
    parsed = _parse_commad_number(s)

    if parsed is None:
        st.caption("Enter a number (commas allowed).")
        return float(current_value)

    return float(parsed)





# -----------------------------
# Display formatting helpers
# -----------------------------

def fmt0(x: float | int) -> str:
    """Comma-separated integer formatting (no currency symbol)."""
    return f"{int(round(float(x))):,}"

def fmt_pct(x: float, decimals: int = 1) -> str:
    return f"{float(x):.{decimals}f}%"
# -----------------------------
# Configurable constants (update for exact tax year)
# -----------------------------

# Personal allowance taper (assumed)
PA_STANDARD = 12_570.0
PA_TAPER_START = 100_000.0
PA_ZERO_AT = 125_140.0  # 100k + 2*12,570

# Income Tax bands (England/Wales/NI style) - placeholders; update as needed.
# Bands defined as (upper_limit, rate). Upper limit is taxable income upper bound for that band.
# Example placeholders: basic up to 37,700 @20%, higher up to 125,140 @40%, additional above @45%.
IT_BANDS = [
    (37_700.0, 0.20),
    (125_140.0, 0.40),
    (float("inf"), 0.45),
]

# Employee NIC (Class 1) thresholds/rates - placeholders; update as needed.
# Example placeholders: 0% below PT, 8% between PT and UEL, 2% above UEL (rates may differ by year).
NIC_PRIMARY_THRESHOLD_ANNUAL = 12_570.0
NIC_UPPER_EARNINGS_LIMIT_ANNUAL = 50_270.0
NIC_MAIN_RATE = 0.08
NIC_ADDITIONAL_RATE = 0.02

# SDLT bands (England/N.I.) - define standard residential bands (placeholders; update as needed).
# Format: (upper_limit, rate) applied marginally.
SDLT_STANDARD_BANDS = [
    # England/N.I. residential SDLT bands (from 1 April 2025)
    # Applied marginally: (upper_limit, rate)
    (125_000.0, 0.00),
    (250_000.0, 0.02),
    (925_000.0, 0.05),
    (1_500_000.0, 0.10),
    (float("inf"), 0.12),
]

# First-time buyer relief rule (England/N.I.)
# Relief applies only if consideration <= £500,000.
# 0% up to £300,000; 5% on £300,001–£500,000. (England/N.I.)
FTB_MAX_PRICE = 500_000.0
FTB_BANDS = [
    (300_000.0, 0.00),
    (500_000.0, 0.05),
]

# Additional property higher rates bump (England/N.I.)
# Modelled as +5% added to each marginal band rate.
ADDITIONAL_RATE_BUMP = 0.05


# -----------------------------
# Data models
# -----------------------------

SCHEMA_VERSION = "1.0.0"


@dataclass
class LumpSum:
    month: int
    amount_gbp: float


@dataclass
class OverpaymentPlan:
    extra_monthly_gbp: float = 0.0
    lump_sums: List[LumpSum] = None

    def __post_init__(self):
        if self.lump_sums is None:
            self.lump_sums = []


@dataclass
class ScenarioConfig:
    name: str = "A"
    property_price_gbp: float = 500_000.0
    deposit_pct: float = 10.0
    annual_interest_rate: float = 0.039
    term_years: int = 30
    annual_property_growth: float = 0.03
    fees_gbp: Dict[str, float] = None
    sdlt_mode: str = "FIRST_TIME_BUYER"  # FIRST_TIME_BUYER | HOME_MOVER | ADDITIONAL_PROPERTY
    overpayment: OverpaymentPlan = None

    def __post_init__(self):
        if self.fees_gbp is None:
            self.fees_gbp = {"lender_fee": 1000.0, "legal": 2000.0}
        if self.overpayment is None:
            self.overpayment = OverpaymentPlan()


@dataclass
class AffordabilityConfig:
    gross_salary_annual_gbp: float = 50_000.0
    other_income_monthly_gbp: float = 0.0
    existing_debts_monthly_gbp: float = 0.0
    pension_pct: float = 5.0  # salary sacrifice %
    # Budget line-items (user can edit)
    food_monthly_gbp: float = 400.0
    utilities_monthly_gbp: float = 250.0
    transport_monthly_gbp: float = 250.0
    other_needs_monthly_gbp: float = 400.0
    wants_monthly_gbp: float = 600.0
    savings_monthly_gbp: float = 400.0


@dataclass
class AppState:
    schema_version: str
    saved_at: str  # ISO date
    global_settings: Dict[str, Any]
    affordability: AffordabilityConfig
    scenarios: List[ScenarioConfig]


# -----------------------------
# Helper math
# -----------------------------

def pmt(rate_per_period: float, nper: int, pv: float) -> float:
    """Excel-like PMT (returns positive payment)."""
    if nper <= 0:
        return 0.0
    if abs(rate_per_period) < 1e-12:
        return pv / nper
    return (rate_per_period * pv) / (1 - (1 + rate_per_period) ** (-nper))


def compute_personal_allowance(gross_adj: float) -> float:
    """2025/26-style: taper above £100k. Reduce £1 per £2 over taper start, to zero."""
    if gross_adj <= PA_TAPER_START:
        return PA_STANDARD
    reduction = (gross_adj - PA_TAPER_START) / 2.0
    pa = max(PA_STANDARD - reduction, 0.0)
    return pa


def income_tax_annual(gross_adj: float) -> float:
    pa = compute_personal_allowance(gross_adj)
    taxable = max(gross_adj - pa, 0.0)

    tax = 0.0
    lower = 0.0
    for upper, rate in IT_BANDS:
        band_amount = max(min(taxable, upper) - lower, 0.0)
        if band_amount <= 0:
            break
        tax += band_amount * rate
        lower = upper
    return tax


def employee_nic_annual(gross_adj: float) -> float:
    if gross_adj <= NIC_PRIMARY_THRESHOLD_ANNUAL:
        return 0.0
    nic = 0.0
    main_band = min(gross_adj, NIC_UPPER_EARNINGS_LIMIT_ANNUAL) - NIC_PRIMARY_THRESHOLD_ANNUAL
    if main_band > 0:
        nic += main_band * NIC_MAIN_RATE
    if gross_adj > NIC_UPPER_EARNINGS_LIMIT_ANNUAL:
        nic += (gross_adj - NIC_UPPER_EARNINGS_LIMIT_ANNUAL) * NIC_ADDITIONAL_RATE
    return max(nic, 0.0)


def estimate_net_monthly(aff: AffordabilityConfig) -> Dict[str, float]:
    """Gross -> (salary sacrifice) -> tax & NI -> net monthly."""
    gross = aff.gross_salary_annual_gbp
    pension_pct = max(min(aff.pension_pct, 100.0), 0.0) / 100.0
    gross_adj = gross * (1.0 - pension_pct)  # salary sacrifice reduces taxable + NI pay

    tax = income_tax_annual(gross_adj)
    nic = employee_nic_annual(gross_adj)

    net_annual = max(gross_adj - tax - nic, 0.0)
    net_monthly = net_annual / 12.0

    return {
        "gross_annual": gross,
        "gross_adj_annual": gross_adj,
        "pension_annual": gross - gross_adj,
        "tax_annual": tax,
        "nic_annual": nic,
        "net_annual": net_annual,
        "net_monthly": net_monthly,
    }


# -----------------------------
# SDLT
# -----------------------------

def marginal_tax(amount: float, bands: List[Tuple[float, float]]) -> float:
    """Marginal calculation over bands."""
    tax = 0.0
    lower = 0.0
    for upper, rate in bands:
        slice_amt = max(min(amount, upper) - lower, 0.0)
        if slice_amt <= 0:
            break
        tax += slice_amt * rate
        lower = upper
    return tax


def bump_bands(bands: List[Tuple[float, float]], bump: float) -> List[Tuple[float, float]]:
    return [(upper, rate + bump) for (upper, rate) in bands]

def sdlt(property_price: float, mode: str) -> Dict[str, float]:
    """
    England/N.I. SDLT:
    - HOME_MOVER: standard bands
    - FIRST_TIME_BUYER: relief if price <= 500k using FTB bands; else standard (relief fully lost)
    - ADDITIONAL_PROPERTY: higher rates (standard rates + 5% on each band)
    """
    mode = (mode or "HOME_MOVER").upper()

    if mode == "FIRST_TIME_BUYER":
        if property_price <= FTB_MAX_PRICE:
            total = marginal_tax(property_price, FTB_BANDS)
            return {
                "sdlt_total": total,
                "ftb_relief_applied": True,
                "mode_used": "FIRST_TIME_BUYER",
            }
        # Relief is lost entirely above 500k -> standard rates on full price
        total = marginal_tax(property_price, SDLT_STANDARD_BANDS)
        return {
            "sdlt_total": total,
            "ftb_relief_applied": False,
            "mode_used": "STANDARD (FTB ineligible >500k)",
        }

    if mode == "ADDITIONAL_PROPERTY":
        higher = bump_bands(SDLT_STANDARD_BANDS, ADDITIONAL_RATE_BUMP)
        total = marginal_tax(property_price, higher)
        return {
            "sdlt_total": total,
            "ftb_relief_applied": False,
            "mode_used": "ADDITIONAL_PROPERTY",
        }

    total = marginal_tax(property_price, SDLT_STANDARD_BANDS)
    return {
        "sdlt_total": total,
        "ftb_relief_applied": False,
        "mode_used": "HOME_MOVER",
    }


# -----------------------------
# Mortgage engine (repayment) + overpayments
# -----------------------------
# -----------------------------

def build_schedule(
    scenario: ScenarioConfig,
    payments_per_year: int = 12,
) -> Dict[str, Any]:
    price = scenario.property_price_gbp
    deposit = price * (scenario.deposit_pct / 100.0)
    loan = max(price - deposit, 0.0)

    r = scenario.annual_interest_rate / payments_per_year
    n = scenario.term_years * payments_per_year
    scheduled_payment = pmt(r, n, loan)

    # Pre-index lump sums by month for quick lookup
    lumps = {}
    for ls in (scenario.overpayment.lump_sums or []):
        if ls.month >= 1:
            lumps.setdefault(int(ls.month), 0.0)
            lumps[int(ls.month)] += float(ls.amount_gbp)

    balance = loan
    prop_val = price

    rows = []
    cum_interest = 0.0
    cum_principal = 0.0
    cum_overpay = 0.0

    for m in range(1, n + 1):
        interest = balance * r
        principal = max(scheduled_payment - interest, 0.0)

        extra = max(float(scenario.overpayment.extra_monthly_gbp or 0.0), 0.0)
        lump = float(lumps.get(m, 0.0))
        overpay = extra + lump

        # Pay down
        total_principal_reduction = principal + overpay
        new_balance = max(balance - total_principal_reduction, 0.0)

        # If last payment overshoots, adjust effective payment down (optional)
        # For reporting, we keep scheduled payment and separate overpay; last row may show balance->0.
        balance = new_balance

        # Property growth monthly
        prop_val *= (1.0 + scenario.annual_property_growth / payments_per_year)
        equity = prop_val - balance

        cum_interest += interest
        cum_principal += principal
        cum_overpay += overpay

        rows.append(
            {
                "month": m,
                "year": int(np.ceil(m / payments_per_year)),
                "scheduled_payment": scheduled_payment,
                "interest": interest,
                "principal": principal,
                "overpayment": overpay,
                "balance": balance,
                "property_value": prop_val,
                "equity": equity,
                "cum_interest": cum_interest,
                "cum_overpayment": cum_overpay,
            }
        )

        if balance <= 0.0 + 1e-6:
            break

    df = pd.DataFrame(rows)

    return {
        "loan_gbp": loan,
        "deposit_gbp": deposit,
        "scheduled_payment_monthly_gbp": scheduled_payment,
        "payoff_month": int(df["month"].iloc[-1]) if len(df) else 0,
        "total_interest_gbp": float(df["interest"].sum()) if len(df) else 0.0,
        "total_overpayment_gbp": float(df["overpayment"].sum()) if len(df) else 0.0,
        "schedule": df,
    }


def baseline_interest_without_overpay(scenario: ScenarioConfig) -> float:
    """Compute interest if overpayment plan is zeroed (for interest-saved comparison)."""
    tmp = ScenarioConfig(**{**asdict(scenario), "overpayment": asdict(OverpaymentPlan(0.0, []))})
    # Rehydrate nested dataclasses properly
    tmp.overpayment = OverpaymentPlan(extra_monthly_gbp=0.0, lump_sums=[])
    out = build_schedule(tmp)
    return float(out["total_interest_gbp"])


# -----------------------------
# JSON save/load (user-managed)
# -----------------------------

def state_to_json(state: AppState) -> str:
    def default(o):
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    return json.dumps(asdict(state), indent=2, default=default)


def json_to_state(raw: str) -> AppState:
    data = json.loads(raw)
    if data.get("schema_version") != SCHEMA_VERSION:
        # Basic forward-compat: accept older schema if fields exist; for MVP, just warn.
        st.warning(f"Profile schema version mismatch: {data.get('schema_version')} vs expected {SCHEMA_VERSION}")

    aff = AffordabilityConfig(**data["affordability"])
    scenarios = []
    for sc in data["scenarios"]:
        # OverpaymentPlan and LumpSum rehydration
        op = sc.get("overpayment") or {}
        lump_objs = [LumpSum(**ls) for ls in (op.get("lump_sums") or [])]
        op_obj = OverpaymentPlan(extra_monthly_gbp=op.get("extra_monthly_gbp", 0.0), lump_sums=lump_objs)

        sc_obj = ScenarioConfig(
            name=sc.get("name", "A"),
            property_price_gbp=sc.get("property_price_gbp", 0.0),
            deposit_pct=sc.get("deposit_pct", 0.0),
            annual_interest_rate=sc.get("annual_interest_rate", 0.0),
            term_years=sc.get("term_years", 0),
            annual_property_growth=sc.get("annual_property_growth", 0.0),
            fees_gbp=sc.get("fees_gbp", {}),
            sdlt_mode=sc.get("sdlt_mode", "HOME_MOVER"),
            overpayment=op_obj,
        )
        scenarios.append(sc_obj)

    return AppState(
        schema_version=data.get("schema_version", SCHEMA_VERSION),
        saved_at=data.get("saved_at", date.today().isoformat()),
        global_settings=data.get("global_settings", {}),
        affordability=aff,
        scenarios=scenarios,
    )



def clear_input_widget_keys():
    """Clear Streamlit widget keys so loaded JSON values don't get overwritten by stale widget state."""
    keys = list(st.session_state.keys())
    prefixes = (
        # affordability
        "gross_salary_annual",
        # scenario inputs
        "price_", "dep_", "rate_pct_", "term_", "growth_pct_",
        "fee_l_", "fee_g_", "op_m_",
        # lump sum editors
        "ls_m_", "ls_a_", "ls_r_", "ls_add_",
        # scenario name/sdlt
        "name_", "sdlt_",
        # comparison selectors
        "Left scenario", "Right scenario",
    )
    exact_keys = {"add_from"}  # add-scenario selector
    for k in keys:
        if k in exact_keys:
            del st.session_state[k]
            continue
        if any(k.startswith(p) for p in prefixes):
            del st.session_state[k]

def ensure_default_session_state():
    if "affordability" not in st.session_state:
        st.session_state.affordability = AffordabilityConfig()
    if "scenarios" not in st.session_state:
        st.session_state.scenarios = [ScenarioConfig(name="A"), ScenarioConfig(name="B")]
        st.session_state.scenarios[1].annual_interest_rate = 0.049  # example alternate scenario
    if "pending_delete_idx" not in st.session_state:
        st.session_state.pending_delete_idx = None
    if "global_settings" not in st.session_state:
        st.session_state.global_settings = {
            "tax_year": "2025/26",
            "sdlt_effective_date": "2025-04-06",
            "payments_per_year": 12,
        }


def next_scenario_name(existing: List[str]) -> str:
    """Generate a human-friendly scenario name (A..Z, then S27, S28...)."""
    existing_set = set(existing)
    for ch in string.ascii_uppercase:
        if ch not in existing_set:
            return ch
    i = 1
    while True:
        cand = f"S{i}"
        if cand not in existing_set:
            return cand
        i += 1




# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="Mortgage Compare", layout="wide")
ensure_default_session_state()

st.title("Mortgage Model (Compare + SDLT + Overpayments + Affordability)")

# --- Profile Save/Load ---
with st.expander("Profile: Save / Load (JSON)", expanded=False):
    col_s, col_l = st.columns(2)

    with col_s:
        if st.button("Prepare JSON for download"):
            state = AppState(
                schema_version=SCHEMA_VERSION,
                saved_at=date.today().isoformat(),
                global_settings=st.session_state.global_settings,
                affordability=st.session_state.affordability,
                scenarios=st.session_state.scenarios,
            )
            st.session_state._profile_json = state_to_json(state)

        profile_json = st.session_state.get("_profile_json", "")
        st.download_button(
            "Download profile.json",
            data=profile_json.encode("utf-8") if profile_json else b"",
            file_name="mortgage_profile.json",
            mime="application/json",
            disabled=not bool(profile_json),
        )
        st.caption("This saves to a file on your device. The app does not store your profile server-side.")

    with col_l:
        up = st.file_uploader("Upload a profile JSON to restore", type=["json"])
        if up is not None:
            raw = up.read().decode("utf-8")
            loaded = json_to_state(raw)
            st.session_state.global_settings = loaded.global_settings
            st.session_state.affordability = loaded.affordability
            st.session_state.scenarios = loaded.scenarios
            st.success("Profile loaded. (Tip: collapse and continue.)")



            clear_input_widget_keys()
            st.rerun()
# --- Sidebar: Global + Affordability ---
with st.sidebar:
    st.header("Global")
    st.session_state.global_settings["tax_year"] = st.text_input(
        "Tax year label", st.session_state.global_settings.get("tax_year", "2025/26")
    )
    st.session_state.global_settings["payments_per_year"] = st.selectbox(
        "Payments per year", [12], index=0
    )

    st.divider()
    st.header("Affordability (Gross → Net)")
    aff: AffordabilityConfig = st.session_state.affordability

    aff.gross_salary_annual_gbp = comma_number_input("Gross salary (annual)", float(aff.gross_salary_annual_gbp), key="gross_salary_annual", decimals=0)
    aff.pension_pct = st.slider("Pension (salary sacrifice, %)", 0.0, 30.0, float(aff.pension_pct), 0.5)
    aff.other_income_monthly_gbp = st.number_input("Other income (monthly)", 0.0, value=float(aff.other_income_monthly_gbp), step=100.0)
    aff.existing_debts_monthly_gbp = st.number_input("Existing debts (monthly)", 0.0, value=float(aff.existing_debts_monthly_gbp), step=50.0)

    st.subheader("Budget (monthly)")
    aff.food_monthly_gbp = st.number_input("Food", 0.0, value=float(aff.food_monthly_gbp), step=25.0)
    aff.utilities_monthly_gbp = st.number_input("Utilities", 0.0, value=float(aff.utilities_monthly_gbp), step=25.0)
    aff.transport_monthly_gbp = st.number_input("Transport", 0.0, value=float(aff.transport_monthly_gbp), step=25.0)
    aff.other_needs_monthly_gbp = st.number_input("Other needs", 0.0, value=float(aff.other_needs_monthly_gbp), step=25.0)
    aff.wants_monthly_gbp = st.number_input("Wants", 0.0, value=float(aff.wants_monthly_gbp), step=25.0)
    aff.savings_monthly_gbp = st.number_input("Savings", 0.0, value=float(aff.savings_monthly_gbp), step=25.0)


# --- Main: Scenario tabs ---
scenarios: List[ScenarioConfig] = st.session_state.scenarios
tab_names = [f"Scenario {sc.name}" for sc in scenarios]
tabs = st.tabs(tab_names + ["➕ Add scenario"])

for idx, sc in enumerate(scenarios):
    with tabs[idx]:
        left, right = st.columns([1, 1])

        with left:
            st.subheader("Inputs")
            name_col, copy_col, del_col = st.columns([1, 0.32, 0.23])
            with name_col:
                sc.name = st.text_input("Scenario name", sc.name, key=f"name_{idx}")
            with copy_col:
                if st.button("Copy scenario", key=f"clone_{idx}"):
                    new_sc = copy.deepcopy(sc)
                    new_sc.name = next_scenario_name([s.name for s in st.session_state.scenarios])
                    st.session_state.scenarios.append(new_sc)
                    st.rerun()
            with del_col:
                if len(st.session_state.scenarios) <= 1:
                    st.caption("Cannot delete last")
                else:
                    if st.button("Delete", key=f"del_{idx}", help="Delete this scenario"):
                        st.session_state.pending_delete_idx = idx

            # Delete confirmation (two-step)
            if st.session_state.get("pending_delete_idx") == idx:
                st.warning("Delete this scenario? This cannot be undone.")
                c1, c2 = st.columns([0.5, 0.5])
                with c1:
                    if st.button("Confirm delete", key=f"confirm_del_{idx}", type="primary"):
                        st.session_state.scenarios.pop(idx)
                        st.session_state.pending_delete_idx = None
                        st.rerun()
                with c2:
                    if st.button("Cancel", key=f"cancel_del_{idx}"):
                        st.session_state.pending_delete_idx = None
                        st.rerun()

            sc.property_price_gbp = comma_number_input("Property price", float(sc.property_price_gbp), key=f"price_{idx}", decimals=0)
            sc.deposit_pct = st.slider("Deposit (%)", 0.0, 100.0, float(sc.deposit_pct), 0.5, key=f"dep_{idx}")
            rate_pct = st.number_input("Interest rate (annual, %)", 0.0, 25.0, value=float(sc.annual_interest_rate)*100.0, step=0.05, format="%.2f", key=f"rate_pct_{idx}")
            sc.annual_interest_rate = float(rate_pct) / 100.0
            sc.term_years = st.number_input("Term (years)", 1, value=int(sc.term_years), step=1, key=f"term_{idx}")
            growth_pct = st.number_input("Property growth (annual, %)", -20.0, 20.0, value=float(sc.annual_property_growth)*100.0, step=0.1, format="%.2f", key=f"growth_pct_{idx}")
            sc.annual_property_growth = float(growth_pct) / 100.0

            st.subheader("Stamp Duty")
            sc.sdlt_mode = st.selectbox(
                "SDLT mode",
                ["FIRST_TIME_BUYER", "HOME_MOVER", "ADDITIONAL_PROPERTY"],
                index=["FIRST_TIME_BUYER", "HOME_MOVER", "ADDITIONAL_PROPERTY"].index(sc.sdlt_mode)
                if sc.sdlt_mode in ["FIRST_TIME_BUYER", "HOME_MOVER", "ADDITIONAL_PROPERTY"]
                else 1,
                key=f"sdlt_{idx}",
            )
            if sc.sdlt_mode == "FIRST_TIME_BUYER" and sc.property_price_gbp > 500_000:
                st.warning("First-time buyer relief does not apply above 500,000. Standard rates will be used.")

            st.subheader("Fees (upfront)")
            if sc.fees_gbp is None:
                sc.fees_gbp = {}
            sc.fees_gbp["lender_fee"] = st.number_input("Lender fee", 0.0, value=float(sc.fees_gbp.get("lender_fee", 0.0)), step=100.0, key=f"fee_l_{idx}")
            sc.fees_gbp["legal"] = st.number_input("Legal", 0.0, value=float(sc.fees_gbp.get("legal", 0.0)), step=100.0, key=f"fee_g_{idx}")

            st.subheader("Overpayments")
            sc.overpayment.extra_monthly_gbp = st.number_input(
                "Extra monthly overpayment", 0.0, value=float(sc.overpayment.extra_monthly_gbp), step=50.0, key=f"op_m_{idx}"
            )

            # Lump sums editor (simple)
            with st.expander("Lump sums (month → amount)", expanded=False):
                # Display current lump sums
                if sc.overpayment.lump_sums is None:
                    sc.overpayment.lump_sums = []
                for j, ls in enumerate(list(sc.overpayment.lump_sums)):
                    c1, c2, c3 = st.columns([1, 1, 0.5])
                    with c1:
                        m = st.number_input("Month", 1, value=int(ls.month), step=1, key=f"ls_m_{idx}_{j}")
                    with c2:
                        a = st.number_input("Amount", 0.0, value=float(ls.amount_gbp), step=100.0, key=f"ls_a_{idx}_{j}")
                    with c3:
                        if st.button("Remove", key=f"ls_r_{idx}_{j}"):
                            sc.overpayment.lump_sums.pop(j)
                            st.rerun()
                    ls.month, ls.amount_gbp = int(m), float(a)

                if st.button("Add lump sum", key=f"ls_add_{idx}"):
                    sc.overpayment.lump_sums.append(LumpSum(month=12, amount_gbp=5_000.0))
                    st.rerun()

        with right:
            # Compute outputs
            sd = sdlt(sc.property_price_gbp, sc.sdlt_mode)
            out = build_schedule(sc, payments_per_year=st.session_state.global_settings["payments_per_year"])

            upfront_fees = sum(float(v) for v in (sc.fees_gbp or {}).values())
            upfront_cash = out["deposit_gbp"] + sd["sdlt_total"] + upfront_fees

            baseline_interest = baseline_interest_without_overpay(sc)
            interest_saved = max(baseline_interest - out["total_interest_gbp"], 0.0)

            st.subheader("Key metrics")
            # Show deposit explicitly (in £) instead of it being hidden inside "Upfront cash"
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Loan", fmt0(out["loan_gbp"]))
            m2.metric("Deposit", fmt0(out["deposit_gbp"]))
            m3.metric("Monthly payment (scheduled)", fmt0(out["scheduled_payment_monthly_gbp"]))
            m4.metric("SDLT", fmt0(sd["sdlt_total"]))
            m5.metric("Upfront cash", fmt0(upfront_cash))

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Payoff month", fmt0(out["payoff_month"]))
            m6.metric("Total interest", fmt0(out["total_interest_gbp"]))
            m7.metric("Interest saved vs baseline", fmt0(interest_saved))
            m8.metric("Total overpayments", fmt0(out["total_overpayment_gbp"]))

            df = out["schedule"].copy()
            if len(df) > 0:
                st.line_chart(df.set_index("month")[["balance", "equity", "property_value"]])

            st.subheader("Schedule")
            display_df = df.copy()
            # Keep numeric df for charts/CSV, but show formatted integers for readability.
            for c in display_df.columns:
                if pd.api.types.is_numeric_dtype(display_df[c]):
                    display_df[c] = display_df[c].map(fmt0)
            st.dataframe(display_df, use_container_width=True, height=280)

            st.download_button(
                f"Download {sc.name} schedule (CSV)",
                df.to_csv(index=False).encode("utf-8"),
                file_name=f"schedule_{sc.name}.csv",
                mime="text/csv",
                key=f"dl_{idx}",
            )

with tabs[-1]:
    st.subheader("Add scenario")
    create_from = ["Blank"] + [sc.name for sc in st.session_state.scenarios]
    base = st.selectbox("Create from", create_from, index=0, key="add_from")
    if st.button("Create scenario"):
        next_name = next_scenario_name([s.name for s in st.session_state.scenarios])
        if base == "Blank":
            st.session_state.scenarios.append(ScenarioConfig(name=next_name))
        else:
            src = next(sc for sc in st.session_state.scenarios if sc.name == base)
            new_sc = copy.deepcopy(src)
            new_sc.name = next_name
            st.session_state.scenarios.append(new_sc)
        st.rerun()

# --- Comparison section (A/B overlay) ---
st.divider()
st.header("Comparison")

if len(scenarios) >= 2:
    names = [sc.name for sc in scenarios]
    c1, c2 = st.columns(2)
    with c1:
        s_left = st.selectbox("Left scenario", names, index=0)
    with c2:
        s_right = st.selectbox("Right scenario", names, index=1)

    left_sc = next(sc for sc in scenarios if sc.name == s_left)
    right_sc = next(sc for sc in scenarios if sc.name == s_right)

    left_out = build_schedule(left_sc)
    right_out = build_schedule(right_sc)

    # Align schedules by month
    ldf = left_out["schedule"][["month", "balance", "equity", "property_value", "cum_interest"]].rename(
        columns=lambda c: f"{c}_{s_left}" if c != "month" else c
    )
    rdf = right_out["schedule"][["month", "balance", "equity", "property_value", "cum_interest"]].rename(
        columns=lambda c: f"{c}_{s_right}" if c != "month" else c
    )
    merged = pd.merge(ldf, rdf, on="month", how="outer").sort_values("month")

    st.subheader("Overlay charts")
    st.line_chart(merged.set_index("month")[[f"balance_{s_left}", f"balance_{s_right}"]])
    st.line_chart(merged.set_index("month")[[f"equity_{s_left}", f"equity_{s_right}"]])

    st.subheader("Delta summary")
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Δ Monthly payment", fmt0(left_out["scheduled_payment_monthly_gbp"] - right_out["scheduled_payment_monthly_gbp"]))
    colB.metric("Δ Total interest", fmt0(left_out["total_interest_gbp"] - right_out["total_interest_gbp"]))
    colC.metric("Δ Payoff month", fmt0(left_out["payoff_month"] - right_out["payoff_month"]))
    colD.metric("Δ Loan", fmt0(left_out["loan_gbp"] - right_out["loan_gbp"]))
else:
    st.info("Add at least two scenarios to compare.")

# --- Affordability outputs ---
st.divider()
st.header("Affordability")

aff: AffordabilityConfig = st.session_state.affordability
net = estimate_net_monthly(aff)

# choose a “primary” scenario for affordability ratios: scenario A by default
primary = scenarios[0] if scenarios else ScenarioConfig(name="A")
primary_out = build_schedule(primary)
mortgage_payment = float(primary_out["scheduled_payment_monthly_gbp"])
other_income = float(aff.other_income_monthly_gbp)
debts = float(aff.existing_debts_monthly_gbp)

budget_total = (
    aff.food_monthly_gbp
    + aff.utilities_monthly_gbp
    + aff.transport_monthly_gbp
    + aff.other_needs_monthly_gbp
    + aff.wants_monthly_gbp
    + aff.savings_monthly_gbp
)

net_monthly_total = net["net_monthly"] + other_income
after_housing = net_monthly_total - mortgage_payment - aff.utilities_monthly_gbp - debts
after_budget = net_monthly_total - mortgage_payment - debts - budget_total

g1, g2, g3, g4 = st.columns(4)
g1.metric("Net pay (est.) / month", fmt0(net["net_monthly"]))
g2.metric("Mortgage as % gross", fmt_pct(mortgage_payment*12/net["gross_annual"]*100))
g3.metric("Mortgage as % net", fmt_pct(mortgage_payment/net_monthly_total*100))
g4.metric("Buffer after housing+debts", fmt0(after_housing))

st.caption(
    "Net pay estimate uses configurable 2025/26-style Income Tax + employee NI + £100k personal allowance taper, "
    "with pension treated as salary sacrifice. Update constants in app.py for exact thresholds/rates."
)

st.subheader("Budget view (monthly)")
bdf = pd.DataFrame(
    [
        ["Net income (est.)", net_monthly_total],
        ["Mortgage (scheduled)", -mortgage_payment],
        ["Existing debts", -debts],
        ["Food", -aff.food_monthly_gbp],
        ["Utilities", -aff.utilities_monthly_gbp],
        ["Transport", -aff.transport_monthly_gbp],
        ["Other needs", -aff.other_needs_monthly_gbp],
        ["Wants", -aff.wants_monthly_gbp],
        ["Savings", -aff.savings_monthly_gbp],
        ["Remaining after budget", after_budget],
    ],
    columns=["Item", "Amount"],
)

bdf["Amount"] = bdf["Amount"].map(fmt0)

st.dataframe(bdf, use_container_width=True, height=320)
