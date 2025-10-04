import streamlit as st
import requests
import pandas as pd
import streamlit.components.v1 as components

# ==============================
# Page config + CSS styles
# ==============================
st.set_page_config(page_title="Lifestyle Disease Risk Predictor", page_icon="🩺", layout="wide")

PRIMARY = "#22d3ee"  # cyan
ACCENT = "#a78bfa"   # purple
GOOD   = "#22c55e"   # green
WARN   = "#f59e0b"   # amber
DANGER = "#ef4444"   # red

st.markdown(
    f"""
    <style>
      .app-header h1 {{
        font-size: 2.2rem; font-weight: 900; letter-spacing:.3px; margin-bottom:.25rem;
      }}
      .sub {{ opacity:.8; margin-bottom:1rem; }}
      .card {{
        border-radius: 14px; padding: 18px 18px; background: #0e1117; border: 1px solid #222633;
      }}
      .pill {{ display:inline-block; padding:6px 12px; border-radius:999px; font-weight:700; color:#fff; }}
      .pill-blue   {{ background:{PRIMARY}; }}
      .pill-green  {{ background:{GOOD}; }}
      .pill-amber  {{ background:{WARN}; }}
      .pill-red    {{ background:{DANGER}; }}
      .result {{
        border-radius: 16px; padding: 20px; border:1px solid #1f2937; background: linear-gradient(135deg, rgba(34,211,238,.10), rgba(167,139,250,.10));
      }}
      .metric {{ font-size: 1.8rem; font-weight: 900; }}
      .muted  {{ opacity:.7; }}
      .section-title {{ font-weight:800; font-size:1.05rem; margin:.25rem 0 .5rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-header"><h1>🩺 Lifestyle Disease Risk Predictor</h1></div>', unsafe_allow_html=True)
st.caption("Only the inputs shown are used by the current reduced model.")

# ---- Step state (0=Vitals, 1=Daily Habits)
if "step" not in st.session_state:
    st.session_state.step = 0

# ==============================
# Ranges for sliders (keep within model scope)
# ==============================
R = {
    "bmi": (10, 50, 22),
    "blood_pressure": (80, 200, 120),
    "cholesterol": (100, 400, 180),
    "sleep_hours": (0, 12, 7),
    "physical_activity": (0, 30, 5),
    "work_hours": (0, 16, 8),
    "calorie_intake": (1000, 5000, 2000),
    "sugar_intake": (0, 300, 50),
    "water_intake": (0, 10, 2),
}

# Features the reduced XGBoost expects
SELECTED_FEATURES = [
    "sugar_intake","bmi","cholesterol","sleep_hours",
    "physical_activity","work_hours","blood_pressure",
    "calorie_intake","water_intake"
]

# Helper: safely get a value from session_state with sensible defaults
def _d(key: str):
    if key in st.session_state and st.session_state.get(key) is not None:
        return st.session_state.get(key)
    # fall back to the default (3rd item) from R if available, else 0
    if key in R and len(R[key]) >= 3:
        return R[key][2]
    return 0

submit = False

# --- Simple stepper header ---
step_labels = ["🩺 Vitals", "🧭 Daily Habits"]
cols = st.columns(2)
for i, c in enumerate(cols):
    with c:
        active = (st.session_state.step == i)
        st.markdown(
            f"<div class='card' style='text-align:center; border:{'2px solid #22d3ee' if active else '1px solid #222633'}'>"
            f"<div style='font-weight:800'>{step_labels[i]}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("---")

# ---------- Step 0: Vitals ----------
if st.session_state.step == 0:
    st.markdown("<div class='section-title'>Vitals</div>", unsafe_allow_html=True)

    # Full-width sliders (no extra empty column)
    st.session_state.bmi = st.slider("BMI", *R["bmi"])                          # used
    st.session_state.blood_pressure = st.slider("Blood Pressure (systolic mmHg)", *R["blood_pressure"])  # used
    st.session_state.cholesterol = st.slider("Cholesterol (mg/dL)", *R["cholesterol"])                  # used

    # Only Next → button (right aligned)
    _, nxt = st.columns([6,1])
    with nxt:
        if st.button("Next →", use_container_width=True):
            st.session_state.step = 1
            st.rerun()

# ---------- Step 1: Daily Habits + Predict ----------
elif st.session_state.step == 1:
    with st.form("risk-form-final"):
        st.markdown("<div class='section-title'>Daily Habits</div>", unsafe_allow_html=True)
        h1, h2 = st.columns(2)
        with h1:
            st.session_state.physical_activity = st.slider("Physical Activity (hours/week)", *R["physical_activity"])   # used
            st.session_state.work_hours = st.slider("Work Hours (per day)", *R["work_hours"])                         # used
            st.session_state.calorie_intake = st.slider("Daily Calorie Intake", *R["calorie_intake"])                 # used
        with h2:
            st.session_state.sugar_intake = st.slider("Daily Sugar Intake (grams)", *R["sugar_intake"])               # used
            st.session_state.water_intake = st.slider("Water Intake (liters/day)", *R["water_intake"])                 # used
            st.session_state.sleep_hours = st.slider("Sleep Hours", *R["sleep_hours"])                                # used

        cols = st.columns([1,1])
        with cols[0]:
            if st.form_submit_button("← Back", use_container_width=True):
                st.session_state.step = 0
                st.rerun()
        with cols[1]:
            submit = st.form_submit_button("🚀 Predict", use_container_width=True)


#
# ==============================
# Submit to FastAPI & render result card
# ==============================
if submit:
    # Anchor to scroll into view
    st.markdown("<div id='result-anchor'></div>", unsafe_allow_html=True)
    payload = {
        "sugar_intake": _d("sugar_intake"),
        "bmi": _d("bmi"),
        "cholesterol": _d("cholesterol"),
        "sleep_hours": _d("sleep_hours"),
        "physical_activity": _d("physical_activity"),
        "work_hours": _d("work_hours"),
        "blood_pressure": _d("blood_pressure"),
        "calorie_intake": _d("calorie_intake"),
        "water_intake": _d("water_intake"),
    }

    try:
        r = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=20)
        if not r.ok:
            st.error(f"API error {r.status_code}: {r.text}")
        else:
            res = r.json()
            prob = float(res.get("probability", 0.0))
            pred = str(res.get("prediction", "Unknown"))
            thr  = float(res.get("threshold", 0.5))
            model_version = res.get("model_version", "-")
            
            st.markdown("---")
            # Color by server-side decision; keep amber for mid band near threshold
            band = "low" if prob < max(0.33, thr - 0.15) else ("mid" if prob < max(0.66, thr + 0.15) else "high")
            pill_class = "pill-green" if pred == "Healthy" else ("pill-amber" if band == "mid" else "pill-red")
            emoji = "🟢" if pred == "Healthy" else ("🟠" if band == "mid" else "🔴")

            st.markdown(
                f"""
                <div class='result'>
                    <div class='pill {pill_class}'>{emoji} {pred}</div>
                    <div style='height:10px'></div>
                    <div style='display:flex; gap:26px; flex-wrap:wrap;'>
                        <div>
                            <div class='muted'>Risk probability</div>
                            <div class='metric'>{prob:.1%}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(min(max(prob, 0.0), 1.0))
            st.caption(f"Model: {model_version}")

            # Scroll to result anchor on predict
            components.html(
                """
                <script>
                    const el = parent.document.getElementById('result-anchor');
                    if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
                </script>
                """,
                height=0,
            )

            st.markdown("### Suggestions (informational, not medical)")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**🍽️ Nutrition**\n\nReduce added sugar; prioritize whole foods and fiber.")
            with c2:
                st.markdown("**🏃 Activity**\n\nAim for 150+ mins/week moderate activity; increase daily steps.")
            with c3:
                st.markdown("**😴 Sleep**\n\nTarget 7–9 hours; keep a consistent schedule and limit screen time at night.")

            st.caption("Disclaimer: This app provides informational insights only and is not a substitute for professional medical advice.")

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")