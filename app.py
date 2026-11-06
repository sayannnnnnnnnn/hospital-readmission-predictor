import json
import os

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "models")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "KNN": "knn.joblib",
}

MODEL_COLORS = {
    "Logistic Regression": "#4da3ff",
    "Decision Tree": "#f6ad55",
    "Random Forest": "#34d399",
    "KNN": "#c084fc",
}

TIER_COLORS = {"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e74c3c"}
TIER_GRADIENTS = {
    "Low": ("#11998e", "#38ef7d", "#ffffff"),
    "Medium": ("#f7971e", "#ffd200", "#1a1a1a"),
    "High": ("#cb2d3e", "#ef473a", "#ffffff"),
}

# test-set results from the notebook: (ROC-AUC, PR-AUC, top-20% capture)
QUALITY = {
    "Random Forest": (0.653, 0.178, 0.384),
    "Logistic Regression": (0.649, 0.170, 0.376),
    "Decision Tree": (0.624, 0.154, 0.365),
    "KNN": (0.570, 0.111, 0.270),
}

# the 10 inputs shown up front; everything else goes in "More details"
KEY_FIELDS = [
    "age", "discharge_disposition_id", "number_inpatient", "number_emergency",
    "number_outpatient", "time_in_hospital", "number_diagnoses", "diag_group",
    "admission_type_id", "admission_source_id",
]

st.set_page_config(page_title="Readmission Risk", page_icon="🏥", layout="wide")

# ---------- custom styling ----------
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d3a 0%, #2a1f4d 100%);
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        padding: 10px 18px;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(167, 139, 250, 0.18);
    }
    div.stButton > button {
        border-radius: 10px;
        border: 1px solid #a78bfa;
        width: 100%;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border-color: transparent;
    }
    div[data-testid="stMetric"] {
        background: rgba(167, 139, 250, 0.10);
        border: 1px solid rgba(167, 139, 250, 0.35);
        border-radius: 14px;
        padding: 14px 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- loading ----------
@st.cache_resource
def load_everything():
    def read_json(name):
        with open(os.path.join(MODELS_DIR, name)) as f:
            return json.load(f)

    columns = read_json("feature_columns.json")
    options = read_json("options.json")
    labels = read_json("labels.json")
    defaults = read_json("defaults.json")
    test_scores = read_json("test_scores.json")
    models = {name: joblib.load(os.path.join(MODELS_DIR, file))
              for name, file in MODEL_FILES.items()}
    return columns, options, labels, defaults, models, test_scores


columns, options, labels, defaults, models, test_scores = load_everything()


# ---------- helper functions ----------
def percentile_of(score, ref_scores):
    """% of test-set patients ranked below this patient (ties count as half)."""
    ref = np.array(ref_scores)
    below = (ref < score).mean()
    ties = (ref == score).mean()
    return float((below + 0.5 * ties) * 100)


def tier_of(pct):
    if pct >= 80:
        return "High"
    if pct >= 50:
        return "Medium"
    return "Low"


def base_value(col):
    """The 'typical patient' value for a column."""
    if col in options["numeric"]:
        return int(options["numeric"][col]["median"])
    return defaults[col]


def apply_preset(name):
    """Fill the form with an example patient."""
    overrides = {
        "typical": {},
        "high": {
            "age": "[80-90)", "discharge_disposition_id": "22",
            "number_inpatient": 3, "number_emergency": 2,
            "time_in_hospital": 8, "number_diagnoses": 9,
        },
        "low": {
            "age": "[30-40)", "discharge_disposition_id": "1",
            "number_inpatient": 0, "time_in_hospital": 2, "number_diagnoses": 5,
        },
    }[name]
    for col in columns:
        st.session_state[f"in_{col}"] = overrides.get(col, base_value(col))


def original_col(feature_name):
    """Map 'cat__discharge_disposition_id_22' back to 'discharge_disposition_id'."""
    if feature_name.startswith("num__"):
        return feature_name[5:]
    rest = feature_name[5:]
    matches = [c for c in columns if rest.startswith(c + "_")]
    return max(matches, key=len)


def explain(patient_df):
    """How much each input moves this patient's risk vs a typical patient (logistic regression)."""
    pipe = models["Logistic Regression"]
    ct = pipe.named_steps["columntransformer"]
    lr = pipe.named_steps["logisticregression"]
    baseline_df = pd.DataFrame([{c: base_value(c) for c in columns}])[columns]

    def dense(m):
        return m.toarray() if hasattr(m, "toarray") else np.asarray(m)

    x = dense(ct.transform(patient_df))[0]
    x0 = dense(ct.transform(baseline_df))[0]
    push = (x - x0) * lr.coef_[0]
    groups = [original_col(n) for n in ct.get_feature_names_out()]
    return pd.Series(push).groupby(groups).sum()


# ---------- session state: give every input a starting value ----------
for col in columns:
    st.session_state.setdefault(f"in_{col}", base_value(col))


def make_input(col):
    key = f"in_{col}"
    label = col.replace("_", " ")
    if col in options["numeric"]:
        r = options["numeric"][col]
        st.slider(label, min_value=r["min"], max_value=r["max"], key=key)
    else:
        names = labels.get(col, {})
        st.selectbox(
            label, options["categorical"][col], key=key,
            format_func=lambda v, names=names: f"{v} - {names[v]}" if v in names else v,
        )
    return st.session_state[key]


# ---------- sidebar: all the inputs ----------
values = {}
with st.sidebar:
    st.header("🧑‍⚕️ Patient setup")
    model_name = st.selectbox("Model", list(MODEL_FILES.keys()))

    st.caption("Quick examples")
    b1, b2, b3 = st.columns(3)
    b1.button("Typical", on_click=apply_preset, args=("typical",))
    b2.button("High", on_click=apply_preset, args=("high",))
    b3.button("Low", on_click=apply_preset, args=("low",))

    st.divider()
    st.subheader("Key details")
    for col in KEY_FIELDS:
        values[col] = make_input(col)

    with st.expander("More details (medications, race, gender...)"):
        for col in columns:
            if col not in KEY_FIELDS:
                values[col] = make_input(col)

# ---------- predict (runs live on every change) ----------
patient = pd.DataFrame([values])[columns]

results = {}
for name, pipe in models.items():
    score = pipe.predict_proba(patient)[0, 1]
    results[name] = percentile_of(score, test_scores[name])

pct = results[model_name]
tier = tier_of(pct)
g1, g2, text_color = TIER_GRADIENTS[tier]
n_high = sum(tier_of(p) == "High" for p in results.values())

# ---------- banner ----------
st.markdown(
    """
    <div style="background:linear-gradient(90deg,#667eea 0%,#764ba2 55%,#f093fb 100%);
                padding:22px 30px;border-radius:16px;color:white;margin-bottom:18px;">
      <div style="font-size:2.1rem;font-weight:800;">🏥 30-Day Readmission Risk</div>
      <div style="font-size:1.05rem;opacity:0.92;">
        Change the patient in the sidebar. The result updates instantly.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_result, tab_why, tab_quality = st.tabs(
    ["📊 Result", "🔍 Why this score?", "✅ How good is it?"]
)

with tab_result:
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,{g1} 0%,{g2} 100%);color:{text_color};
                    padding:22px 28px;border-radius:16px;box-shadow:0 6px 18px rgba(0,0,0,0.35);">
          <div style="font-size:0.95rem;opacity:0.85;">{model_name}</div>
          <div style="font-size:2.8rem;font-weight:800;line-height:1.15;">{tier} risk</div>
          <div style="font-size:1.15rem;">Ranks above about <b>{pct:.0f}%</b> of the test patients</div>
        </div>
        <div style="position:relative;height:14px;border-radius:7px;margin:36px 0 6px 0;
             background:linear-gradient(90deg,#2ecc71 0%,#2ecc71 50%,#f1c40f 50%,#f1c40f 80%,#e74c3c 80%,#e74c3c 100%);">
          <div style="position:absolute;left:{pct}%;top:-7px;width:5px;height:28px;background:white;
               border:1px solid #333;border-radius:3px;transform:translateX(-50%);"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.8rem;opacity:0.75;">
          <span>Low (bottom 50%)</span><span>Medium (next 30%)</span><span>High (top 20%)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # one colored chip per model, showing its tier
    chips = ""
    for name, p in results.items():
        t = tier_of(p)
        chips += (
            f'<span style="display:inline-block;background:{TIER_COLORS[t]};color:#111;'
            f'padding:5px 14px;border-radius:16px;margin:4px 8px 4px 0;font-weight:600;">'
            f'{name}: {t}</span>'
        )
    st.markdown(f"<div style='margin-top:18px;'>{chips}</div>", unsafe_allow_html=True)

    st.write("")
    if n_high == 4:
        st.error("All 4 models place this patient in the **High** tier.")
    elif n_high > 0:
        st.warning(f"{n_high} of 4 models place this patient in the **High** tier. The models disagree.")
    else:
        st.info("No model places this patient in the High tier.")
    st.caption("High = the top 20% by risk, the group a care team would follow up with. "
               "In our test set, that group contained about 38% of all readmissions.")

    st.subheader("All four models, same patient")
    chart_df = pd.DataFrame({"Model": list(results), "Percentile": list(results.values())})
    chart_df["Tier"] = chart_df["Percentile"].apply(tier_of)
    bars = alt.Chart(chart_df).mark_bar(cornerRadiusEnd=6).encode(
        x=alt.X("Percentile:Q", scale=alt.Scale(domain=[0, 100]), title="Risk percentile"),
        y=alt.Y("Model:N", sort=None, title=None),
        color=alt.Color("Tier:N",
                        scale=alt.Scale(domain=list(TIER_COLORS), range=list(TIER_COLORS.values())),
                        legend=None),
        tooltip=["Model", alt.Tooltip("Percentile:Q", format=".0f"), "Tier"],
    )
    lines = alt.Chart(pd.DataFrame({"x": [50, 80]})).mark_rule(
        strokeDash=[5, 4], color="gray").encode(x="x:Q")
    st.altair_chart(bars + lines)
    st.caption("Dashed lines mark the tier cutoffs at 50 and 80. Logistic Regression and Random Forest "
               "are the most reliable models; KNN and the Decision Tree are weaker.")

with tab_why:
    st.subheader("What moves this patient's risk?")
    st.write("Compared with a **typical patient**, these inputs push the risk up (red) or down (green). "
             "A longer bar is a bigger push. This view always uses Logistic Regression, "
             "because its weights can be read directly.")
    push = explain(patient)
    top = push.reindex(push.abs().sort_values(ascending=False).index).head(8)
    if top.abs().max() < 1e-9:
        st.info("This patient matches the typical profile, so nothing stands out. "
                "Try the **High** example button in the sidebar.")
    else:
        why_df = pd.DataFrame({"Factor": [f.replace("_", " ") for f in top.index],
                               "Push": top.values})
        why_df["Direction"] = np.where(why_df["Push"] > 0, "Raises risk", "Lowers risk")
        why_chart = alt.Chart(why_df).mark_bar(cornerRadiusEnd=6).encode(
            x=alt.X("Push:Q", title="lowers risk  ←   →  raises risk"),
            y=alt.Y("Factor:N", sort=None, title=None),
            color=alt.Color("Direction:N",
                            scale=alt.Scale(domain=["Lowers risk", "Raises risk"],
                                            range=["#2ecc71", "#e74c3c"]),
                            legend=None),
            tooltip=["Factor", alt.Tooltip("Push:Q", format=".2f"), "Direction"],
        )
        st.altair_chart(why_chart)
    st.caption("These show association, not cause. For example, a rehab discharge is a marker of a sicker "
               "patient, not something that itself causes readmission.")

with tab_quality:
    roc, pr, cap = QUALITY[model_name]
    st.subheader(f"{model_name}: results on 13,998 unseen test patients")
    m1, m2, m3 = st.columns(3)
    m1.metric("ROC-AUC", f"{roc:.3f}", f"{roc - 0.5:+.3f} vs random (0.50)")
    m2.metric("PR-AUC", f"{pr:.3f}", f"{pr - 0.09:+.3f} vs random (0.09)")
    m3.metric("Top-20% capture", f"{cap:.1%}", f"{(cap - 0.2) * 100:+.1f} pts vs random (20%)")

    st.subheader("All models compared (ROC-AUC)")
    q_df = pd.DataFrame({"Model": list(QUALITY), "ROC-AUC": [v[0] for v in QUALITY.values()]})
    q_bars = alt.Chart(q_df).mark_bar(cornerRadiusEnd=6).encode(
        x=alt.X("ROC-AUC:Q", scale=alt.Scale(domain=[0.4, 0.7])),
        y=alt.Y("Model:N", sort=None, title=None),
        color=alt.Color("Model:N",
                        scale=alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values())),
                        legend=None),
        tooltip=["Model", alt.Tooltip("ROC-AUC:Q", format=".3f")],
    )
    q_rule = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(
        strokeDash=[5, 4], color="gray").encode(x="x:Q")
    st.altair_chart(q_bars + q_rule)
    st.caption("The dashed line at 0.50 is random guessing. The axis starts at 0.40 so the differences are visible.")

    st.subheader("What flagging more patients buys (Logistic Regression)")
    lift = pd.DataFrame({
        "Patients flagged": ["Top 5%", "Top 10%", "Top 20%", "Top 30%", "Top 50%"],
        "Readmissions caught": ["15.0%", "23.6%", "37.6%", "48.4%", "68.5%"],
        "Lift vs random": ["3.01x", "2.36x", "1.88x", "1.61x", "1.37x"],
    })
    st.dataframe(lift, hide_index=True)
    st.info("Readmission is hard to predict, so all models score around 0.65 ROC-AUC. "
            "The value is in ranking: the riskiest 20% hold nearly twice their fair share of readmissions.")

st.divider()
st.caption("Learning project built on the UCI 'Diabetes 130-US hospitals' dataset (one encounter per patient). "
           "Not a clinical tool and not medical advice.")