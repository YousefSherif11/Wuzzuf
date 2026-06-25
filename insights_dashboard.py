
import os, warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import joblib
from sklearn.metrics import confusion_matrix, classification_report
warnings.filterwarnings("ignore")

try:
    BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE = os.path.abspath('wuzzuf_model_output')

st.set_page_config(layout="wide", page_title="Wuzzuf Insights", page_icon="📊")

@st.cache_resource
def load_artifacts():
    model     = joblib.load(os.path.join(BASE, "wuzzuf_experience_model.pkl"))
    le        = joblib.load(os.path.join(BASE, "label_encoder.pkl"))
    oof_proba = np.load(os.path.join(BASE, "oof_proba.npy"))
    feat_imp  = pd.read_csv(os.path.join(BASE, "feature_importances.csv"))
    return model, le, oof_proba, feat_imp

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE, "..", "wuzzuf_cleaned.csv"))

model, le, oof_proba, feat_imp_df = load_artifacts()
df        = load_data()
classes   = le.classes_
y_true    = le.transform(df["experience_level"])
oof_preds = oof_proba.argmax(axis=1)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1A2E44 0%,#E35A1E 100%);
            padding:28px 32px;border-radius:12px;margin-bottom:1.5rem">
  <h1 style="color:#fff;margin:0;font-size:1.8rem">📊 Wuzzuf Insights Dashboard</h1>
  <p style="color:rgba(255,255,255,.75);margin:6px 0 0;font-size:.95rem">
      تحليل عميق لسوق الشغل المصري وأداء الموديل
  </p>
</div>
""", unsafe_allow_html=True)

# ── KPIs ────────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("إجمالي الوظائف",      f"{len(df):,}")
k2.metric("متوسط سنوات الخبرة",  f"{df['exp_years'].mean():.1f} سنة")
k3.metric("أكثر مستوى خبرة",     df["experience_level"].value_counts().index[0])
k4.metric("OOF Accuracy",        f"{(oof_preds == y_true).mean():.1%}")
k5.metric("متوسط Confidence",    f"{oof_proba.max(axis=1).mean():.1%}")
st.markdown("---")

# ── Tabs ────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4 = st.tabs(
    ["🎯 Feature Importance","📊 Model Performance","🛤️ Career Path","🔬 Model Confidence"]
)

# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("أهم الـ Features في الموديل")
    col_a, col_b = st.columns([3, 1])
    with col_a:
        top_n = st.slider("عدد الـ Features", 5, min(30, len(feat_imp_df)), 15, key="fn")
        fig = px.bar(
            feat_imp_df.head(top_n),
            x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale="Blues",
            title=f"Top {top_n} Feature Importances"
        )
        fig.update_layout(yaxis={"categoryorder":"total ascending"}, height=520)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.markdown("#### Top 10")
        st.dataframe(
            feat_imp_df.head(10)
              .rename(columns={"feature":"Feature","importance":"Score"})
              .reset_index(drop=True),
            use_container_width=True, height=380
        )

# ────────────────────────────────────────────────────────────────────────────
with tab2:
    left, right = st.columns(2)
    with left:
        st.subheader("Confusion Matrix (OOF)")
        cm     = confusion_matrix(y_true, oof_preds)
        cm_pct = (cm.astype(float) / cm.sum(axis=1, keepdims=True)).round(3)
        hover  = [[f"{cm_pct[r,c]:.1%}  ({cm[r,c]})" for c in range(len(classes))]
                  for r in range(len(classes))]
        fig_cm = go.Figure(go.Heatmap(
            z=cm_pct, x=list(classes), y=list(classes),
            colorscale="Blues", text=hover, texttemplate="%{text}", showscale=True
        ))
        fig_cm.update_layout(
            xaxis_title="Predicted", yaxis_title="Actual",
            yaxis={"autorange":"reversed"}, height=420
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    with right:
        st.subheader("Per-Class F1 / Precision / Recall")
        rpt   = classification_report(y_true, oof_preds, target_names=classes, output_dict=True)
        f1_df = pd.DataFrame(
            [{"Class":c, "F1":rpt[c]["f1-score"],
              "Precision":rpt[c]["precision"], "Recall":rpt[c]["recall"]}
             for c in classes]
        )
        fig_f1 = px.bar(
            f1_df.melt(id_vars="Class", var_name="Metric", value_name="Score"),
            x="Class", y="Score", color="Metric", barmode="group",
            title="Metrics per Class",
            color_discrete_sequence=["#1D4ED8","#16A34A","#EA580C"]
        )
        fig_f1.update_layout(height=420, yaxis_range=[0, 1])
        st.plotly_chart(fig_f1, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
with tab3:
    LEVEL_ORDER = ["Student","Entry Level","Experienced","Senior Management","Manager"]
    LEVEL_ORDER = [l for l in LEVEL_ORDER if l in df["experience_level"].unique()]

    st.subheader("متوسط الخبرة والـ Skills لكل مستوى")
    career = (
        df.groupby("experience_level")
          .agg(avg_exp=("exp_years","mean"), avg_sk=("skills_count","mean"),
               count=("employment_type","count"))
          .reindex(LEVEL_ORDER).reset_index()
    )
    COLORS = ["#93C5FD","#60A5FA","#3B82F6","#1D4ED8","#1E3A8A"]
    fig_c  = make_subplots(1, 3,
        subplot_titles=("متوسط سنوات الخبرة","متوسط عدد الـ Skills","عدد الوظائف"))
    for j, row in career.iterrows():
        c = COLORS[j % len(COLORS)]
        for col_i, val in enumerate([row.avg_exp, row.avg_sk, row["count"]], 1):
            fig_c.add_trace(
                go.Bar(x=[row.experience_level], y=[val], marker_color=c,
                       showlegend=False, name=row.experience_level),
                row=1, col=col_i
            )
    fig_c.update_layout(height=360)
    st.plotly_chart(fig_c, use_container_width=True)

    st.subheader("Skills Gap — Top Skills لكل مستوى")
    sel     = st.selectbox("اختر مستوى الخبرة", LEVEL_ORDER, key="lvl")
    sk_vc   = (df[df["experience_level"] == sel]["top_skill"]
               .value_counts().head(10)
               .rename_axis("skill").reset_index(name="count"))
    fig_sk  = px.bar(
        sk_vc, x="count", y="skill", orientation="h",
        color="count", color_continuous_scale="Oranges",
        title=f"Top 10 Skills — {sel}"
    )
    fig_sk.update_layout(yaxis={"categoryorder":"total ascending"}, height=400)
    st.plotly_chart(fig_sk, use_container_width=True)

    st.subheader("توزيع Work Mode لكل مستوى")
    wm = (df.groupby(["experience_level","work_mode"])
            .size().reset_index(name="count"))
    fig_wm = px.bar(
        wm, x="experience_level", y="count", color="work_mode",
        barmode="group",
        category_orders={"experience_level": LEVEL_ORDER},
        title="Work Mode by Experience Level",
        color_discrete_sequence=["#1D4ED8","#16A34A","#EA580C","#7C3AED"]
    )
    st.plotly_chart(fig_wm, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
with tab4:
    conf_df = pd.DataFrame({
        "max_probability": oof_proba.max(axis=1),
        "predicted_class": classes[oof_preds],
        "correct":         (oof_preds == y_true)
    })

    st.subheader("Confidence Violin — كل الـ predictions")
    fig_vio = px.violin(
        conf_df, x="predicted_class", y="max_probability",
        color="correct", box=True, points=False,
        color_discrete_map={True:"#16A34A", False:"#DC2626"},
        category_orders={"predicted_class": list(classes)},
        title="Model Confidence Distribution  🟢 Correct  🔴 Wrong",
        labels={"max_probability":"Max Probability",
                "predicted_class":"Predicted Class","correct":"Correct"}
    )
    fig_vio.update_layout(height=450)
    st.plotly_chart(fig_vio, use_container_width=True)

    st.subheader("Low Confidence Predictions Explorer")
    thr    = st.slider("حد الـ Confidence", 0.30, 0.90, 0.50, 0.05)
    low    = conf_df[conf_df["max_probability"] < thr]
    c1, c2 = st.columns(2)
    c1.metric(f"Predictions تحت {thr:.0%}", f"{len(low):,}",
              f"{len(low)/len(conf_df)*100:.1f}% من الكل")
    c2.metric("Avg Confidence (low group)", f"{low.max_probability.mean():.1%}")
    if len(low):
        fig_pie = px.pie(
            low["predicted_class"].value_counts().rename_axis("class").reset_index(name="count"),
            names="class", values="count",
            title=f"Distribution of Low-Confidence Predictions (< {thr:.0%})",
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        st.plotly_chart(fig_pie, use_container_width=True)
