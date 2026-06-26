
import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib

BASE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(layout='wide', page_title='Wuzzuf Experience Level Predictor')

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE, 'wuzzuf_experience_model.pkl'))
    le    = joblib.load(os.path.join(BASE, 'label_encoder.pkl'))
    return model, le

model, le = load_model()

@st.cache_data
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE, 'wuzzuf_cleaned.csv'))

df = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dataset Information", "Data Analysis", "Prediction", "Project Insights", "Business Insights"])

# ─────────────────────────────────────────────
# ✦  INSIGHTS PAGE
# ─────────────────────────────────────────────
if page == "Project Insights":
    st.title("✦ Project Insights")
    st.markdown("Key decisions and highlights across every stage of the ML pipeline.", unsafe_allow_html=False)
    st.markdown("---")

    INSIGHTS_CSS = """
    <style>
    .ins-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 14px;
        margin-top: 8px;
    }
    .ins-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        overflow: hidden;
        display: flex;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .ins-strip { width: 4px; flex-shrink: 0; }
    .ins-body  { padding: 14px 16px; flex: 1; }
    .ins-head  { font-size: 13.5px; font-weight: 600; color: #111827; margin-bottom: 10px; display: flex; align-items: center; gap: 7px; }
    .ins-list  { list-style: none; padding: 0; margin: 0; }
    .ins-item  { font-size: 13px; color: #6b7280; margin-bottom: 6px; display: flex; gap: 8px; align-items: flex-start; line-height: 1.55; }
    .ins-dot   { font-size: 8px; margin-top: 5px; flex-shrink: 0; }
    </style>
    """

    INSIGHTS_HTML = """
    <div class="ins-grid">

      <div class="ins-card">
        <div class="ins-strip" style="background:#3B82F6;"></div>
        <div class="ins-body">
          <div class="ins-head">📊 Dataset overview</div>
          <ul class="ins-list">
            <li class="ins-item"><span class="ins-dot" style="color:#3B82F6;">●</span>Scraped from Wuzzuf — Egypt's top job platform</li>
            <li class="ins-item"><span class="ins-dot" style="color:#3B82F6;">●</span>Target: 5 classes — Entry Level, Experienced, Manager, Senior Management, Student</li>
            <li class="ins-item"><span class="ins-dot" style="color:#3B82F6;">●</span>Supervised multi-class classification task</li>
          </ul>
        </div>
      </div>

      <div class="ins-card">
        <div class="ins-strip" style="background:#10B981;"></div>
        <div class="ins-body">
          <div class="ins-head">🧹 Data cleaning</div>
          <ul class="ins-list">
            <li class="ins-item"><span class="ins-dot" style="color:#10B981;">●</span>Dropped all URL columns, misc, job_title &amp; company_name</li>
            <li class="ins-item"><span class="ins-dot" style="color:#10B981;">●</span>Removed duplicates and reset index</li>
            <li class="ins-item"><span class="ins-dot" style="color:#10B981;">●</span>Stripped bullet chars ( · - * • ) from all skill and category columns</li>
          </ul>
        </div>
      </div>

      <div class="ins-card">
        <div class="ins-strip" style="background:#F59E0B;"></div>
        <div class="ins-body">
          <div class="ins-head">⚙️ Feature engineering</div>
          <ul class="ins-list">
            <li class="ins-item"><span class="ins-dot" style="color:#F59E0B;">●</span>Parsed experience range → numeric average (exp_years)</li>
            <li class="ins-item"><span class="ins-dot" style="color:#F59E0B;">●</span>Extracted city &amp; country from location string</li>
            <li class="ins-item"><span class="ins-dot" style="color:#F59E0B;">●</span>8 skill columns → skills_count + top_skill + second_skill</li>
            <li class="ins-item"><span class="ins-dot" style="color:#F59E0B;">●</span>Skills longer than 40 chars mapped to "Unknown"</li>
          </ul>
        </div>
      </div>

      <div class="ins-card">
        <div class="ins-strip" style="background:#EF4444;"></div>
        <div class="ins-body">
          <div class="ins-head">⚖️ Class imbalance</div>
          <ul class="ins-list">
            <li class="ins-item"><span class="ins-dot" style="color:#EF4444;">●</span>compute_sample_weight(class_weight='balanced')</li>
            <li class="ins-item"><span class="ins-dot" style="color:#EF4444;">●</span>Student class gets significantly higher weight vs Experienced</li>
            <li class="ins-item"><span class="ins-dot" style="color:#EF4444;">●</span>Weights applied only to training folds — never test folds</li>
          </ul>
        </div>
      </div>

      <div class="ins-card">
        <div class="ins-strip" style="background:#8B5CF6;"></div>
        <div class="ins-body">
          <div class="ins-head">🔬 Preprocessing pipeline</div>
          <ul class="ins-list">
            <li class="ins-item"><span class="ins-dot" style="color:#8B5CF6;">●</span>Numerical: Median Imputer → Standard Scaler</li>
            <li class="ins-item"><span class="ins-dot" style="color:#8B5CF6;">●</span>Categorical: Simple Imputer → OHE / Ordinal Encoder</li>
            <li class="ins-item"><span class="ins-dot" style="color:#8B5CF6;">●</span>Full sklearn Pipeline — zero data leakage guaranteed</li>
          </ul>
        </div>
      </div>

      <div class="ins-card">
        <div class="ins-strip" style="background:#6B7280;"></div>
        <div class="ins-body">
          <div class="ins-head">🤖 Models compared</div>
          <ul class="ins-list">
            <li class="ins-item"><span class="ins-dot" style="color:#6B7280;">●</span>KNN — neighbors, distance metric, weight scheme</li>
            <li class="ins-item"><span class="ins-dot" style="color:#6B7280;">●</span>Decision Tree — max_depth, min_samples tuning</li>
            <li class="ins-item"><span class="ins-dot" style="color:#6B7280;">●</span>Random Forest — n_estimators &amp; depth tuning</li>
            <li class="ins-item"><span class="ins-dot" style="color:#6B7280;">●</span>All tuned with RandomizedSearchCV · 5-fold CV · n_iter=10</li>
          </ul>
        </div>
      </div>

      <div class="ins-card">
        <div class="ins-strip" style="background:#059669;"></div>
        <div class="ins-body">
          <div class="ins-head">🏆 Best model — XGBoost</div>
          <ul class="ins-list">
            <li class="ins-item"><span class="ins-dot" style="color:#059669;">●</span>n_estimators=500, max_depth=7, learning_rate=0.01</li>
            <li class="ins-item"><span class="ins-dot" style="color:#059669;">●</span>subsample=0.7, colsample_bytree=0.7</li>
            <li class="ins-item"><span class="ins-dot" style="color:#059669;">●</span>StratifiedKFold (5 splits) with out-of-fold predictions</li>
            <li class="ins-item"><span class="ins-dot" style="color:#059669;">●</span>Each row evaluated on a model that never trained on it</li>
          </ul>
        </div>
      </div>

      <div class="ins-card">
        <div class="ins-strip" style="background:#EC4899;"></div>
        <div class="ins-body">
          <div class="ins-head">🚀 Deployment</div>
          <ul class="ins-list">
            <li class="ins-item"><span class="ins-dot" style="color:#EC4899;">●</span>Streamlit app — 3 pages: Dataset Info, EDA, Prediction</li>
            <li class="ins-item"><span class="ins-dot" style="color:#EC4899;">●</span>Model + LabelEncoder saved via joblib</li>
            <li class="ins-item"><span class="ins-dot" style="color:#EC4899;">●</span>Real-time prediction with probability bar chart per class</li>
          </ul>
        </div>
      </div>

    </div>
    """

    st.markdown(INSIGHTS_CSS + INSIGHTS_HTML, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 💡  BUSINESS INSIGHTS PAGE
# ─────────────────────────────────────────────
elif page == "Business Insights":
    st.title("💡 Business Insights")
    st.markdown("Live KPIs computed directly from the Wuzzuf dataset.")
    st.markdown("---")

    # ── Compute all KPIs from data ──────────────────────────────
    total_jobs     = len(df)

    vc_level       = df['experience_level'].value_counts()
    top_level      = vc_level.idxmax()
    top_level_pct  = vc_level.max() / total_jobs * 100
    least_level    = vc_level.idxmin()
    least_pct      = vc_level.min() / total_jobs * 100

    avg_exp        = df['exp_years'].dropna().mean()
    max_exp_level  = df.groupby('experience_level')['exp_years'].mean().idxmax()

    top_city       = df['city'].value_counts().idxmax()
    top_city_pct   = df['city'].value_counts().max() / total_jobs * 100

    remote_pct     = df['work_mode'].str.contains('Remote|Hybrid', case=False, na=False).mean() * 100
    onsite_pct     = 100 - remote_pct

    fulltime_pct   = df['employment_type'].str.contains('Full Time', case=False, na=False).mean() * 100

    top_category   = df['job_category'].value_counts().idxmax()
    top_cat_pct    = df['job_category'].value_counts().max() / total_jobs * 100

    top_skill_val  = df['top_skill'].dropna().value_counts().idxmax()
    top_skill_pct  = df['top_skill'].dropna().value_counts().max() / df['top_skill'].notna().sum() * 100

    avg_skills     = df['skills_count'].mean()
    max_skills_lvl = df.groupby('experience_level')['skills_count'].mean().idxmax()

    # ── ROW 1: Volume & Demand ───────────────────────────────────
    st.subheader("📦 Volume & Demand")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total job listings",     f"{total_jobs:,}")
    c2.metric("Dominant level",         top_level,         f"{top_level_pct:.1f}% of listings")
    c3.metric("Rarest level",           least_level,       f"{least_pct:.1f}% of listings")
    c4.metric("Avg. experience needed", f"{avg_exp:.1f} yrs", f"Highest: {max_exp_level}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROW 2: Location & Work Mode ──────────────────────────────
    st.subheader("🌍 Location & Work mode")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Top hiring city",   top_city,       f"{top_city_pct:.1f}% of listings")
    c6.metric("Remote / Hybrid",   f"{remote_pct:.1f}%", "of all openings")
    c7.metric("On-site",           f"{onsite_pct:.1f}%", "of all openings")
    c8.metric("Full-time share",   f"{fulltime_pct:.1f}%", "of employment types")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROW 3: Skills & Categories ───────────────────────────────
    st.subheader("🛠️ Skills & Categories")
    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Top job category",   top_category,      f"{top_cat_pct:.1f}% of listings")
    c10.metric("Most demanded skill", top_skill_val,   f"{top_skill_pct:.1f}% appear first")
    c11.metric("Avg. skills per listing", f"{avg_skills:.1f}", "skills required")
    c12.metric("Most skills demanded by", max_skills_lvl, "experience level")

    st.markdown("---")

    # ── EXPERIENCE LEVEL BREAKDOWN ───────────────────────────────
    st.subheader("📊 Experience level breakdown")
    col_a, col_b = st.columns(2)

    with col_a:
        fig_level = px.pie(
            vc_level.reset_index(),
            names='experience_level', values='count',
            title='Distribution of experience levels',
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_level.update_traces(textposition='outside', textinfo='percent+label')
        fig_level.update_layout(showlegend=False)
        st.plotly_chart(fig_level, use_container_width=True)

    with col_b:
        exp_by_level = (
            df.groupby('experience_level')['exp_years']
            .mean().sort_values(ascending=False).reset_index()
        )
        fig_exp = px.bar(
            exp_by_level, x='experience_level', y='exp_years',
            title='Avg. years of experience per level',
            text_auto='.1f',
            color='exp_years', color_continuous_scale='Blues'
        )
        fig_exp.update_layout(xaxis_title='', yaxis_title='Years', coloraxis_showscale=False)
        st.plotly_chart(fig_exp, use_container_width=True)

    # ── TOP SKILLS BAR ───────────────────────────────────────────
    st.subheader("🔑 Top 10 most demanded skills")
    top_skills_df = df['top_skill'].dropna().value_counts().head(10).reset_index()
    fig_skills = px.bar(
        top_skills_df, x='top_skill', y='count',
        text_auto=True,
        color='count', color_continuous_scale='Teal'
    )
    fig_skills.update_layout(
        xaxis={'categoryorder': 'total descending'},
        xaxis_title='', yaxis_title='Listings',
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_skills, use_container_width=True)


# ─────────────────────────────────────────────
# ORIGINAL PAGES (unchanged)
# ─────────────────────────────────────────────
elif page == "Dataset Information":
    st.title("💼 Wuzzuf Jobs - Dataset Information")
    st.image(
        'https://images.unsplash.com/photo-1486312338219-ce68d2c6f44d?w=1200',
        caption='Egyptian Job Market Analysis', use_column_width=True
    )
    st.write("### Dataset Preview")
    st.dataframe(df)
    st.write("### Column Descriptions")
    cols_desc = {
        "employment_type" : "Full Time / Part Time / Internship / Freelance",
        "work_mode"       : "On-site / Remote / Hybrid",
        "experience_level": "Our Target🎯",
        "job_category"    : "Accounting / IT / Engineering/ ...",
        "exp_years"       : "Average years of experience required",
        "skills_count"    : "Number of skills required in the job",
        "top_skill"       : "Most demanded skill",
        "second_skill"    : "Second most demanded skill",
    }
    for col, desc in cols_desc.items():
        with st.expander(f"**{col}**"):
            st.write(desc)

elif page == "Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    tabs = st.tabs(["Univariate", "Bivariate", "Multivariate"])

    with tabs[0]:
        st.subheader("Univariate Analysis")
        for col in df.select_dtypes(include='number').columns:
            fig = px.histogram(df, x=col, title=f'Distribution of {col}')
            st.plotly_chart(fig, use_column_width=True)
        for col in ['experience_level', 'employment_type', 'work_mode']:
            vc  = df[col].value_counts().reset_index()
            fig = px.bar(vc, x=col, y='count', title=f'{col} Distribution', text_auto=True)
            st.plotly_chart(fig, use_column_width=True)

    with tabs[1]:
        st.subheader("Bivariate Analysis")
        df_exp = df.groupby('experience_level')['exp_years'].mean().sort_values(ascending=False).reset_index()
        fig1   = px.bar(df_exp, x='experience_level', y='exp_years',
                        title='Average Experience Years per Level', text_auto='.2f',
                        color='exp_years', color_continuous_scale='Blues')
        st.plotly_chart(fig1, use_column_width=True)

        df_wm = df.groupby(['experience_level', 'work_mode']).size().reset_index(name='count')
        fig2  = px.bar(df_wm, x='experience_level', y='count', color='work_mode',
                       barmode='group', title='Work Mode by Experience Level', text_auto=True)
        fig2.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig2, use_column_width=True)

        top_cats = df['job_category'].value_counts().head(10).reset_index()
        fig3     = px.bar(top_cats, x='job_category', y='count',
                          title='Top 10 Job Categories', text_auto=True,
                          color='count', color_continuous_scale='Reds')
        fig3.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig3, use_column_width=True)

    with tabs[2]:
        st.subheader("Multivariate Analysis")
        fig4 = px.box(df, x='experience_level', y='skills_count', color='employment_type',
                      title='Skills Count by Experience Level & Employment Type')
        fig4.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig4, use_column_width=True)

        top_cities  = df['city'].value_counts().head(10).index
        df_city     = df[df['city'].isin(top_cities)]
        df_city_grp = df_city.groupby(['city', 'experience_level']).size().reset_index(name='count')
        fig5        = px.bar(df_city_grp, x='city', y='count', color='experience_level',
                             barmode='group', title='Experience Levels in Top 10 Cities', text_auto=True)
        fig5.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig5, use_column_width=True)

elif page == "Prediction":
    st.title("🔮 Experience Level Prediction")
    st.sidebar.markdown("### Job Specifications")

    employment_type   = st.sidebar.selectbox('Employment Type', sorted(df['employment_type'].unique()))
    work_mode         = st.sidebar.selectbox('Work Mode',       sorted(df['work_mode'].unique()))
    job_category      = st.sidebar.selectbox('Job Category',    sorted(df['job_category'].unique()))
    country           = st.sidebar.selectbox('Country',         sorted(df['country'].unique()))
    cities_in_country = sorted(df[df['country'] == country]['city'].unique())
    city              = st.sidebar.selectbox('City', cities_in_country)
    top_skill         = st.sidebar.selectbox('Top Skill',    sorted(df['top_skill'].dropna().unique()))
    second_skill      = st.sidebar.selectbox('Second Skill', sorted(df['second_skill'].dropna().unique()))
    exp_years         = st.sidebar.slider('Years of Experience', 0.0, 20.0, 3.0, 0.5)
    skills_count      = st.sidebar.slider('Number of Required Skills', 2, 8, 5)

    input_df = pd.DataFrame({
        'employment_type': [employment_type],
        'work_mode':       [work_mode],
        'job_category':    [job_category],
        'exp_years':       [exp_years],
        'city':            [city],
        'country':         [country],
        'skills_count':    [skills_count],
        'top_skill':       [top_skill],
        'second_skill':    [second_skill]
    })

    st.markdown("### 📋 Input Summary")
    st.table(input_df)

    if st.button("Predict Experience Level"):
        pred_idx        = model.predict(input_df)[0]
        probas          = model.predict_proba(input_df)[0]
        predicted_class = le.classes_[pred_idx]

        st.success(f"🎯 Predicted Experience Level: **{predicted_class}**")

        proba_df = pd.DataFrame({'Level': le.classes_, 'Probability': probas})
        fig_prob = px.bar(proba_df, x='Level', y='Probability',
                          title='Prediction Probabilities', text_auto='.2%',
                          color='Probability', color_continuous_scale='Greens')
        st.plotly_chart(fig_prob, use_column_width=True)
