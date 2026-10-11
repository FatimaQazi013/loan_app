import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ============ 1. PAGE SETUP ============
st.set_page_config(page_title="Loan Mining App", page_icon="🏦", layout="wide")
TEAL, CORAL, NAVY = "#0F766E", "#E4572E", "#1E3A5F"
COLORS = {"Y": TEAL, "N": CORAL}          # Y = approved (teal), N = rejected (coral)
sns.set_theme(style="whitegrid")

st.title("🏦 :blue[Loan Approval] :green[Data Mining App]")
st.caption("Explore the data, find patterns, and predict whether a loan will be approved.")

# ============ 2. LOAD + PREPROCESS ============
df = pd.read_csv("loan.csv")

# Sidebar: user decides how many rows to see in every table
st.sidebar.header("⚙️ Settings")
rows = st.sidebar.slider("Rows to show in tables", 5, len(df), 10)
st.sidebar.info(f"Dataset: {df.shape[0]} applicants, {df.shape[1]} columns")

# Missing values: median for numbers, mode (most common) for categories
df_clean = df.copy()
missing_before = df.isnull().sum()
df_clean["LoanAmount"] = df_clean["LoanAmount"].fillna(df_clean["LoanAmount"].median())
for col in ["Gender", "Married", "Dependents", "Self_Employed", "Loan_Amount_Term", "Credit_History"]:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# Outliers: find with IQR rule, then cap them (rows are not deleted)
num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
before_cap = df_clean.copy()
outliers, limits = {}, {}
for col in num_cols:
    q1, q3 = df_clean[col].quantile([0.25, 0.75])
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    limits[col] = (low, high)
    outliers[col] = int(((df_clean[col] < low) | (df_clean[col] > high)).sum())
    df_clean[col] = df_clean[col].clip(low, high)

# Encoding: text -> 0/1 numbers
df_model = df_clean.drop(columns="Loan_ID")
df_model["Loan_Status"] = df_model["Loan_Status"].map({"Y": 1, "N": 0})
cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
df_enc = pd.get_dummies(df_model, columns=cat_cols, drop_first=True, dtype=int)

# Scaling: same range for all numbers (Logistic Regression needs this)
scale_cols = num_cols + ["Loan_Amount_Term"]
scaler = StandardScaler()
df_scaled = df_enc.copy()
df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])

# ============ 3. TRAIN THE MODEL (once, used by two tabs) ============
X, y = df_scaled.drop(columns="Loan_Status"), df_scaled["Loan_Status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

# ============ 4. TABS ============
t_home, t_pre, t_eda, t_model, t_pred = st.tabs(
    ["🏠 Overview", "🧹 Preprocessing", "📈 EDA", "🤖 Model", "🔮 Prediction"])

# ---------- Overview ----------
with t_home:
    st.subheader("Dataset at a glance", divider="green")
    stats = [("Applicants", len(df)),
             ("Approval rate", f"{(df['Loan_Status'] == 'Y').mean() * 100:.1f}%"),
             ("Avg loan (thousands)", f"{df_clean['LoanAmount'].mean():.0f}"),
             ("Avg income", f"{df_clean['ApplicantIncome'].mean():,.0f}")]
    for box, (label, value) in zip(st.columns(4), stats):
        with box.container(border=True):
            st.metric(label, value)

    st.dataframe(df.head(rows), use_container_width=True)
    rate = df_clean.groupby("Credit_History")["Loan_Status"].apply(lambda s: (s == "Y").mean() * 100)
    st.info(f"💡 Applicants with good credit history are approved {rate[1.0]:.0f}% of the time, "
            f"against only {rate[0.0]:.0f}% with bad credit history.")

# ---------- Preprocessing ----------
with t_pre:
    st.subheader("1. Missing values", divider="blue")
    c1, c2 = st.columns(2)
    c1.write("Before cleaning")
    c1.dataframe(missing_before[missing_before > 0].rename("missing"))
    c2.write("After cleaning")
    c2.success("All missing values are filled ✅")

    st.subheader("2. Outliers (IQR rule)", divider="orange")
    col = st.selectbox("Choose a column", num_cols)
    high = limits[col][1]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    sns.boxplot(y=before_cap[col], color=CORAL, ax=ax[0])
    ax[0].axhline(high, color=NAVY, linestyle="--", label=f"Upper limit = {high:,.0f}")
    ax[0].legend()
    ax[0].set_title(f"Before capping ({outliers[col]} outliers)")
    sns.boxplot(y=df_clean[col], color=TEAL, ax=ax[1])
    ax[1].set_title("After capping")
    ax[1].set_ylabel("")
    fig.tight_layout()
    st.pyplot(fig)
    st.info(f"📝 {col} had {outliers[col]} outliers. Every value above the dashed line "
            f"({high:,.0f}) was brought down to that limit, so no rows were deleted. "
            f"Note: the two charts use different y-axis scales.")

    st.subheader("3. Encoding (text to 0/1)", divider="green")
    st.dataframe(df_enc.head(rows), use_container_width=True)

    st.subheader("4. Scaling (same range for all numbers)", divider="violet")
    st.dataframe(df_scaled.head(rows), use_container_width=True)

    st.download_button("⬇️ Download cleaned data (CSV)", df_clean.to_csv(index=False),
                       "loan_cleaned.csv", "text/csv")

# ---------- EDA ----------
with t_eda:
    plot = st.radio("Choose a plot", ["Histogram", "Boxplot", "Scatterplot", "Correlation Heatmap"],
                    horizontal=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    if plot == "Histogram":
        col = st.selectbox("Column", num_cols, key="h")
        sns.histplot(data=df_clean, x=col, hue="Loan_Status", palette=COLORS, kde=True, ax=ax)
        obs = f"{col} has skewness {df_clean[col].skew():.2f}. Above 0 means a few applicants have very large values."
    elif plot == "Boxplot":
        col = st.selectbox("Column", num_cols, key="b")
        sns.boxplot(data=df_clean, x="Loan_Status", y=col, palette=COLORS, ax=ax)
        m = df_clean.groupby("Loan_Status")[col].median()
        obs = f"Median {col}: approved = {m['Y']:.0f}, rejected = {m['N']:.0f}."
    elif plot == "Scatterplot":
        x = st.selectbox("X axis", num_cols, index=0, key="x")
        y_axis = st.selectbox("Y axis", num_cols, index=2, key="y")
        sns.scatterplot(data=df_clean, x=x, y=y_axis, hue="Loan_Status", palette=COLORS, alpha=0.6, ax=ax)
        obs = f"Correlation between {x} and {y_axis} is {df_clean[x].corr(df_clean[y_axis]):.2f}."
    else:
        cols = num_cols + ["Loan_Amount_Term", "Credit_History", "Loan_Status"]
        corr = df_model[cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        top = corr["Loan_Status"].drop("Loan_Status").abs().idxmax()
        obs = f"{top} is the feature most related to Loan_Status (r = {corr.loc[top, 'Loan_Status']:.2f})."

    fig.tight_layout()
    st.pyplot(fig)
    st.info("📝 Observation: " + obs)

# ---------- Data Mining: Logistic Regression ----------
with t_model:
    st.write("**Logistic Regression** gives every feature a *weight*, adds them up into a score, "
             "and turns the score into a probability between 0% and 100%. "
             "A positive weight pushes towards **approval**, a negative weight towards **rejection**.")

    train_acc, test_acc = model.score(X_train, y_train), model.score(X_test, y_test)
    baseline = max(y.mean(), 1 - y.mean())
    for box, (label, value) in zip(st.columns(3), [
            ("Train accuracy", f"{train_acc * 100:.1f}%"),
            ("Test accuracy", f"{test_acc * 100:.1f}%"),
            ("Guessing 'Approved' always", f"{baseline * 100:.1f}%")]):
        with box.container(border=True):
            st.metric(label, value)

    st.subheader("What matters most? (feature weights)", divider="green")
    weights = pd.Series(model.coef_[0], index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(weights.index, weights.values, color=[TEAL if w > 0 else CORAL for w in weights])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Weight (right = towards approval, left = towards rejection)")
    fig.tight_layout()
    st.pyplot(fig)
    big = weights.abs().idxmax()
    st.info(f"📝 Observation: {big} has the biggest weight "
            f"({'towards approval' if weights[big] > 0 else 'towards rejection'}), so it influences "
            f"the decision the most. The model scores {test_acc * 100:.1f}% on unseen test data, "
            f"compared with {baseline * 100:.1f}% for always guessing 'Approved'.")

# ---------- Prediction ----------
with t_pred:
    st.write("Fill in the applicant details and press **Predict**.")
    terms = sorted(df_clean["Loan_Amount_Term"].unique())

    with st.form("predict"):
        c1, c2, c3 = st.columns(3)
        gender = c1.selectbox("Gender", sorted(df_clean["Gender"].unique()))
        married = c1.selectbox("Married", sorted(df_clean["Married"].unique()))
        deps = c1.selectbox("Dependents", sorted(df_clean["Dependents"].unique()))
        edu = c2.selectbox("Education", sorted(df_clean["Education"].unique()))
        selfemp = c2.selectbox("Self employed", sorted(df_clean["Self_Employed"].unique()))
        area = c2.selectbox("Property area", sorted(df_clean["Property_Area"].unique()))
        credit = c3.selectbox("Credit history", [1.0, 0.0], format_func=lambda v: "Good" if v == 1.0 else "Bad")
        inc = c3.number_input("Applicant income", 0, value=5000, step=500)
        coinc = c3.number_input("Coapplicant income", 0, value=0, step=500)
        loan = c3.number_input("Loan amount (thousands)", 1, value=130, step=10)
        term = c3.selectbox("Loan term (months)", terms, index=terms.index(360.0) if 360.0 in terms else 0)
        go = st.form_submit_button("Predict", type="primary")

    if go:
        new = pd.DataFrame([{"Gender": gender, "Married": married, "Dependents": deps,
                             "Education": edu, "Self_Employed": selfemp,
                             "ApplicantIncome": float(inc), "CoapplicantIncome": float(coinc),
                             "LoanAmount": float(loan), "Loan_Amount_Term": float(term),
                             "Credit_History": float(credit), "Property_Area": area}])
        # Step 1: encode the new applicant exactly like the training data
        both = pd.concat([df_model.drop(columns="Loan_Status"), new], ignore_index=True)
        row = pd.get_dummies(both, columns=cat_cols, drop_first=True, dtype=int).iloc[[-1]][X.columns].copy()
        # Step 2: scale with the SAME scaler used for training
        row[scale_cols] = scaler.transform(row[scale_cols])

        p = model.predict_proba(row)[0][1]
        if p >= 0.5:
            st.success(f"✅ Likely APPROVED (approval probability {p * 100:.0f}%)")
        else:
            st.error(f"❌ Likely REJECTED (approval probability {p * 100:.0f}%)")
        st.progress(float(p))

        # Why? weight x value for each feature = how much it pushed the decision
        push = pd.Series(row.iloc[0].astype(float).values * model.coef_[0], index=X.columns)
        top6 = push.reindex(push.abs().sort_values(ascending=False).index)[:6].sort_values()
        st.subheader("Why did the model decide this?", divider="green")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.barh(top6.index, top6.values, color=[TEAL if v > 0 else CORAL for v in top6])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Push (right = towards approval, left = towards rejection)")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Shows the 6 features that influenced this applicant's result the most. "
                   "This is a model trained on past data, not a real bank decision.")
