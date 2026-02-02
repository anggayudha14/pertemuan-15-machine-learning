import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score
)

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Prediksi Bantuan Sosial",
    page_icon="🎯",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎯 Prediksi Kelayakan Penerima Bantuan Sosial")
st.caption(
    "Aplikasi machine learning untuk memprediksi kelayakan penerima bantuan sosial "
    "berdasarkan kriteria ekonomi dan sosial."
)

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    df = pd.read_excel("dataset.xlsx")
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    return df

df = load_data()

with st.expander("📊 Lihat Dataset"):
    st.dataframe(df, use_container_width=True)

# ======================
# PREPROCESSING
# ======================
target_col = df.columns[-1]
X = df.drop(columns=[target_col]).apply(pd.to_numeric)
y = df[target_col].map({"Yes": 1, "No": 0})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ======================
# TRAIN MODELS
# ======================
lr_model = LogisticRegression()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

lr_prob = lr_model.predict_proba(X_test)[:, 1]
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# ======================
# METRICS SUMMARY
# ======================
st.subheader("📈 Evaluasi Model")

acc_lr = accuracy_score(y_test, lr_pred)
acc_rf = accuracy_score(y_test, rf_pred)

col1, col2, col3 = st.columns(3)
col1.metric("Akurasi Logistic Regression", f"{acc_lr:.2f}")
col2.metric("Akurasi Random Forest", f"{acc_rf:.2f}")
col3.metric("Jumlah Data", df.shape[0])

# ======================
# BAR CHART AKURASI
# ======================
st.subheader("📊 Perbandingan Akurasi Model")

acc_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Akurasi": [acc_lr, acc_rf]
})

fig_acc, ax_acc = plt.subplots()
sns.barplot(data=acc_df, x="Model", y="Akurasi", ax=ax_acc)
ax_acc.set_ylim(0, 1)
st.pyplot(fig_acc)

# ======================
# CONFUSION MATRIX
# ======================
st.subheader("📊 Confusion Matrix")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Logistic Regression**")
    cm_lr = confusion_matrix(y_test, lr_pred)
    fig_lr, ax_lr = plt.subplots()
    sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", ax=ax_lr)
    ax_lr.set_xlabel("Predicted")
    ax_lr.set_ylabel("Actual")
    st.pyplot(fig_lr)

with col2:
    st.markdown("**Random Forest**")
    cm_rf = confusion_matrix(y_test, rf_pred)
    fig_rf, ax_rf = plt.subplots()
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens", ax=ax_rf)
    ax_rf.set_xlabel("Predicted")
    ax_rf.set_ylabel("Actual")
    st.pyplot(fig_rf)

# ======================
# ROC CURVE (2 MODEL)
# ======================
st.subheader("📉 ROC Curve")

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)

fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={roc_auc_score(y_test, lr_prob):.2f})")
ax_roc.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={roc_auc_score(y_test, rf_prob):.2f})")
ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend()
st.pyplot(fig_roc)

# ======================
# FEATURE IMPORTANCE
# ======================
st.subheader("⭐ Feature Importance")

tab1, tab2 = st.tabs(["Logistic Regression", "Random Forest"])

with tab1:
    lr_importance = np.abs(lr_model.coef_[0])
    fi_lr = pd.DataFrame({
        "Fitur": X.columns,
        "Importance": lr_importance
    }).sort_values(by="Importance", ascending=False)

    fig_fi_lr, ax_fi_lr = plt.subplots()
    sns.barplot(data=fi_lr, x="Importance", y="Fitur", ax=ax_fi_lr)
    st.pyplot(fig_fi_lr)

with tab2:
    rf_importance = rf_model.feature_importances_
    fi_rf = pd.DataFrame({
        "Fitur": X.columns,
        "Importance": rf_importance
    }).sort_values(by="Importance", ascending=False)

    fig_fi_rf, ax_fi_rf = plt.subplots()
    sns.barplot(data=fi_rf, x="Importance", y="Fitur", ax=ax_fi_rf)
    st.pyplot(fig_fi_rf)

# ======================
# PREDIKSI DATA BARU
# ======================
st.subheader("🧮 Prediksi Kelayakan")

model_choice = st.selectbox(
    "Pilih Model Prediksi",
    ["Logistic Regression", "Random Forest"]
)

input_data = []
for col in X.columns:
    val = st.number_input(f"{col}", value=0.0)
    input_data.append(val)

if st.button("Prediksi"):
    input_scaled = scaler.transform([input_data])

    if model_choice == "Logistic Regression":
        result = lr_model.predict(input_scaled)[0]
    else:
        result = rf_model.predict(input_scaled)[0]

    if result == 1:
        st.success("✅ Layak Menerima Bantuan Sosial")
    else:
        st.error("❌ Tidak Layak Menerima Bantuan Sosial")
