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
    classification_report,
    roc_curve,
    roc_auc_score
)

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Prediksi Bantuan Sosial",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 Prediksi Kelayakan Penerima Bantuan Sosial")
st.markdown(
    "Aplikasi ini digunakan untuk memprediksi kelayakan penerima bantuan sosial "
    "berdasarkan data kriteria ekonomi dan sosial."
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

# ======================
# DATA PREVIEW
# ======================
with st.expander("📊 Lihat Data"):
    st.dataframe(df, use_container_width=True)

# ======================
# PREPROCESSING
# ======================
target_col = df.columns[-1]  # kolom terakhir sebagai target
X = df.drop(columns=[target_col])
y = df[target_col]

X = X.apply(pd.to_numeric)
y = y.map({"Yes": 1, "No": 0})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ======================
# MODEL SELECTION
# ======================
model_choice = st.sidebar.selectbox(
    "🧠 Pilih Model",
    ["Logistic Regression", "Random Forest"]
)

if model_choice == "Logistic Regression":
    model = LogisticRegression()
else:
    model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ======================
# METRICS
# ======================
st.subheader("📈 Evaluasi Model")

col1, col2, col3 = st.columns(3)
col1.metric("Akurasi", f"{model.score(X_test, y_test):.2f}")
col2.metric("ROC AUC", f"{roc_auc_score(y_test, y_prob):.2f}")
col3.metric("Jumlah Data", df.shape[0])

# ======================
# CONFUSION MATRIX
# ======================
st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

# ======================
# ROC CURVE
# ======================
st.subheader("📉 ROC Curve")

fpr, tpr, _ = roc_curve(y_test, y_prob)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
ax_roc.plot([0, 1], [0, 1], linestyle="--")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend()
st.pyplot(fig_roc)

# ======================
# FEATURE IMPORTANCE
# ======================
st.subheader("⭐ Feature Importance")

if model_choice == "Random Forest":
    importance = model.feature_importances_
else:
    importance = np.abs(model.coef_[0])

fi_df = pd.DataFrame({
    "Fitur": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

fig_fi, ax_fi = plt.subplots()
sns.barplot(data=fi_df, x="Importance", y="Fitur", ax=ax_fi)
st.pyplot(fig_fi)

# ======================
# DISTRIBUTION PLOT
# ======================
st.subheader("📦 Distribusi Data")

selected_feature = st.selectbox(
    "Pilih Fitur",
    X.columns
)

fig_dist, ax_dist = plt.subplots()
sns.histplot(df[selected_feature], kde=True, ax=ax_dist)
st.pyplot(fig_dist)

# ======================
# PREDICTION INPUT
# ======================
st.subheader("🧮 Prediksi Data Baru")

input_data = []
for col in X.columns:
    val = st.number_input(f"Masukkan {col}", value=0.0)
    input_data.append(val)

if st.button("Prediksi Kelayakan"):
    input_scaled = scaler.transform([input_data])
    result = model.predict(input_scaled)[0]

    if result == 1:
        st.success("✅ Layak Menerima Bantuan Sosial")
    else:
        st.error("❌ Tidak Layak Menerima Bantuan Sosial")
