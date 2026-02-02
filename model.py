import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

data = {
    "Pendapatan": np.random.randint(500000, 8000000, 200),
    "Jumlah_Tanggungan": np.random.randint(1, 6, 200),
    "Kondisi_Rumah": np.random.choice(["Baik", "Sedang", "Buruk"], 200),
    "Status_Pekerjaan": np.random.choice(
        ["Tidak Bekerja", "Tidak Tetap", "Tetap"], 200
    ),
    "Kepemilikan_Aset": np.random.choice(["Tidak Ada", "Ada"], 200),
    "Kelayakan": np.random.choice(["Tidak Layak", "Layak"], 200)
}

df = pd.DataFrame(data)

le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

X = df.drop("Kelayakan", axis=1)
y = df["Kelayakan"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=150, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

feature_names = X.columns
