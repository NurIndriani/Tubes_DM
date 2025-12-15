import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, silhouette_score

st.set_page_config(
    page_title="Online Retail Clustering & Regression",
    layout="wide"
)

st.title("📊 Online Retail Clustering & Regression")

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_data():
    return pd.read_excel("Online Retail.xlsx")

df = load_data()

st.subheader("📄 Dataset Awal")
st.dataframe(df.head())

# =====================
# PREPROCESSING
# =====================
st.subheader("🧹 Preprocessing Data")

df = df.dropna(subset=["CustomerID"])
df = df[df["Quantity"] > 0]
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

st.write("✔ Menghapus CustomerID kosong")
st.write("✔ Menghapus Quantity ≤ 0")
st.write("✔ Membuat fitur TotalPrice")

# =====================
# FEATURE ENGINEERING
# =====================
customer_df = df.groupby("CustomerID").agg({
    "Quantity": "sum",
    "TotalPrice": "sum"
}).reset_index()

customer_df.columns = [
    "CustomerID",
    "TotalQuantity",
    "TotalSpending"
]

st.subheader("📌 Data Pelanggan (Hasil Agregasi)")
st.dataframe(customer_df.head())

# =====================
# SCALING
# =====================
scaler = StandardScaler()
scaled_data = scaler.fit_transform(
    customer_df[["TotalQuantity", "TotalSpending"]]
)

# =====================
# CLUSTERING
# =====================
st.subheader("📍 Clustering Pelanggan")

k = st.slider("Jumlah Cluster (K)", min_value=2, max_value=6, value=3)

kmeans = KMeans(n_clusters=k, random_state=42)
customer_df["Cluster"] = kmeans.fit_predict(scaled_data)

# Silhouette Score
sil_score = silhouette_score(scaled_data, customer_df["Cluster"])

st.metric("Silhouette Score", f"{sil_score:.3f}")

# Visualisasi clustering
fig, ax = plt.subplots()
scatter = ax.scatter(
    customer_df["TotalQuantity"],
    customer_df["TotalSpending"],
    c=customer_df["Cluster"]
)
ax.set_xlabel("Total Quantity")
ax.set_ylabel("Total Spending")
ax.set_title("Customer Segmentation (K-Means)")
st.pyplot(fig)

# =====================
# REGRESSION (ENSEMBLE)
# =====================
st.subheader("📈 Regression (Random Forest)")

X = customer_df[["TotalQuantity"]]
y = customer_df["TotalSpending"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# =====================
# EVALUASI REGRESI
# =====================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.metric("RMSE (Root Mean Squared Error)", f"{rmse:,.2f}")

# Visualisasi regresi
fig2, ax2 = plt.subplots()
ax2.scatter(X_test, y_test, label="Actual")
ax2.scatter(X_test, y_pred, label="Predicted")
ax2.set_xlabel("Total Quantity")
ax2.set_ylabel("Total Spending")
ax2.set_title("Random Forest Regression")
ax2.legend()
st.pyplot(fig2)

# =====================
# INPUT USER
# =====================
st.subheader("🧮 Prediksi Total Spending (Input User)")

input_qty = st.number_input(
    "Masukkan Total Quantity",
    min_value=1,
    value=10
)

if st.button("Prediksi"):
    pred_spending = rf.predict([[input_qty]])
    st.success(
        f"💰 Perkiraan Total Spending: {pred_spending[0]:,.2f}"
    )

st.success("✅ Analisis Clustering & Regression Berhasil!")
