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

k = st.slider("Jumlah Cluster (K)", 2, 6, 3)

kmeans = KMeans(n_clusters=k, random_state=42)
customer_df["Cluster"] = kmeans.fit_predict(scaled_data)

sil_score = silhouette_score(scaled_data, customer_df["Cluster"])
st.metric("Silhouette Score", f"{sil_score:.3f}")

# Visualisasi cluster
fig, ax = plt.subplots()
ax.scatter(
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
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
st.metric("RMSE", f"{rmse:,.2f}")

# =====================
# INPUT USER (PREDIKSI REGRESI + CLUSTER)
# =====================
st.subheader("🧮 Prediksi Total Spending & Cluster (Input User)")

input_qty = st.number_input(
    "Masukkan Total Quantity",
    min_value=1,
    value=10
)

if st.button("Prediksi"):
    # Prediksi total spending
    pred_spending = rf.predict([[input_qty]])[0]

    # Gabungkan fitur untuk clustering
    new_data = np.array([[input_qty, pred_spending]])
    new_data_scaled = scaler.transform(new_data)

    # Prediksi cluster
    pred_cluster = kmeans.predict(new_data_scaled)[0]

    st.success(f"💰 Perkiraan Total Spending: {pred_spending:,.2f}")
    st.info(f"📌 Pelanggan diprediksi masuk ke Cluster: {pred_cluster}")

st.success("✅ Analisis Clustering & Regression Berhasil!")
