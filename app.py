import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Online Retail Clustering & Regression",
    layout="wide"
)

st.title("📊 Clustering & Regression Online Retail")

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_data():
    return pd.read_excel("Online Retail.xlsx")

df = load_data()

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# =====================
# PREPROCESSING
# =====================
st.subheader("🧹 Preprocessing Data")

# 1. Hapus missing CustomerID
df = df.dropna(subset=["CustomerID"])

# 2. Hapus transaksi retur
df = df[df["Quantity"] > 0]

# 3. Hapus duplikasi
df = df.drop_duplicates()

# 4. Feature Engineering
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

st.write("Jumlah data setelah preprocessing:", df.shape[0])

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

st.subheader("📌 Data Pelanggan (Agregasi)")
st.dataframe(customer_df.head())

# =====================
# SCALING
# =====================
scaler = StandardScaler()
scaled_data = scaler.fit_transform(
    customer_df[["TotalQuantity", "TotalSpending"]]
)

# =====================
# CLUSTERING (K-MEANS)
# =====================
st.subheader("📍 Clustering Pelanggan (K-Means)")

k = st.slider("Jumlah Cluster", 2, 6, 3)

kmeans = KMeans(n_clusters=k, random_state=42)
customer_df["Cluster"] = kmeans.fit_predict(scaled_data)

fig1, ax1 = plt.subplots()
ax1.scatter(
    customer_df["TotalQuantity"],
    customer_df["TotalSpending"],
    c=customer_df["Cluster"]
)
ax1.set_xlabel("Total Quantity")
ax1.set_ylabel("Total Spending")
ax1.set_title("Customer Segmentation")
st.pyplot(fig1)

# =====================
# REGRESSION (ENSEMBLE)
# =====================
st.subheader("📈 Regresi (Random Forest - Ensemble Method)")

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

# =====================
# EVALUASI RMSE
# =====================
y_pred_test = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

st.metric(
    label="RMSE (Root Mean Squared Error)",
    value=f"{rmse:,.2f}"
)

# =====================
# INPUT USER
# =====================
st.subheader("🧮 Prediksi Pelanggan Baru")

input_quantity = st.number_input(
    "Masukkan Total Quantity Pembelian",
    min_value=1,
    step=1
)

if st.button("🔮 Prediksi & Tentukan Cluster"):
    # Prediksi regresi
    pred_spending = rf.predict([[input_quantity]])[0]

    # Prediksi cluster
    new_data = pd.DataFrame(
        [[input_quantity, pred_spending]],
        columns=["TotalQuantity", "TotalSpending"]
    )

    new_scaled = scaler.transform(new_data)
    cluster_result = kmeans.predict(new_scaled)[0]

    st.success("✅ Hasil Prediksi")
    st.write(f"💰 **Prediksi Total Spending:** {pred_spending:,.2f}")
    st.write(f"📍 **Cluster Pelanggan:** {cluster_result}")

# =====================
# VISUALISASI REGRESI
# =====================
st.subheader("📉 Visualisasi Regresi")

fig2, ax2 = plt.subplots()
ax2.scatter(X_test, y_test, label="Actual")
ax2.scatter(X_test, y_pred_test, label="Predicted")
ax2.set_xlabel("Total Quantity")
ax2.set_ylabel("Total Spending")
ax2.set_title("Random Forest Regression")
ax2.legend()
st.pyplot(fig2)

st.success("🎉 Aplikasi berhasil dijalankan!")
