import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Online Retail Clustering & Regression", layout="wide")
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
# DATA CLEANING
# =====================
df = df.dropna(subset=["CustomerID"])
df = df[df["Quantity"] > 0]
df = df.drop_duplicates()

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# =====================
# FEATURE ENGINEERING
# =====================
customer_df = df.groupby("CustomerID").agg({
    "Quantity": "sum",
    "TotalPrice": "sum"
}).reset_index()

customer_df.columns = ["CustomerID", "TotalQuantity", "TotalSpending"]

st.subheader("📌 Data Pelanggan (Setelah Agregasi)")
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
k = st.slider("Jumlah Cluster (K-Means)", 2, 6, 3)

kmeans = KMeans(n_clusters=k, random_state=42)
customer_df["Cluster"] = kmeans.fit_predict(scaled_data)

st.subheader("📍 Hasil Clustering")

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
# REGRESSION (ENSEMBLE METHOD)
# =====================
st.subheader("📈 Regression: Random Forest (Ensemble Method)")

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
# INPUT USER
# =====================
st.subheader("🧮 Input Data Pelanggan Baru")

input_quantity = st.number_input(
    "Masukkan Total Quantity Pembelian",
    min_value=1,
    step=1
)

if st.button("🔮 Prediksi & Tentukan Cluster"):
    # ---- REGRESSION PREDICTION
    pred_spending = rf.predict([[input_quantity]])[0]

    # ---- CLUSTER PREDICTION
    new_data = pd.DataFrame(
        [[input_quantity, pred_spending]],
        columns=["TotalQuantity", "TotalSpending"]
    )

    new_scaled = scaler.transform(new_data)
    cluster_result = kmeans.predict(new_scaled)[0]

    # =====================
    # OUTPUT
    # =====================
    st.success("✅ Hasil Prediksi")
    st.write(f"💰 **Prediksi Total Spending:** {pred_spending:,.2f}")
    st.write(f"📍 **Masuk ke Cluster:** {cluster_result}")

# =====================
# VISUALISASI REGRESI
# =====================
st.subheader("📉 Visualisasi Regresi")

y_pred = rf.predict(X_test)

fig2, ax2 = plt.subplots()
ax2.scatter(X_test, y_test, label="Actual")
ax2.scatter(X_test, y_pred, label="Predicted")
ax2.set_xlabel("Total Quantity")
ax2.set_ylabel("Total Spending")
ax2.set_title("Random Forest Regression")
ax2.legend()
st.pyplot(fig2)

st.success("🎉 Aplikasi berhasil dijalankan!")
