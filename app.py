import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import StringIO
import datetime

st.set_page_config(page_title="Sales Forecasting System", page_icon="📈", layout="wide")

st.title("📈 Sales Forecasting System")
st.markdown("Upload your sales data to get predictions for the next 30 days using Linear Regression.")

# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a **Linear Regression** model to forecast future sales based on your historical data.\n\n"
    "**Required CSV columns:**\n"
    "- `Date` — date of sale (e.g. 2024-01-01)\n"
    "- `Sales` — numeric sales value"
)
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Python, Streamlit & Scikit-learn")


# --- File Upload ---
st.header("1. Upload Your Sales Data")
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Your CSV must have a 'Date' column and a 'Sales' column."
)

# Helper: load sample data
def load_sample():
    df = pd.read_csv("sample_sales.csv")
    return df

# Use sample data toggle
use_sample = st.checkbox("Use sample data instead", value=False)

if uploaded_file is not None:
    raw = uploaded_file.read().decode("utf-8")
    df = pd.read_csv(StringIO(raw))
elif use_sample:
    df = load_sample()
    st.info("Loaded built-in sample_sales.csv")
else:
    st.warning("Please upload a CSV file or check 'Use sample data instead' to get started.")
    st.stop()


# --- Preprocessing ---
st.header("2. Data Preview & Preprocessing")

# Validate columns
required_cols = {"Date", "Sales"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing column(s): {missing}. Your CSV must contain 'Date' and 'Sales'.")
    st.stop()

# Convert date
try:
    df["Date"] = pd.to_datetime(df["Date"])
except Exception as e:
    st.error(f"Could not parse 'Date' column: {e}")
    st.stop()

# Handle missing values
before = len(df)
df = df.dropna(subset=["Date", "Sales"])
after = len(df)
if before != after:
    st.warning(f"Removed {before - after} row(s) with missing values.")

# Convert sales to numeric
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
df = df.dropna(subset=["Sales"])

# Sort by date
df = df.sort_values("Date").reset_index(drop=True)

# Extract features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayIndex"] = (df["Date"] - df["Date"].min()).dt.days

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Dataset Preview")
    st.dataframe(df[["Date", "Sales", "Year", "Month", "Day"]].head(20), use_container_width=True)
with col2:
    st.subheader("Summary Statistics")
    total_sales = df["Sales"].sum()
    avg_sales = df["Sales"].mean()
    max_sales = df["Sales"].max()
    min_sales = df["Sales"].min()
    num_records = len(df)

    st.metric("Total Records", f"{num_records:,}")
    st.metric("Total Sales", f"${total_sales:,.2f}")
    st.metric("Average Daily Sales", f"${avg_sales:,.2f}")
    st.metric("Max Sales", f"${max_sales:,.2f}")
    st.metric("Min Sales", f"${min_sales:,.2f}")


# --- Sales Trend Chart ---
st.header("3. Sales Trend")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["Date"], df["Sales"], color="#4C9BE8", linewidth=1.8, label="Actual Sales")
ax.fill_between(df["Date"], df["Sales"], alpha=0.15, color="#4C9BE8")
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Sales ($)", fontsize=11)
ax.set_title("Historical Sales Trend", fontsize=13, fontweight="bold")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
st.pyplot(fig)


# --- Model Training ---
st.header("4. Linear Regression Model")

X = df[["DayIndex"]].values
y = df["Sales"].values

model = LinearRegression()
model.fit(X, y)
train_score = model.score(X, y)

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Model R² Score", f"{train_score:.4f}")
col_m2.metric("Coefficient (slope)", f"{model.coef_[0]:.4f}")
col_m3.metric("Intercept", f"{model.intercept_:.4f}")

with st.expander("ℹ️ What does R² mean?"):
    st.write(
        "The **R² (R-squared) score** measures how well the model explains the variance in your data. "
        "A score of 1.0 means a perfect fit; a score close to 0 means the model explains very little. "
        "For real-world sales data with seasonality or noise, 0.5–0.8 is generally acceptable."
    )


# --- Future Forecast ---
st.header("5. 30-Day Sales Forecast")

last_day_index = int(df["DayIndex"].max())
last_date = df["Date"].max()

future_days = np.arange(last_day_index + 1, last_day_index + 31).reshape(-1, 1)
future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 31)]
future_sales = model.predict(future_days)

# Clip negatives (sales can't be negative)
future_sales = np.clip(future_sales, 0, None)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Sales ($)": np.round(future_sales, 2)
})

col_a, col_b = st.columns([1, 1])

with col_a:
    st.subheader("Forecast Table")
    st.dataframe(forecast_df.style.format({"Predicted Sales ($)": "${:,.2f}"}), use_container_width=True)

with col_b:
    st.subheader("Forecast Chart")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(df["Date"], df["Sales"], color="#4C9BE8", linewidth=1.5, label="Historical Sales")
    ax2.plot(forecast_df["Date"], forecast_df["Predicted Sales ($)"],
             color="#E84C4C", linewidth=2, linestyle="--", label="Predicted Sales")
    ax2.axvline(x=last_date, color="gray", linestyle=":", linewidth=1.2, label="Today")
    ax2.set_xlabel("Date", fontsize=10)
    ax2.set_ylabel("Sales ($)", fontsize=10)
    ax2.set_title("Historical + 30-Day Forecast", fontsize=12, fontweight="bold")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig2)


# --- Download Button ---
st.header("6. Download Forecast Results")

csv_out = forecast_df.to_csv(index=False)
st.download_button(
    label="⬇️ Download Forecast as CSV",
    data=csv_out,
    file_name="sales_forecast_30days.csv",
    mime="text/csv",
    help="Download the predicted sales for the next 30 days as a CSV file."
)

st.success("Forecast complete! Use the download button above to save your results.")
