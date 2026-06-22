# Sales Forecasting System

A beginner-friendly web app that uses **Linear Regression** to forecast your sales for the next 30 days, built with Python and Streamlit.

---

## Features

- Upload your own CSV sales data or use the included sample
- Automatic preprocessing: date parsing, feature extraction, missing value handling
- Dataset preview with key summary statistics (total, average, max, min sales)
- Interactive sales trend chart
- 30-day sales forecast with chart and table
- Download forecast results as a CSV file

---

## Requirements

- Python 3.8+
- pip packages listed in `requirements.txt`

---

## Quick Start (Local)

### 1. Clone or download this project

```bash
git clone https://github.com/your-username/sales-forecasting.git
cd sales-forecasting
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## CSV Format

Your CSV file must include these two columns:

| Date       | Sales   |
|------------|---------|
| 2024-01-01 | 1520.50 |
| 2024-01-02 | 1380.00 |
| ...        | ...     |

- **Date** — any standard date format (e.g. `YYYY-MM-DD`)
- **Sales** — numeric sales values (no currency symbols)

A ready-to-use sample file (`sample_sales.csv`) is included.

---

## Project Structure

```
sales-forecasting/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── sample_sales.csv    # Sample data for testing
├── README.md           # This file
└── .streamlit/
    └── config.toml     # Streamlit server configuration
```

---

## How It Works

1. **Upload** your CSV or use sample data.
2. **Preprocessing**: dates are parsed, year/month/day columns are extracted, missing values are dropped.
3. **Training**: a `LinearRegression` model from scikit-learn is fitted on a sequential day index vs. sales.
4. **Forecast**: the model predicts sales for the next 30 days and displays the results as a chart and table.
5. **Download**: click the download button to save the forecast as `sales_forecast_30days.csv`.

---

## Deployment

### GitHub

1. Create a new repository on [github.com](https://github.com)
2. Push your files:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/your-username/sales-forecasting.git
git push -u origin main
```

### Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**.
3. Select your repository, branch (`main`), and set the main file to `app.py`.
4. Click **Deploy** — Streamlit Cloud installs dependencies from `requirements.txt` automatically.
5. Your app is live at `https://your-username-sales-forecasting-app-xxxx.streamlit.app`.

> **Tip:** Make sure `requirements.txt` is committed to your repository. Streamlit Cloud reads it to install packages.

---

## Tech Stack

| Library       | Purpose                        |
|---------------|--------------------------------|
| Streamlit     | Web UI framework               |
| Pandas        | Data loading and manipulation  |
| NumPy         | Numerical computations         |
| Scikit-learn  | Linear Regression model        |
| Matplotlib    | Charts and visualizations      |
