import csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


FEATURE_COLUMNS = [
    "Company",
    "TypeName",
    "Ram",
    "Gpu",
    "OpSys",
    "Weight",
    "TouchScreen",
    "IPS",
    "PPI",
    "Cpu_Brand",
    "HDD",
    "SSD",
]

GPU_MAP = {"Intel": 0, "AMD": 1, "Nvidia": 2}
OS_MAP = {"Windows": 0, "Mac": 1, "Other": 2}
CPU_MAP = {
    "Intel Core i3": 3,
    "Intel Core i5": 5,
    "Intel Core i7": 7,
    "Other Intel": 1,
    "AMD": 2,
    "Other": 0,
}


@st.cache_resource
def load_model():
    return joblib.load("xgb_full_pipeline.pkl")


def load_choices():
    companies = set()
    types = set()
    train_path = Path("train-data.csv")

    if not train_path.exists():
        return sorted(companies), sorted(types)

    with train_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get("Company"):
                companies.add(row["Company"])
            if row.get("TypeName"):
                types.add(row["TypeName"])

    return sorted(companies), sorted(types)


st.set_page_config(page_title="Laptop Price Predictor", page_icon=":computer:", layout="centered")
st.title("Laptop Price Predictor")

model = load_model()
company_options, type_options = load_choices()

col1, col2 = st.columns(2)
with col1:
    company = st.selectbox("Company", company_options or ["Apple"])
    laptop_type = st.selectbox("Type", type_options or ["Ultrabook"])
    ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64], index=3)
    cpu_brand = st.selectbox("CPU Brand", list(CPU_MAP.keys()), index=1)
    gpu_brand = st.selectbox("GPU Brand", list(GPU_MAP.keys()), index=0)
    os_name = st.selectbox("Operating System", list(OS_MAP.keys()), index=0)

with col2:
    weight = st.number_input("Weight (kg)", min_value=0.5, max_value=6.0, value=1.8, step=0.05)
    resolution_options = [
        "1366x768",
        "1600x900",
        "1920x1080",
        "1920x1200",
        "2160x1440",
        "2304x1440",
        "2560x1600",
        "2880x1800"
    ]
    resolution = st.selectbox("Resolution (WxH)", resolution_options, index=2)
    inches = st.number_input("Screen Size (Inches)", min_value=8.0, max_value=25.0, value=15.6, step=0.1)
    touchscreen = st.selectbox("TouchScreen", ["No", "Yes"], index=0)
    ips = st.selectbox("IPS", ["No", "Yes"], index=0)
    hdd = st.selectbox("HDD (GB)", [0, 128, 256, 500, 512, 1000, 2000], index=0)
    ssd = st.selectbox("SSD (GB)", [0, 8, 16, 24, 32, 64, 128, 256, 512, 1000, 2000], index=6)

if st.button("Predict Price", type="primary"):
    width_str, height_str = resolution.lower().split("x")
    width = int(width_str)
    height = int(height_str)
    ppi = float(round(((width ** 2 + height ** 2) ** 0.5) / float(inches), 2))

    input_df = pd.DataFrame(
        [
            {
                "Company": company,
                "TypeName": laptop_type,
                "Ram": int(ram),
                "Gpu": int(GPU_MAP[gpu_brand]),
                "OpSys": int(OS_MAP[os_name]),
                "Weight": float(weight),
                "TouchScreen": 1 if touchscreen == "Yes" else 0,
                "IPS": 1 if ips == "Yes" else 0,
                "PPI": ppi,
                "Cpu_Brand": int(CPU_MAP[cpu_brand]),
                "HDD": int(hdd),
                "SSD": int(ssd),
            }
        ]
    )[FEATURE_COLUMNS]

    try:
        pred_log = float(model.predict(input_df)[0])
        pred_price = float(np.expm1(pred_log))

        st.success(f"Estimated Price: INR {pred_price:,.0f}")
        st.caption(f"Computed PPI: {ppi}")
        st.caption(f"Model output (log price): {pred_log:.4f}")
    except Exception as err:
        st.error(f"Prediction failed: {err}")
