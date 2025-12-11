# src/dashboard.py
import streamlit as st
import pandas as pd
import subprocess
import os
from pathlib import Path
import joblib
import json

st.set_page_config(page_title="Factory Forecast Dashboard", layout="wide")

st.title("Factory Forecast Dashboard")

uploaded = st.file_uploader("Upload CSV (hourly factory file)", type=["csv"])
if uploaded is not None:
    data_path = Path("data") / "uploaded_for_inference.csv"
    os.makedirs("data", exist_ok=True)
    with open(data_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success("Saved uploaded file for inference.")

    if st.button("Run inference"):
        # call inference script
        st.info("Running inference...")
        subprocess.run(["python", "src/inference.py", "--data", str(data_path)])
        st.success("Inference completed. Results saved to outputs/predictions.csv")

    if Path("outputs/predictions.csv").exists():
        df_out = pd.read_csv("outputs/predictions.csv")
        st.subheader("Sample predictions")
        st.dataframe(df_out.head(200))

        st.markdown("### Download predictions")
        st.download_button("Download CSV", df_out.to_csv(index=False).encode("utf-8"), file_name="predictions.csv")

        # show feature importance if model exists
        if Path("models/factory_model.joblib").exists() and Path("models/feature_columns.json").exists():
            model = joblib.load("models/factory_model.joblib")
            with open("models/feature_columns.json", "r") as f:
                feat_cols = json.load(f)
            # feature importances
            import pandas as pd
            fi = pd.DataFrame({"feature": feat_cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
            st.subheader("Top 20 features")
            st.table(fi.head(20))
        else:
            st.info("No trained model found yet. Run training first.")
