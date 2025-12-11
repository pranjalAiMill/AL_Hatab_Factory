# 📦 AL Hatab Factory – Demand Forecasting (LightGBM)

This project builds an hourly demand forecasting model for AL Hatab Factory using **LightGBM**. It includes a clean modular codebase, automatic feature engineering, model training, and forecast generation.

---

## 🚀 Project Structure

```
AL_Hatab_Factory/
│
├── data/                      # Raw dataset (not pushed to GitHub)
│   └── factory_hourly_synthetic.csv
│
├── src/
│   ├── train_model.py         # Train LightGBM model
│   ├── inference.py             # Run inference on new data
│   └── dashboard.py           # Streamlit dashboard
│
├── models/                    # Saved models (ignored in GitHub)
│
├── outputs/
│   ├── test_predictions_sample.csv   # Sample predictions (public)
│   └── predictions.csv               # New forecasts from predict.py
│
└── README.md
```

---

## 🧠 Features

✔ LightGBM regression model
✔ Auto feature generation
✔ Train/test split
✔ Rolling & lag features
✔ Saves model + metrics
✔ Inference script to forecast future hourly quantities
✔ Outputs clean CSV files

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/AL_Hatab_Factory.git
cd AL_Hatab_Factory

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## 📘 Usage

### 1. Train Model

Place your CSV file inside the `data/` folder.

Then run:

```bash
python src/train_model.py
```

This will:

* Load your dataset
* Engineer features
* Train LightGBM
* Save the model into `models/`
* Generate sample predictions into `outputs/`

---

### 2. Generate Forecasts (Inference)

Prepare a new CSV with future timestamps and run:

```bash
python src/inference.py --input new_future_data.csv
```

Outputs:

```
outputs/predictions.csv
```

---

# 📈 Forecast Example

The model produced the following sample predictions found in:

```
outputs/test_predictions_sample.csv
```

| hour | sku_id  | y_true | y_pred |
| ---- | ------- | ------ | ------ |
| 14   | SKU_202 | 185    | 193.82 |
| 16   | SKU_202 | 192    | 195.67 |
| 18   | SKU_202 | 210    | 200.78 |
| 20   | SKU_202 | 205    | 198.32 |
| ...  | ...     | ...    | ...    |

You can open the complete file inside the `outputs/` folder.

---

## 📊 Sample Output File

The forecast CSV file includes:

```
timestamp, sku_id, predicted_qty
2025-01-01 13:00, SKU_101, 187.22
2025-01-01 14:00, SKU_101, 193.81
2025-01-01 15:00, SKU_101, 201.44
...
```

---

## 🛠 Technologies Used

* Python 3.10
* LightGBM
* Pandas
* Scikit-Learn
* NumPy

