from pathlib import Path
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent
model = joblib.load(BASE / "retail_product_demand_model.pkl")

sample = pd.DataFrame([{
    "product_price": 45.00,
    "promotion_active": 1,
    "season": "High",
    "stock_available": 250,
    "competitor_price": 42.50
}])

prediction = max(0, float(model.predict(sample)[0]))
print(f"Predicted Product Demand: {prediction:.0f} units")
