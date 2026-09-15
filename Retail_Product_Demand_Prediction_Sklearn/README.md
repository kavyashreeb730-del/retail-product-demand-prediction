# Retail Product Demand Prediction using Python and Scikit-learn

## Project Overview
This machine learning project predicts retail product demand in units using product price, promotion status, season, stock availability, and competitor price.

## Technology
- Python
- Pandas
- Scikit-learn
- Random Forest Regressor
- OneHotEncoder
- Joblib
- Matplotlib

## Files
- `train_model.py` - trains and evaluates the model
- `predict.py` - predicts demand for a sample product
- `retail_product_demand_model.pkl` - trained model
- `data/retail_product_demand.csv` - synthetic dataset
- `actual_vs_predicted.png` - evaluation chart
- `requirements.txt` - required libraries

## Run
```bash
pip install -r requirements.txt
python train_model.py
python predict.py
streamlit run app.py
```

## Output
The model predicts expected product demand in units.

Note: The dataset is synthetic and intended for educational purposes only.
