# retail-product-demand-prediction
Retail Product Demand Prediction using Python, Scikit-learn, Random Forest Regression, and Streamlit to forecast product demand based on pricing, promotions, seasonality, inventory, and competitor pricing.
Project Overview

Accurate demand forecasting is important for retail businesses because it helps with inventory planning, pricing decisions, promotional strategies, and revenue optimization.

This project uses historical retail data to predict expected product demand based on five key factors:

Product price
Promotion status
Season
Available stock
Competitor price

The trained Random Forest model predicts the expected number of units that may be demanded for a given retail scenario.

The project also includes a Streamlit dashboard that allows users to interactively simulate different pricing, promotion, seasonal, and inventory conditions.

Objectives

The main objectives of this project are:

Predict retail product demand using machine learning.
Analyze the relationship between pricing and demand.
Understand the effect of promotions and seasonality.
Compare product pricing with competitor pricing.
Estimate expected sales revenue.
Identify potential stockout situations.
Support batch demand forecasting for multiple products.
Evaluate machine learning model performance.
Provide an interactive forecasting dashboard.
Dataset

The project uses a synthetic retail dataset containing 800 records and 6 columns.

Dataset Features
Feature	Description
product_price	Retail selling price of the product
promotion_active	Indicates whether a promotion is active
season	Demand season such as High, Medium, or Low
stock_available	Current inventory available for sale
competitor_price	Price of a comparable competitor product
demand_units	Actual product demand in units and target variable
Dataset Statistics
Property	Value
Number of records	800
Number of features	5 input features
Target variable	demand_units
Missing values	0
Season categories	3

The dataset is synthetic and is intended for educational and machine learning demonstration purposes.

Machine Learning Model

The project uses:

RandomForestRegressor

The model is configured with:

RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

Random Forest Regression combines multiple decision trees to predict a continuous numerical value.

The target variable is:

demand_units

The final prediction represents the expected number of product units demanded.

Data Preprocessing

The project uses a Scikit-learn ColumnTransformer to process numerical and categorical data.

Categorical Feature

The season column is converted using:

OneHotEncoder(handle_unknown="ignore")

This converts the three season categories into numerical representations suitable for the machine learning model.

Numerical Features

The following features are passed directly to the Random Forest model:

product_price
promotion_active
stock_available
competitor_price

The complete preprocessing and prediction process is implemented as a Scikit-learn Pipeline.

Machine Learning Workflow
Retail Dataset
      |
      v
Separate Features and Target
      |
      v
Categorical Encoding
      |
      v
Train-Test Split
      |
      v
Random Forest Regression
      |
      v
Model Evaluation
      |
      v
Save Trained Model
      |
      v
Demand Prediction
      |
      v
Business and Inventory Analysis
Model Training

The dataset is divided into:

80% Training Data
20% Testing Data

The split uses:

train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

The Random Forest model contains 200 decision trees.

Model Performance

The model was evaluated using three standard regression metrics.

R² Score
R² Score: 0.9227

The R² score indicates that the model explains approximately 92.27% of the variation in the test-set demand values.

Mean Absolute Error
MAE: 11.24 units

On average, the model's prediction differs from the actual demand by approximately 11.24 units.

Root Mean Squared Error
RMSE: 14.12 units

RMSE gives greater weight to larger prediction errors.

Performance Summary
Metric	Result
R² Score	0.9227
MAE	11.24 units
RMSE	14.12 units
Model	Random Forest Regressor
Number of Trees	200

These results are based on the included dataset and the project's fixed train-test split.

Streamlit Application

The project includes an interactive Streamlit dashboard called:

Retail Product Demand & Revenue AI

The application provides three major sections.

1. Demand and Revenue Simulator

The simulator allows users to enter or modify:

Product price
Competitor price
Promotion status
Season
Available stock

The model then predicts expected product demand.

The application also calculates:

Predicted demand
Expected sales revenue
Price difference compared with competitors
Promotional demand uplift
Fulfilled units
Unmet demand
Potential lost revenue
Inventory coverage
2. Inventory Analysis

The application compares predicted demand with available inventory.

It identifies three possible inventory conditions:

Potential Stockout

When predicted demand is greater than available stock.

The application calculates the number of units that may be unavailable and estimates the corresponding lost revenue.

Elevated Inventory Buffer

When available stock is significantly higher than expected demand.

This can indicate excess inventory and may suggest the need for promotional strategies.

Well-Calibrated Inventory

When available stock is sufficient to meet predicted demand without creating a significant excess.

3. Batch Demand Forecasting

The application supports uploading a CSV file containing multiple products.

Required columns:

product_price
promotion_active
season
stock_available
competitor_price

The application generates predictions for all uploaded products.

For each product, it can calculate:

Predicted demand
Projected revenue
Stock deficit
Lost revenue risk
Inventory alignment

The resulting forecast can be downloaded as a CSV file.

Commercial Analysis

The application extends the machine learning prediction into basic retail business analysis.

Price Positioning

The application compares the product price with the competitor benchmark price.

Price Difference =
Competitor Price - Product Price

The percentage difference is also calculated to understand relative price positioning.

Revenue Estimation

Potential revenue is calculated as:

Potential Revenue =
Predicted Demand × Product Price

Realized revenue is limited by available inventory:

Realized Revenue =
Fulfilled Units × Product Price
Unmet Demand

When demand exceeds available inventory:

Unmet Demand =
Predicted Demand - Available Stock

The application then estimates potential lost revenue:

Lost Revenue =
Unmet Demand × Product Price
Promotional Impact Analysis

The application performs a counterfactual prediction by comparing demand under the current promotion setting with demand under the opposite promotion setting.

This provides an estimated promotional demand uplift.

For example:

Demand with Promotion
        vs.
Demand without Promotion

The difference is used as an estimated change in demand associated with the promotion setting.

This should be interpreted as a model-based comparison rather than a causal estimate.

Example Prediction

The included command-line prediction script uses the following sample scenario:

Product Price:       $45.00
Promotion Active:   Yes
Season:             High
Stock Available:    250 units
Competitor Price:   $42.50

The trained model generates the expected product demand for this scenario.

Project Structure
Retail_Product_Demand_Prediction_Sklearn/
|
├── app.py
├── predict.py
├── train_model.py
├── requirements.txt
├── retail_product_demand_model.pkl
├── actual_vs_predicted.png
├── data/
│   └── retail_product_demand.csv
└── README.md
File Description
train_model.py

Trains and evaluates the Random Forest Regression model.

The script:

Loads the retail dataset.
Separates input features and target.
Encodes the categorical season feature.
Splits the data into training and testing sets.
Trains the Random Forest model.
Generates predictions.
Calculates MAE, RMSE, and R².
Saves the trained model.
Generates the actual-versus-predicted chart.
predict.py

Loads the trained model and generates a demand prediction for a predefined sample retail scenario.

app.py

Runs the Streamlit web application.

It provides:

Individual demand prediction
Revenue estimation
Inventory analysis
Promotion comparison
Batch CSV forecasting
Model performance information
Actual-versus-predicted visualization
retail_product_demand_model.pkl

Contains the trained Scikit-learn pipeline, including preprocessing and the Random Forest Regression model.

retail_product_demand.csv

Contains the synthetic retail dataset used for model training and evaluation.

actual_vs_predicted.png

A scatter plot comparing actual demand values with the model's predicted demand values on the test dataset.

requirements.txt

Contains the Python packages required to run the project.

Technologies Used
Python
Pandas
NumPy
Scikit-learn
Random Forest Regression
OneHotEncoder
Joblib
Matplotlib
Streamlit
Installation
1. Clone the Repository
git clone https://github.com/your-username/Retail_Product_Demand_Prediction_Sklearn.git
2. Navigate to the Project Directory
cd Retail_Product_Demand_Prediction_Sklearn
3. Install Dependencies
pip install -r requirements.txt
Running the Project
Train the Model
python train_model.py

This trains the Random Forest model and generates:

retail_product_demand_model.pkl
actual_vs_predicted.png
Run the Prediction Script
python predict.py
Run the Streamlit Application
streamlit run app.py

The application will open in your browser.

Input Format for Batch Prediction

A CSV file uploaded to the Streamlit application should contain:

product_price,promotion_active,season,stock_available,competitor_price
45.00,1,High,250,42.50
112.40,0,Medium,90,91.35
18.59,0,High,180,15.07
84.02,0,Low,110,76.57
131.05,1,High,140,122.68

The application will add the predicted demand and business-analysis columns to the uploaded data.

Key Advantages
Uses an ensemble machine learning algorithm.
Handles both numerical and categorical variables.
Provides strong predictive performance on the included dataset.
Offers an interactive Streamlit interface.
Supports individual demand forecasting.
Supports batch product forecasting.
Provides inventory-risk analysis.
Estimates potential revenue and lost revenue.
Includes promotional scenario comparison.
Provides model evaluation metrics.
Includes actual-versus-predicted visualization.
Future Enhancements

Possible improvements include:

Add time-series forecasting models.
Include historical sales dates and trends.
Add product categories and SKU-level information.
Add customer behavior features.
Compare Random Forest with Gradient Boosting, XGBoost, and other regression models.
Add hyperparameter tuning.
Add cross-validation.
Add feature-importance visualization.
Add interactive demand-vs-price curves.
Add automated inventory reorder recommendations.
Add database integration.
Deploy the application online.
Add real-world retail datasets.
Limitations

The dataset is synthetic and may not represent real-world retail purchasing behavior.

Demand is influenced by many factors that are not included in the dataset, such as:

Product category
Brand
Customer demographics
Historical sales trends
Geographic location
Competitor availability
Marketing expenditure
Holidays
Economic conditions

Therefore, predictions should be treated as model-based estimates rather than guaranteed future sales.

The promotional uplift displayed by the application is based on changing the promotion input and comparing model predictions. It should not be interpreted as proof that promotions causally increase demand.
