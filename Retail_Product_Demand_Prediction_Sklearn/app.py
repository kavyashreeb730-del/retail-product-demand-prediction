import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Retail Product Demand AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Commercial & Retail Merchandising Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #4338CA 100%);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 16px;
        padding: 26px 32px;
        margin-bottom: 24px;
        box-shadow: 0 14px 30px -8px rgba(0, 0, 0, 0.45);
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        background: rgba(99, 102, 241, 0.2);
        color: #A5B4FC;
        border: 1px solid rgba(165, 180, 252, 0.4);
        margin-bottom: 10px;
        text-transform: uppercase;
    }

    .hero-stat-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.75) 100%);
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        padding: 22px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .hero-stat-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.6);
    }

    .status-stockout {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(185, 28, 28, 0.08) 100%);
        border: 1px solid rgba(239, 68, 68, 0.45);
    }

    .status-healthy {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.18) 0%, rgba(5, 150, 105, 0.08) 100%);
        border: 1px solid rgba(16, 185, 129, 0.45);
    }

    .status-surplus {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.18) 0%, rgba(217, 119, 6, 0.08) 100%);
        border: 1px solid rgba(245, 158, 11, 0.45);
    }

    .insight-card {
        background: rgba(30, 41, 59, 0.55);
        border-left: 4px solid #6366F1;
        padding: 16px 20px;
        border-radius: 0 12px 12px 0;
        margin-top: 14px;
        color: #E2E8F0;
    }

    .sub-metric {
        font-size: 0.85rem;
        color: #94A3B8;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to find assets
def get_asset_path(filename):
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path(filename),
        script_dir / filename,
        script_dir / "Retail_Product_Demand_Prediction_Sklearn" / filename,
        script_dir.parent / filename,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return filename

# Load Model Pipeline
@st.cache_resource
def load_model():
    model_path = get_asset_path("retail_product_demand_model.pkl")
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Error loading product demand model: {e}")
    st.stop()

# Header
st.markdown("""
<div class="main-header">
    <div class="badge-pill">Commercial Demand Forecasting Engine</div>
    <h1 style="color: white; margin: 0; font-size: 2.3rem; font-weight: 800;">
        📈 Retail Product Demand & Revenue AI
    </h1>
    <p style="color: #C7D2FE; margin-top: 8px; margin-bottom: 0; font-size: 1.05rem;">
        Forecast unit sales volume, assess price elasticity vs competitors, and prevent lost revenue using Random Forest AI (R² 0.923).
    </p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab_simulator, tab_batch, tab_analytics = st.tabs([
    "🎯 Demand & Revenue Simulator",
    "📑 Batch Multi-SKU Demand Forecast",
    "📊 Model Performance & Price Elasticity"
])

# -------------------------------------------------------------
# TAB 1: Demand & Revenue Simulator
# -------------------------------------------------------------
with tab_simulator:
    st.markdown("### 🛍️ SKU Demand Prediction & Inventory Coverage")
    st.caption("Adjust merchandising parameters or select an industry retail scenario to simulate unit demand and expected sales revenue.")

    # Industry Quick Presets
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        load_holiday = st.button("⚡ Holiday Flash Sale", width="stretch")
    with p2:
        load_value = st.button("🏷️ Everyday Value Leader", width="stretch")
    with p3:
        load_luxury = st.button("💎 Premium Luxury SKU", width="stretch")
    with p4:
        load_clearance = st.button("📉 Off-Season Clearance", width="stretch")

    # Defaults
    s_price = 45.00
    s_promo = 1
    s_season = "High"
    s_stock = 250
    s_comp_price = 42.50

    if load_holiday:
        s_price = 39.99
        s_promo = 1
        s_season = "High"
        s_stock = 320
        s_comp_price = 49.99
    elif load_value:
        s_price = 22.00
        s_promo = 0
        s_season = "Medium"
        s_stock = 200
        s_comp_price = 28.50
    elif load_luxury:
        s_price = 135.00
        s_promo = 0
        s_season = "High"
        s_stock = 150
        s_comp_price = 120.00
    elif load_clearance:
        s_price = 35.00
        s_promo = 1
        s_season = "Low"
        s_stock = 120
        s_comp_price = 40.00

    with st.form("demand_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏷️ Pricing & Competitive Positioning")
            product_price = st.number_input(
                "Our Retail Product Price ($)",
                min_value=1.0,
                max_value=250.0,
                value=float(s_price),
                step=1.0,
                help="Selling price per unit."
            )

            competitor_price = st.number_input(
                "Competitor Benchmark Price ($)",
                min_value=1.0,
                max_value=300.0,
                value=float(s_comp_price),
                step=1.0,
                help="Price of the nearest comparable competitor product."
            )

            promo_choice = st.radio(
                "Active Marketing Promotion / Discount Campaign",
                options=[1, 0],
                format_func=lambda x: "Active Campaign (Yes)" if x == 1 else "Standard Full Price (No)",
                index=0 if s_promo == 1 else 1,
                horizontal=True
            )

        with col2:
            st.subheader("📦 Inventory Availability & Calendar Cycle")
            season_options = ["High", "Medium", "Low"]
            selected_season = st.selectbox(
                "Demand Seasonality Cycle",
                options=season_options,
                index=season_options.index(s_season) if s_season in season_options else 0,
                format_func=lambda s: f"📅 {s} Season ({'Peak Festivities / Q4' if s == 'High' else 'Standard Retail Period' if s == 'Medium' else 'Off-Peak / Clearance Cycle'})"
            )

            stock_available = st.number_input(
                "Warehouse / Store Available Stock (Units)",
                min_value=0,
                max_value=1000,
                value=int(s_stock),
                step=10,
                help="Current shelf inventory available to fulfill incoming demand."
            )

        calc_submitted = st.form_submit_button("⚡ Compute Expected Product Demand", width="stretch")

    # Inferences
    input_sample = pd.DataFrame([{
        "product_price": product_price,
        "promotion_active": promo_choice,
        "season": selected_season,
        "stock_available": stock_available,
        "competitor_price": competitor_price
    }])

    pred_demand_raw = float(model.predict(input_sample)[0])
    pred_demand = max(0.0, pred_demand_raw)
    pred_units = int(round(pred_demand))

    # Counterfactual check: What if promotion was opposite?
    opp_promo = 0 if promo_choice == 1 else 1
    counter_sample = input_sample.copy()
    counter_sample["promotion_active"] = opp_promo
    counter_units = int(round(max(0.0, float(model.predict(counter_sample)[0]))))
    promo_uplift = pred_units - counter_units if promo_choice == 1 else counter_units - pred_units

    # Commercial Analysis
    price_diff = competitor_price - product_price
    price_pct_diff = (price_diff / competitor_price) * 100 if competitor_price > 0 else 0.0

    fulfilled_units = min(stock_available, pred_units)
    potential_revenue = pred_units * product_price
    realized_revenue = fulfilled_units * product_price
    unmet_demand = max(0, pred_units - stock_available)
    lost_revenue = unmet_demand * product_price

    st.markdown("---")
    st.markdown("### 📊 Demand Forecast & Commercial Financials")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="hero-stat-card">
            <div class="sub-metric">PREDICTED DEMAND</div>
            <div style="font-size: 2.4rem; font-weight: 800; color: #6366F1; margin: 4px 0;">{pred_units:,}</div>
            <div class="sub-metric">Forecasted Units</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="hero-stat-card">
            <div class="sub-metric">EXPECTED GROSS SALES</div>
            <div style="font-size: 2.4rem; font-weight: 800; color: #10B981; margin: 4px 0;">${realized_revenue:,.2f}</div>
            <div class="sub-metric">At ${product_price:.2f} / unit</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        price_color = "#10B981" if price_pct_diff > 0 else "#EF4444" if price_pct_diff < -5 else "#F59E0B"
        st.markdown(f"""
        <div class="hero-stat-card">
            <div class="sub-metric">PRICE POSITIONING</div>
            <div style="font-size: 2.4rem; font-weight: 800; color: {price_color}; margin: 4px 0;">{price_pct_diff:+.1f}%</div>
            <div class="sub-metric">vs Competitor (${competitor_price:.2f})</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="hero-stat-card">
            <div class="sub-metric">PROMO DEMAND UPLIFT</div>
            <div style="font-size: 2.4rem; font-weight: 800; color: #38BDF8; margin: 4px 0;">+{promo_uplift:,}</div>
            <div class="sub-metric">Units gained via campaign</div>
        </div>
        """, unsafe_allow_html=True)

    # Stock Fulfillment Evaluation
    if unmet_demand > 0:
        status_banner = "status-stockout"
        s_title = f"🚨 Potential Stockout Detected — {unmet_demand:,} Units Short!"
        s_desc = f"Predicted consumer demand ({pred_units:,} units) exceeds warehouse availability ({stock_available:,} units). Estimated uncaptured revenue: **${lost_revenue:,.2f}**. Urgent inventory reorder required."
    elif pred_units > 0 and (stock_available / pred_units) > 2.2:
        status_banner = "status-surplus"
        s_title = "⚠️ Elevated Inventory Buffer / Surplus Stock"
        s_desc = f"Available inventory ({stock_available:,} units) provides over {stock_available / pred_units:.1f}x coverage of expected demand ({pred_units:,} units). Consider promotional bundling to accelerate turnover."
    else:
        status_banner = "status-healthy"
        s_title = "✅ Well-Calibrated Inventory Alignment"
        s_desc = f"Stock level of {stock_available:,} units adequately satisfies consumer demand ({pred_units:,} units) with a healthy safety reserve."

    st.markdown(f"""
    <div class="hero-stat-card {status_banner}" style="margin-top: 18px; text-align: left;">
        <div style="font-size: 1.15rem; font-weight: 700; margin-bottom: 6px;">
            {s_title}
        </div>
        <p style="margin: 0; color: #F1F5F9; font-size: 0.95rem;">
            {s_desc}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Merchandising Strategy Insight
    st.markdown(f"""
    <div class="insight-card">
        <strong>💡 Strategic Merchandising Recommendations:</strong>
        <ul style="margin-top: 6px; margin-bottom: 0; padding-left: 20px;">
            <li><strong>Pricing Advantage:</strong> Our price of ${product_price:.2f} is {f'{abs(price_pct_diff):.1f}% more competitive than' if price_pct_diff > 0 else f'{abs(price_pct_diff):.1f}% higher than'} the competitor's ${competitor_price:.2f}.</li>
            <li><strong>Seasonality Dynamics:</strong> Operating in <em>{selected_season} Season</em> shifts the demand curve significantly; combine active promotions with peak holiday periods for maximum revenue velocity.</li>
            <li><strong>Fulfillment Rate:</strong> Projected sell-through rate is <strong>{(fulfilled_units / stock_available * 100) if stock_available > 0 else 0.0:.1f}%</strong> of current warehouse on-hand stock.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 2: Batch Optimization
# -------------------------------------------------------------
with tab_batch:
    st.markdown("### 📑 Multi-SKU Catalog Demand Forecast")
    st.caption("Upload a catalog CSV to evaluate customer demand across all your retail products simultaneously.")

    sample_batch_df = pd.DataFrame([
        {"sku_code": "SKU-401", "product_price": 45.00, "promotion_active": 1, "season": "High", "stock_available": 250, "competitor_price": 42.50},
        {"sku_code": "SKU-402", "product_price": 112.40, "promotion_active": 0, "season": "Medium", "stock_available": 90, "competitor_price": 91.35},
        {"sku_code": "SKU-403", "product_price": 18.59, "promotion_active": 0, "season": "High", "stock_available": 180, "competitor_price": 15.07},
        {"sku_code": "SKU-404", "product_price": 84.02, "promotion_active": 0, "season": "Low", "stock_available": 110, "competitor_price": 76.57},
        {"sku_code": "SKU-405", "product_price": 131.05, "promotion_active": 1, "season": "High", "stock_available": 140, "competitor_price": 122.68},
    ])

    b1, b2 = st.columns([1, 2])
    with b1:
        st.download_button(
            label="📥 Download Sample Catalog CSV",
            data=sample_batch_df.to_csv(index=False),
            file_name="sample_retail_demand_catalog.csv",
            mime="text/csv",
            width="stretch"
        )

    uploaded_file = st.file_uploader(
        "Upload Retail Catalog CSV",
        type=["csv"],
        help="CSV must include columns: product_price, promotion_active, season, stock_available, competitor_price"
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            req_cols = ["product_price", "promotion_active", "season", "stock_available", "competitor_price"]
            missing_cols = [c for c in req_cols if c not in batch_df.columns]

            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            else:
                features_df = batch_df[req_cols].copy()
                raw_preds = model.predict(features_df)

                batch_df["predicted_demand"] = [max(0, int(round(p))) for p in raw_preds]
                batch_df["projected_revenue"] = batch_df["predicted_demand"] * batch_df["product_price"]
                batch_df["stock_deficit"] = [max(0, d - s) for d, s in zip(batch_df["predicted_demand"], batch_df["stock_available"])]
                batch_df["lost_revenue_risk"] = batch_df["stock_deficit"] * batch_df["product_price"]

                def flag_row(r):
                    if r["stock_deficit"] > 0:
                        return "🚨 Stockout Risk"
                    elif r["stock_available"] > 2.5 * max(1, r["predicted_demand"]):
                        return "⚠️ High Surplus"
                    return "✅ Optimal Stock"

                batch_df["inventory_alignment"] = batch_df.apply(flag_row, axis=1)

                st.success(f"Successfully processed {len(batch_df)} SKUs!")

                # Aggregate Metrics
                tot_demand = batch_df["predicted_demand"].sum()
                tot_revenue = batch_df["projected_revenue"].sum()
                tot_lost = batch_df["lost_revenue_risk"].sum()
                at_risk_skus = (batch_df["stock_deficit"] > 0).sum()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Catalog Demand", f"{tot_demand:,} units")
                m2.metric("Total Projected Sales", f"${tot_revenue:,.2f}")
                m3.metric("At-Risk SKUs (Stockout)", f"{at_risk_skus} SKUs")
                m4.metric("Unrealized Revenue Risk", f"${tot_lost:,.2f}")

                # Dataframe view
                display_cols = [c for c in ["sku_code", "product_price", "season", "stock_available", "predicted_demand", "projected_revenue", "inventory_alignment"] if c in batch_df.columns]
                st.dataframe(batch_df[display_cols], width="stretch")

                # Export
                st.download_button(
                    label="📥 Download Demand Forecast Report CSV",
                    data=batch_df.to_csv(index=False),
                    file_name="retail_product_demand_forecast_results.csv",
                    mime="text/csv",
                    width="stretch"
                )
        except Exception as err:
            st.error(f"Error reading uploaded batch file: {err}")
    else:
        st.info("Upload a CSV file or download the template above to forecast demand across entire product lines.")

# -------------------------------------------------------------
# TAB 3: Model Diagnostics & Price Elasticity
# -------------------------------------------------------------
with tab_analytics:
    st.markdown("### 📈 Model Evaluation & Elasticity Diagnostics")
    st.caption("Statistical performance benchmarks of the Random Forest Regressor on consumer purchase behavior.")

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("R² Score (Variance Explained)", "0.9227", delta="+92.3% fit accuracy")
    with d2:
        st.metric("Mean Absolute Error (MAE)", "11.24 units", delta="Average deviation per SKU")
    with d3:
        st.metric("Root Mean Squared Error (RMSE)", "14.12 units", delta="Penalty for outliers")
    with d4:
        st.metric("Model Architecture", "Random Forest", delta="200 decision trees")

    st.markdown("---")

    chart_c1, chart_c2 = st.columns([3, 2])

    with chart_c1:
        st.subheader("🎯 Actual vs. Predicted Demand Parity")
        chart_path = get_asset_path("actual_vs_predicted.png")
        if os.path.exists(chart_path):
            st.image(chart_path, caption="Strong linear correlation between empirical customer demand and Random Forest forecasts.", width="stretch")
        else:
            st.warning("Parity chart not found.")

    with chart_c2:
        st.subheader("💡 Factors Governing Consumer Demand")
        st.markdown("""
        The feature engineering and Random Forest model capture key retail dynamics:
        
        1. **Relative Price Difference:** The spread between own price and competitor benchmark is the most sensitive driver of unit sales velocity.
        2. **Promotion Multiplier:** Active promotional flags increase consumer propensity to purchase by an average of 18-35 units.
        3. **Seasonal Waves:** High season yields concentrated purchasing volume across all price tiers.
        4. **Shelf Availability Factor:** Prevents phantom demand forecasting when inventory constraints suppress observed purchasing.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.85rem; padding: 10px 0;">
    Retail Product Demand & Revenue AI • Powered by Scikit-learn Random Forest & Streamlit
</div>
""", unsafe_allow_html=True)
