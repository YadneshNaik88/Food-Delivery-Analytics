# 🍔 Food Delivery Analytics & Intelligent Recommendation System

> A complete end-to-end Machine Learning pipeline built on 50,000 food delivery orders — covering data integration, pattern discovery, predictive modeling, customer segmentation, and an interactive web application.

---

## 📌 Project Overview

This project applies **5 core Machine Learning techniques** to a real-world food delivery dataset consisting of 7 relational tables. The goal is to extract business insights, predict delivery outcomes, segment customers, and recommend food based on ordering behavior.

Built as a **Minor Project** at MIT Academy of Engineering, Pune.

---

## 🎯 Objectives

- Integrate and preprocess data from 7 relational Excel sheets (50,000 orders)
- Discover customer cuisine & payment preferences using Association Rule Mining
- Predict delivery status (OnTime / Late / Cancelled) using classification models
- Segment customers into behavioral groups using clustering
- Predict delivery time in minutes using a Neural Network
- Deploy all results as an interactive Streamlit web application

---

## 🗂️ Dataset

| Sheet | Rows | Description |
|-------|------|-------------|
| Customers | 50,000 | Age, Gender, City |
| Restaurants | 500 | Cuisine, City, Rating |
| Menu | 50,000 | Item Name, Price |
| Orders | 50,000 | CustomerID, RestaurantID, Date |
| OrderItems | 50,000 | ItemID, Quantity |
| Payments | 50,000 | Amount, Payment Mode |
| Delivery | 50,000 | Delivery Time, Status |

---

## 🧠 ML Techniques Used

### Part 1 — Data Integration & Preprocessing
- Merged 7 relational sheets using Pandas merge (left joins)
- Engineered 6 new features: `CustomerOrderFrequency`, `TotalOrderValue`, `ItemsPerOrder`, `OrderMonth`, `OrderDayOfWeek`, `AvgDeliveryTime`
- Label encoded categorical variables (Cuisine, City, Payment Mode, Gender)
- Normalized features using `StandardScaler`

### Part 2 — Association Rule Mining
- Built **customer-level transaction baskets** (cuisine + payment mode history per customer)
- Implemented **Apriori algorithm from scratch** — no external library
- Implemented **FP-Growth algorithm from scratch** — no external library
- Discovered 36 rules with min_support=0.05, min_confidence=0.30
- Key finding: Card users prefer Indian cuisine (Lift=1.19), Wallet users prefer Italian (Lift=1.18)

### Part 3 — Classification
- Target variable: Delivery Status (OnTime / Late / Cancelled)
- Models trained: **Logistic Regression, Decision Tree, Random Forest, SVM**
- Evaluated using: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- Feature importance analysis to identify causes of late delivery

### Part 4 — Clustering
- Customer segmentation using **K-Means** (K=3, Elbow Method)
- Validated with **Hierarchical Clustering** (Ward linkage, Dendrogram)
- PCA used for 2D cluster visualization
- Segments identified: 💎 High-Value, 🔁 Frequent, 👤 Regular

### Part 5 — ANN (Neural Network)
- Architecture: `Input(9) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(32, ReLU) → Output(1)`
- Optimizer: Adam | Loss: MSE | Early Stopping
- Compared against Linear Regression and Random Forest Regressor
- Evaluated using RMSE, MAE, R²

---

## 📊 Key Results

| Part | Model | Best Metric |
|------|-------|-------------|
| Classification | SVM | Accuracy: 34.5% |
| Classification | Random Forest | F1: 0.34 (weighted) |
| ARM | Apriori + FP-Growth | 36 rules, max Lift: 1.19 |
| Clustering | K-Means | 3 meaningful segments |
| ANN | MLP Regressor | RMSE: 13.4 min |

> **Note:** Classification accuracy of ~34% and ANN R²≈0 reflect a data limitation — delivery status and time are nearly uniformly distributed in this dataset, indicating that additional features like GPS distance, traffic data, and restaurant prep time would significantly improve predictions.

---

## 🌐 Streamlit Web Application

The project includes a fully interactive web app with 5 pages:

| Page | Feature |
|------|---------|
| 📊 Dashboard | KPIs, status distribution, cuisine trends, monthly volume |
| 🔍 Association Rules | Interactive rule explorer + food recommendation engine |
| 🚦 Predict Status | Enter order details → predict OnTime/Late/Cancelled |
| 👥 Customer Segmentation | Classify a new customer into a segment |
| ⏱️ Predict Delivery Time | Estimate delivery minutes using ANN |

### Run the app locally:
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn openpyxl
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Notebook | Jupyter Notebook |
| Dataset | Microsoft Excel (.xlsx) |

---

## 📁 Project Structure

```
Food-Delivery-Analytics/
│
├── Food_Delivery_Analytics.ipynb        # Main Jupyter notebook (all 5 parts)
├── app.py                               # Streamlit web application
├── Food_Delivery_STRICT_RELATIONAL.xlsx # Dataset (7 sheets)
└── README.md                            # Project documentation
```

---

## 🚀 How to Run

### Jupyter Notebook
```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl jupyter
jupyter notebook Food_Delivery_Analytics.ipynb
```

### Streamlit App
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn openpyxl
streamlit run app.py
```

> Make sure `Food_Delivery_STRICT_RELATIONAL.xlsx` is in the same folder as both files.

---

## 📈 Business Insights

- **33% of orders are late or cancelled** — highlights urgent need for delivery optimization
- **Card users prefer Indian food** — enables targeted recommendations at checkout
- **Top 42% repeat customers** drive most of the revenue — segment-specific retention campaigns recommended
- **Italian cuisine** generates the highest average order value — premium upsell opportunity

---

## 👨‍💻 Author

**Yadnesh Naik**
MIT Academy of Engineering, Pune
B.Tech — Computer Engineering

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
