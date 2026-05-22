import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations
from collections import defaultdict

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    r2_score
)

import warnings
warnings.filterwarnings("ignore")

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Food Delivery Analytics",
    page_icon="🍔",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
.main-header {
    font-size: 2.2rem;
    font-weight: 700;
    color: #16a34a;
}

.metric-box {
    background: #f3f4f6;
    padding: 1rem;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    FILE = "Food_Delivery_STRICT_RELATIONAL.xlsx"

    sheets = pd.read_excel(FILE, sheet_name=None)

    customers = sheets['Customers']
    restaurants = sheets['Restaurants']
    menu = sheets['Menu']
    orders = sheets['Orders']
    order_items = sheets['OrderItems']
    payments = sheets['Payments']
    delivery = sheets['Delivery']

    # MERGE
    df = orders.merge(customers, on='CustomerID', how='left')

    df = df.merge(
        restaurants,
        on='RestaurantID',
        how='left',
        suffixes=('_customer', '_restaurant')
    )

    df = df.merge(payments, on='OrderID', how='left')

    df = df.merge(delivery, on='OrderID', how='left')

    # ORDER SUMMARY
    oi = order_items.merge(menu, on='ItemID', how='left')

    order_summary = oi.groupby('OrderID').agg(
        ItemsPerOrder=('Quantity', 'sum'),
        UniqueItems=('ItemID', 'nunique'),
        AvgItemPrice=('Price', 'mean')
    ).reset_index()

    df = df.merge(order_summary, on='OrderID', how='left')

    # FEATURES
    df['TotalOrderValue'] = df['Amount']

    df['OrderDate'] = pd.to_datetime(df['OrderDate'])

    df['OrderMonth'] = df['OrderDate'].dt.month
    df['OrderDayOfWeek'] = df['OrderDate'].dt.dayofweek
    df['OrderHour'] = df['OrderDate'].dt.hour

    cust_freq = df.groupby('CustomerID')['OrderID'].count().rename('CustomerOrderFrequency')
    df = df.merge(cust_freq, on='CustomerID', how='left')

    avg_del = df.groupby('CustomerID')['DeliveryTime(min)'].mean().rename('AvgDeliveryTime')
    df = df.merge(avg_del, on='CustomerID', how='left')

    df.rename(columns={'DeliveryTime(min)': 'DeliveryTime'}, inplace=True)

    # ENCODING
    encoders = {}

    cat_cols = ['Cuisine', 'City_customer', 'Mode', 'Gender']

    for col in cat_cols:
        le = LabelEncoder()

        df[col + '_enc'] = le.fit_transform(
            df[col].astype(str)
        )

        encoders[col] = le

    df['Status_enc'] = LabelEncoder().fit_transform(
        df['Status'].astype(str)
    )

    df.dropna(
        subset=['DeliveryTime', 'TotalOrderValue', 'Status'],
        inplace=True
    )

    df.reset_index(drop=True, inplace=True)

    return (
        df,
        customers,
        restaurants,
        menu,
        orders,
        order_items,
        payments,
        delivery,
        encoders
    )

# =====================================================
# LOAD DATA
# =====================================================

try:

    (
        df,
        customers,
        restaurants,
        menu,
        orders,
        order_items,
        payments,
        delivery,
        encoders
    ) = load_data()

except Exception as e:

    st.error(
        "Place Food_Delivery_STRICT_RELATIONAL.xlsx "
        "in same folder as app.py"
    )

    st.stop()

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("🍔 Food Delivery Analytics")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Association Rule Mining",
            "Classification",
            "Customer Segmentation",
            "ANN Prediction",
            "Business Insights"
        ]
    )

# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-header">Food Delivery Dashboard</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Orders", len(df))
    c2.metric("Customers", df['CustomerID'].nunique())
    c3.metric("Restaurants", df['RestaurantID'].nunique())
    c4.metric("Avg Delivery", round(df['DeliveryTime'].mean(), 2))

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Delivery Status")

        fig, ax = plt.subplots()

        df['Status'].value_counts().plot(
            kind='bar',
            ax=ax,
            color=['green', 'orange', 'red']
        )

        st.pyplot(fig)

    with col2:

        st.subheader("Top Cuisines")

        fig, ax = plt.subplots()

        df['Cuisine'].value_counts().head(5).plot(
            kind='barh',
            ax=ax
        )

        st.pyplot(fig)

    st.subheader("Monthly Order Trend")

    monthly = df.groupby('OrderMonth')['OrderID'].count()

    fig, ax = plt.subplots(figsize=(8,4))

    ax.plot(
        monthly.index,
        monthly.values,
        marker='o'
    )

    st.pyplot(fig)

# =====================================================
# ASSOCIATION RULE MINING
# =====================================================

elif page == "Association Rule Mining":

    st.header("Association Rule Mining")

    oi_full = order_items.merge(menu, on='ItemID', how='left')

    oi_full = oi_full.merge(
        restaurants[['RestaurantID', 'Cuisine']],
        on='RestaurantID',
        how='left'
    )

    oi_full = oi_full.merge(
        orders[['OrderID', 'CustomerID']],
        on='OrderID',
        how='left'
    )

    oi_full = oi_full.merge(
        payments[['OrderID', 'Mode']],
        on='OrderID',
        how='left'
    )

    oi_full['CuisineItem'] = (
        'Cuisine_' + oi_full['Cuisine'].astype(str)
    )

    oi_full['PayItem'] = (
        'Pay_' + oi_full['Mode'].astype(str)
    )

    def build_basket(group):

        return list(
            set(
                group['CuisineItem'].tolist() +
                group['PayItem'].tolist()
            )
        )

    transactions = oi_full.groupby(
        'CustomerID'
    ).apply(build_basket).tolist()

    def get_support(itemset, transactions):

        count = sum(
            1 for t in transactions
            if set(itemset).issubset(set(t))
        )

        return count / len(transactions)

    rules = []

    all_items = set(
        item for t in transactions for item in t
    )

    for item1 in all_items:

        for item2 in all_items:

            if item1 != item2:

                support = get_support(
                    [item1, item2],
                    transactions
                )

                if support >= 0.05:

                    confidence = (
                        support /
                        get_support([item1], transactions)
                    )

                    lift = (
                        confidence /
                        get_support([item2], transactions)
                    )

                    rules.append([
                        item1,
                        item2,
                        round(support, 4),
                        round(confidence, 4),
                        round(lift, 4)
                    ])

    rules_df = pd.DataFrame(
        rules,
        columns=[
            'Antecedent',
            'Consequent',
            'Support',
            'Confidence',
            'Lift'
        ]
    )

    rules_df = rules_df.sort_values(
        'Lift',
        ascending=False
    )

    st.dataframe(
        rules_df.head(20),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(8,5))

    sc = ax.scatter(
        rules_df['Support'],
        rules_df['Confidence'],
        c=rules_df['Lift'],
        cmap='viridis'
    )

    plt.colorbar(sc)

    st.pyplot(fig)

# =====================================================
# CLASSIFICATION
# =====================================================

elif page == "Classification":

    st.header("Delivery Status Classification")

    clf_features = [
        'Age',
        'Rating',
        'ItemsPerOrder',
        'TotalOrderValue',
        'OrderMonth',
        'OrderDayOfWeek',
        'Cuisine_enc',
        'City_customer_enc',
        'Mode_enc',
        'Gender_enc'
    ]

    clf_df = df[
        clf_features + ['Status']
    ].dropna()

    X = np.asarray(
        clf_df[clf_features]
    ).astype(np.float64)

    y = np.asarray(
        clf_df['Status'].astype(str).tolist()
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        'Logistic Regression': LogisticRegression(max_iter=500),
        'Decision Tree': DecisionTreeClassifier(max_depth=8),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'SVM': SVC()
    }

    results = []

    best_model = None
    best_acc = 0

    for name, model in models.items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        if acc > best_acc:
            best_acc = acc
            best_model = model

        results.append([
            name,
            acc,
            precision_score(
                y_test,
                y_pred,
                average='weighted'
            ),
            recall_score(
                y_test,
                y_pred,
                average='weighted'
            ),
            f1_score(
                y_test,
                y_pred,
                average='weighted'
            )
        ])

    results_df = pd.DataFrame(
        results,
        columns=[
            'Model',
            'Accuracy',
            'Precision',
            'Recall',
            'F1'
        ]
    )

    st.dataframe(results_df)

    y_pred = best_model.predict(X_test)

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(5,4))

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax
    )

    st.pyplot(fig)

# =====================================================
# CUSTOMER SEGMENTATION
# =====================================================

elif page == "Customer Segmentation":

    st.header("Customer Segmentation")

    seg_features = [
        'CustomerOrderFrequency',
        'TotalOrderValue',
        'AvgDeliveryTime',
        'Rating'
    ]

    seg_df = df.groupby('CustomerID').agg({
        'CustomerOrderFrequency':'mean',
        'TotalOrderValue':'mean',
        'AvgDeliveryTime':'mean',
        'Rating':'mean'
    }).reset_index()

    scaler = StandardScaler()

    X_seg = scaler.fit_transform(
        seg_df[seg_features]
    )

    inertia = []

    for k in range(1, 10):

        km = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        km.fit(X_seg)

        inertia.append(km.inertia_)

    st.subheader("Elbow Method")

    fig, ax = plt.subplots()

    ax.plot(
        range(1,10),
        inertia,
        marker='o'
    )

    st.pyplot(fig)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    seg_df['Cluster'] = kmeans.fit_predict(X_seg)

    st.subheader("Cluster Plot")

    fig, ax = plt.subplots(figsize=(7,5))

    scatter = ax.scatter(
        seg_df['CustomerOrderFrequency'],
        seg_df['TotalOrderValue'],
        c=seg_df['Cluster']
    )

    st.pyplot(fig)

# =====================================================
# ANN PREDICTION
# =====================================================

elif page == "ANN Prediction":

    st.header("Delivery Time Prediction")

    ann_features = [
        'ItemsPerOrder',
        'Rating',
        'TotalOrderValue',
        'Cuisine_enc',
        'City_customer_enc',
        'Mode_enc',
        'Age',
        'OrderMonth',
        'OrderDayOfWeek'
    ]

    ann_df = df[
        ann_features + ['DeliveryTime']
    ].dropna()

    # SAFE NUMPY CONVERSION
    X_ann = np.asarray(
        ann_df[ann_features]
    ).astype(np.float64)

    y_ann = np.asarray(
        ann_df['DeliveryTime']
    ).astype(np.float64)

    # SCALER
    scaler_ann = StandardScaler()

    X_ann_scaled = scaler_ann.fit_transform(X_ann)

    # TRAIN TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X_ann_scaled,
        y_ann,
        test_size=0.2,
        random_state=42
    )

    # ANN MODEL
    ann = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        max_iter=300,
        early_stopping=True,
        random_state=42
    )

    ann.fit(X_train, y_train)

    # EVALUATION
    y_pred = ann.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    r2 = r2_score(y_test, y_pred)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "RMSE",
            round(rmse, 2)
        )

    with col2:
        st.metric(
            "R² Score",
            round(r2, 4)
        )

    st.markdown("---")

    st.subheader("Predict New Delivery Time")

    cuisine = st.selectbox(
        "Cuisine",
        sorted(df['Cuisine'].dropna().unique()),
        key='ann_cuisine'
    )

    city = st.selectbox(
        "City",
        sorted(df['City_customer'].dropna().unique()),
        key='ann_city'
    )

    mode = st.selectbox(
        "Payment Mode",
        sorted(df['Mode'].dropna().unique()),
        key='ann_mode'
    )

    age = st.slider(
        "Customer Age",
        18,
        70,
        25
    )

    rating = st.slider(
        "Restaurant Rating",
        1.0,
        5.0,
        4.0,
        0.1
    )

    items = st.slider(
        "Items Ordered",
        1,
        10,
        2
    )

    order_val = st.number_input(
        "Order Value",
        50,
        5000,
        500
    )

    month = st.slider(
        "Order Month",
        1,
        12,
        6
    )

    dow = st.slider(
        "Day Of Week",
        0,
        6,
        3
    )

    if st.button("Predict Delivery Time"):

        # ENCODE VALUES
        cuisine_enc = encoders['Cuisine'].transform(
            [cuisine]
        )[0]

        city_enc = encoders['City_customer'].transform(
            [city]
        )[0]

        mode_enc = encoders['Mode'].transform(
            [mode]
        )[0]

        # EXACT SAME FEATURE ORDER
        input_data = np.array([[
            items,
            rating,
            order_val,
            cuisine_enc,
            city_enc,
            mode_enc,
            age,
            month,
            dow
        ]], dtype=np.float64)

        # SCALE INPUT
        input_scaled = scaler_ann.transform(
            input_data
        )

        # PREDICT
        prediction = ann.predict(
            input_scaled
        )[0]

        # SAFETY
        prediction = max(5, prediction)

        st.success(
            f"Estimated Delivery Time: {prediction:.1f} minutes"
        )

    st.markdown("---")

    st.subheader("Actual vs Predicted")

    fig, ax = plt.subplots(figsize=(6,4))

    ax.scatter(
        y_test[:200],
        y_pred[:200],
        alpha=0.6
    )

    ax.set_xlabel("Actual Delivery Time")
    ax.set_ylabel("Predicted Delivery Time")

    st.pyplot(fig)

# =====================================================
# BUSINESS INSIGHTS
# =====================================================

elif page == "Business Insights":

    st.header("Business Insights")

    st.subheader("Top Restaurants")

    top_rest = df.groupby(
        'RestaurantID'
    )['Rating'].mean().sort_values(
        ascending=False
    ).head(10)

    st.dataframe(top_rest)

    st.subheader("High Value Customers")

    high_val = df.groupby(
        'CustomerID'
    )['TotalOrderValue'].sum().sort_values(
        ascending=False
    ).head(10)

    st.dataframe(high_val)

    st.subheader("Delivery Bottlenecks")

    bottlenecks = df.groupby(
        'Cuisine'
    )['DeliveryTime'].mean().sort_values(
        ascending=False
    )

    st.bar_chart(bottlenecks)