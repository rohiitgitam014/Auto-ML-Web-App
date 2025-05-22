import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report as cr, r2_score, mean_squared_error

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ----- Styling -----
st.set_page_config(page_title="Auto ML App", layout="wide")

st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
            padding: 1rem;
        }
        h1 {
            color: #4b4b4b;
        }
        .stTextInput > label {
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ----- Helper Functions -----
def is_continuous(series):
    return pd.api.types.is_numeric_dtype(series) and series.nunique() > 10

def clean_data(df):
    for col in df.columns:
        null_percentage = df[col].isnull().mean() * 100
        if null_percentage > 50:
            df.drop(columns=[col], inplace=True)
        elif null_percentage > 0:
            if is_continuous(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    return df

def get_model(model_name, is_regression):
    if is_regression:
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            "KNeighbors Regressor": KNeighborsRegressor(),
            "AdaBoost Regressor": AdaBoostRegressor(),
            "Decision Tree Regressor (max_depth=3)": DecisionTreeRegressor(max_depth=3)
        }[model_name]
    else:
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(),
            "Gradient Boosting Classifier": GradientBoostingClassifier(),
            "AdaBoost Classifier": AdaBoostClassifier(),
            "KNeighbors Classifier": KNeighborsClassifier(),
            "Gaussian Naive Bayes": GaussianNB(),
            "Decision Tree Classifier (max_depth=3)": DecisionTreeClassifier(max_depth=3)
        }[model_name]

# ----- App Header -----
st.title("🔮 Auto ML Prediction App")
st.markdown("Upload a CSV dataset, choose a target, and let the app clean data, select models, and predict!")

# ----- File Upload -----
st.header("📂 Step 1: Upload Dataset")
uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Raw Dataset")
    st.dataframe(df, use_container_width=True)

    # ----- Clean Data -----
    df = clean_data(df)
    st.subheader("🧼 Cleaned Dataset")
    st.dataframe(df, use_container_width=True)

    # ----- Target Column -----
    st.header("🎯 Step 2: Select Target Variable")
    target_col = st.selectbox("Choose the column you want to predict", df.columns)

    if target_col:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X = pd.get_dummies(X)
        X = X.astype(float)

        is_reg = is_continuous(y)

        # ----- Model Selection -----
        st.header("🤖 Step 3: Choose a Machine Learning Model")

        if is_reg:
            model_choice = st.selectbox("Regression Models", [
                "Linear Regression",
                "Random Forest Regressor",
                "Gradient Boosting Regressor",
                "KNeighbors Regressor",
                "AdaBoost Regressor",
                "Decision Tree Regressor (max_depth=3)"
            ])
        else:
            model_choice = st.selectbox("Classification Models", [
                "Logistic Regression",
                "Random Forest Classifier",
                "Gradient Boosting Classifier",
                "AdaBoost Classifier",
                "KNeighbors Classifier",
                "Gaussian Naive Bayes",
                "Decision Tree Classifier (max_depth=3)"
            ])

        # ----- Train Model -----
        st.header("⚙️ Step 4: Training & Evaluation")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = get_model(model_choice, is_reg)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ----- Results -----
        st.subheader("📊 Model Performance")

        if is_reg:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2score = r2_score(y_test, y_pred)
            st.success(f"Model: **{model_choice}**")
            st.metric("R² Score", f"{r2score:.2f}")
            st.metric("MSE", f"{mse:.2f}")
            st.metric("RMSE", f"{rmse:.2f}")
        else:
            report = cr(y_test, y_pred, output_dict=True)
            report = pd.DataFrame(report).transpose()
            st.success(f"Model: **{model_choice}**")
            st.text("Classification Report:")
            st.table(report)

        # ----- Predictions -----
        st.subheader("📈 Predictions on Test Set")
        st.dataframe(pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}), use_container_width=True)
