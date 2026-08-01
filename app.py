import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
st.title("Auto Data Cleaning & EDA Dashboard")

def classify_and_clean(df, target_col=None, id_threshold=0.9, missing_threshold=0.5, cardinality_threshold=10):
    if target_col:
        df = df.drop(columns=[target_col])
    nrows = len(df)
    drop_cols, numeric_cols, categorical_cols = [], [], []
    for cols in df.columns:
        if df[cols].nunique()/nrows > id_threshold:
            drop_cols.append(cols); continue
        if df[cols].isnull().sum()/nrows > missing_threshold:
            drop_cols.append(cols); continue
        if df[cols].dtype in ['int64','float64','int32','float32']:
            numeric_cols.append(cols)
        else:
            if df[cols].nunique() <= cardinality_threshold:
                categorical_cols.append(cols)
            else:
                drop_cols.append(cols)
    return numeric_cols, categorical_cols, drop_cols

def build_auto_pipeline(numeric_cols, categorical_cols):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])
    return preprocessor

uploaded_file = st.file_uploader("Upload your CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of your data:")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    st.subheader("Exploratory Data Analysis (before cleaning)")

    # Missing values chart
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        st.write("Missing values by column:")
        fig1, ax1 = plt.subplots(figsize=(6,3))
        missing.sort_values(ascending=False).plot(kind='bar', color='coral', ax=ax1)
        plt.tight_layout()
        st.pyplot(fig1)
    else:
        st.write("No missing values found.")

    # Numeric distributions
    numeric_preview = df.select_dtypes(include=['int64','float64','int32','float32']).columns.tolist()
    if numeric_preview:
        st.write("Numeric column distributions:")
        fig2 = df[numeric_preview].hist(figsize=(10,6), bins=20, color='steelblue', edgecolor='black')
        plt.tight_layout()
        st.pyplot(fig2[0][0].figure)

    # Categorical distributions
    categorical_preview = df.select_dtypes(include='object').columns.tolist()
    categorical_preview = [c for c in categorical_preview if df[c].nunique() <= 10]
    if categorical_preview:
        st.write("Categorical column distributions:")
        fig3, axes3 = plt.subplots(1, len(categorical_preview), figsize=(5*len(categorical_preview), 4))
        if len(categorical_preview) == 1:
            axes3 = [axes3]
        for ax, col in zip(axes3, categorical_preview):
            df[col].value_counts().plot(kind='bar', ax=ax, color='teal')
            ax.set_title(col)
        plt.tight_layout()
        st.pyplot(fig3)
    target_col = st.selectbox("Select target column (optional)", [None] + list(df.columns))

    numeric_cols, categorical_cols, drop_cols = classify_and_clean(df, target_col=target_col)

    st.write("Numeric columns:", numeric_cols)
    st.write("Categorical columns:", categorical_cols)
    st.write("Dropped columns:", drop_cols)

    preprocessor = build_auto_pipeline(numeric_cols, categorical_cols)
    X = df.drop(columns=drop_cols + ([target_col] if target_col else []))
    X_transformed = preprocessor.fit_transform(X)
    # convert back to dataframe for download
    encoded_cat_cols = []
    if categorical_cols:
        encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        encoded_cat_cols = list(encoder.get_feature_names_out(categorical_cols))
    
    all_cols = numeric_cols + encoded_cat_cols
    cleaned_df = pd.DataFrame(X_transformed, columns=all_cols)
    
    st.write("Cleaned data preview:")
    st.dataframe(cleaned_df.head())
    
    csv = cleaned_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Cleaned CSV", csv, "cleaned_data.csv", "text/csv")
    st.write("Cleaned data shape:", X_transformed.shape)