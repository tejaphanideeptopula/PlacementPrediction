import io
from contextlib import redirect_stdout
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pathlib import Path

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    csv_path = BASE_DIR / "placement_predict_50k Dataset.csv"
    df = pd.read_csv(csv_path)

def normalize(df):
    num_cols = ["CGPA", "AttendencePercent", "AptitudeTestScore", "CodingTestScore", "Interships"]
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["PlacementStatus"])
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    columns = train_df.columns.tolist()
    train_df = pd.DataFrame(train_scaled, columns=columns, index=train_df.index)
    test_df = pd.DataFrame(test_scaled, columns=columns, index=test_df.index)
    return train_df, test_df

def normalization(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    columns = train_df.columns.tolist()
    train_df = pd.DataFrame(train_scaled, columns=columns, index=train_df.index)
    test_df = pd.DataFrame(test_scaled, columns=columns, index=test_df.index)
    return train_df, test_df

def standardization(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    num_cols = ["CGPA", "Internships", "Projects", "Workshops"]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[num_cols])
    X_test_scaled = scaler.transform(test_df[num_cols])
    train_scaled_df = pd.DataFrame(X_train_scaled, columns=num_cols, index=train_df.index)
    test_scaled_df = pd.DataFrame(X_test_scaled, columns=num_cols, index=test_df.index)
    return train_scaled_df, test_scaled_df

def onehot(df):
    nominal_cols = ["Gender"]
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["PlacementStatus"])
    try:
        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    except TypeError:
        ohe = OneHotEncoder(drop="first", sparse=False, handle_unknown="ignore")
    train_ohe = ohe.fit_transform(train_df[nominal_cols])
    test_ohe = ohe.transform(test_df[nominal_cols])
    feature_names = ohe.get_feature_names_out(nominal_cols)
    train_ohe_df = pd.DataFrame(train_ohe, columns=feature_names, index=train_df.index)
    test_ohe_df = pd.DataFrame(test_ohe, columns=feature_names, index=test_df.index)
    return train_ohe_df, test_ohe_df

def ordinal(df):
    df["CGPA_Category"] = pd.cut(df["CGPA"], bins=[0, 6, 8, 10], labels=["low", "medium", "high"], ordered=True)
    ordinal_cols = ["CGPA_Category"]
    categories_order = [["low", "medium", "high"]]
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["PlacementStatus"])
    encoder = OrdinalEncoder(categories=categories_order, handle_unknown="use_encoded_value", unknown_value=-1)
    train_df_encoded = train_df.copy()
    test_df_encoded = test_df.copy()
    train_df_encoded[ordinal_cols] = encoder.fit_transform(train_df[ordinal_cols])
    test_df_encoded[ordinal_cols] = encoder.transform(test_df[ordinal_cols])
    return train_df_encoded, test_df_encoded

def label_encode_branch(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["PlacementStatus"])
    encoder = LabelEncoder()
    train_df_encoded = train_df.copy()
    test_df_encoded = test_df.copy()
    train_df_encoded["Stream"] = encoder.fit_transform(train_df["Stream"])
    test_df_encoded["Stream"] = encoder.transform(test_df["Stream"])
    mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    return train_df_encoded, test_df_encoded, mapping

def create_cgpa_category(df):
    df["CGPA_Category"] = pd.cut(df["CGPA"], bins=[0, 6, 8, 10], labels=["low", "medium", "high"], ordered=True)
    return df

def find_null_values(df, column):
    null_values = df[column].isnull()
    print("Number of NULL values:", null_values.sum())
    print("\nRows where value is NULL:")
    print(df[null_values])

def data_cleaning(df):
    df["Workshops"] = df["Workshops"].fillna(df["Workshops"].mean())
    df["AptitudeTestScore"] = df["AptitudeTestScore"].fillna(df["AptitudeTestScore"].median())
    numerical_cols = ["CGPA", "AttendancePercent", "Internships", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore"]
    imputer = KNNImputer(n_neighbors=3)
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    return df

def get_numerical_features(df):
    num_cols = df.select_dtypes(include="number").columns
    return df[num_cols]

def get_categorical_features(df):
    cat_cols = df.select_dtypes(include="object").columns
    return df[cat_cols]

def detect_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def select_model_features(df):
    features = ["CGPA", "Internships", "Projects", "CodingTestScore"]
    X = df[features]
    y = df["PlacementStatus"]
    return X, y

def simple_linear_regression(df):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    X = df[['CGPA']]
    y = df['Salary Package']
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    linear_regression = LinearRegression()
    linear_regression.fit(train_X, train_y)
    y_pred = linear_regression.predict(test_X)
    mse1 = mean_squared_error(test_y, y_pred)
    rsq1 = r2_score(test_y, y_pred)
    print("MSE:", mse1)
    print("RSQ:", rsq1)

def knn_impute(df, n_neighbors=3):
    filled = df.copy()
    if "Workshops" in filled.columns:
        filled["Workshops"] = filled["Workshops"].fillna(filled["Workshops"].mean())
    if "AptitudeTestScore" in filled.columns:
        filled["AptitudeTestScore"] = filled["AptitudeTestScore"].fillna(filled["AptitudeTestScore"].median())
    numerical_cols = ["CGPA", "AttendancePercent", "Internships", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore"]
    numerical_cols = [c for c in numerical_cols if c in filled.columns]
    imputer = KNNImputer(n_neighbors=n_neighbors)
    filled[numerical_cols] = imputer.fit_transform(filled[numerical_cols])
    return {"before": int(df.isnull().sum().sum()), "after": int(filled.isnull().sum().sum()), "n_neighbors": n_neighbors, "columns": numerical_cols, "df": filled}

def gradient_descent(X, y, learning_rate=0.001, iterations=100):
    X = np.asarray(X, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = len(y)
    theta0 = 0.0
    theta1 = 0.0
    mse_history = []
    for i in range(iterations):
        y_pred = theta0 + theta1 * X
        error = y_pred - y
        mse = np.mean(error ** 2)
        mse_history.append(mse)
        d_theta0 = (2 / m) * np.sum(error)
        d_theta1 = (2 / m) * np.sum(error * X)
        theta0 = theta0 - learning_rate * d_theta0
        theta1 = theta1 - learning_rate * d_theta1
    return theta0, theta1, mse_history

def simple_linear_regression_GD(df):
    X = df[['CGPA']]
    y = df['PlacementStatus']
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    theta0, theta1, mse_history = gradient_descent(train_X, train_y, learning_rate=0.001, iterations=100)
    y_pred = theta0 + theta1 * test_X
    mse1 = mean_squared_error(test_y, y_pred)
    rsq1 = r2_score(test_y, y_pred)
    print("MSE:", mse1)
    print("RSQ:", rsq1)
    return theta0, theta1, mse1, rsq1

def run_regression_models(df):
    features = ["CGPA", "Internships", "Projects", "Workshops", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore", "SoftSkillsRating", "AttendancePercent"]
    target = "Salary Package"
    features = [c for c in features if c in df.columns]
    data = df[features + [target]].dropna()
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results.append({
            "model": name,
            "r2": round(float(r2_score(y_test, preds)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
            "mae": round(float(mean_absolute_error(y_test, preds)), 4)
        })
    linear = LinearRegression().fit(X_train, y_train)
    coefficients = [{"feature": col, "coefficient": round(float(coef), 4)} for col, coef in zip(features, linear.coef_)]
    return {
        "target": target,
        "features": features,
        "samples": int(len(data)),
        "results": results,
        "coefficients": coefficients,
        "intercept": round(float(linear.intercept_), 4)
    }

def simple_logistic_regression_GD(df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    X = df[["CGPA"]]
    y = df[["PlacementStatus"]]
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(train_X, train_y)
    theta0 = model.intercept_[0]
    theta1 = model.coef_[0]
    predictions = model.predict(test_X)
    accuracy = accuracy_score(test_y.values.ravel(), predictions.ravel())
    return {
        "target": "PlacementStatus",
        "feature": "CGPA",
        "theta0": round(float(theta0), 4),
        "theta1": round(float(theta1[0]), 4),
        "accuracy": round(float(accuracy), 4),
        "train_samples": int(len(train_X)),
        "test_samples": int(len(test_X))
    }

def run_simple_linear(df):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        simple_linear_regression(df)
    output = buffer.getvalue()
    mse = None
    rsq = None
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("MSE:"):
            mse = float(line.split(":", 1)[1])
        elif line.startswith("RSQ:"):
            rsq = float(line.split(":", 1)[1])
    return {"target": "Salary Package", "feature": "CGPA", "mse": mse, "rsq": rsq, "samples": int(len(df))}

def run_linear_gd(df):
    theta0, theta1, mse, rsq = simple_linear_regression_GD(df)
    return {
        "target": "PlacementStatus",
        "feature": "CGPA",
        "theta0": round(float(theta0), 4),
        "theta1": round(float(theta1), 4),
        "mse": round(float(mse), 4),
        "rsq": round(float(rsq), 4),
        "samples": int(len(df))
    }

def builtin_linear_sample():
    data = {
        'StudyHours': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        'ExamScore': [35, 45, 50, 60, 65, 75, 80, 90, 92, 98]
    }
    df = pd.DataFrame(data)
    X = df[['StudyHours']]
    y = df['ExamScore']
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    mse = mean_squared_error(test_y, y_pred)
    rsq = r2_score(test_y, y_pred)
    return {
        "x": "StudyHours",
        "y": "ExamScore",
        "theta0": round(float(model.intercept_), 4),
        "theta1": round(float(model.coef_[0]), 4),
        "mse": round(float(mse), 4),
        "rsq": round(float(rsq), 4)
    }

def builtin_logistic_sample():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    data = {
        'StudyHours': [1.0, 1.5, 2.0, 3.5, 4.0, 5.5, 6.0, 7.5, 8.0, 9.0],
        'PassStatus': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    }
    df = pd.DataFrame(data)
    X = df[['StudyHours']]
    y = df['PassStatus']
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    accuracy = accuracy_score(test_y, y_pred)
    return {
        "x": "StudyHours",
        "y": "PassStatus",
        "theta0": round(float(model.intercept_[0]), 4),
        "theta1": round(float(model.coef_[0][0]), 4),
        "accuracy": round(float(accuracy), 4),
        "predictions": y_pred.tolist(),
        "actual": test_y.values.tolist()
    }
def run_random_forest_classifier(df):
    X = df.drop(columns=['StudentID', 'PlacementStatus', 'Salary Package', 'IsAnomaly'], errors='ignore')
    y = df['PlacementStatus']
    numerical_cols = X.select_dtypes(include=np.number).columns
    categorical_cols = X.select_dtypes(include=['str', 'category']).columns
    X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())
    X[categorical_cols] = X[categorical_cols].fillna('Unknown')
    X = pd.get_dummies(X, drop_first=True)
    X = X.astype(float)
    if y.dtype == object:
        y = y.astype(str).str.strip()
        y = y.map({"Yes": 1, "No": 0})
    if y.isnull().any():
        raise ValueError("Invalid values found in PlacementStatus.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Model Accuracy: {accuracy:.2f}")
    return rf, accuracy
if __name__ == "__main__":
    cleaned_df = data_cleaning(df)
    run_random_forest_classifier(cleaned_df)