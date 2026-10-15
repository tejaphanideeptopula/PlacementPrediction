import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
# =========================================================
# 1. LOAD DATA
# =========================================================
def load_data():
    df = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "placement_predict_50k Dataset.csv"
        )
    )
    return df
# =========================================================
# 2. SUMMARY
# =========================================================
def summarize(df):
    print("\n========== DATA SUMMARY ==========")
    print("\n--- Describe ---")
    print(df.describe())
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Shape ---")
    print(df.shape)
    print("\n--- First 5 Rows ---")
    print(df.head())
# =========================================================
# 3. STATISTICAL ANALYSIS
# =========================================================
def stat(df):
    print("\n========== STATISTICS ==========")
    num_cols = df.select_dtypes(include=np.number)
    print("\nNumerical Columns:")
    print(num_cols.columns.tolist())
    mean_values = num_cols.mean()
    print("\nMean:")
    print(mean_values)
    std_values = num_cols.std()
    print("\nStandard Deviation:")
    print(std_values)
    skewness = num_cols.skew()
    print("\nSkewness:")
    print(skewness)
    kurtosis = num_cols.kurt()
    print("\nKurtosis:")
    print(kurtosis)
    correlation = num_cols.corr()
    print("\nCorrelation:")
    print(correlation)
    return (    num_cols,    mean_values,    std_values,    skewness,    kurtosis,    correlation)
# =========================================================
# 4. VISUALIZATION
# =========================================================
def visualize(df):
    numeric_df = df.select_dtypes(include=np.number)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # 1. Correlation heatmap'
    sns.heatmap(    numeric_df.corr(),    annot=True,    cmap="YlGnBu",    ax=axes[0, 0])
    axes[0, 0].set_title("Correlation Heatmap")
    # 2. PlacementStatus count
    if "PlacementStatus" in df.columns:
        sns.countplot(        data=df,        x="PlacementStatus",        ax=axes[0, 1]    )
        axes[0, 1].set_title("Placement Status")
    # 3. Salary Package boxplot
    if "Salary Package" in df.columns:
        sns.boxplot(        y=df["Salary Package"],        ax=axes[0, 2]    )
        axes[0, 2].set_title("Salary Package")
    # 4. CGPA distribution
    if "CGPA" in df.columns:    sns.histplot(        df["CGPA"],        kde=True,        ax=axes[1, 0]    )
    axes[1, 0].set_title("CGPA Distribution")
    # 5. CGPA boxplot
    if "CGPA" in df.columns:    sns.boxplot(        y=df["CGPA"],        ax=axes[1, 1]    )
    axes[1, 1].set_title("CGPA Boxplot")
    # 6. CGPA vs Salary Package
    if "CGPA" in df.columns and "Salary Package" in df.columns:    sns.scatterplot(        data=df,        x="CGPA",        y="Salary Package",        ax=axes[1, 2]    )
    axes[1, 2].set_title("CGPA vs Salary Package")
    plt.tight_layout()
    plt.show()

# =========================================================
# 5. NORMALIZATION AND STANDARDIZATION
# =========================================================
def normalize(df, exclude_cols=None):
    print("\n========== SCALING ==========")

    if exclude_cols is None:
        exclude_cols = []

    # Numeric columns that are eligible for scaling (exclude targets)
    numeric_cols_all = df.select_dtypes(include=np.number).columns
    scale_cols = [c for c in numeric_cols_all if c not in exclude_cols]

    numeric_df = df[scale_cols].copy()
    # Everything else — non-numeric columns AND excluded target columns — stays untouched
    keep_df = df.drop(columns=scale_cols).copy()

    print("\nOriginal Data:")
    print(df.head())

    # Min-Max Normalization
    min_max_scaler = MinMaxScaler()
    normalized_array = min_max_scaler.fit_transform(numeric_df)
    normalized_numeric_df = pd.DataFrame(
        normalized_array,
        columns=numeric_df.columns,
        index=numeric_df.index
    )
    normalized_df = pd.concat([keep_df, normalized_numeric_df], axis=1)[df.columns]

    print("\n========== MIN-MAX NORMALIZED DATA (FULL DATASET) ==========")
    print(normalized_df.head())

    # Standardization
    standard_scaler = StandardScaler()
    standardized_array = standard_scaler.fit_transform(numeric_df)
    standardized_numeric_df = pd.DataFrame(
        standardized_array,
        columns=numeric_df.columns,
        index=numeric_df.index
    )
    standardized_df = pd.concat([keep_df, standardized_numeric_df], axis=1)[df.columns]

    print("\n========== STANDARDIZED DATA (FULL DATASET) ==========")
    print(standardized_df.head())

    return normalized_df, standardized_df# =========================================================
# 6. OUTLIER HANDLING
# =========================================================

def outlier(df):
    if "Salary Package" not in df.columns:
        print("Salary Package column not found.")
        return df
    df = df.copy()
    print("\n========== OUTLIER HANDLING ==========")
    print("\nBefore Clipping:")
    print(df["Salary Package"].describe())
    q1 = df["Salary Package"].quantile(0.25)
    q3 = df["Salary Package"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    df["Salary Package"] = df["Salary Package"].clip(    upper=upper)
    print("\nAfter Clipping:")
    print(df["Salary Package"].describe())
    return df


# =========================================================
# 7. ONE-HOT ENCODING
# =========================================================

def onehot(df):
    nominal_cols = [    "Gender",    "City",    "Stream",    "Specialisation",    "Hostel",    "HistoryOfBacklogs"]
    # Check whether columns exist
    nominal_cols = [    col for col in nominal_cols    if col in df.columns]
    train_df, test_df = train_test_split(    df,    test_size=0.2,    random_state=42,    stratify=df["PlacementStatus"])
    ohe = OneHotEncoder(    drop="first",    sparse_output=False,    handle_unknown="ignore")
    # Fit ONLY on training data
    train_ohe = ohe.fit_transform(    train_df[nominal_cols])
    # Transform test data
    test_ohe = ohe.transform(    test_df[nominal_cols])
    ohe_cols = ohe.get_feature_names_out(    nominal_cols)
    train_ohe_df = pd.DataFrame(    train_ohe,    columns=ohe_cols,    index=train_df.index)
    test_ohe_df = pd.DataFrame(    test_ohe,    columns=ohe_cols,    index=test_df.index)
    print("\n========== ONE-HOT ENCODED DATA ==========")
    print("\nTraining Data:")
    print(train_ohe_df.head())
    print("\nTesting Data:")
    print(test_ohe_df.head())
    return train_ohe_df, test_ohe_df


# =========================================================
# 8. GRADIENT DESCENT
# =========================================================

def gradient_descent(    X,    y,    learning_rate=0.001,    iterations=1000
):
    # Convert to 1-D NumPy arrays
    X = np.asarray(X, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = len(y)
    # Initial parameters
    theta0 = 0.0
    theta1 = 0.0
    mse_history = []
    for i in range(iterations):
        # Prediction
        y_pred = theta0 + theta1 * X
        # Error
        error = y_pred - y
        # MSE
        mse = np.mean(error ** 2)
        mse_history.append(mse)
        # Gradients
        d_theta0 = (2 / m) * np.sum(error)
        d_theta1 = (2 / m) * np.sum(        error * X    )
        # Update parameters
        theta0 = theta0 - learning_rate * d_theta0
        theta1 = theta1 - learning_rate * d_theta1
    return theta0, theta1, mse_history


# =========================================================
# 9. SIMPLE LINEAR REGRESSION USING GRADIENT DESCENT
# =========================================================

def simple_linear_regression_GD(df):
    print("\n========== LINEAR REGRESSION USING GRADIENT DESCENT ==========")
    # Check required columnsif "CGPA" not in df.columns:    raise ValueError("CGPA column not found.")
    if "Salary Package" not in df.columns:    raise ValueError("Salary Package column not found.")
    X = df[["CGPA"]]
    y = df["Salary Package"]
    # Train-test split
    train_X, test_X, train_y, test_y = train_test_split(    X,    y,    test_size=0.2,    random_state=42)
    # Gradient Descent
    theta0, theta1, mse_history = gradient_descent(    train_X,    train_y,    learning_rate=0.001,    iterations=1000)
    print("\nTheta0 (Intercept):", theta0)
    print("Theta1 (Slope):", theta1)
    # Prediction
    test_X_array = test_X["CGPA"].values
    y_pred = theta0 + theta1 * test_X_array
    # Evaluation
    mse1 = mean_squared_error(   test_y,    y_pred)
    rsq1 = r2_score(    test_y,    y_pred)
    print("\nMSE:", mse1)
    print("R-Square Score:", rsq1)
    # Plot MSE vs Iterations
    plt.figure(figsize=(8, 5))
    plt.plot(    mse_history)
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.title(    "Gradient Descent: MSE vs Iterations")
    plt.grid(True)
    plt.show()
    # Regression Line
    plt.figure(figsize=(8, 5))
    plt.scatter(    test_X["CGPA"],    test_y,    alpha=0.5,    label="Actual")
    sorted_indices = np.argsort(    test_X["CGPA"].values)
    plt.plot(    test_X["CGPA"].values[sorted_indices],    y_pred[sorted_indices],    linewidth=2,    label="Regression Line")
    plt.xlabel("CGPA")
    plt.ylabel("Salary Package")
    plt.title(    "Simple Linear Regression: CGPA vs Salary Package")
    plt.legend()
    plt.grid(True)
    plt.show()
    return theta0, theta1, mse1, rsq1


# =========================================================
# 10. SIMPLE LINEAR REGRESSION USING SKLEARN
# =========================================================

def simple_linear_regression(df):
    from sklearn.linear_model import LinearRegression
    print("\n========== SKLEARN LINEAR REGRESSION ==========")
    X = df[["CGPA"]]
    y = df["Salary Package"]
    train_X, test_X, train_y, test_y = train_test_split(    X,    y,    test_size=0.2,    random_state=42)
    model = LinearRegression()
    model.fit(    train_X,    train_y)
    y_pred = model.predict(    test_X)
    mse1 = mean_squared_error(    test_y,    y_pred)
    rsq1 = r2_score(    test_y,    y_pred)
    print("Intercept:", model.intercept_)
    print("Coefficient:", model.coef_[0])
    print("MSE:", mse1)
    print("R-Square Score:", rsq1)
    return model
def simple_logistic_regression(df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    X = df[["CGPA"]]
    y = df["PlacementStatus"]
    print("Placement status values::::::")
    print(y)
    train_X, test_X, train_y, test_y = train_test_split(
        X, y,
        test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(
        max_iter=1000, random_state=42)
    model.fit(train_X, train_y)
    theta0 = model.intercept_[0]
    theta1 = model.coef_[0][0]
    print("Theta0 / Intercept:", theta0)
    print("Theta1 / Coefficient:", theta1)
    y_pred = model.predict(test_X)
    y_prob = model.predict_proba(test_X)[:, 1]
    print(y_pred)
    accuracy = accuracy_score(test_y, y_pred)
    print("Accuracy:", accuracy)
    # Plot as a heatmap
    cm = confusion_matrix(test_y, y_pred)
    print("\n========== CONFUSION MATRIX ==========")
    print(cm)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix — Logistic Regression")
    plt.show()
    print("\n========== SAMPLE PREDICTIONS ==========")
    for i in range(min(10, len(test_X))):
        print(
            "CGPA:", test_X.iloc[i, 0],
            "| Actual:", test_y.iloc[i],
            "| Probability:", round(y_prob[i], 4),
            "| Prediction:", y_pred[i]        )
    return model
# plt.figure(figsize=(18, 10))
# plot_tree(
#     model,
#     feature_names=[
#         "CGPA",
#         "HistoryOfBacklogs",
#         "Internships",
#         "AptitudeTestScore"
#     ],
#     class_names=[str(c) for c in model.classes_],
#     filled=True,
#     rounded=True
# )
# plt.title("Decision Tree - Placement Prediction")
# plt.show()
def decision_tree(df):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
    from sklearn.model_selection import train_test_split
    import numpy as np

    X = df[["CGPA", "Internships", "AptitudeTestScore"]]
    y = df["PlacementStatus"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    p_val = model.predict_proba(X_val)[:, 1]
    accuracy = accuracy_score(y_val, y_pred)
    print("Decision Tree Accuracy:", accuracy)
    print(classification_report(y_val, y_pred))
    def best_threshold(y, p, beta=1.0):
        prec, rec, thr = precision_recall_curve(y, p)
        prec = prec[:-1]
        rec = rec[:-1]
        fb = (1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-12)
        k = np.argmax(fb)
        return thr[k], fb[k]
    print("\n========== BEST THRESHOLD ==========")
    for b in (1, 2):
        tau, f = best_threshold(y_val, p_val, beta=b)
        print(f"beta = {b} : tau = {tau:.2f} F{b} = {f:.3f}")
    return model
def decision_tree_pruned(df):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score , classification_report
    from sklearn.tree import export_graphviz
    X = df[["CGPA", "Internships","AptitudeTestScore"]]
    y = df[["PlacementStatus"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    print(classification_report(y_test, y_pred))
    return model
print("====================================================================================================")
def random_forest_model(df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score , classification_report
    from sklearn.model_selection import train_test_split
    X = df[["CGPA", "Internships","AptitudeTestScore"]]
    y = df["PlacementStatus"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=5,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    return model
print("====================================================================================================")
def compare(sample_df, m1, m2, m3):
    X_sample = sample_df[["CGPA", "Internships", "AptitudeTestScore"]]
    y_actual = sample_df['PlacementStatus']
    pred_dt = m1.predict(X_sample)
    pred_pruned = m2.predict(X_sample)
    pred_rf = m3.predict(X_sample)
    results = pd.DataFrame({
        'Student' : sample_df.index,
        'Actual' : y_actual.values,
        'Decision Tree' : pred_dt,
        'Pruned Decision Tree' : pred_pruned,
        'Random Forest' : pred_rf
    })
    print("\n Prediction Comparison for 5 Random Students")
    print("=" * 80)
    print(results.to_string(index=False))

def adaboost_model(df):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    X = df[["CGPA", "Internships", "AptitudeTestScore"]].copy()
    y = df["PlacementStatus"].copy()
    # Handle missing values
    X = X.fillna(X.median())
    # Convert target to 0 and 1 if required
    if y.dtype == "object":
        y = y.map({"No": 0, "Yes": 1})
    # Remove rows where target is missing
    valid_rows = y.notna()
    X = X[valid_rows]
    y = y[valid_rows]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=1
        ),
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model

def gradient_boosting_model(df):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    X = df[["CGPA", "Internships", "AptitudeTestScore"]]
    X = X.fillna(X.median())
    y = df["PlacementStatus"]
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42,stratify=y)
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        loss="log_loss",
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Gradient Boosting Accuracy:",
          accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model

def xgboost_model(df):
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    X = df[["CGPA", "Internships", "AptitudeTestScore"]]
    X = X.fillna(X.median())
    y = df["PlacementStatus"]
    # Convert Yes/No into 1/0 if required
    if y.dtype == "object":
        y = y.map({"No": 0, "Yes": 1})
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42,stratify=y)
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("XGBoost Accuracy:",accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model

def xgboost_model(df):
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    X = df[["CGPA", "Internships", "AptitudeTestScore"]]
    X = X.fillna(X.median())
    y = df["PlacementStatus"]
    if y.dtype == "object":
        y = y.map({"No": 0, "Yes": 1})
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("XGBoost Accuracy:",accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model

def kmeans_model(df):
    from sklearn.cluster import KMeans
    X = df[["CGPA", "Internships", "AptitudeTestScore"]]
    X = X.fillna(X.median())
    model = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10
    )
    model.fit(X)
    labels = model.labels_
    centers = model.cluster_centers_
    print("\n========== K-MEANS CLUSTERING ==========")
    print("\nCluster Labels:")
    print(labels)
    print("\nCluster Centers:")
    print(centers)
    return model

def dbscan_model(df):
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    X = df[["CGPA", "Internships", "AptitudeTestScore"]]
    X = X.fillna(X.median())
    model = DBSCAN(eps=0.5,min_samples=5)
    model.fit(X)
    labels = model.labels_
    print("\n========== DBSCAN CLUSTERING ==========")
    print("\nCluster Labels:")
    print(labels)
    unique_clusters = set(labels.tolist())
    print("\nUnique Clusters:")
    print(unique_clusters)
    n_clusters = len(unique_clusters - {-1})
    print("\nNumber of Clusters:")
    print(n_clusters)
    n_noise = list(labels).count(-1)
    print("Number of Noise Points:")
    print(n_noise)
    mask = labels != -1
    if n_clusters >= 2 and sum(mask) > n_clusters:
        score = silhouette_score(X[mask], labels[mask])
        print("Silhouette Score:")
        print(score)
    else:
        score = None
        print("Silhouette Score:")
        print("Cannot be calculated")
    return model
# =========================================================
# PCA USING SKLEARN
# =========================================================
def pca_model(df):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X=df[['CGPA','Internships','AptitudeTestScore']].copy()
    X=X.fillna(X.median())
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)
    pca=PCA(n_components=2)
    X_pca=pca.fit_transform(X_scaled)
    print("\n========== PCA ==========")
    print("Explained Variance Ratio:",pca.explained_variance_ratio_)
    print("Total Explained Variance:",pca.explained_variance_ratio_.sum())
    print("PCA Components:")
    print(pca.components_)
    pca_df=pd.DataFrame(X_pca,columns=['PC1','PC2'],index=df.index)
    print("\nPCA Data:")
    print(pca_df.head())

    return pca,pca_df
def onesvm(df):
    print("\n========== ONESVM CLUSTERING ==========")
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    X=df[['CGPA','Internships','AptitudeTestScore']].copy()
    X=X.fillna(X.median())
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)
    model=OneClassSVM(kernel='rbf',gamma='scale',nu=0.05)
    model.fit(X_scaled)
    predictions=model.predict(X_scaled)
    df['OneClassSVM_Label']=predictions
    print(df[['CGPA','Internships','AptitudeTestScore','OneClassSVM_Label']])
    print("Normal Samples:",(predictions==1).sum())
    print("Anomaly Samples:",(predictions==-1).sum())
    print(df[df['OneClassSVM_Label']==-1])
    return model

def anomaly_detection(df):
    anomalies=df[df['OneClassSVM_Label']==-1]
    print("\n========== ANOMALY DETECTION ==========")
    print("Total Anomalies:",len(anomalies))
    print("Total Normal:",(df['OneClassSVM_Label']==1).sum())
    print("\nAnomaly Records:")
    print(anomalies[['CGPA','Internships','AptitudeTestScore','OneClassSVM_Label']])
    return anomalies

def platt_scaling(df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    X = df[["CGPA", "Internships", "AptitudeTestScore"]]
    y = df["PlacementStatus"]
    X = X.fillna(X.median())
    if y.dtype == "object":
        y = y.map({"No": 0, "Yes": 1})
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42,stratify=y)
    model = RandomForestClassifier( n_estimators=100,random_state=42)
    model.fit(X_train, y_train)
    p_raw = model.predict_proba(X_test)[:, 1]
    p_raw = np.clip(p_raw, 1e-10, 1 - 1e-10)
    f = np.log(p_raw / (1 - p_raw))
    platt = LogisticRegression(C=1e6)
    platt.fit(f.reshape(-1, 1), y_test)
    p_calibrated = platt.predict_proba(f.reshape(-1, 1))[:, 1]
    print("Raw Probability:")
    print(p_raw[:10])
    print("Platt Calibrated Probability:")
    print(p_calibrated[:10])
    return platt

def grid_search(df):
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV,train_test_split
    X=df[["CGPA","Internships","AptitudeTestScore"]]
    y=df["PlacementStatus"]
    X=X.fillna(X.median())
    if y.dtype=="object":
        y=y.map({"No":0,"Yes":1})
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    X_train=X_train.sample(n=5000,random_state=42)
    y_train=y_train.loc[X_train.index]
    param_grid={"C":[0.1,1,10],"gamma":[0.001,0.01,0.1,1]}
    grid=GridSearchCV(SVC(kernel="rbf"),param_grid=param_grid,cv=5,scoring="accuracy",n_jobs=-1,verbose=1)
    grid.fit(X_train,y_train)
    print("Grid Best Parameters:",grid.best_params_)
    print("Grid Best Score:",grid.best_score_)
    print("Test Accuracy:",grid.score(X_test,y_test))
    return grid

def random_search(df):
    from sklearn.svm import SVC
    from sklearn.model_selection import RandomizedSearchCV,train_test_split
    from scipy.stats import loguniform
    X=df[["CGPA","Internships","AptitudeTestScore"]]
    y=df["PlacementStatus"]
    X=X.fillna(X.median())
    if y.dtype=="object":
        y=y.map({"No":0,"Yes":1})
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    X_train=X_train.sample(n=5000,random_state=42)
    y_train=y_train.loc[X_train.index]
    param_dist={"C":loguniform(1e-3,1e1),"gamma":loguniform(1e-4,1e0)}
    rand=RandomizedSearchCV(SVC(kernel="rbf"),param_distributions=param_dist,n_iter=10,cv=5,scoring="accuracy",random_state=42,n_jobs=-1)
    rand.fit(X_train,y_train)
    print("Random Best Parameters:",rand.best_params_)
    print("Random Best Score:",rand.best_score_)
    print("Test Accuracy:",rand.score(X_test,y_test))
    return rand

# MAIN PROGRAM
if __name__ == "__main__":
# Load
    df = load_data()
    print("Dataset Shape:", df.shape)
    print("\nFirst 5 Rows:")
    print(df.head())
    print(df['PlacementStatus'])
    # Uncomment when required
    # summarize(df)
    # stat(df)
    # visualize(df)
    # Scaling
    X=df.drop(["PlacementStatus"], axis=1)
    target_cols = ["PlacementStatus", "IsAnomaly", "Salary Package"]
    normalized_df, standardized_df = normalize(df, target_cols)
    # Outlier handling
    standardized_df = outlier(standardized_df)
    # One-hot encoding
    #train_ohe_df, test_ohe_df = onehot(standardized_df)
    # Linear Regression using original data
    #simple_linear_regression_GD(df)
    # Linear Regression using standardized data
    #simple_linear_regression_GD(standardized_df)
    print(standardized_df.head())
    simple_logistic_regression(standardized_df)
    m1 = decision_tree(standardized_df)
    m2 = decision_tree_pruned(standardized_df)
    m3 = random_forest_model(standardized_df)
    sample_df = standardized_df.sample(n=10, random_state=42)
    compare(sample_df, m1, m2, m3)
    print("====================================================================================================")
    m4 = adaboost_model(standardized_df)
    print("====================================================================================================")
    m5 = gradient_boosting_model(standardized_df)
    print("====================================================================================================")
    m6 = xgboost_model(standardized_df)
    print("====================================================================================================")
    # m7 = lightgbm_model(standardized_df)
    m8 = kmeans_model(standardized_df)
    print("====================================================================================================")
    m9 = dbscan_model(standardized_df)
    print("====================================================================================================")
    m10, pca_df = pca_model(standardized_df)
    print("====================================================================================================")
    svm_df = standardized_df.sample(n=5000, random_state=42)
    m11 = onesvm(svm_df)
    print("====================================================================================================")
    anomalies = anomaly_detection(svm_df)
    print("====================================================================================================")
    print("Platt Scaling:")
    m12 = platt_scaling(standardized_df)
    print("====================================================================================================")
    print("Grid Search:")
    m13 = grid_search(standardized_df)
    print("====================================================================================================")
    print("Random Search:")
    m14 = random_search(standardized_df)
