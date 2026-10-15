import ast
import io
import os
import re
from contextlib import redirect_stdout

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request

from models.load import load_data
from models import eda as eda_module
from models import feature_engg as fe_module
from models.predict import predict_student, get_feature_info

from models.complete_code import (
    summarize as cc_summarize,
    stat as cc_stat,
    visualize as cc_visualize,
    normalize as cc_normalize,
    outlier as cc_outlier,
    onehot as cc_onehot,
    simple_linear_regression_GD as cc_simple_linear_regression_GD,
    simple_linear_regression as cc_simple_linear_regression,
    simple_logistic_regression as cc_simple_logistic_regression,
    decision_tree as cc_decision_tree,
    decision_tree_pruned as cc_decision_tree_pruned,
    random_forest_model as cc_random_forest_model,
    compare as cc_compare,
    adaboost_model as cc_adaboost_model,
    gradient_boosting_model as cc_gradient_boosting_model,
    xgboost_model as cc_xgboost_model,
    kmeans_model as cc_kmeans_model,
    dbscan_model as cc_dbscan_model,
    pca_model as cc_pca_model,
    onesvm as cc_onesvm,
    anomaly_detection as cc_anomaly_detection,
    platt_scaling as cc_platt_scaling,
    grid_search as cc_grid_search,
    random_search as cc_random_search,
)

app = Flask(__name__)
app.secret_key = "dev-secret-key"

DATA_PATH = "placement_predict_50k Dataset.csv"
TARGET_COLS = ["PlacementStatus", "IsAnomaly", "Salary Package"]

_cache = {}


def _cached(key, fn):
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]


def get_df():
    if "df" not in _cache:
        _cache["df"] = load_data(DATA_PATH)
    return _cache["df"]


def _load_and_cache():
    df = get_df()
    knn = _cached("knn", lambda: fe_module.knn_impute(df))
    regression = _cached("regression_models", lambda: fe_module.run_regression_models(df))
    return df, knn, regression


def capture(fn, *args, **kwargs):
    buf = io.StringIO()
    plt.close("all")
    try:
        with redirect_stdout(buf):
            result = fn(*args, **kwargs)
    except Exception as e:
        plt.close("all")
        return None, buf.getvalue() + "\nError: " + str(e)
    return result, buf.getvalue()


def save_figures(names):
    os.makedirs(os.path.join("static", "plots"), exist_ok=True)
    saved = []
    for num, name in zip(list(plt.get_fignums()), names):
        plt.figure(num).savefig(
            os.path.join("static", "plots", name),
            dpi=100,
            bbox_inches="tight"
        )
        saved.append(name)
    plt.close("all")
    return saved


def df_head(df, n=10):
    if df is None:
        return ""
    return df.head(n).to_html(classes="table")


def parse_float(output, label):
    m = re.search(
        re.escape(label) + r"\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        output
    )
    return round(float(m.group(1)), 4) if m else None


def parse_int(output, label):
    m = re.search(re.escape(label) + r"\s*(\d+)", output)
    return int(m.group(1)) if m else None


def parse_value(output, label):
    m = re.search(re.escape(label) + r"\s*(.+)", output)
    return m.group(1).strip() if m else None


def parse_literal(output, label):
    text = parse_value(output, label)
    if not text:
        return None
    text = text.splitlines()[0]
    text = re.sub(r"np\.\w+\(([^)]*)\)", r"\1", text)
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def parse_array(output, label):
    idx = output.find(label)
    if idx < 0:
        return None
    m = re.search(r"\[([^\]]+)\]", output[idx + len(label):])
    if not m:
        return None
    return [round(float(x), 4) for x in m.group(1).split()]


def parse_accuracy(output):
    m = re.search(r"Accuracy:\s*([\d.]+)", output)
    return round(float(m.group(1)), 4) if m else None


def parse_clf_accuracy(output):
    m = re.search(r"accuracy\s+([\d.]+)", output, re.IGNORECASE)
    return round(float(m.group(1)), 4) if m else None


def parse_sample_predictions(output):
    rows = []
    pattern = re.compile(
        r"CGPA:\s*([\d.]+)\s*\|\s*Actual:\s*(\S+)\s*"
        r"\|\s*Probability:\s*([\d.]+)\s*\|\s*Prediction:\s*(\S+)"
    )
    for m in pattern.finditer(output):
        rows.append({
            "cgpa": m.group(1),
            "actual": m.group(2),
            "probability": m.group(3),
            "prediction": m.group(4)
        })
    return rows[:10]


@app.route("/")
def index():
    data_available = os.path.exists(DATA_PATH)
    return render_template(
        "index.html",
        data_available=data_available,
        data_path=DATA_PATH
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    feature_info = None
    result = None
    error = None
    values = {}

    try:
        feature_info = get_feature_info()
    except Exception as e:
        error = str(e)

    if request.method == "POST":
        values = request.form.to_dict()
        try:
            result = predict_student(values)
        except (ValueError, TypeError, FileNotFoundError) as e:
            error = f"Invalid input: {e}"

    if feature_info is None:
        return render_template(
            "predict.html",
            numeric_cols=[],
            categorical_cols=[],
            categories={},
            values=values,
            result=result,
            error=error
        )

    return render_template(
        "predict.html",
        numeric_cols=feature_info["numeric_cols"],
        categorical_cols=feature_info["categorical_cols"],
        categories=feature_info["categories"],
        values=values,
        result=result,
        error=error
    )


@app.route("/load")
def load_page():
    df, knn, _ = _load_and_cache()
    cleaned_df = knn["df"]
    summary = eda_module.summarize(df)
    cleaned_summary = eda_module.summarize(cleaned_df)
    return render_template(
        "load.html",
        shape=summary["shape"],
        columns=summary["columns"],
        preview=df.head(10).to_dict(orient="records"),
        cleaned_shape=cleaned_summary["shape"],
        cleaned_columns=cleaned_summary["columns"],
        cleaned_preview=cleaned_df.head(10).to_dict(orient="records")
    )


def _compute_stat(df):
    result, _ = capture(cc_stat, df)
    if result is None:
        return {"stat_summary": [], "stat_corr": None}
    _, mean_values, std_values, skewness, kurtosis, correlation = result
    stat_summary = []
    for col in mean_values.index:
        stat_summary.append({
            "feature": col,
            "mean": round(float(mean_values[col]), 3),
            "std": round(float(std_values[col]), 3),
            "skew": round(float(skewness[col]), 3),
            "kurtosis": round(float(kurtosis[col]), 3)
        })
    stat_corr = {
        "columns": correlation.columns.tolist(),
        "rows": [
            {
                "feature": idx,
                "values": [round(float(v), 2) for v in row]
            }
            for idx, row in correlation.iterrows()
        ]
    }
    return {"stat_summary": stat_summary, "stat_corr": stat_corr}


def _compute_visualize(df):
    capture(cc_visualize, df)
    saved = save_figures(["visualize_overview.png"])
    return saved[0] if saved else None


@app.route("/eda")
def eda_page():
    df, knn, regression = _load_and_cache()

    summary = eda_module.summarize(df)

    _cached("eda_plots", lambda: eda_module.generate_full_eda_plots(df))

    placement_values = df["PlacementStatus"].astype(str).str.lower()
    placed_count = placement_values.isin(["yes", "1", "placed"]).sum()
    placement_rate = round((placed_count / len(df)) * 100, 2)

    missing_values = eda_module.get_missing_summary(df)
    duplicate_stats = eda_module.get_duplicate_stats(df)

    filled_df = eda_module.median_fill_numeric(df)
    missing_before = int(df.isnull().sum().sum())
    missing_after = int(filled_df.isnull().sum().sum())

    unique_values = []
    for column in df.columns:
        unique_values.append({
            "column": column,
            "count": int(df[column].nunique())
        })

    descriptive_statistics = eda_module.get_univariate_stats(df)
    numeric_describe = eda_module.get_numeric_describe(df)
    categorical_summary = eda_module.get_categorical_summary(df)

    cgpa_values = eda_module.column_summary(df, "CGPA")
    cgpa_mode = cgpa_values["mode"]
    if len(cgpa_mode) > 0:
        cgpa_mode = cgpa_mode[0]
    else:
        cgpa_mode = "N/A"

    cgpa = {
        "mean": round(cgpa_values["mean"], 2),
        "median": round(cgpa_values["median"], 2),
        "mode": cgpa_mode,
        "range": round(cgpa_values["range"], 2),
        "variance": round(cgpa_values["variance"], 2),
        "std": round(cgpa_values["standard_deviation"], 2),
        "q1": round(cgpa_values["q1"], 2),
        "q3": round(cgpa_values["q3"], 2),
        "iqr": round(cgpa_values["iqr"], 2)
    }

    ttest_result = eda_module.cgpa_placement_ttest(df)
    ttest = {
        "statistic": round(float(ttest_result.statistic), 4),
        "pvalue": round(float(ttest_result.pvalue), 6)
    }

    gender_breakdown = eda_module.get_placement_breakdown(df, "Gender")
    college_breakdown = eda_module.get_placement_breakdown(df, "CollegeTier")
    stream_breakdown = eda_module.get_placement_breakdown(df, "Stream")
    sgpa_trend = eda_module.get_sgpa_trend_data(df)
    salary_stats = eda_module.get_salary_stats(df)

    train_scaled, test_scaled = fe_module.standardization(knn["df"])
    std_columns = train_scaled.columns.tolist()
    std_train_rows = train_scaled.head(5).round(3).values.tolist()
    std_test_rows = test_scaled.head(5).round(3).values.tolist()

    cc_summary_output = _cached(
        "cc_summary",
        lambda: capture(cc_summarize, df)[1]
    )
    stat_data = _cached("cc_stat", lambda: _compute_stat(df))
    visualize_image = _cached("cc_visualize", lambda: _compute_visualize(df))

    return render_template(
        "eda.html",
        shape=summary["shape"],
        columns=summary["columns"],
        numeric_columns=summary["numeric_columns"],
        categorical_columns=summary["categorical_columns"],
        duplicates=summary["duplicates"],
        total_missing=missing_before,
        total_unique=int(df.nunique().sum()),
        missing_values=missing_values,
        missing_before=missing_before,
        missing_after=missing_after,
        duplicate_stats=duplicate_stats,
        unique_values=unique_values,
        descriptive_statistics=descriptive_statistics,
        numeric_describe=numeric_describe,
        categorical_summary=categorical_summary,
        correlations=eda_module.get_correlation_with_target(df),
        cgpa=cgpa,
        ttest=ttest,
        placement_rate=placement_rate,
        preview=df.head(10).to_dict(orient="records"),
        gender_breakdown=gender_breakdown,
        college_breakdown=college_breakdown,
        stream_breakdown=stream_breakdown,
        sgpa_trend=sgpa_trend,
        salary_stats=salary_stats,
        knn=knn,
        regression=regression,
        std_columns=std_columns,
        std_train_rows=std_train_rows,
        std_test_rows=std_test_rows,
        cc_summary_output=cc_summary_output,
        stat_summary=stat_data["stat_summary"],
        stat_corr=stat_data["stat_corr"],
        visualize_image=visualize_image
    )


def _compute_feature_engg(df):
    data = {}

    norm = capture(cc_normalize, df, TARGET_COLS)[0]
    if norm is not None:
        normalized_df, standardized_df = norm
        standardized_df = capture(cc_outlier, standardized_df)[0]
    else:
        normalized_df = standardized_df = None

    ohe = capture(cc_onehot, df)[0]
    train_ohe_df = test_ohe_df = None
    if ohe is not None:
        train_ohe_df, test_ohe_df = ohe

    numeric_df = df.select_dtypes(include="number")

    fe_norm = capture(fe_module.normalize, numeric_df)[0]
    fe_norma = capture(fe_module.normalization, numeric_df)[0]
    fe_std = capture(fe_module.standardization, df)[0]
    fe_ohe = capture(fe_module.onehot, df)[0]
    fe_ord = capture(fe_module.ordinal, df.copy())[0]
    fe_label = capture(fe_module.label_encode_branch, df)[0]
    cleaned = capture(fe_module.data_cleaning, df.copy())[0]
    outliers = capture(fe_module.detect_outliers, df, "Salary Package")[0]
    numerical = capture(fe_module.get_numerical_features, df)[0]
    categorical = capture(fe_module.get_categorical_features, df)[0]
    selected = capture(fe_module.select_model_features, df)[0]
    null_output = capture(fe_module.find_null_values, df, "Workshops")[1]
    cgpa_df = capture(fe_module.create_cgpa_category, df.copy())[0]

    cgpa_counts = []
    if cgpa_df is not None and "CGPA_Category" in cgpa_df.columns:
        for name, count in cgpa_df["CGPA_Category"].value_counts().items():
            cgpa_counts.append({"name": str(name), "count": int(count)})

    label_mapping = {}
    ordinal_train = ordinal_test = None
    label_train = label_test = None
    if fe_ord is not None:
        ordinal_train, ordinal_test = fe_ord
    if fe_label is not None:
        label_train, label_test, label_mapping = fe_label

    selected_features = []
    target_name = None
    if selected is not None:
        X, y = selected
        selected_features = X.columns.tolist()
        target_name = y.name

    missing_before = int(df.isnull().sum().sum())
    missing_after = int(cleaned.isnull().sum().sum()) if cleaned is not None else None

    data.update({
        "normalized_head": df_head(normalized_df),
        "standardized_head": df_head(standardized_df),
        "ohe_train_head": df_head(train_ohe_df),
        "ohe_test_head": df_head(test_ohe_df),
        "fe_norm_train_head": df_head(fe_norm[0] if fe_norm else None),
        "fe_norm_test_head": df_head(fe_norm[1] if fe_norm else None),
        "fe_norma_train_head": df_head(fe_norma[0] if fe_norma else None),
        "fe_norma_test_head": df_head(fe_norma[1] if fe_norma else None),
        "fe_std_train_head": df_head(fe_std[0] if fe_std else None),
        "fe_std_test_head": df_head(fe_std[1] if fe_std else None),
        "fe_ohe_head": df_head(fe_ohe[0] if fe_ohe else None),
        "ordinal_train_head": df_head(ordinal_train),
        "ordinal_test_head": df_head(ordinal_test),
        "label_train_head": df_head(label_train),
        "label_test_head": df_head(label_test),
        "label_mapping": label_mapping,
        "cgpa_counts": cgpa_counts,
        "outlier_count": len(outliers) if outliers is not None else 0,
        "outliers_head": df_head(outliers),
        "cleaned_head": df_head(cleaned),
        "missing_before": missing_before,
        "missing_after": missing_after,
        "numerical_cols": numerical.columns.tolist() if numerical is not None else [],
        "categorical_cols": categorical.columns.tolist() if categorical is not None else [],
        "selected_features": selected_features,
        "target_name": target_name,
        "null_output": "\n".join(null_output.splitlines()[:5])
    })
    return data


@app.route("/feature-engg")
def feature_engg_page():
    df = get_df()
    data = _cached("fe_page", lambda: _compute_feature_engg(df))
    return render_template("feature_engg.html", **data)


def _compute_linear(df):
    linear_df = df.dropna(subset=["Salary Package"])

    gd_result, _ = capture(cc_simple_linear_regression_GD, df)
    gd_figs = save_figures(
        ["gd_mse_curve.png", "gd_regression_line.png"]
    )
    gd = None
    if gd_result is not None:
        theta0, theta1, mse1, rsq1 = gd_result
        gd = {
            "theta0": round(float(theta0), 4),
            "theta1": round(float(theta1), 4),
            "mse": round(float(mse1), 4),
            "rsq": round(float(rsq1), 4),
            "figures": gd_figs
        }

    sk_model, sk_out = capture(cc_simple_linear_regression, linear_df)
    sk = None
    if sk_model is not None:
        sk = {
            "intercept": round(float(sk_model.intercept_), 4),
            "coefficient": round(float(sk_model.coef_[0]), 4),
            "mse": parse_float(sk_out, "MSE:"),
            "rsq": parse_float(sk_out, "R-Square Score:")
        }

    return {
        "gd": gd,
        "sk": sk,
        "fe_linear": capture(fe_module.run_simple_linear, linear_df)[0],
        "fe_gd": capture(fe_module.run_linear_gd, df)[0],
        "builtin": capture(fe_module.builtin_linear_sample)[0]
    }


@app.route("/linear-regression")
def linear_regression_page():
    df, knn, _ = _load_and_cache()
    cleaned_df = knn["df"]
    data = _cached("linreg_page", lambda: _compute_linear(cleaned_df))
    return render_template("linear_regression.html", **data)


def _compute_logistic(df):
    model, out = capture(cc_simple_logistic_regression, df)
    figs = save_figures(["logistic_confusion_matrix.png"])
    logit = None
    if model is not None:
        logit = {
            "theta0": round(float(model.intercept_[0]), 4),
            "theta1": round(float(model.coef_[0][0]), 4),
            "accuracy": parse_accuracy(out),
            "confusion_figure": figs[0] if figs else None,
            "samples": parse_sample_predictions(out)
        }

    return {
        "logit": logit,
        "fe_logit": capture(fe_module.simple_logistic_regression_GD, df)[0],
        "builtin": capture(fe_module.builtin_logistic_sample)[0]
    }


@app.route("/logistic-regression")
def logistic_regression_page():
    df, knn, _ = _load_and_cache()
    cleaned_df = knn["df"]
    data = _cached(
        "logit_page",
        lambda: _compute_logistic(cleaned_df)
    )
    return render_template("logistic_regression.html", **data)


def _compute_classification(cleaned_df):
    dt_model, dt_out = capture(cc_decision_tree, cleaned_df)
    dt_accuracy = parse_accuracy(dt_out)

    dt_pruned_model, dt_pruned_out = capture(
        cc_decision_tree_pruned, cleaned_df
    )
    dt_pruned_accuracy = parse_clf_accuracy(dt_pruned_out)

    rf_model, rf_out = capture(cc_random_forest_model, cleaned_df)
    rf_accuracy = parse_accuracy(rf_out)

    sample_df = cleaned_df.sample(
        n=min(10, len(cleaned_df)),
        random_state=42
    )
    _, compare_out = capture(
        cc_compare, sample_df, dt_model, dt_pruned_model, rf_model
    )

    ab_model, ab_out = capture(cc_adaboost_model, cleaned_df)
    ab_accuracy = parse_accuracy(ab_out) or parse_clf_accuracy(ab_out)

    gb_model, gb_out = capture(cc_gradient_boosting_model, cleaned_df)
    gb_accuracy = parse_accuracy(gb_out) or parse_clf_accuracy(gb_out)

    xgb_model, xgb_out = capture(cc_xgboost_model, cleaned_df)
    xgb_accuracy = parse_accuracy(xgb_out) or parse_clf_accuracy(xgb_out)

    return {
        "dt_accuracy": dt_accuracy,
        "dt_pruned_accuracy": dt_pruned_accuracy,
        "rf_accuracy": rf_accuracy,
        "compare_output": compare_out.strip(),
        "ab_accuracy": ab_accuracy,
        "gb_accuracy": gb_accuracy,
        "xgb_accuracy": xgb_accuracy,
    }


def _compute_regression_extras(cleaned_df):
    platt_model, platt_out = capture(cc_platt_scaling, cleaned_df)
    platt = None
    if platt_model is not None:
        platt = {
            "raw": parse_array(platt_out, "Raw Probability:"),
            "calibrated": parse_array(
                platt_out,
                "Platt Calibrated Probability:"
            )
        }

    grid_model, grid_out = capture(cc_grid_search, cleaned_df)
    if grid_model is not None:
        grid = {
            "best_params": parse_literal(
                grid_out, "Grid Best Parameters:"
            ),
            "best_score": parse_float(grid_out, "Grid Best Score:"),
            "test_accuracy": parse_float(grid_out, "Test Accuracy:"),
            "error": None
        }
    else:
        grid = {
            "best_params": None,
            "best_score": None,
            "test_accuracy": None,
            "error": grid_out.strip()[-500:]
        }

    rand_model, rand_out = capture(cc_random_search, cleaned_df)
    if rand_model is not None:
        random_search_result = {
            "best_params": parse_literal(
                rand_out, "Random Best Parameters:"
            ),
            "best_score": parse_float(rand_out, "Random Best Score:"),
            "test_accuracy": parse_float(rand_out, "Test Accuracy:"),
            "error": None
        }
    else:
        random_search_result = {
            "best_params": None,
            "best_score": None,
            "test_accuracy": None,
            "error": rand_out.strip()[-500:]
        }

    rf_all, rf_all_out = capture(
        fe_module.run_random_forest_classifier, cleaned_df
    )
    rf_all_accuracy = parse_accuracy(rf_all_out)
    if rf_all_accuracy is None and rf_all is not None:
        try:
            rf_all_accuracy = round(float(rf_all[1]), 4)
        except (TypeError, ValueError, IndexError):
            rf_all_accuracy = None

    return {
        "platt": platt,
        "grid": grid,
        "random_search": random_search_result,
        "rf_all_accuracy": rf_all_accuracy
    }


@app.route("/regression")
def regression_page():
    df, knn, multi = _load_and_cache()
    cleaned_df = knn["df"]
    data = _cached(
        "regression_page",
        lambda: _compute_regression_page(cleaned_df, multi)
    )
    return render_template("regression.html", **data)


def _compute_regression_page(cleaned_df, multi):
    linear_df = cleaned_df.dropna(subset=["Salary Package"])
    data = {
        "linear": capture(fe_module.run_simple_linear, linear_df)[0],
        "gd": capture(fe_module.run_linear_gd, cleaned_df)[0],
        "logistic": capture(
            fe_module.simple_logistic_regression_GD, cleaned_df
        )[0],
        "builtin_linear": capture(fe_module.builtin_linear_sample)[0],
        "builtin_logistic": capture(fe_module.builtin_logistic_sample)[0],
        "multi": multi,
    }
    data.update(_compute_classification(cleaned_df))
    data.update(_compute_regression_extras(cleaned_df))
    return data


def _compute_unsupervised(cleaned_df):
    km_model, km_out = capture(cc_kmeans_model, cleaned_df)

    km_centers = None
    km_labels_count = None
    if km_model is not None:
        km_centers = [
            [round(float(c), 4) for c in center]
            for center in km_model.cluster_centers_
        ]
        km_labels_count = len(set(km_model.labels_.tolist()))

    db_model, db_out = capture(cc_dbscan_model, cleaned_df)

    db_clusters = None
    db_noise = None
    db_silhouette = None
    if db_model is not None:
        labels = db_model.labels_
        unique = set(labels.tolist())
        db_clusters = len(unique - {-1})
        db_noise = list(labels).count(-1)
        from sklearn.metrics import silhouette_score
        mask = labels != -1
        if db_clusters >= 2 and sum(mask) > db_clusters:
            db_silhouette = round(float(silhouette_score(
                cleaned_df[
                    ["CGPA", "Internships", "AptitudeTestScore"]
                ][mask],
                labels[mask]
            )), 4)

    pca_features = ["CGPA", "Internships", "AptitudeTestScore"]
    pca_explained = []
    pca_components = []
    pca_res, pca_out = capture(cc_pca_model, cleaned_df)
    if pca_res is not None:
        pca, pca_df = pca_res
        pca_explained = [
            round(float(v), 4)
            for v in pca.explained_variance_ratio_
        ]
        pca_components = [
            {
                "feature": feat,
                "values": [round(float(c), 4) for c in comp]
            }
            for feat, comp in zip(pca_features, pca.components_)
        ]

    svm_df = cleaned_df.sample(
        n=min(5000, len(cleaned_df)),
        random_state=42
    ).copy()
    svm_model, svm_out = capture(cc_onesvm, svm_df)
    svm_normal = parse_int(svm_out, "Normal Samples:")
    svm_anomaly = parse_int(svm_out, "Anomaly Samples:")
    if svm_model is not None and "OneClassSVM_Label" in svm_df.columns:
        if svm_normal is None:
            svm_normal = int((svm_df["OneClassSVM_Label"] == 1).sum())
        if svm_anomaly is None:
            svm_anomaly = int((svm_df["OneClassSVM_Label"] == -1).sum())

    anomalies, anomaly_out = capture(cc_anomaly_detection, svm_df)
    anomaly_total = parse_int(anomaly_out, "Total Anomalies:")
    normal_total = parse_int(anomaly_out, "Total Normal:")
    anomaly_records = []
    if anomalies is not None and len(anomalies) > 0:
        record_cols = [
            c for c in [
                "CGPA",
                "Internships",
                "AptitudeTestScore",
                "OneClassSVM_Label"
            ]
            if c in anomalies.columns
        ]
        anomaly_records = (
            anomalies[record_cols]
            .head(10)
            .round(3)
            .to_dict(orient="records")
        )

    return {
        "km_centers": km_centers,
        "km_labels_count": km_labels_count,
        "km_output": km_out.strip(),
        "db_clusters": db_clusters,
        "db_noise": db_noise,
        "db_silhouette": db_silhouette,
        "db_output": db_out.strip(),
        "pca_explained": pca_explained,
        "pca_features": pca_features,
        "pca_components": pca_components,
        "svm_normal": svm_normal,
        "svm_anomaly": svm_anomaly,
        "svm_output": svm_out.strip()[:2000],
        "anomaly_total": anomaly_total,
        "normal_total": normal_total,
        "anomaly_records": anomaly_records,
        "anomaly_output": anomaly_out.strip()[:2000],
    }


@app.route("/unsupervised-learning")
def unsupervised_page():
    df, knn, _ = _load_and_cache()
    cleaned_df = knn["df"]
    unsup = _cached(
        "unsup",
        lambda: _compute_unsupervised(cleaned_df)
    )
    return render_template("unsupervised-learning.html", **unsup)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
