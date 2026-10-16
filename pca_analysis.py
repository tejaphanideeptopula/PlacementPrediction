"""Standalone PCA analysis for the placement prediction dataset.

Loads the full numeric feature set (excluding ID/target columns),
imputes missing values, standardizes, runs PCA with sklearn, prints
explained-variance and loading tables, and saves two plots:

    static/plots/pca_scree.png    - per-component + cumulative explained variance
    static/plots/pca_scatter.png  - PC1 vs PC2 colored by PlacementStatus

Run:  python pca_analysis.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "placement_predict_50k Dataset.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "static" / "plots"
EXCLUDE_COLS = ["StudentID", "PlacementStatus", "IsAnomaly", "Salary Package"]
N_TARGET_VARIANCE = 0.95
N_TARGET_VARIANCE_ALT = 0.90
SCATTER_SAMPLE = 5000
RANDOM_STATE = 42


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the placement dataset CSV and basic validation."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Select all numeric columns except ID/target columns for PCA."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in EXCLUDE_COLS]
    if not feature_cols:
        raise ValueError("No numeric feature columns available for PCA.")
    return df[feature_cols].copy(), feature_cols


def preprocess(X: pd.DataFrame) -> np.ndarray:
    """Median-impute missing values, then standardize to zero mean / unit variance."""
    null_counts = X.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if len(null_counts) > 0:
        print("\nNulls imputed (median):")
        for col, count in null_counts.items():
            print(f"  {col}: {count}")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X)
    return scaler.fit_transform(X_imputed)


def run_pca(X_scaled: np.ndarray) -> PCA:
    """Fit PCA on all components of the scaled feature matrix."""
    pca = PCA(n_components=X_scaled.shape[1], random_state=RANDOM_STATE)
    pca.fit(X_scaled)
    return pca


def print_explained_variance(pca: PCA) -> None:
    """Print eigenvalue table, cumulative variance, and components for 90%/95%."""
    var = pca.explained_variance_ratio_
    cum = np.cumsum(var)
    table = pd.DataFrame({
        "Component": [f"PC{i + 1}" for i in range(len(var))],
        "Eigenvalue": pca.explained_variance_.round(4),
        "Variance %": (var * 100).round(2),
        "Cumulative %": (cum * 100).round(2),
    })
    print("\n========== EXPLAINED VARIANCE ==========")
    print(table.to_string(index=False))
    n95 = int(np.searchsorted(cum, N_TARGET_VARIANCE) + 1)
    n90 = int(np.searchsorted(cum, N_TARGET_VARIANCE_ALT) + 1)
    print(f"\nComponents needed for {int(N_TARGET_VARIANCE_ALT * 100)}% variance: {n90}")
    print(f"Components needed for {int(N_TARGET_VARIANCE * 100)}% variance: {n95}")


def print_loadings(pca: PCA, features: list[str]) -> None:
    """Print the full loadings matrix and the top-3 features per leading PC."""
    columns = [f"PC{i + 1}" for i in range(pca.n_components_)]
    loadings = pd.DataFrame(pca.components_.T, index=features, columns=columns)
    print("\n========== LOADINGS (features x components) ==========")
    print(loadings.round(3).to_string())
    print("\n========== TOP FEATURES BY ABSOLUTE LOADING ==========")
    for i in range(min(5, pca.n_components_)):
        top = loadings.iloc[:, i].abs().sort_values(ascending=False).head(3)
        ranked = ", ".join(
            f"{feat} ({loadings.loc[feat, columns[i]]:+.3f})"
            for feat in top.index
        )
        print(f"{columns[i]}: {ranked}")


def plot_scree(pca: PCA) -> Path:
    """Save scree plot: variance bar + cumulative line with 95% marker."""
    var = pca.explained_variance_ratio_ * 100
    cum = np.cumsum(var)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, len(var) + 1), var, alpha=0.7, label="Variance %")
    ax.plot(range(1, len(cum) + 1), cum, "o-", color="darkorange",
            label="Cumulative %")
    ax.axhline(95, color="red", linestyle="--", linewidth=1, label="95% threshold")
    n95 = int(np.searchsorted(cum / 100, N_TARGET_VARIANCE) + 1)
    ax.annotate(f"95% at PC{n95}", xy=(n95, 95), xytext=(n95 + 1, 80),
                arrowprops=dict(arrowstyle="->", color="red"), color="red")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("PCA Scree Plot - Placement Dataset")
    ax.set_xticks(range(1, len(var) + 1))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "pca_scree.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_scatter(df: pd.DataFrame, scores: np.ndarray,
                 var_ratios: np.ndarray) -> Path:
    """Save PC1 vs PC2 scatter of a 5000-row sample colored by PlacementStatus."""
    scores_df = pd.DataFrame(
        scores[:, :2], index=df.index, columns=["PC1", "PC2"]
    )
    sample = scores_df.sample(n=min(SCATTER_SAMPLE, len(scores_df)),
                              random_state=RANDOM_STATE)
    target = df.loc[sample.index, "PlacementStatus"]
    fig, ax = plt.subplots(figsize=(9, 6))
    for value, label, color in [(0, "Not Placed (0)", "steelblue"),
                                (1, "Placed (1)", "darkorange")]:
        mask = target == value
        ax.scatter(sample.loc[mask, "PC1"], sample.loc[mask, "PC2"],
                   s=12, alpha=0.3, label=label, color=color)
    ax.set_xlabel(f"PC1 ({var_ratios[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_ratios[1] * 100:.1f}%)")
    ax.set_title("PCA Projection - PC1 vs PC2 (5000 sample)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "pca_scatter.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    """Run the full PCA pipeline end to end."""
    df = load_dataset()
    print("Dataset Shape:", df.shape)

    X, feature_cols = get_feature_matrix(df)
    print(f"\nFeatures used for PCA ({len(feature_cols)}):")
    print(", ".join(feature_cols))

    X_scaled = preprocess(X)
    pca = run_pca(X_scaled)

    print_explained_variance(pca)
    print_loadings(pca, feature_cols)

    scree_path = plot_scree(pca)
    scores = pca.transform(X_scaled)
    scatter_path = plot_scatter(df, scores, pca.explained_variance_ratio_)

    print("\n========== SAVED PLOTS ==========")
    print("Scree plot:", scree_path)
    print("PC1/PC2 scatter:", scatter_path)


if __name__ == "__main__":
    main()
