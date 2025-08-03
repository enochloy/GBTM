import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def impute_for_kmeans(
    data_long: pd.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: list,
    fill_method="interpolate",
) -> np.ndarray:
    """
    Transforms long-format data to wide format (N x T) for KMeans clustering,
    imputing missing values within each individual's trajectory.

    Args:
        data_long (pd.DataFrame): The input DataFrame in long format.
        unit (str): Name of the column representing the individual/unit ID.
        time (str): Name of the column representing the time point.
        outcome (str): Name of the column representing the outcome variable.
        fill_method (str): Method to fill NaNs. Options:
                           'interpolate' (linear interpolation),
                           'ffill' (forward fill then backward fill),
                           'bfill' (backward fill then forward fill),
                           'mean' (mean of non-missing values for each unit),
                           'median' (median of non-missing values for each unit).

    Returns:
        np.ndarray: A N x T array of imputed outcomes, suitable for KMeans.
    """

    # Pivot to wide format. This will automatically create NaNs for missing (unit, time) combinations.
    data_wide = data_long.pivot_table(
        index=unit_col, columns=time_col, values=outcome_col
    )

    # Impute missing values
    if fill_method == "interpolate":
        imputed_data_wide = data_wide.interpolate(
            method="linear", limit_direction="both", axis=1
        )
    elif fill_method == "ffill":
        imputed_data_wide = data_wide.ffill(axis=1).bfill(axis=1)
    elif fill_method == "bfill":
        imputed_data_wide = data_wide.bfill(axis=1).ffill(axis=1)
    elif fill_method == "mean":
        imputed_data_wide = data_wide.apply(lambda row: row.fillna(row.mean()), axis=1)
    elif fill_method == "median":
        imputed_data_wide = data_wide.apply(
            lambda row: row.fillna(row.median()), axis=1
        )
    else:
        raise ValueError(f"Unsupported fill_method: {fill_method}")

    # Return as NumPy array, as KMeans expects this
    return imputed_data_wide


# CREATE FULL-GRID FOR KMEANS
def create_fullgrid(data, unit_col, time_col):
    # Create a complete grid of all possible (id, t) combinations
    unique_ids = data[unit_col].unique()
    unique_ts = np.sort(data[time_col].unique())
    full_grid = pd.MultiIndex.from_product(
        [unique_ids, unique_ts], names=[unit_col, time_col]
    ).to_frame(index=False)
    full_grid = pd.merge(full_grid, data, how="left", on=[unit_col, time_col])

    return full_grid


def design_matrix(data, degree, time_col, static_cov, tv_cov):
    df = pd.DataFrame(index=data.index)

    # create design matrix
    for d in range(degree + 1):
        df[f"{time_col}_{d}"] = data[time_col].values ** d

    df[static_cov + tv_cov] = data[static_cov + tv_cov].values

    return df


# create dropout
def create_dropout(full_grid, unit_col, time_col, outcome_col):
    # find maximum time observed for each unit for specific outcome
    dropout = (
        full_grid.dropna(subset=[outcome_col])
        .groupby(unit_col)[time_col]
        .max()
        .reset_index()
    )

    # create dropout time (t + 1)
    dropout["dropout_time"] = dropout[time_col] + 1

    full_grid = pd.merge(
        full_grid, dropout.drop(columns=time_col), how="left", on=unit_col
    )
    full_grid[f"dropout_{outcome_col}"] = np.where(
        full_grid[time_col] < full_grid["dropout_time"], 0, 1
    )
    full_grid[f"dropout_{outcome_col}"] = np.where(
        full_grid[time_col] > full_grid["dropout_time"],
        np.nan,
        full_grid[f"dropout_{outcome_col}"],
    )
    full_grid = full_grid.drop(columns="dropout_time")

    return full_grid


def kmeans_data_prep(full_grid, unit_col, time_col, trajectory_models, dropout_models):
    full_model_list = list(trajectory_models.keys()) + [
        f"dropout_{outcome}" for outcome in dropout_models.keys()
    ]

    for i, var in enumerate(full_model_list):
        df = impute_for_kmeans(full_grid, unit_col, time_col, var)
        df.columns = [f"{var}_{col}" for col in df.columns]
        if i == 0:
            combined_df = df
        else:
            combined_df = combined_df.join(df)

    wide_data_index = combined_df.index

    ss = StandardScaler()
    scaled_df = ss.fit_transform(combined_df)

    return wide_data_index, scaled_df


def kmeans_clustering(data, index, K, seed):
    kmeans = KMeans(n_clusters=K, random_state=seed, n_init="auto")
    cluster_assignments = kmeans.fit_predict(data)
    unit_to_cluster_map = pd.Series(cluster_assignments, index=index)

    return unit_to_cluster_map
