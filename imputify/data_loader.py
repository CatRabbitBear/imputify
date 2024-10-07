from typing import Optional
import pandas as pd


def handle_missing_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    # Create a copy to ensure immutability
    df_copy = df.copy()

    missing_values = ['N/A', 'NA', 'na', 'None', 'none', '', '?', '--']
    df_copy = df_copy.replace(missing_values, pd.NA)

    return df_copy


def infer_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()

    for column in df_copy.columns:
        if df_copy[column].dtype == 'object':
            try:
                df_copy[column] = pd.to_datetime(df_copy[column], errors='raise')
            except (ValueError, TypeError):
                pass

    return df_copy


def convert_numeric_with_threshold(df: pd.DataFrame, column: str, threshold: float = 0.9) -> pd.DataFrame:
    df_copy = df.copy()

    coerced_col = pd.to_numeric(df_copy[column], errors='coerce')
    valid_values_ratio = coerced_col.notna().mean()

    if valid_values_ratio >= threshold:
        df_copy[column] = coerced_col
    else:
        df_copy[column] = df_copy[column].astype('object')

    return df_copy


def downcast_numeric_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Downcast numeric column to save memory if possible.
    - If the column contains integer-like float values, cast to int.
    - If the column contains float values, downcast to float32.
    - NaN values will be preserved and kept as floats.
    """
    # Make a copy of the column for immutability
    col: pd.Series = df[column].copy()

    # Check if the column is float and has integer-like values (e.g., years)
    if pd.api.types.is_float_dtype(col):
        # Check if the float values are integer-like (ignoring NaN)
        int_like_col = col.dropna().apply(lambda x: x.is_integer())
        print(int_like_col.mean())

        # If most of the values are integers, cast to int64 but keep NaNs as float
        if int_like_col.mean() > 0.95:  # If over 95% of the non-NaN values are integer-like
            # Convert only non-NaN values to integers, leave NaNs as float
            col = col.apply(lambda x: int(x) if pd.notna(x) and x.is_integer() else x)
            df[column] = col  # Assign back to the DataFrame
        else:
            # Otherwise, downcast to float32 for memory savings
            df[column] = pd.to_numeric(col, downcast=None)

    return df


def infer_categorical_columns(df: pd.DataFrame, threshold: float = 0.05, exclude_columns: Optional[list[str]] = None,
                              user_defined: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Convert Object (string) columns to categorical if they meet the criteria for repeated values,
    unless they are in the exclusion list or unless specified by the user as categorical.

    Parameters:
    - threshold: proportion of unique values below which a column will be converted to categorical.
    - exclude_columns: list of columns that should not be converted to categorical.
    - user_defined: list of columns that the user explicitly wants to convert to categorical.
    """
    if exclude_columns is None:
        exclude_columns = []

    if user_defined is None:
        user_defined = []

    df_copy = df.copy()

    # Convert only Object (string) columns or user-specified columns
    for column in df_copy.columns:
        if column in exclude_columns:
            continue  # Skip columns in the exclusion list

        # Check for Object columns first or user-defined categorical columns
        if df_copy[column].dtype == 'object' or column in user_defined:
            unique_ratio = df_copy[column].nunique() / len(df_copy[column])
            if unique_ratio < threshold or column in user_defined:
                df_copy[column] = df_copy[column].astype('category')

    return df_copy


def infer_column_types(df: pd.DataFrame, numeric_threshold: float = 0.9) -> pd.DataFrame:
    """
    Full type inference workflow with numeric coercion threshold and downcasting.
    """
    # Step 1: Handle missing value placeholders
    df = handle_missing_placeholders(df)

    # Step 2: Convert datetime columns
    df = infer_datetime_columns(df)

    # Step 3: Convert numeric columns with a threshold
    for column in df.columns:
        if df[column].dtype == 'object':
            df = convert_numeric_with_threshold(df, column, threshold=numeric_threshold)

    # Step 4: Convert categorical columns (if needed)
    df = infer_categorical_columns(df)

    return df


def impute_from_csv(filepath, return_type=pd.DataFrame, **kwargs):
    """
    Load a CSV file and process it using Imputify's pipeline.

    Parameters:
    - filepath: str, path to the CSV file.
    - return_type: Type to return (default is pd.DataFrame)
    - **kwargs: additional arguments passed to pd.read_csv.
    """
    # This will only pass the kwargs not handled explicitly (e.g., return_type will not be passed)
    df = pd.read_csv(filepath, **kwargs)
    df = infer_column_types(df)
    # Process the data and return as the specified type
    #TODO: implement ENUM for valid return types
    if return_type == pd.DataFrame:
        return df
    elif return_type == 'dict':
        return df.to_dict()

