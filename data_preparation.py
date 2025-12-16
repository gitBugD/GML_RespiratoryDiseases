import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def preprocessing(data: pd.DataFrame):
    """
    Removes non numeric values and correctly sets nan values 
    """
    result = data.copy()
    
    # Replace '..' and 'nan' strings with real NaN
    result = result.replace(['..', 'nan'], np.nan)

    return result


def normalize(data: pd.DataFrame):
    """
    Normalizes columns
    """

    numeric_columns = data.select_dtypes(include=np.number).columns
    if len(numeric_columns) != len(data.columns) :
        raise Exception("data contains non-numeric values")

    result = data.copy()

    #TODO

    return result


def trim_outliers(data: pd.DataFrame):
    """
    Removes outliers in the Value column based on F-score (for now)

    returns the data without outliers AND the outliers
    """
    result = data.copy()

    #TODO

    return result, outliers_result


def impute_nans(data: pd.DataFrame):
    """
    Impute NaNs values with meaningful ones, and encode data on the way. This requires NOT ENCODED DATA (Country Code must still be a string). Encoding will be done inside.
    """
    def impute_values(df, cols_to_fill):
        df_first_imputation = df.copy()

        df_first_imputation["Year"] = pd.to_numeric(df_first_imputation["Year"], errors="coerce")
        df_first_imputation = df_first_imputation.sort_values(["Country Code", "Year"])

        # Forward/backward fill within each country
        df_first_imputation[cols_to_fill] = (
            df_first_imputation.groupby("Country Code")[cols_to_fill]
                    .transform(lambda g: g.ffill().bfill())
        )

        return df_first_imputation

    def encode(data: pd.DataFrame):
        """
        Ensures all columns are numeric
        """
        # One-hot encode 'Country Code'
        result = pd.get_dummies(data, columns=["Country Code"], drop_first=True)
        
        # Convert all columns to numeric (coerce errors to NaN)
        result = result.apply(pd.to_numeric, errors='coerce')
    
        return result
        
    # Automatically detect numeric columns that need filling
    # Exclude metadata columns like Country Code, Country Name, Year
    exclude_cols = {"Country Code", "Country Name", "Year", "Disease", "Measure", "Metric"}
    
    # Get all columns that are not in the exclusion list
    cols_to_fill = [col for col in data.columns 
                   if col not in exclude_cols]
    
    # Filter to only include columns that exist and can be converted to numeric
    df_test = data[cols_to_fill].apply(pd.to_numeric, errors='coerce')
    cols_to_fill = [col for col in cols_to_fill if df_test[col].notna().any()]
    
    print(f"📊 Colonnes détectées pour l'imputation : {len(cols_to_fill)} colonnes")
    
    df_first_imputation = impute_values(data, cols_to_fill)

    df_encoded = encode(df_first_imputation)
    
    # Initialize imputer
    imp_mean = IterativeImputer(random_state=0, max_iter=10, n_nearest_features=15)

    # Fit / transform
    df_imputed = imp_mean.fit_transform(df_encoded)

    # Rebuild the DataFrame after imputation
    df_imputed = pd.DataFrame(df_imputed, columns=df_encoded.columns, index=df_encoded.index)
    return df_imputed

def impute_nans_no_encoding(data: pd.DataFrame) -> pd.DataFrame:
    """
    Impute NaN values for numeric columns only.
    Categorical columns are left untouched and returned as-is.
    """

    df = data.copy()

    # -------------------------
    # Identify columns
    # -------------------------
    exclude_cols = {"Year", "Disease", "Measure", "Metric"}

    categorical_cols = [
        c for c in df.columns
        if df[c].dtype == "object" and c not in exclude_cols
    ]

    numeric_cols = [
        c for c in df.columns
        if c not in categorical_cols and c not in exclude_cols
    ]

    # Ensure numeric dtype
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # -------------------------
    # Sort & group-wise ffill/bfill
    # -------------------------
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    df = df.sort_values(["Country Code", "Year"])

    df[numeric_cols] = (
        df.groupby("Country Code")[numeric_cols]
          .transform(lambda g: g.ffill().bfill())
    )

    # -------------------------
    # Iterative imputation (numeric only)
    # -------------------------
    imp = IterativeImputer(
        random_state=0,
        max_iter=5,
        n_nearest_features=15
    )

    df[numeric_cols] = imp.fit_transform(df[numeric_cols])
    assert not df[numeric_cols].isnull().any().any()
    if df[numeric_cols].isnull().any().any():
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # -------------------------
    # Return original structure
    # -------------------------
    df_final = df[data.columns]

    return df_final