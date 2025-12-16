import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocessing(data: pd.DataFrame):
    """
    Removes non numeric values and correctly sets nan values 
    """
    result = data.copy()

    # Replace '..' and 'nan' strings with real NaN
    result = result.replace(['..', 'nan'], np.nan)

    # Try to convert each object column to numeric
    for col in df.select_dtypes(include=['object']).columns:
        # First try to convert to numeric
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # If conversion failed for all values, keep as object
        if df[col].isna().all():
            # Revert to original values (excluding "..")
            df[col] = df[col].fillna('').astype(str).replace('nan', np.nan)
    

    result = result.select_dtypes(include=np.number)
    

    return result


def normalize(data: pd.DataFrame):
    """
    Normalizes columns, values are scaled so that each columns min = 0 and max = 1
    """

    numeric_columns = data.select_dtypes(include=np.number).columns
    if len(numeric_columns) != len(data.columns) :
        raise Exception("data contains non-numeric values")

    result = data.copy()

    result = MinMaxScaler().fit_transform(result)
    
    return result

def standardize(data: pd.DataFrame):
    """
    Standardize columns, values are changed so that each columns mean = 0 and std = 1
    """

    numeric_columns = data.select_dtypes(include=np.number).columns
    if len(numeric_columns) != len(data.columns) :
        raise Exception("data contains non-numeric values")
    
    result = data.copy()

    result = StandardScaler().fit_transform(result)
    
    return result


def trim_outliers(data: pd.DataFrame):
    """
    Removes outliers in the Value column based on IQR

    returns the data without outliers AND the outliers
    """
    result = data.copy()

    values = result['Value']
    
    # Calculate quartiles and IQR
    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Create mask for outliers
    outlier_mask = (values < lower_bound) | (values > upper_bound)
    
    # Split the data
    outliers_result = result[outlier_mask].copy()
    cleaned_result = result[~outlier_mask].copy()
    
    return cleaned_result, outliers_result


def impute_nans(data: pd.DataFrame):
    """
    Impute NaNs values with meaningful ones, and encode data on the way. This requires NOT ENCODED DATA (Country Code must still be a string). Encoding will be done inside.
    """
    def impute_values(df, cols_to_fill):
        df_first_imputation = df.copy()

        df_first_imputation = df_first_imputation.sort_values(["Country Code", "Year"])

        print("eeee")
        
        # Forward/backward fill within each country
        df_first_imputation[cols_to_fill] = (
            df_first_imputation.groupby("Country Code")[cols_to_fill]
                    .transform(lambda g: g.ffill().bfill())
        )

        print("eeee")

        df_first_imputation.fillna(df_first_imputation.median(numeric_only=True), inplace=True)

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
        
    cols_to_fill = [c for c in data.columns if data[c].count() != len(data.index)]
    print(cols_to_fill)
    df_first_imputation = impute_values(data, cols_to_fill)

    # Rebuild the DataFrame after imputation
    df_first_imputation = pd.DataFrame(df_first_imputation, columns=df_first_imputation.columns, index=df_first_imputation.index)
    return df_first_imputation