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
        
    cols_to_fill = [
        "Access to clean fuels and technologies for cooking (% of population)",
        "Carbon dioxide (CO2) emissions excluding LULUCF per capita (t CO2e/capita)",
        "Gini index",
        "Poverty headcount ratio at national poverty lines (% of population)",
        "Renewable electricity output (% of total electricity output)",
        "Unemployment, total (% of total labor force) (national estimate)",
        "Total area (Square Km)",
        "PM10_ConcentrationAvg",
        "PM25_ConcentrationAvg",
        "NO2_ConcentrationAvg",
        "Greenhouse gases (Kg CO2-equivalent Per Person)",
        "Sulphur oxides (tonnes)",
        "Total sales of agricultural pesticides (tonnes)",
        "Share of population who are daily smokers (Pct population)"
    ]
    df_first_imputation = impute_values(data, cols_to_fill)
    df_first_imputation.isnull().sum()

    df_encoded = encode(df_first_imputation)
    
    # Initialize imputer
    imp_mean = IterativeImputer(random_state=0)

    # Fit / transform
    df_imputed = imp_mean.fit_transform(df_encoded)

    # Rebuild the DataFrame after imputation
    df_imputed = pd.DataFrame(df_imputed, columns=df_encoded.columns, index=df_encoded.index)
    return df_imputed