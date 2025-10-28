import numpy as np
import pandas as pd

def preprocessing(data: pd.DataFrame):
    """
    Removes non numeric values, correctly sets nan values and ensures 
    all columns are numeric
    """
    result = data.copy()

    result = result.select_dtypes(include=np.number)
    
    # Replace '..' and 'nan' strings with real NaN
    result = result.replace(['..', 'nan'], np.nan)
    
    # One-hot encode 'Country Code'
    result = pd.get_dummies(result, columns=["Country Code"], drop_first=True)
    
    # Convert all columns to numeric (coerce errors to NaN)
    result = result.apply(pd.to_numeric, errors='coerce')

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

def inpute_nans(data: pd.DataFrame):
    """
    Inpute NaNs values with meaningful ones.
    """
    result = data.copy()

    #TODO

    return result