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
    Impute NaNs values with meaningful ones, and encode data on the way. 
    This requires NOT ENCODED DATA (Country Code must still be a string). 
    Encoding will be done inside.
    
    This function automatically detects numeric columns that need imputation.
    """
    def impute_values(df, cols_to_fill):
        df_first_imputation = df.copy()

        # Convert Year to numeric if it exists
        if "Year" in df_first_imputation.columns:
            df_first_imputation["Year"] = pd.to_numeric(df_first_imputation["Year"], errors="coerce")
        
        # Sort by Country Code and Year if both exist
        if "Country Code" in df_first_imputation.columns and "Year" in df_first_imputation.columns:
            df_first_imputation = df_first_imputation.sort_values(["Country Code", "Year"])
            
            # Forward/backward fill within each country for numeric columns that exist
            if cols_to_fill:
                df_first_imputation[cols_to_fill] = (
                    df_first_imputation.groupby("Country Code")[cols_to_fill]
                            .transform(lambda g: g.ffill().bfill())
                )

        return df_first_imputation

    def encode(data: pd.DataFrame):
        """
        Ensures all columns are numeric
        """
        # One-hot encode 'Country Code' if it exists
        if "Country Code" in data.columns:
            result = pd.get_dummies(data, columns=["Country Code"], drop_first=True)
        else:
            result = data.copy()
        
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
    # Optimisation for bigger datasets.
    imp_mean = IterativeImputer(random_state=0, max_iter=5, n_nearest_features=15)

    # Fit / transform
    df_imputed = imp_mean.fit_transform(df_encoded)

    # Rebuild the DataFrame after imputation
    df_imputed = pd.DataFrame(df_imputed, columns=df_encoded.columns, index=df_encoded.index)
    return df_imputed


def display_columns(df: pd.DataFrame):
    """
    Affiche la liste de toutes les colonnes présentes dans le DataFrame.
    """
    print(f"--- Liste des {len(df.columns)} colonnes ---")
    for col in df.columns:
        print(f"• {col}")
    print("---------------------------------")


def create_per_capita_features(df: pd.DataFrame, columns_to_scale: list, population_col: str = 'Population, total'):
    """
    Crée de nouvelles colonnes normalisées par habitant.
    
    Args:
        df: Le DataFrame source.
        columns_to_scale: Liste des colonnes (numérateurs) à diviser par la population.
        population_col: Le nom de la colonne population (dénominateur).
    
    Returns:
        DataFrame avec les nouvelles colonnes ajoutées.
    """
    df_result = df.copy()
    
    # Vérification que la colonne population existe
    if population_col not in df_result.columns:
        raise ValueError(f"La colonne de population '{population_col}' n'existe pas dans le DataFrame.")

    count = 0
    for col in columns_to_scale:
        if col in df_result.columns:
            # Nettoyage du nom pour la nouvelle colonne (ex: "Sulphur (tonnes)" -> "Sulphur_per_capita")
            clean_name = col.split('(')[0].strip().replace(' ', '_')
            new_col_name = f"{clean_name}_per_capita"
            
            # Calcul du ratio
            df_result[new_col_name] = df_result[col] / df_result[population_col]
            count += 1
        else:
            print(f"⚠️ Attention : La colonne '{col}' n'existe pas et a été ignorée.")
            
    print(f"✅ {count} colonnes 'per_capita' créées.")
    return df_result


def remove_columns(df: pd.DataFrame, columns_to_drop: list):
    """
    Supprime une liste de colonnes du DataFrame si elles existent.
    
    Args:
        df: Le DataFrame source.
        columns_to_drop: Liste des noms de colonnes à supprimer.
    
    Returns:
        DataFrame nettoyé.
    """
    df_result = df.copy()
    
    # On ne garde que les colonnes qui existent vraiment dans le DF pour éviter les erreurs
    cols_present = [c for c in columns_to_drop if c in df_result.columns]
    
    if cols_present:
        df_result = df_result.drop(columns=cols_present)
        print(f"🗑️ {len(cols_present)} colonnes supprimées : {cols_present}")
    else:
        print("ℹ️ Aucune colonne à supprimer n'a été trouvée.")
        
    return df_result


def round_features(df: pd.DataFrame, columns_to_round: list, decimals: int = 2):
    """
    Arrondit les valeurs des colonnes spécifiées à un nombre de décimales donné.
    
    Args:
        df: Le DataFrame source.
        columns_to_round: Liste des colonnes à arrondir.
        decimals: Nombre de décimales (entier).
    
    Returns:
        DataFrame avec valeurs arrondies.
    """
    df_result = df.copy()
    
    count = 0
    for col in columns_to_round:
        if col in df_result.columns:
            df_result[col] = df_result[col].round(decimals)
            count += 1
    
    print(f"✅ Arrondi effectué sur {count} colonnes à {decimals} décimales.")
    return df_result


def process_wind_speed(df: pd.DataFrame):
    """
    Calcule la magnitude de la vitesse du vent à partir des vecteurs u10 et v10,
    puis supprime les colonnes originales u10 et v10.
    
    Formule : Wind_Speed = sqrt(u10² + v10²)
    
    Returns:
        DataFrame avec la nouvelle colonne 'Wind_Speed' et sans 'u10'/'v10'.
    """
    df_result = df.copy()
    
    if 'u10' in df_result.columns and 'v10' in df_result.columns:
        # Calcul de la magnitude (Vitesse vectorielle)
        df_result['Wind_Speed'] = np.sqrt(df_result['u10']**2 + df_result['v10']**2)
        
        # Suppression des composantes vectorielles (sources de biais géographique)
        #df_result = df_result.drop(columns=['u10', 'v10'])
        print("✅ 'Wind_Speed' calculée. Colonnes 'u10' et 'v10' non supprimées.")
    else:
        print("⚠️ Impossible de calculer la vitesse du vent : 'u10' ou 'v10' manquants.")
        
    return df_result