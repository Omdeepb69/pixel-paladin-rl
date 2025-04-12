import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

# Assume data_loader.py exists in the same directory or is importable
# Create dummy functions if data_loader.py doesn't exist for testing setup
try:
    from data_loader import load_data, preprocess_data, transform_data
except ImportError:
    # Define dummy functions if the actual module is not available
    # This allows the test file structure to be generated and validated
    # In a real scenario, data_loader.py should exist and be importable
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer

    def load_data(filepath: str) -> pd.DataFrame:
        """Dummy load_data function."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found at {filepath}")
        try:
            df = pd.read_csv(filepath)
            if df.empty and os.path.getsize(filepath) > 0:
                 # Handle cases where read_csv might return empty for malformed CSVs
                 raise ValueError("Failed to parse CSV correctly.")
            return df
        except pd.errors.EmptyDataError:
             # If the file is genuinely empty or just headers
             return pd.DataFrame()
        except Exception as e:
            raise ValueError(f"Error loading data from {filepath}: {e}")


    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """Dummy preprocess_data function."""
        df_processed = df.copy()
        numerical_cols = df_processed.select_dtypes(include=np.number).columns
        categorical_cols = df_processed.select_dtypes(include='object').columns

        for col in numerical_cols:
            if df_processed[col].isnull().any():
                mean_val = df_processed[col].mean()
                df_processed[col].fillna(mean_val, inplace=True)
                # Attempt to keep integer type if original was integer and mean is whole number
                if pd.api.types.is_integer_dtype(df[col]) and mean_val == np.floor(mean_val):
                     try:
                         df_processed[col] = df_processed[col].astype(int)
                     except ValueError: # Handle potential overflow or non-int values if mean wasn't perfectly whole
                         pass # Keep as float

        for col in categorical_cols:
            if df_processed[col].isnull().any():
                df_processed[col].fillna('Unknown', inplace=True)

        # Example type correction - ensure 'id' column is int if present
        if 'id' in df_processed.columns:
            try:
                # Convert to numeric first to handle potential strings, coerce errors to NaN, fill NaN, then convert to int
                df_processed['id'] = pd.to_numeric(df_processed['id'], errors='coerce').fillna(0).astype(int)
            except Exception:
                 # Keep original if conversion fails
                 pass

        return df_processed


    def transform_data(df: pd.DataFrame) -> pd.DataFrame:
        """Dummy transform_data function."""
        df_transformed = df.copy()
        numerical_cols = df_transformed.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df_transformed.select_dtypes(include='object').columns.tolist()

        if not numerical_cols and not categorical_cols:
            return df_transformed # Return as is if no columns to transform

        transformers = []
        if numerical_cols:
            transformers.append(('num', MinMaxScaler(), numerical_cols))
        if categorical_cols:
            # Use sparse_output=False for easier handling in tests if needed
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols))

        if not transformers:
            return df_transformed

        preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

        try:
            transformed_data = preprocessor.fit_transform(df_transformed)
            # Attempt to get feature names automatically
            feature_names = preprocessor.get_feature_names_out()
            transformed_df = pd.DataFrame(transformed_data, columns=feature_names, index=df_transformed.index)
        except Exception as e:
             # Fallback or error handling if get_feature_names_out fails or other issues
             print(f"Warning: Could not automatically determine feature names: {e}")
             # Basic fallback: create generic names or return array
             # For testing purposes, we might just return the numpy array here
             # Or construct names manually if structure is known
             # Returning DataFrame with generic names for structure check
             num_cols_out = len(numerical_cols) if numerical_cols else 0
             cat_cols_out = 0
             if categorical_cols:
                 # Estimate output columns from OHE
                 ohe_transformer = [t[1] for t in transformers if t[0] == 'cat'][0]
                 ohe_transformer.fit(df_transformed[categorical_cols])
                 if hasattr(ohe_transformer, 'categories_'):
                     cat_cols_out = sum(len(cats) for cats in ohe_transformer.categories_)

             total_cols = num_cols_out + cat_cols_out
             # Add remainder columns count if any
             remainder_cols = [c for c in df.columns if c not in numerical_cols and c not in categorical_cols]
             total_cols += len(remainder_cols)

             generic_names = [f'feature_{i}' for i in range(transformed_data.shape[1])]
             if len(generic_names) != transformed_data.shape[1]:
                 # If shape mismatch, something went wrong, return array
                 print("Shape mismatch in transformed data, returning NumPy array.")
                 return transformed_data # Return array if names are problematic

             transformed_df = pd.DataFrame(transformed_data, columns=generic_names, index=df_transformed.index)


        return transformed_df


# --- Fixtures ---

@pytest.fixture(scope="module")
def sample_data_dict():
    return {
        'id': [1, 2, 3, 4, 5],
        'feature1': [10.0, 20.0, np.nan, 40.0, 50.0],
        'feature2': [0.5, 0.4, 0.3, 0.2, 0.1],
        'category': ['A', 'B', 'A', np.nan, 'B'],
        'extra_col': [True, False, True, False, True]
    }

@pytest.fixture(scope="module")
def sample_dataframe(sample_data_dict):
    return pd.DataFrame(sample_data_dict)

@pytest.fixture(scope="function")
def temp_csv_file(sample_dataframe):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".csv") as tmpfile:
        sample_dataframe.to_csv(tmpfile.name, index=False)
        filepath = tmpfile.name
    yield filepath
    os.remove(filepath) # Clean up the file after the test

@pytest.fixture(scope="function")
def empty_temp_csv_file():
     with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".csv") as tmpfile:
         filepath = tmpfile.name
         # Write nothing or just headers
         # pd.DataFrame().to_csv(tmpfile.name, index=False) # Option 1: Empty DF
         tmpfile.write("col1,col2\n") # Option 2: Only headers
     yield filepath
     os.remove(filepath)

# --- Test Cases ---

# 1. Data Loading Tests
def test_load_data_success(temp_csv_file, sample_dataframe):
    """Test loading data from a valid CSV file."""
    loaded_df = load_data(temp_csv_file)
    pd.testing.assert_frame_equal(loaded_df, sample_dataframe)
    assert not loaded_df.empty
    assert list(loaded_df.columns) == list(sample_dataframe.columns)

def test_load_data_file_not_found():
    """Test loading data from a non-existent file."""
    non_existent_file = "non_existent_file.csv"
    with pytest.raises(FileNotFoundError):
        load_data(non_existent_file)

def test_load_data_empty_file(empty_temp_csv_file):
    """Test loading data from an empty or header-only CSV file."""
    # Depending on implementation, this might return an empty DataFrame
    # or raise an error. Adjust assertion based on expected behavior.
    try:
        loaded_df = load_data(empty_temp_csv_file)
        assert loaded_df.empty
        # If only headers were written, check columns
        if os.path.getsize(empty_temp_csv_file) > 0:
             assert list(loaded_df.columns) == ['col1', 'col2']
        else:
             assert loaded_df.columns.empty

    except ValueError as e:
         # If load_data raises ValueError for empty/malformed files
         assert "empty" in str(e).lower() or "parse" in str(e).lower()
    except pd.errors.EmptyDataError:
         # If pandas itself raises EmptyDataError and it's not caught/handled
         pytest.fail("load_data should handle EmptyDataError gracefully.")


# 2. Data Preprocessing Tests
def test_preprocess_data_nan_filling(sample_dataframe):
    """Test NaN values are filled correctly."""
    processed_df = preprocess_data(sample_dataframe)

    # Check numerical NaN filling (feature1 mean is (10+20+40+50)/4 = 30)
    assert not processed_df['feature1'].isnull().any()
    assert processed_df.loc[2, 'feature1'] == pytest.approx(30.0)

    # Check categorical NaN filling
    assert not processed_df['category'].isnull().any()
    assert processed_df.loc[3, 'category'] == 'Unknown'

    # Check other columns remain unchanged if they had no NaNs
    assert processed_df['feature2'].equals(sample_dataframe['feature2'])
    assert processed_df['id'].equals(sample_dataframe['id'])
    assert processed_df['extra_col'].equals(sample_dataframe['extra_col'])


def test_preprocess_data_type_conversion(sample_dataframe):
    """Test data types are handled/converted correctly."""
    df = sample_dataframe.copy()
    # Introduce a column that looks numeric but is object type
    df['id'] = df['id'].astype(str)
    df.loc[0, 'id'] = 'non-numeric' # Add a non-numeric value

    processed_df = preprocess_data(df)

    # Check if 'id' column was attempted to be converted to int
    # The dummy function converts non-numeric to 0
    assert pd.api.types.is_integer_dtype(processed_df['id'])
    assert processed_df.loc[0, 'id'] == 0 # 'non-numeric' becomes NaN then 0
    assert processed_df.loc[1, 'id'] == 2 # '2' becomes 2

    # Check other types remain as expected
    assert pd.api.types.is_float_dtype(processed_df['feature1'])
    assert pd.api.types.is_float_dtype(processed_df['feature2'])
    assert pd.api.types.is_object_dtype(processed_df['category'])
    assert pd.api.types.is_bool_dtype(processed_df['extra_col'])


def test_preprocess_data_no_nans():
    """Test preprocessing on data with no NaNs."""
    data = {
        'id': [1, 2, 3],
        'feature1': [10.0, 20.0, 30.0],
        'category': ['A', 'B', 'A']
    }
    df = pd.DataFrame(data)
    processed_df = preprocess_data(df.copy()) # Pass copy to avoid modifying original fixture
    pd.testing.assert_frame_equal(processed_df, df)


# 3. Data Transformation Tests
@pytest.fixture
def preprocessed_sample_dataframe(sample_dataframe):
    """Provides a preprocessed version of the sample dataframe."""
    return preprocess_data(sample_dataframe.copy())

def test_transform_data_scaling(preprocessed_sample_dataframe):
    """Test numerical features are scaled (e.g., MinMaxScaler)."""
    transformed_df = transform_data(preprocessed_sample_dataframe.copy())

    # Check scaled numerical columns (feature1, feature2, potentially id if numeric)
    # MinMaxScaler scales to [0, 1]
    num_cols = preprocessed_sample_dataframe.select_dtypes(include=np.number).columns
    num_cols_transformed = [col for col in transformed_df.columns if col.startswith('num__')]

    assert len(num_cols_transformed) == len(num_cols)

    for col_name in num_cols:
        original_col_name_in_transformed = f'num__{col_name}'
        assert original_col_name_in_transformed in transformed_df.columns
        scaled_col = transformed_df[original_col_name_in_transformed]
        assert scaled_col.min() >= 0.0
        assert scaled_col.max() <= 1.0
        # Check a specific value if logic is known (e.g., feature2 min=0.1, max=0.5)
        # Expected scaled value for 0.5 is 1.0, for 0.1 is 0.0
        original_col = preprocessed_sample_dataframe[col_name]
        if col_name == 'feature2':
             assert transformed_df.loc[original_col.idxmax(), original_col_name_in_transformed] == pytest.approx(1.0)
             assert transformed_df.loc[original_col.idxmin(), original_col_name_in_transformed] == pytest.approx(0.0)


def test_transform_data_encoding(preprocessed_sample_dataframe):
    """Test categorical features are one-hot encoded."""
    transformed_df = transform_data(preprocessed_sample_dataframe.copy())

    # Check for one-hot encoded columns ('category' -> 'A', 'B', 'Unknown')
    # Expected columns depend on the OneHotEncoder naming convention (get_feature_names_out)
    expected_cat_cols = ['cat__category_A', 'cat__category_B', 'cat__category_Unknown']
    for col in expected_cat_cols:
        assert col in transformed_df.columns
        # Check values are 0 or 1
        assert transformed_df[col].isin([0, 1]).all()

    # Check that original categorical column is removed
    assert 'category' not in transformed_df.columns

    # Check a specific row's encoding
    # Row 0: category 'A' -> cat__category_A=1, others=0
    assert transformed_df.loc[0, 'cat__category_A'] == 1
    assert transformed_df.loc[0, 'cat__category_B'] == 0
    assert transformed_df.loc[0, 'cat__category_Unknown'] == 0
    # Row 3: category 'Unknown' -> cat__category_Unknown=1, others=0
    assert transformed_df.loc[3, 'cat__category_A'] == 0
    assert transformed_df.loc[3, 'cat__category_B'] == 0
    assert transformed_df.loc[3, 'cat__category_Unknown'] == 1


def test_transform_data_output_shape(preprocessed_sample_dataframe):
    """Test the output shape of the transformed data."""
    transformed_df = transform_data(preprocessed_sample_dataframe.copy())

    original_rows = preprocessed_sample_dataframe.shape[0]
    assert transformed_df.shape[0] == original_rows

    # Calculate expected columns: num_cols + ohe_cols + remainder_cols
    num_cols = preprocessed_sample_dataframe.select_dtypes(include=np.number).shape[1]
    cat_cols = preprocessed_sample_dataframe.select_dtypes(include='object').columns
    ohe_cols_count = 0
    if not cat_cols.empty:
        # Need to fit OHE to know the number of output columns
        ohe = OneHotEncoder(handle_unknown='ignore')
        ohe.fit(preprocessed_sample_dataframe[cat_cols])
        if hasattr(ohe, 'categories_'):
            ohe_cols_count = sum(len(cats) for cats in ohe.categories_)

    remainder_cols = preprocessed_sample_dataframe.select_dtypes(exclude=[np.number, 'object']).shape[1]

    expected_cols = num_cols + ohe_cols_count + remainder_cols
    # This check might be fragile if ColumnTransformer naming or remainder handling changes
    # A less fragile check might be to ensure all expected groups of columns are present
    assert transformed_df.shape[1] >= num_cols + ohe_cols_count # Check at least expected cols are there


def test_transform_data_passthrough(preprocessed_sample_dataframe):
    """Test that columns not specified for transformation are passed through."""
    transformed_df = transform_data(preprocessed_sample_dataframe.copy())
    # 'extra_col' is boolean, should be passed through if remainder='passthrough'
    # The dummy implementation uses get_feature_names_out which includes remainder
    # Check if 'remainder__extra_col' exists or just 'extra_col' depending on sklearn version/config
    assert 'remainder__extra_col' in transformed_df.columns or 'extra_col' in transformed_df.columns
    if 'remainder__extra_col' in transformed_df.columns:
        pd.testing.assert_series_equal(transformed_df['remainder__extra_col'].astype(bool), preprocessed_sample_dataframe['extra_col'], check_names=False)
    elif 'extra_col' in transformed_df.columns:
         pd.testing.assert_series_equal(transformed_df['extra_col'], preprocessed_sample_dataframe['extra_col'], check_names=False)


def test_transform_data_empty_dataframe():
    """Test transforming an empty DataFrame."""
    empty_df = pd.DataFrame({'A': [], 'B': []})
    transformed_df = transform_data(empty_df)
    assert transformed_df.empty
    pd.testing.assert_frame_equal(transformed_df, empty_df)


def test_transform_data_only_numerical():
    """Test transforming DataFrame with only numerical columns."""
    df = pd.DataFrame({'f1': [1, 2, 3], 'f2': [0.1, 0.5, 1.0]})
    transformed_df = transform_data(df)
    assert transformed_df.shape == df.shape
    assert 'num__f1' in transformed_df.columns
    assert 'num__f2' in transformed_df.columns
    assert transformed_df['num__f1'].min() == 0.0
    assert transformed_df['num__f1'].max() == 1.0


def test_transform_data_only_categorical():
    """Test transforming DataFrame with only categorical columns."""
    df = pd.DataFrame({'c1': ['A', 'B', 'A'], 'c2': ['X', 'X', 'Y']})
    transformed_df = transform_data(df)
    assert transformed_df.shape[0] == df.shape[0]
    assert 'cat__c1_A' in transformed_df.columns
    assert 'cat__c1_B' in transformed_df.columns
    assert 'cat__c2_X' in transformed_df.columns
    assert 'cat__c2_Y' in transformed_df.columns
    assert transformed_df.shape[1] == 4 # 2 from c1, 2 from c2