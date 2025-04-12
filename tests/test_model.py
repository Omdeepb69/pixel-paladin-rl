```python
import pytest
import numpy as np

try:
    from model import Model
except ImportError:
    # Fallback dummy Model class if model.py is not found
    class Model:
        def __init__(self, param1=1, param2='default'):
            self.param1 = param1
            self.param2 = param2
            self._is_trained = False
            self._internal_state = None
            self.n_features_ = None

        def train(self, X, y):
            if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
                raise ValueError("X and y must be numpy arrays")
            if X.ndim != 2:
                 raise ValueError(f"X must be a 2D array, but got shape {X.shape}")
            if y.ndim != 1:
                 # Allowing y=(n,1) might be needed depending on model specifics
                 if y.ndim != 2 or y.shape[1] != 1:
                     raise ValueError(f"y must be a 1D array or 2D column vector, but got shape {y.shape}")
                 if y.ndim == 2:
                     y = y.ravel() # Convert (n,1) to (n,)

            if X.shape[0] != y.shape[0]:
                raise ValueError(f"X and y must have the same number of samples ({X.shape[0]} != {y.shape[0]})")

            self.n_features_ = X.shape[1]
            self._internal_state = np.random.rand(self.n_features_)
            self._is_trained = True

        def predict(self, X):
            if not self._is_trained:
                raise RuntimeError("Model must be trained before prediction.")
            if not isinstance(X, np.ndarray):
                raise ValueError("X must be a numpy array")

            original_ndim = X.ndim
            if original_ndim == 1:
                 if self.n_features_ != X.shape[0]:
                     raise ValueError(f"Input feature dimension {X.shape[0]} does not match trained model dimension {self.n_features_}")
                 X = X.reshape(1, -1)
            elif X.ndim == 2:
                if X.shape[1] != self.n_features_:
                    raise ValueError(f"Input feature dimension {X.shape[1]} does not match trained model dimension {self.n_features_}")
            else:
                raise ValueError(f"Input X must be 1D or 2D, but got {X.ndim} dimensions")

            predictions = np.dot(X, self._internal_state)

            # Return shape consistent with input: (n,) for 2D input, scalar for 1D input
            return predictions[0] if original_ndim == 1 else predictions


        def evaluate(self, X, y):
            if not self._is_trained:
                raise RuntimeError("Model must be trained before evaluation.")
            if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
                raise ValueError("X and y must be numpy arrays")

            if X.ndim != 2:
                 raise ValueError(f"X must be a 2D array, but got shape {X.shape}")
            if y.ndim != 1:
                 if y.ndim != 2 or y.shape[1] != 1:
                     raise ValueError(f"y must be a 1D array or 2D column vector, but got shape {y.shape}")
                 if y.ndim == 2:
                     y = y.ravel() # Convert (n,1) to (n,)

            if X.shape[0] != y.shape[0]:
                raise ValueError(f"X and y must have the same number of samples ({X.shape[0]} != {y.shape[0]})")
            if X.shape[1] != self.n_features_:
                 raise ValueError(f"Input feature dimension {X.shape[1]} does not match trained model dimension {self.n_features_}")


            predictions = self.predict(X) # predict handles feature check

            if predictions.shape != y.shape:
                 raise ValueError(f"Shape mismatch during evaluation: predictions {predictions.shape}, y {y.shape}")

            mse = np.mean((predictions - y) ** 2)
            return mse

        @property
        def is_trained(self):
            return self._is_trained


N_SAMPLES = 100
N_FEATURES = 5

@pytest.fixture(scope="module")
def sample_data():
    """Provides sample data for testing."""
    rng = np.random.RandomState(42)
    X = rng.rand(N_SAMPLES, N_FEATURES)
    # Generate y based on a simple linear relationship + noise
    true_coeffs = rng.rand(N_FEATURES)
    y = X @ true_coeffs + rng.normal(0, 0.1, N_SAMPLES)
    return X, y

@pytest.fixture
def untrained_model():
    """Provides an untrained instance of the Model."""
    return Model(param1=10, param2='test')

@pytest.fixture
def trained_model(untrained_model, sample_data):
    """Provides a trained instance of the Model."""
    X, y = sample_data
    untrained_model.train(X, y)
    return untrained_model


# 1. Test Model Initialization
def test_model_initialization_defaults():
    model = Model()
    assert model.param1 == 1
    assert model.param2 == 'default'
    assert not model.is_trained
    assert model._internal_state is None
    assert model.n_features_ is None

def test_model_initialization_custom_params(untrained_model):
    assert untrained_model.param1 == 10
    assert untrained_model.param2 == 'test'
    assert not untrained_model.is_trained
    assert untrained_model._internal_state is None
    assert untrained_model.n_features_ is None

# 2. Test Model Training
def test_model_train(untrained_model, sample_data):
    X, y = sample_data
    assert not untrained_model.is_trained
    untrained_model.train(X, y)
    assert untrained_model.is_trained
    assert untrained_model._internal_state is not None
    assert isinstance(untrained_model._internal_state, np.ndarray)
    assert untrained_model.n_features_ == N_FEATURES
    assert len(untrained_model._internal_state) == N_FEATURES

def test_model_train_accepts_y_column_vector(untrained_model, sample_data):
    X, y = sample_data
    y_col = y.reshape(-1, 1) # Convert y to (N_SAMPLES, 1)
    assert not untrained_model.is_trained
    untrained_model.train(X, y_col) # Should work
    assert untrained_model.is_trained
    assert untrained_model.n_features_ == N_FEATURES

def test_model_train_raises_error_on_invalid_input_type(untrained_model):
    X_list = [[1, 2], [3, 4]]
    y_array = np.array([0, 1])
    with pytest.raises(ValueError, match="X and y must be numpy arrays"):
        untrained_model.train(X_list, y_array)

    X_array = np.array([[1, 2], [3, 4]])
    y_list = [0, 1]
    with pytest.raises(ValueError, match="X and y must be numpy arrays"):
        untrained_model.train(X_array, y_list)

def test_model_train_raises_error_on_shape_mismatch(untrained_model):
    X = np.random.rand(10, N_FEATURES)
    y = np.random.rand(9) # Mismatched number of samples
    with pytest.raises(ValueError, match="X and y must have the same number of samples"):
        untrained_model.train(X, y)

def test_model_train_raises_error_on_invalid_dims(untrained_model):
    X_1d = np.random.rand(N_SAMPLES)
    y_1d = np.random.rand(N_SAMPLES)
    with pytest.raises(ValueError, match="X must be a 2D array"):
        untrained_model.train(X_1d, y_1d)

    X_2d = np.random.rand(N_SAMPLES, N_FEATURES)
    y_2d = np.random.rand(N_SAMPLES, 2) # y has more than 1 column
    with pytest.raises(ValueError, match="y must be a 1D array or 2D column vector"):
        untrained_model.train(X_2d, y_2d)

    X_3d = np.random.rand(N_SAMPLES, N_FEATURES, 1)
    with pytest.raises(ValueError, match="X must be a 2D array"):
        untrained_model.train(X_3d, y_1d)


# 3. Test Model Prediction
def test_model_predict_before_train(untrained_model, sample_data):
    X, _ = sample_data
    with pytest.raises(RuntimeError, match="Model must be trained before prediction."):
        untrained_model.predict(X[:10])

def test_model_predict_2d_input(trained_model, sample_data):
    X, _ = sample_data
    n_test = 10
    X_test = X[:n_test]
    predictions = trained_model.predict(X_test)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (n_test,)

def test_model_predict_1d_input(trained_model, sample_data):
    X, _ = sample_data
    X_single = X[0] # Get a single sample (1D array)
    prediction = trained_model.predict(X_single)
    assert isinstance(prediction, (float, np.number)) # Should return a scalar for 1D input
    assert np.isscalar(prediction)

def test_model_predict_raises_error_on_invalid_input_type(trained_model):
    X_list = [[1] * N_FEATURES]
    with pytest.raises(ValueError, match="X must be a numpy array"):
        trained_model.predict(X_list)

def test_model_predict_raises_error_on_feature_mismatch(trained_model, sample_data):
    X, _ = sample_data
    # Test with 2D input
    X_wrong_features_2d = np.random.rand(10, N_FEATURES + 1)
    with pytest.raises(ValueError, match="Input feature dimension"):
        trained_model.predict(X_wrong_features_2d)

    # Test with 1D input
    X_wrong_features_1d = np.random.rand(N_FEATURES + 1)
    with pytest.raises(ValueError, match="Input feature dimension"):
        trained_model.predict(X_wrong_features_1d)

def test_model_predict_raises_error_on_invalid_dims(trained_model):
    X_3d = np.random.rand(10, N_FEATURES, 1)
    with pytest.raises(ValueError, match="Input X must be 1D or 2D"):
        trained_model.predict(X_3d)


# 4. Test Model Evaluation
def test_model_evaluate_before_train(untrained_model, sample_data):
    X, y = sample_data
    with pytest.raises(RuntimeError, match="Model must be trained before evaluation."):
        untrained_model.evaluate(X, y)

def test_model_evaluate(trained_model, sample_data):
    X, y = sample_data
    n_eval = 20
    X_eval, y_eval = X[:n_eval], y[:n_eval]
    metric = trained_model.evaluate(X_eval, y_eval)
    assert isinstance(metric, (float, np.float_))
    assert metric >= 0.0 # Assuming a non-negative metric like MSE

def test_model_evaluate_accepts_y_column_vector(trained_model, sample_data):
    X, y = sample_data
    n_eval = 20
    X_eval, y_eval = X[:n_eval], y[:n_eval]
    y_eval_col = y_eval.reshape(-1, 1)
    metric = trained_model.evaluate(X_eval, y_eval_col) # Should work
    assert isinstance(metric, (float, np.float_))
    assert metric >= 0.0

def test_model_evaluate_raises_error_on_invalid_input_type(trained_model):
    X_list = [[1] * N_FEATURES] * 10
    y_array = np.random.rand(10)
    with pytest.raises(ValueError, match="X and y must be numpy arrays"):
        trained_model.evaluate(X_list, y_array)

    X_array = np.random.rand(10, N_FEATURES)
    y_list = list(range(10))
    with pytest.raises(ValueError, match="X and y must be numpy arrays"):
        trained_model.evaluate(X_array, y_list)

def test_model_evaluate_raises_error_on_shape_mismatch(trained_model, sample_data):
    X, y = sample_data
    X_eval = X[:10]
    y_eval_mismatch = y[:9] # Mismatched number of samples
    with pytest.raises(ValueError, match="X and y must have the same number of samples"):
        trained_model.evaluate(X_eval, y_eval_mismatch)

def test_model_evaluate_raises_error_on_feature_mismatch(trained_model, sample_data):
     X, y = sample_data
     X_wrong_features = np.random.rand(10, N_FEATURES + 1)
     y_correct_shape = y[:10]
     with pytest.raises(ValueError, match="Input feature dimension"):
         trained_model.evaluate(X_wrong_features, y_correct_shape)

def test_model_evaluate_raises_error_on_invalid_dims(trained_model, sample_data):
    X, y = sample_data
    X_1d = X[0]
    y_1d = y[0]
    with pytest.raises(ValueError, match="X must be a 2D array"):
        trained_model.evaluate(X_1d, np.array([y_1d])) # y needs to be array-like

    X_2d = X[:10]
    y_2d_wrong = np.random.rand(10, 2) # y has more than 1 column
    with pytest.raises(ValueError, match="y must be a 1D array or 2D column vector"):
        trained_model.evaluate(X_2d, y_2d_wrong)

    X_3