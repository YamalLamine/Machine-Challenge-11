"""
Unit tests for the Polynomial Regression implementation.
"""

import numpy as np
import pytest
from polynomial_regression import (
    generate_sample_data,
    fit_polynomial_regression,
    predict,
    evaluate_model,
)


class TestGenerateSampleData:
    """Tests for the generate_sample_data function."""
    
    def test_returns_correct_shape(self):
        """Test that generated data has correct shape."""
        X, y = generate_sample_data(n_samples=50)
        assert X.shape == (50, 1)
        assert y.shape == (50,)
    
    def test_reproducible_with_random_state(self):
        """Test that same random_state produces same data."""
        X1, y1 = generate_sample_data(n_samples=10, random_state=42)
        X2, y2 = generate_sample_data(n_samples=10, random_state=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
    
    def test_different_random_states_produce_different_data(self):
        """Test that different random states produce different data."""
        X1, y1 = generate_sample_data(n_samples=10, random_state=42)
        X2, y2 = generate_sample_data(n_samples=10, random_state=123)
        assert not np.array_equal(X1, X2)


class TestFitPolynomialRegression:
    """Tests for the fit_polynomial_regression function."""
    
    def test_returns_model_and_features(self):
        """Test that function returns model and poly_features."""
        X, y = generate_sample_data(n_samples=50)
        model, poly_features = fit_polynomial_regression(X, y, degree=2)
        assert model is not None
        assert poly_features is not None
    
    def test_model_has_coefficients(self):
        """Test that fitted model has coefficients."""
        X, y = generate_sample_data(n_samples=50)
        model, poly_features = fit_polynomial_regression(X, y, degree=2)
        assert hasattr(model, 'coef_')
        assert len(model.coef_) == 2  # degree 2 with include_bias=False


class TestPredict:
    """Tests for the predict function."""
    
    def test_returns_predictions(self):
        """Test that predict returns predictions."""
        X, y = generate_sample_data(n_samples=50)
        model, poly_features = fit_polynomial_regression(X, y, degree=2)
        predictions = predict(model, poly_features, X)
        assert predictions.shape == (50,)
    
    def test_predictions_are_numeric(self):
        """Test that predictions are numeric values."""
        X, y = generate_sample_data(n_samples=50)
        model, poly_features = fit_polynomial_regression(X, y, degree=2)
        predictions = predict(model, poly_features, X)
        assert np.isfinite(predictions).all()


class TestEvaluateModel:
    """Tests for the evaluate_model function."""
    
    def test_returns_mse_and_r2(self):
        """Test that evaluate_model returns MSE and R2."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        mse, r2 = evaluate_model(y_true, y_pred)
        assert mse >= 0
        assert -1 <= r2 <= 1
    
    def test_perfect_predictions_give_zero_mse(self):
        """Test that perfect predictions result in zero MSE."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        mse, r2 = evaluate_model(y_true, y_pred)
        assert mse == 0
        assert r2 == 1.0


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_pipeline_works(self):
        """Test that the full pipeline runs without errors."""
        # Generate data
        X, y = generate_sample_data(n_samples=100)
        
        # Split manually
        train_size = 80
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Fit model
        model, poly_features = fit_polynomial_regression(X_train, y_train, degree=2)
        
        # Predict
        y_pred = predict(model, poly_features, X_test)
        
        # Evaluate
        mse, r2 = evaluate_model(y_test, y_pred)
        
        # Check that model performs reasonably well
        assert r2 > 0.5  # Should explain at least 50% of variance
    
    def test_higher_degree_polynomial(self):
        """Test that higher degree polynomials work."""
        X, y = generate_sample_data(n_samples=100)
        model, poly_features = fit_polynomial_regression(X, y, degree=3)
        predictions = predict(model, poly_features, X)
        mse, r2 = evaluate_model(y, predictions)
        assert r2 > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
