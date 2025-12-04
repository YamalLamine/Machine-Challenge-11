"""
Polynomial Regression Machine Learning Challenge

This module implements polynomial regression using scikit-learn.
It demonstrates how to fit polynomial models to non-linear data.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def generate_sample_data(n_samples=100, noise=10, random_state=42):
    """
    Generate sample data for polynomial regression.
    
    The true underlying function is: y = 0.5*x^2 + x + 2 + noise
    
    Args:
        n_samples: Number of samples to generate
        noise: Standard deviation of Gaussian noise
        random_state: Random seed for reproducibility
        
    Returns:
        X: Feature array of shape (n_samples, 1)
        y: Target array of shape (n_samples,)
    """
    np.random.seed(random_state)
    X = np.random.uniform(-10, 10, n_samples).reshape(-1, 1)
    # True function: y = 0.5*x^2 + x + 2
    y = 0.5 * X.flatten()**2 + X.flatten() + 2 + np.random.normal(0, noise, n_samples)
    return X, y


def fit_polynomial_regression(X_train, y_train, degree=2):
    """
    Fit a polynomial regression model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        degree: Degree of the polynomial
        
    Returns:
        model: Fitted LinearRegression model
        poly_features: Fitted PolynomialFeatures transformer
    """
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X_train)
    
    model = LinearRegression()
    model.fit(X_poly, y_train)
    
    return model, poly_features


def predict(model, poly_features, X):
    """
    Make predictions using the polynomial regression model.
    
    Args:
        model: Fitted LinearRegression model
        poly_features: Fitted PolynomialFeatures transformer
        X: Features to predict on
        
    Returns:
        y_pred: Predicted values
    """
    X_poly = poly_features.transform(X)
    return model.predict(X_poly)


def evaluate_model(y_true, y_pred):
    """
    Evaluate the model performance.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        mse: Mean Squared Error
        r2: R-squared score
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2


def plot_results(X_train, y_train, X_test, y_test, model, poly_features, degree):
    """
    Plot the polynomial regression results.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        model: Fitted model
        poly_features: Fitted polynomial features transformer
        degree: Polynomial degree
    """
    # Create a smooth line for plotting
    X_line = np.linspace(X_train.min() - 1, X_train.max() + 1, 200).reshape(-1, 1)
    y_line = predict(model, poly_features, X_line)
    
    plt.figure(figsize=(10, 6))
    
    # Plot training data
    plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training data')
    
    # Plot test data
    plt.scatter(X_test, y_test, color='green', alpha=0.5, label='Test data')
    
    # Plot the polynomial regression curve
    plt.plot(X_line, y_line, color='red', linewidth=2, 
             label=f'Polynomial Regression (degree={degree})')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Polynomial Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt


def main():
    """Main function to run the polynomial regression challenge."""
    print("=" * 60)
    print("Polynomial Regression Machine Learning Challenge")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    X, y = generate_sample_data(n_samples=100, noise=10)
    print(f"   Generated {len(X)} samples")
    
    # Split data into training and test sets
    print("\n2. Splitting data into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Fit polynomial regression model
    degree = 2
    print(f"\n3. Fitting polynomial regression model (degree={degree})...")
    model, poly_features = fit_polynomial_regression(X_train, y_train, degree=degree)
    
    # Print model coefficients
    print(f"   Model coefficients: {model.coef_}")
    print(f"   Model intercept: {model.intercept_:.4f}")
    
    # Make predictions
    print("\n4. Making predictions...")
    y_train_pred = predict(model, poly_features, X_train)
    y_test_pred = predict(model, poly_features, X_test)
    
    # Evaluate model
    print("\n5. Evaluating model performance...")
    train_mse, train_r2 = evaluate_model(y_train, y_train_pred)
    test_mse, test_r2 = evaluate_model(y_test, y_test_pred)
    
    print(f"   Training MSE: {train_mse:.4f}")
    print(f"   Training R²: {train_r2:.4f}")
    print(f"   Test MSE: {test_mse:.4f}")
    print(f"   Test R²: {test_r2:.4f}")
    
    # Plot results
    print("\n6. Generating visualization...")
    plot = plot_results(X_train, y_train, X_test, y_test, model, poly_features, degree)
    plot.savefig('polynomial_regression_results.png', dpi=150)
    print("   Saved plot to 'polynomial_regression_results.png'")
    
    print("\n" + "=" * 60)
    print("Challenge completed successfully!")
    print("=" * 60)
    
    return model, poly_features


if __name__ == "__main__":
    main()
