# Machine-Challenge-11: Polynomial Regression

A machine learning challenge implementing polynomial regression using Python and scikit-learn.

## Overview

Polynomial regression is a form of linear regression in which the relationship between the independent variable `x` and dependent variable `y` is modeled as an nth degree polynomial. This challenge demonstrates:

- Generating synthetic data with a polynomial relationship
- Fitting polynomial regression models of various degrees
- Evaluating model performance using MSE and R² metrics
- Visualizing the results

## Requirements

- Python 3.8+
- NumPy
- scikit-learn
- matplotlib
- pandas

Install dependencies:

```bash
pip install numpy scikit-learn matplotlib pandas
```

## Usage

Run the main script:

```bash
python polynomial_regression.py
```

This will:
1. Generate synthetic data following the function `y = 0.5*x² + x + 2 + noise`
2. Split data into training (80%) and test (20%) sets
3. Fit a polynomial regression model (degree=2)
4. Evaluate the model and display metrics
5. Save a visualization to `polynomial_regression_results.png`

## Running Tests

Install pytest and run the tests:

```bash
pip install pytest
pytest test_polynomial_regression.py -v
```

## Module Functions

### `generate_sample_data(n_samples, noise, random_state)`
Generates synthetic data for polynomial regression.

### `fit_polynomial_regression(X_train, y_train, degree)`
Fits a polynomial regression model to the training data.

### `predict(model, poly_features, X)`
Makes predictions using the fitted model.

### `evaluate_model(y_true, y_pred)`
Calculates MSE and R² scores for model evaluation.

### `plot_results(...)`
Creates a visualization of the polynomial regression results.

## Example Output

```
============================================================
Polynomial Regression Machine Learning Challenge
============================================================

1. Generating sample data...
   Generated 100 samples

2. Splitting data into train/test sets (80/20)...
   Training samples: 80
   Test samples: 20

3. Fitting polynomial regression model (degree=2)...
   Model coefficients: [0.82855152 0.56053601]
   Model intercept: -0.6374

4. Making predictions...

5. Evaluating model performance...
   Training MSE: 81.4715
   Training R²: 0.7825
   Test MSE: 63.5841
   Test R²: 0.7642

6. Generating visualization...
   Saved plot to 'polynomial_regression_results.png'

============================================================
Challenge completed successfully!
============================================================
```

## License

This project is for educational purposes as part of the Machine Challenge series.
