import math
import numpy as np
import os
import platform
from typing import List, Tuple

def clear_console(): #this just clears the console so it doesnt look cluttered
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear') #this is in case if terminal is in linux or non windows

def main():
    print("\nPolynomial Curve Fitting")
    
    while True:
        print("\nMenu:")
        print("1) Enter 6 points → predict 7th")
        print("2) Predict world population from dataset of 100 years")
        print("Q) Quit")

        choice = input("Choose: ").strip().upper()

        if choice == "1":
            clear_console()
            run_choice1()
        elif choice == "2":
            clear_console()
            run_choice2()
        elif choice == "Q":
            print("Exiting.")
            break
        else:
            print("Invalid choice.")

# ---------- Dataset ----------
def population_data(n=100):
    xs = [i - 1924 for i in range(1924, 2024)]
    ys = [
        1.96, 1.98, 2, 2.01, 2.03, 2.05, 2.07, 2.09, 2.11, 2.13,
        2.16, 2.18, 2.2, 2.22, 2.25, 2.27, 2.29, 2.31, 2.33, 2.35,
        2.37, 2.38, 2.4, 2.42, 2.44, 2.47, 2.49, 2.54, 2.58, 2.63,
        2.69, 2.74, 2.8, 2.85, 2.91, 2.97, 3.02, 3.06, 3.12, 3.19,
        3.26, 3.33, 3.4, 3.47, 3.55, 3.62, 3.69, 3.77, 3.84, 3.92,
        4, 4.07, 4.14, 4.22, 4.29, 4.37, 4.45, 4.53, 4.61, 4.7,
        4.78, 4.87, 4.96, 5.05, 5.14, 5.23, 5.33, 5.42, 5.51, 5.59,
        5.68, 5.76, 5.84, 5.92, 6.01, 6.09, 6.17, 6.25, 6.34, 6.42,
        6.5, 6.59, 6.67, 6.76, 6.84, 6.93, 7.02, 7.11, 7.2, 7.29,
        7.38, 7.47, 7.56, 7.65, 7.73, 7.81, 7.89, 7.95, 8.02, 8.09
    ]
    return xs, ys

# ---------- Choice 1  ----------
def run_choice1():
    print("\nOption 1: Enter 6 data points.")
    print("\nProgram will fit a degree 5 polynomial based on your points and predicts y at your wanted x7. \nEnter your x and y values per point separated by a space: (x y) \nOr enter M at any time to return to Menu\n")  

    while True:
        # --- Get 6 data points from user ---
        input_points = []
        i = 0
        while i < 6:
            txt = input(f"Point {i+1} (x y): ").strip()
            if txt.upper() == 'M':
                clear_console()
                return main()
            try:
                x_s, y_s = txt.split()
                input_points.append((float(x_s), float(y_s)))
                i += 1
            except ValueError:
                print("Invalid format. Please enter two numbers separated by a space.")

        # Convert to NumPy arrays
        x_vals = np.array([p[0] for p in input_points])
        y_vals = np.array([p[1] for p in input_points])

        # --- Solve polynomial ---
        try:
            coeffs = fit_polynomial_qr(x_vals, y_vals, degree=5)
        except np.linalg.LinAlgError as e:
            print("\nError solving system:", e)
            print("Matrix may be singular or nearly singular.\n")
            continue

        # --- Get x7 from user ---
        while True: # while loop is here so we can have an option for the user to reuse the same points
            txt = input("\n(Enter 'N' to use new set points, or 'M' to return to Menu) \nEnter x7 to predict y7: ").strip().upper()
            if txt == 'M':
                clear_console()
                return
            elif txt == 'N':
                clear_console()
                return run_choice1()  # break inner loop to input new points
            try:
                x7 = float(txt)
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
                continue

            # Predict y7
            y7 = np.polyval(coeffs, x7)
            print(f"\nPredicted y7 at x={x7}:    {y7:.4f}")
            # Print polynomial nicely
            print("\nPolynomial (Ax^n format):")
            print(format_polynomial(coeffs))

            while True:
                use_same_points = input("\nUse the same points for new x7? (Y/N): ").strip().upper()
                if use_same_points == 'Y':
                    break  # continue inner loop to predict new x7
                elif use_same_points == 'N':
                    clear_console()
                    return run_choice1()  # break inner loop to input new points
                else:
                    print("Invalid input. Please enter 'Y' or 'N'.")
                    continue

# ---------- Choice 2  ----------
def run_choice2():
    xs, ys = population_data()
    years = np.arange(1924, 2024)

    print("\nChoice 2: 100 years of global population data (1924–2023).")
    print("1924 to 2023 global population data obtained from https://ourworldindata.org/grapher/population-long-run-with-projections")
    print("\nChoice 2 uses:")
    print("- Scaled x-values for numerical stability")
    print("- Singular value decomposition of Vandermonde matrix")
    print("  from data points for polynomial fitting")
    print("- User-selected year range as data points")
    print("- User-selected polynomial degree")

    while True:
        # --- Select range of years ---
        ui = input("\nEnter start year and end year (e.g., '1950 2000') from 1924 to 2023, \nOr 'M' to return: ").strip().upper()
        if ui == 'M':
            clear_console()
            return main()

        try:
            start_year, end_year = map(int, ui.split())
            if start_year < 1924 or end_year > 2023 or start_year >= end_year:
                print("Invalid range. Must be within 1924–2023 and start < end.")
                continue
        except:
            print("Invalid input.\n")
            continue

        # Extract data slice
        idx_start = start_year - 1924
        idx_end = end_year - 1924 + 1
        x_used_raw = np.array(xs[idx_start:idx_end], dtype=float)
        y_used = np.array(ys[idx_start:idx_end], dtype=float)
        n_points = len(x_used_raw)

        # --- Degree Selection ---
        while True:
            degree_input = input(f"\n\nChoose polynomial degree (0 to {n_points-1}), recommended of 4, \nOr 'B' to go back to year range: ").strip().upper()
            if degree_input == 'B':
                clear_console()
                return run_choice2()

            try:
                degree = int(degree_input)
                if not (0 <= degree <= n_points - 1):
                    print(f"Degree must be between 0 and {n_points-1}\n")
                    continue
            except:
                print("Invalid degree.\n")
                continue

            # Warning for high degree
            if degree >= 8:
                print("\n⚠ WARNING: High-degree polynomials may oscillate and give unstable predictions.\n")

            # Scale x-values to [0,1]
            xmin = x_used_raw[0]
            xmax = x_used_raw[-1]
            x_used = (x_used_raw - xmin) / (xmax - xmin)

            # Fit using SVD
            coeffs, cond = fit_polynomial_svd(x_used, y_used, degree)

            # Condition number warning
            if cond > 1e10:
                print(f"\n⚠ WARNING: Ill-conditioned matrix (cond = {cond:.2e}).")
                print("Predictions may be unstable.\n")

            # --- Prediction Loop ---
            while True:
                ptxt = input("\n'N' for new degree/range,\n'M' for menu\nEnter a year to predict: ").strip().upper()

                if ptxt == 'M':
                    clear_console()
                    return main()
                if ptxt == 'N':
                    break

                try:
                    year_predict = int(ptxt)
                except:
                    print("Invalid year.\n")
                    continue

                if year_predict < 1924 or year_predict > 2100:
                    print("⚠ Warning: Predicting far outside the data range may be inaccurate.\n")

                # Compute scaled x for prediction
                x_scaled = (year_predict - xmin - 1924) / (xmax - xmin)

                # Evaluate polynomial
                y_pred = np.polyval(coeffs, x_scaled)

                # Output results
                print(f"\nPredicted population for {year_predict}: {y_pred:.3f} billion\n")

                # Print polynomial in Ax^n form
                print("Polynomial (Ax^n format):")
                print(format_polynomial(coeffs))

                # Desmos helper
                xd = desmos_x_value(year_predict, start_year, end_year)
                print(f"\nFor Desmos: Input x = {xd:.6f} to evaluate the polynomial for year {year_predict}.\n")


# ---------- QR decomposition for polynomial fitting in Choice 1 ----------
def fit_polynomial_qr(x_vals, y_vals, degree):
    """
    Fits a polynomial of given degree to the points (x_vals, y_vals)
    using Vandermonde matrix and QR decomposition.
    
    Returns:
        coeffs: NumPy array of polynomial coefficients (highest degree first)
    """
    # Construct Vandermonde matrix
    # Ax^2 + Bx + C -> C would be ignored if there isnt a + 1 
    if degree is None:
        degree = len(x_vals) - 1  # full-degree polynomial

    A = np.vander(x_vals, N=degree + 1) # + 1 is there because the .vander method would ignore that value with no degree

    # QR decomposition
    Q, R = np.linalg.qr(A)

    # Solve R * coeffs = Q^T * y_vals
    Qt_y = Q.T @ y_vals
    coeffs = np.linalg.solve(R, Qt_y)

    return coeffs

# ---------- SVD decomposition for polynomial fitting in Choice 2 ----------
def fit_polynomial_svd(x_vals, y_vals, degree):
    """
    Fits a polynomial using scaled x-values and SVD to minimize
    numerical instability. Returns coefficients (highest degree first),
    and a condition number for warnings.
    """
    # Build Vandermonde matrix
    A = np.vander(x_vals, degree + 1)

    # SVD decomposition
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    # Compute coefficients using pseudoinverse
    coeffs = Vt.T @ ( (U.T @ y_vals) / s )

    # Condition number for warnings
    cond = s[0] / s[-1] if s[-1] != 0 else float("inf")

    return coeffs, cond

# ---------- Desmos x-value helper ----------
def desmos_x_value(year_predict, start_year, end_year):
    # since we scaled down x from 0 to 1, they can just put the year like 2024 = 100 into desmos directly
    return (year_predict - start_year) / (end_year - start_year)

# ---------- Polynomial string formatter ----------
def format_polynomial(coeffs):
    # for formatting polynomials into Ax^b format
    degree = len(coeffs) - 1
    terms = []

    for i, c in enumerate(coeffs):
        power = degree - i
        if abs(c) < 1e-12:
            continue

        # Build term string
        if power == 0:
            term = f"{c:.6g}"
        elif power == 1:
            term = f"{c:.6g}x"
        else:
            term = f"{c:.6g}x^{power}"

        terms.append(term)

    # Join terms with proper signs
    poly_str = " + ".join(terms)
    poly_str = poly_str.replace("+ -", "- ")

    return poly_str

if __name__ == "__main__":
    clear_console()
    main()
# A_reconstructed = Q @ R
# print("Original A:\n", A)
# print("Reconstructed A:\n", A_reconstructed)

# Check difference
#print("Difference:", np.max(np.abs(A - A_reconstructed)))


