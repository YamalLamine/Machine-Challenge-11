#!/usr/bin/env python3
"""
Polynomial regression via normal equations solved with LUP (LU + partial pivoting).
Supports two modes:
1) User inputs 6 points -> degree 5 -> predict 7th y at given x7.
2) Use 100 baked points -> user inputs n (1..100) -> use first n points (degree n-1)
   to predict next point.
All linear algebra (LUP, solve) implemented from scratch (no numpy.linalg).
"""

from typing import List, Tuple
import math
import random

# ---------- Basic matrix helpers ----------
def zeros_matrix(rows:int, cols:int) -> List[List[float]]:
    return [[0.0]*cols for _ in range(rows)]

def mat_mul(A:List[List[float]], B:List[List[float]]) -> List[List[float]]:
    r, m = len(A), len(A[0])
    m2, c = len(B), len(B[0])
    assert m == m2
    C = zeros_matrix(r, c)
    for i in range(r):
        Ai = A[i]
        for k in range(m):
            aik = Ai[k]
            rowB = B[k]
            for j in range(c):
                C[i][j] += aik * rowB[j]
    return C

def transpose(A:List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*A)]

def mat_vec_mul(A:List[List[float]], v:List[float]) -> List[float]:
    r = len(A)
    m = len(A[0])
    assert m == len(v)
    out = [0.0]*r
    for i in range(r):
        s = 0.0
        Ai = A[i]
        for j in range(m):
            s += Ai[j]*v[j]
        out[i] = s
    return out

# ---------- Vandermonde (highest degree first) ----------
def vandermonde(x_list:List[float], degree:int) -> List[List[float]]:
    # returns matrix with shape (len(x_list), degree+1)
    V = []
    for x in x_list:
        row = [x**k for k in range(degree, -1, -1)]  # highest -> lowest
        V.append(row)
    return V

# ---------- LUP decomposition and solver ----------
def lup_decompose(A:List[List[float]]) -> Tuple[List[List[float]], List[int]]:
    """Return LU matrix (combined) and pivot permutation P as list.
       A is modified in-place (but we will copy before calling)."""
    n = len(A)
    # create pivot array
    P = list(range(n))
    for k in range(n):
        # find pivot (max abs in column k below/including k)
        pivot_row = max(range(k, n), key=lambda i: abs(A[i][k]))
        if abs(A[pivot_row][k]) < 1e-14:
            raise ValueError("Matrix is singular or nearly singular")
        # swap rows k and pivot_row in A and in P
        if pivot_row != k:
            A[k], A[pivot_row] = A[pivot_row], A[k]
            P[k], P[pivot_row] = P[pivot_row], P[k]
        # elimination
        for i in range(k+1, n):
            A[i][k] /= A[k][k]
            factor = A[i][k]
            rowk = A[k]
            rowi = A[i]
            for j in range(k+1, n):
                rowi[j] -= factor * rowk[j]
    return A, P

def lup_solve(LU:List[List[float]], P:List[int], b:List[float]) -> List[float]:
    n = len(LU)
    # apply permutation to b -> Pb
    Pb = [0.0]*n
    for i in range(n):
        Pb[i] = b[P[i]]
    # forward solve Ly = Pb (L has 1s on diagonal and lower part stored)
    y = [0.0]*n
    for i in range(n):
        s = Pb[i]
        for j in range(i):
            s -= LU[i][j] * y[j]
        y[i] = s
    # backward solve Ux = y
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        s = y[i]
        for j in range(i+1, n):
            s -= LU[i][j] * x[j]
        x[i] = s / LU[i][i]
    return x

# ---------- Regression pipeline ----------
def solve_normal_equation_using_lup(V:List[List[float]], y:List[float]) -> List[float]:
    # V has shape (m, p) where p = degree+1
    VT = transpose(V)
    ATA = mat_mul(VT, V)     # (p x p)
    ATy_mat = mat_vec_mul(VT, y)  # length p
    # copy ATA (because lup_decompose modifies in place)
    ATA_copy = [row[:] for row in ATA]
    LU, P = lup_decompose(ATA_copy)
    coeffs = lup_solve(LU, P, ATy_mat)
    return coeffs  # highest-degree first to match polyval

def polyval(coeffs:List[float], x:float) -> float:
    # coeffs highest->lowest
    s = 0.0
    for c in coeffs:
        s = s*x + c
    return s

# ---------- Example baked 100 points (synthetic) ----------
def population_data(n=100):
    # Years shifted so 1924 -> x = 0
    xs = [i - 1924 for i in range(1924, 2024)]
    
    # Global population in billions
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

# ---------- Interaction ----------
def run_choice1():
    print("Choice 1: enter 6 data points (x y). Program fits degree 5 polynomial and predicts y at x7.")
    input_points = []
    for i in range(6):
        txt = input(f"Point {i+1} (format: x y): ").strip()
        x_s, y_s = txt.split()
        input_points.append((float(x_s), float(y_s)))
    x_vals = [p[0] for p in input_points]
    y_vals = [p[1] for p in input_points]
    # degree = 5 (since 6 points -> degree 5 per your rule)
    degree = 5
    V = vandermonde(x_vals, degree)
    coeffs = solve_normal_equation_using_lup(V, y_vals)
    x7 = float(input("Enter x7 (the x at which to predict y7): ").strip())
    y7 = polyval(coeffs, x7)
    print(f"Predicted y7 at x={x7}: {y7:.4f}")
    print("Coefficients (highest->lowest):")
    print([round(c, 6) for c in coeffs])

def run_choice2():
    xs, ys = population_data(100)  # xs = 0..99 (1924 -> 0)
    print("Choice 2: program has 100 baked points (1924..2023).")
    
    # Ask the user which year they want to predict
    x_year = int(input("Enter the year you want to predict (1924..2028): ").strip())
    if x_year < 1924 or x_year > 2028:
        print("Year must be between 1924 and 2028")
        return

    # Shift year to x in dataset
    x_next = x_year - 1924

    # Determine points to use
    if x_next <= 99:
        # Interpolation: pick 10 points centered around x_next
        idx = min(range(len(xs)), key=lambda i: abs(xs[i]-x_next))
        start = max(0, min(idx-4, len(xs)-10))  # ensures 10 points without going out of bounds
        end = start + 10
        x_used = xs[start:end]
        y_used = ys[start:end]
        degree = len(x_used) - 1  # full-degree polynomial
    else:
        # Extrapolation: use last 10 points for stability
        x_used = xs[-10:]
        y_used = ys[-10:]
        degree = 3  # degree 5 polynomial for extrapolation

    # Rescale x to [0,1] for numerical stability
    x_min = x_used[0]
    x_max = x_used[-1]
    x_scaled = [(xi - x_min) / (x_max - x_min) for xi in x_used]
    x_next_scaled = (x_next - x_min) / (x_max - x_min)

    print(f"Using {len(x_used)} points -> degree {degree}")

    # Fit polynomial
    V = vandermonde(x_scaled, degree)
    coeffs = solve_normal_equation_using_lup(V, y_used)

    # Predict
    y_next = polyval(coeffs, x_next_scaled)
    print(f"Predicted population for year {x_year}: {y_next:.2f} billion")

    # Show all coefficients used
    print("Coefficients used (highest->lowest, rounded 6 decimals):")
    print([round(c, 6) for c in coeffs])



def main():
    print("Polynomial regression program (LUP solver).")
    print("1) Enter 6 points -> predict 7th (degree 5).")
    print("2) Use baked 100 points -> enter n (1..100) -> fit first n points (degree n-1) -> predict next.")
    choice = input("Choose 1 or 2: ").strip()
    if choice == "1":
        run_choice1()
    elif choice == "2":
        run_choice2()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
