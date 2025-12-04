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
def make_baked_data(n=100):
    # Example: daily flu counts with a smooth-ish underlying function + noise.
    rng = random.Random(0)
    xs = [float(i) for i in range(n)]
    ys = []
    for i in xs:
        # synthetic pattern: a slow seasonal bump + local oscillation + noise
        base = 50 + 30*math.exp(-((i-40)/25.0)**2)      # bump around 40
        small_wave = 8*math.sin(i*0.2)
        noise = rng.uniform(-4, 4)
        ys.append(base + small_wave + noise)
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
    print(f"Predicted y7 at x={x7}: {y7}")
    print("Coefficients (highest->lowest):")
    print(coeffs)

def run_choice2():
    xs, ys = make_baked_data(100)
    print("Choice 2: program has 100 baked points (x0..x99).")
    n_str = input("Enter n (use first n points, 1..100): ").strip()
    n = int(n_str)
    if n < 1 or n > 100:
        print("n must be between 1 and 100")
        return
    x_used = xs[:n]
    y_used = ys[:n]
    degree = n - 1  # per assumption
    print(f"Using {n} points -> degree {degree}")
    V = vandermonde(x_used, degree)
    coeffs = solve_normal_equation_using_lup(V, y_used)
    # predict next x: we'll use xs[n] if exists; else ask user
    if n < len(xs):
        x_next = xs[n]
        print(f"Predicting y at baked x_next = {x_next}")
    else:
        x_next = float(input("Enter x at which to predict next y: ").strip())
    y_next = polyval(coeffs, x_next)
    print(f"Predicted y_next: {y_next}")
    print("Sample coefficients (first 10 highest->lowest):")
    print([round(c,6) for c in coeffs[:10]])

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
