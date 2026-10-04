import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize

# ---------- аппроксимация плоскости ----------
def appr_f(x, y, z):
    # прямой МНК: z ≈ -A x - B y - C
    M = np.column_stack([x, y, np.ones_like(x)])
    A, B, C = np.linalg.lstsq(M, -z, rcond=None)[0]
    return np.array([A, B, C])


# ---------- аппроксимация окружности (в плоскости XY) ----------
def circle_residuals(params, x, y):
    x0, y0, R = params
    return np.sum(((x - x0)**2 + (y - y0)**2 - R**2)**2)


def appr_circle(x, y):
    x0, y0 = x.mean(), y.mean()
    R = np.hypot(x - x0, y - y0).mean()
    return minimize(circle_residuals, [x0, y0, R], args=(x, y)).x


# ---------- аппроксимация окружности в плоскости A*x+B*y+z+C=0 ----------
def appr_circle_in_plane(x, y, z, A, B, C):
    """
    Подгоняет окружность, лежащую в плоскости A*x + B*y + z + C = 0.
    Возвращает: center (3,), rho, e1 (3,), e2 (3,) — всё в локальных координатах.
    """
    n = np.array([A, B, 1.0])
    nhat = n / np.linalg.norm(n)
    p0 = np.array([0.0, 0.0, -C])              # точка на плоскости

    # базис в плоскости
    e1 = np.array([1.0, 0.0, -A])
    e1 -= np.dot(e1, nhat) * nhat
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nhat, e1)

    # проецируем точки на плоскость
    P = np.column_stack([x, y, z]) - p0
    P = P - np.outer(P @ nhat, nhat)

    # 2D-координаты в базисе
    u = P @ e1
    v = P @ e2

    # обычная 2D-окружность
    u0, v0, rho = appr_circle(u, v)

    # центр в 3D
    center = p0 + u0 * e1 + v0 * e2

    return center, rho, e1, e2