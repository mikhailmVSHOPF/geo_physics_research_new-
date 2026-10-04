import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import bags_array
import itertools
import random

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


# ---------- данные ----------
folder = r"C:\Users\misha\Desktop\data\data\clear_water\F33\F33_20000"

bags_array = bags_array.bags_array(folder)
df = bags_array[12].copy()


for name, group in df.groupby(by='frame_start', sort=False):
    if len(group) < 3:
        continue
    x0, y0, R = 0.0 , 0.0 , 0.0
    x0, y0, R = appr_circle(group['x_start'], group['y_start'])
    df.loc[(df['frame_start'] == name), ['x0', 'y0', 'R']] = x0, y0, R


for name, group in df.groupby(by='frame_start', sort=False):

    x0 = group['x0'].iloc[0]    
    y0 = group['y0'].iloc[0]  
    R  = group['R'].iloc[0]  
    subgroup = group[group['name'].str.contains(r'r\d+')]

    group_x = (subgroup['x_start'] - x0).to_numpy()
    group_y = (subgroup['y_start'] - y0).to_numpy()
    group_z = np.sqrt(np.maximum(R**2 - group_x**2 - group_y**2, 0.0))

    coeff = appr_f(group_x, group_y, group_z)

    mask = df['frame_start'] == name
    df.loc[(df['frame_start'] == name) & (df['name'].str.contains(r'r\d+')), 'A0']      = coeff[0]
    df.loc[(df['frame_start'] == name) & (df['name'].str.contains(r'r\d+')), 'B0']      = coeff[1]
    df.loc[(df['frame_start'] == name) & (df['name'].str.contains(r'r\d+')), 'C0']      = coeff[2]
    df.loc[(df['frame_start'] == name) & (df['name'].str.contains(r'r\d+')), 'z_start'] = group_z

print(df['name'].str.contains(r'r\d+'))