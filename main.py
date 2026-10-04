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

df_temp = df[df['name'] == ''].copy()

for name, group in df_temp.groupby(by='frame_start', sort=False):
    if len(group) < 3:
        continue
    x0, y0, R = appr_circle(group['x_start'], group['y_start'])
    df.loc[df['frame_start'] == name, ['x0', 'y0', 'R']] = x0, y0, R

df_temp = df[df['name'].str.contains(r"r\d+")].copy()
df_temp[['A0', 'B0', 'C0', 'z_start']] = np.nan

# ---------- аппроксимация плоскостей ----------
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


Rho = []
Time = []
Radius = []

# ---------- отрисовка ----------
for name, group in df.groupby(by='frame_start', sort=False):
    if group[['A0', 'B0', 'C0']].isna().any().any():
        continue

    print(group)

    A  = group['A0'].iloc[0]
    B  = group['B0'].iloc[0]
    C  = group['C0'].iloc[0]
    x0 = group['x0'].iloc[0]
    y0 = group['y0'].iloc[0]
    R  = group['R'].iloc[0]

    gx = (group['x_start'] - x0).to_numpy()
    gy = (group['y_start'] - y0).to_numpy()
    gz = group['z_start'].to_numpy()

    # --- сфера ---
    phi  = np.linspace(0, 2*np.pi, 100)
    teta = np.linspace(0, np.pi, 100)
    PHI, TETA = np.meshgrid(phi, teta)
    xs = x0 + R*np.sin(TETA)*np.cos(PHI)
    ys = y0 + R*np.sin(TETA)*np.sin(PHI)
    zs =      R*np.cos(TETA)

    # --- плоскость ---
    XL, YL = np.meshgrid(np.linspace(-R, R, 30), np.linspace(-R, R, 30))
    ZL = -A*XL - B*YL - C
    Xp, Yp, Zp = XL + x0, YL + y0, ZL

    # --- окружность в плоскости (по реальным точкам) ---
    center, rho, e1, e2 = appr_circle_in_plane(gx, gy, gz, A, B, C)
    Rho.append(rho)
    Time.append(name)
    Radius.append(R)


    t = np.linspace(0, 2*np.pi, 200)
    circ = center[:, None] + rho * (
        np.outer(e1, np.cos(t)) + np.outer(e2, np.sin(t))
    )
    cx, cy, cz = circ

    # --- диагностика ---
    d_center = np.sqrt((gx - center[0])**2 + (gy - center[1])**2 + (gz - center[2])**2)

    # --- фигура ---
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')

    ax.plot_surface(xs, ys, zs, alpha=0.20, edgecolor='none', cmap='viridis')
    ax.plot_surface(Xp, Yp, Zp, alpha=0.35, edgecolor='none', cmap='Reds')

    # окружность — линия, не surface
    ax.plot(cx + x0, cy + y0, cz, color='black', lw=2)

    # точки
    ax.scatter(group['x_start'], group['y_start'], group['z_start'],
               color='red', s=8)

    # центр окружности — отдельная точка
    ax.scatter([center[0] + x0], [center[1] + y0], [center[2]],
               color='black', s=40, marker='x')

    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(name)
    plt.show()

Teta = np.arcsin(np.array(Rho)/np.array(Radius))
k, b = np.polyfit(Time, Teta, 1)
x = np.linspace(min(Time), max(Time), 10)

plt.plot(x, k*x + b)
plt.plot(Time, np.arcsin(np.array(Rho)/np.array(Radius)))
plt.show()