import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bags_array
from scipy.optimize import minimize


# ---------- аппроксимация плоскости ----------
def appr_f(x, y, z):
    M = np.column_stack([x, y, np.ones_like(x)])
    A, B, C = np.linalg.lstsq(M, -z, rcond=None)[0]
    return np.array([A, B, C])


# ---------- аппроксимация окружности (в плоскости XY) ----------
def circle_residuals(params, x, y):
    x0, y0, R = params
    return np.sum(((x - x0) ** 2 + (y - y0) ** 2 - R ** 2) ** 2)


def appr_circle(x, y):
    x0, y0 = x.mean(), y.mean()
    R = np.hypot(x - x0, y - y0).mean()
    return minimize(circle_residuals, [x0, y0, R], args=(x, y)).x


# ---------- аппроксимация окружности в плоскости A*x+B*y+z+C=0 ----------
def appr_circle_in_plane(x, y, z, A, B, C):
    n = np.array([A, B, 1.0])
    nhat = n / np.linalg.norm(n)
    p0 = np.array([0.0, 0.0, -C])

    e1 = np.array([1.0, 0.0, -A])
    e1 -= np.dot(e1, nhat) * nhat
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nhat, e1)

    P = np.column_stack([x, y, z]) - p0
    P = P - np.outer(P @ nhat, nhat)

    u = P @ e1
    v = P @ e2

    u0, v0, rho = appr_circle(u, v)
    center = p0 + u0 * e1 + v0 * e2
    return center, rho, e1, e2


# ---------- обработка одной папки ----------
def process_folder(folder, source_id):
    """
    Для каждого bag'а считает точки (t, theta), делает линейный фит
    theta ≈ k*t + b и возвращает список записей:
        (k, b, r2, n_points, bag_idx, source_id)
    Время нормируется так, что старт каждого bag'а = 0.
    """
    bags = bags_array.bags_array(folder)
    records = []

    for bag_idx, df_raw in enumerate(bags):
        df = df_raw.copy()
        df[['x0', 'y0', 'R']] = np.nan

        # --- центр/радиус по кадрам с name == '' ---
        df_temp = df[df['name'] == ''].copy()
        for name, group in df_temp.groupby(by='frame_start', sort=False):
            if len(group) < 3:
                continue
            try:
                x0, y0, R = appr_circle(group['x_start'], group['y_start'])
            except Exception:
                continue
            mask = df['frame_start'] == name
            df.loc[mask, 'x0'] = x0
            df.loc[mask, 'y0'] = y0
            df.loc[mask, 'R']  = R

        # --- оставляем точки "r\d+" ---
        df = df[df['name'].astype(str).str.contains(r"r\d+", regex=True, na=False)].copy()
        if df.empty:
            continue
        df[['A0', 'B0', 'C0', 'z_start']] = np.nan

        # --- аппроксимация плоскостей ---
        for name, group in df.groupby(by='frame_start', sort=False):
            if group[['x0', 'y0', 'R']].isna().any().any():
                continue

            x0 = group['x0'].iloc[0]
            y0 = group['y0'].iloc[0]
            R  = group['R'].iloc[0]

            gx = (group['x_start'] - x0).to_numpy()
            gy = (group['y_start'] - y0).to_numpy()
            gz = np.sqrt(np.maximum(R ** 2 - gx ** 2 - gy ** 2, 0.0))

            coeff = appr_f(gx, gy, gz)
            mask = df['frame_start'] == name
            df.loc[mask, 'A0']      = coeff[0]
            df.loc[mask, 'B0']      = coeff[1]
            df.loc[mask, 'C0']      = coeff[2]
            df.loc[mask, 'z_start'] = gz

        # --- стартовое время текущего bag'а ---
        bag_frame_times = []
        for name, group in df.groupby(by='frame_start', sort=False):
            if group[['A0', 'B0', 'C0']].isna().any().any():
                continue
            try:
                bag_frame_times.append(float(name))
            except (TypeError, ValueError):
                bag_frame_times.append(float(group.index[0]))

        if not bag_frame_times:
            continue
        t0 = min(bag_frame_times)

        # --- собираем (t, theta) по кадрам ---
        ts, ths = [], []
        for name, group in df.groupby(by='frame_start', sort=False):
            if group[['A0', 'B0', 'C0']].isna().any().any():
                continue

            A  = group['A0'].iloc[0]
            B  = group['B0'].iloc[0]
            C  = group['C0'].iloc[0]
            x0 = group['x0'].iloc[0]
            y0 = group['y0'].iloc[0]
            R  = group['R'].iloc[0]

            gx = (group['x_start'] - x0).to_numpy()
            gy = (group['y_start'] - y0).to_numpy()
            gz = group['z_start'].to_numpy()

            try:
                center, rho, e1, e2 = appr_circle_in_plane(gx, gy, gz, A, B, C)
            except Exception:
                continue

            if R <= 0 or not np.isfinite(rho):
                continue

            theta = np.arcsin(np.clip(rho / R, -1.0, 1.0))

            try:
                t_val = float(name)
            except (TypeError, ValueError):
                t_val = float(group.index[0])

            ts.append(t_val - t0)
            ths.append(theta)

        # --- линейный фит theta = k*t + b ---
        if len(ts) < 2:
            continue
        ts  = np.array(ts)
        ths = np.array(ths)

        if np.ptp(ts) == 0:   # все времена одинаковые — фит бессмыслен
            continue

        k, b = np.polyfit(ts, ths, 1)

        # R^2
        pred = k * ts + b
        ss_res = np.sum((ths - pred) ** 2)
        ss_tot = np.sum((ths - ths.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        records.append({
            'k': k,
            'b': b,
            'r2': r2,
            'n_points': len(ts),
            'bag_idx': bag_idx,
            'source_id': source_id,
        })

    return records


# ---------- список папок ----------
folders = [
    r"C:\Users\misha\Desktop\data\data\clear_water\F33\F33_20000",
    r"C:\Users\misha\Desktop\data\data\clear_water\F33\F33_12000",
    r"D:\2026_onebag(mini channel)\nac_cam\F33\F33_6000",
    r"d:\2026_onebag(mini channel)\nac_cam\F33\F33_10000_21_11_2025",
    r"d:\2026_onebag(mini channel)\nac_cam\F33\F33_15000_20_11_2025",
    # добавляйте сколько нужно
]
labels = [f.split("\\")[-1] for f in folders]


# ---------- обработка ----------
all_records = []

for src_id, folder_path in enumerate(folders):
    print(f"[{src_id + 1}/{len(folders)}] {folder_path}")
    recs = process_folder(folder_path, src_id)
    all_records += recs
    print(f"    bag'ов с валидным фитом: {len(recs)}")


# ---------- DataFrame для удобства ----------
rec_df = pd.DataFrame(all_records)

# --- (опционально) фильтр по качеству фита ---
# оставляем только bag'и, где theta действительно линейна
rec_df = rec_df[rec_df['r2'] > 0.8].copy()

print("\nСтатистика по наклонам k:")
print(rec_df.groupby('source_id')['k'].describe())


# ---------- гистограмма распределения наклона k ----------
plt.figure(figsize=(11, 6))

cmap = plt.get_cmap('tab10')
n_src = len(folders)

# общий диапазон значений k
k_min, k_max = rec_df['k'].min(), rec_df['k'].max()
bins = np.linspace(k_min, k_max, 40)

for src in sorted(rec_df['source_id'].unique()):
    m = rec_df['source_id'] == src
    plt.hist(
        rec_df.loc[m, 'k'],
        bins=bins,
        alpha=0.5,
        color=cmap(src % 10),
        label=labels[src] if src < len(labels) else f'folder {src}',
        edgecolor='black',
        linewidth=0.4,
    )

plt.xlabel(r'наклон $k = d\theta/dt$')
plt.ylabel('число bag\'ов')
plt.title(r'Распределение наклона $\theta(t)$ по bag\'ам')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ---------- на всякий случай: разброс k в зависимости от R^2 ----------
plt.figure(figsize=(9, 5))
for src in sorted(rec_df['source_id'].unique()):
    m = rec_df['source_id'] == src
    plt.scatter(
        rec_df.loc[m, 'r2'], rec_df.loc[m, 'k'],
        s=18, alpha=0.6,
        color=cmap(src % 10),
        label=labels[src] if src < len(labels) else f'folder {src}',
    )
plt.axvline(0.8, color='k', ls='--', lw=1)
plt.xlabel(r'$R^2$ линейного фита $\theta(t)$')
plt.ylabel(r'наклон $k$')
plt.title('Качество фита vs наклон')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()