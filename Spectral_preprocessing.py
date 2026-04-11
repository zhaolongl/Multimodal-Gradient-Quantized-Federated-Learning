import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# =========================
# 你可以在这里改参数
# =========================
WN_FILE = "波数.xlsx"
Y_FILE  = "分类.xlsx"
X_FILE  = "光谱数据.xlsx"

OUT_DIR = "preprocess_output"   # 输出文件夹
SG_WINDOW = 11                  # SG窗口长度（必须为奇数，且 <= 波数点数）
SG_POLY = 3                     # SG多项式阶数（< SG_WINDOW）
SG_MODE = "interp"              # 边界处理方式：'interp' 通常比较稳


def ensure_odd_window(win: int, n_points: int) -> int:
    """确保 SG window 合法：奇数且 <= n_points"""
    win = int(win)
    if win < 3:
        win = 3
    if win > n_points:
        win = n_points if n_points % 2 == 1 else n_points - 1
    if win % 2 == 0:
        win -= 1
    if win < 3:
        raise ValueError(f"波数点数太少（n_points={n_points}），无法进行SG平滑。")
    return win


def load_inputs(wn_file=WN_FILE, y_file=Y_FILE, x_file=X_FILE):
    # 读波数（一列）
    wn = pd.read_excel(wn_file, header=None).iloc[:, 0].values
    wn = pd.to_numeric(pd.Series(wn), errors="coerce").values
    if np.any(np.isnan(wn)):
        raise ValueError("波数.xlsx 中存在无法解析为数值的内容，请检查。")

    # 读分类（一列）
    y = pd.read_excel(y_file, header=None).iloc[:, 0].astype(str).values

    # 读光谱矩阵（每行样本，每列波数）
    # 你说明第一行就是第一个样品，所以必须 header=None
    X_df_raw = pd.read_excel(x_file, header=None)

    # 处理可能存在的“样本ID列/非数值列”
    X_df_num = X_df_raw.copy()
    for c in X_df_num.columns:
        X_df_num[c] = pd.to_numeric(X_df_num[c], errors="coerce")

    # 若有整列都是 NaN（比如字符串ID列），删掉
    X_df_num = X_df_num.dropna(axis=1, how="all")

    X = X_df_num.values

    # 对齐维度：列数应等于波数长度
    if X.shape[1] != len(wn):
        raise ValueError(
            f"维度不匹配：光谱数据列数={X.shape[1]}，波数长度={len(wn)}。\n"
            f"请检查：光谱数据.xlsx 是否包含多余列（如样本编号），或波数.xlsx 是否与光谱列一一对应。"
        )

    # 行数应等于分类长度
    if X.shape[0] != len(y):
        raise ValueError(
            f"样本数不匹配：光谱样本数={X.shape[0]}，分类数={len(y)}。\n"
            f"请检查 分类.xlsx 行数是否与 光谱数据.xlsx 行数一致。"
        )

    # 缺失值检查
    if np.any(np.isnan(X)):
        r, c = np.where(np.isnan(X))
        raise ValueError(
            f"光谱数据.xlsx 中存在空值/非数值导致 NaN，例如第 {r[0]+1} 行、第 {c[0]+1} 列附近。请检查并补齐。"
        )

    return wn, y, X


def sg_smooth(X: np.ndarray, window: int, poly: int, mode: str = "interp") -> np.ndarray:
    n_points = X.shape[1]
    window = ensure_odd_window(window, n_points)
    if poly >= window:
        raise ValueError(f"SG_POLY 必须 < SG_WINDOW（当前 poly={poly}, window={window}）")
    return savgol_filter(X, window_length=window, polyorder=poly, axis=1, mode=mode)


def mean_center(X: np.ndarray) -> np.ndarray:
    """按变量（每个波数点）做均值中心化：X - mean(X, axis=0)"""
    return X - X.mean(axis=0, keepdims=True)


def save_excel(wn, y, X, out_path: str):
    """保存为 Excel：第一列为分类，其余列为波数对应光谱"""
    out_df = pd.DataFrame(X, columns=wn)
    out_df.insert(0, "class", y)
    out_df.to_excel(out_path, index=False)


def plot_spectra(wn, y, X, title: str, out_png: str, max_lines: int = None):
    """
    画光谱图：
    - 默认画全部曲线（样本多会很密），你也可设置 max_lines 只画前N条
    """
    if max_lines is not None and X.shape[0] > max_lines:
        Xp = X[:max_lines]
        yp = y[:max_lines]
    else:
        Xp = X
        yp = y

    descending = wn[0] > wn[-1]  # 波数从大到小显示更常见（如红外）

    plt.figure(figsize=(10, 5))
    classes = pd.unique(yp)
    for cls in classes:
        idx = np.where(yp == cls)[0]
        for i in idx:
            plt.plot(wn, Xp[i, :], alpha=0.25, linewidth=1)

    plt.title(title)
    plt.xlabel("Wavenumber")
    plt.ylabel("Intensity")

    if descending:
        plt.gca().invert_xaxis()

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    wn, y, X = load_inputs()

    # 1) SG平滑
    X_sg = sg_smooth(X, SG_WINDOW, SG_POLY, SG_MODE)

    # 2) 均值中心化（MC）
    X_mc = mean_center(X)

    # 3) SG平滑 + 均值中心化（先SG后MC）
    X_sg_mc = mean_center(X_sg)

    # 保存结果
    save_excel(wn, y, X_sg,    os.path.join(OUT_DIR, "光谱_SG平滑.xlsx"))
    save_excel(wn, y, X_mc,    os.path.join(OUT_DIR, "光谱_均值中心化.xlsx"))
    save_excel(wn, y, X_sg_mc, os.path.join(OUT_DIR, "光谱_SG平滑_均值中心化.xlsx"))

    # 作图（如果样本很多可以限制 max_lines，例如 200）
    plot_spectra(wn, y, X_sg,    "Spectra after SG smoothing",       os.path.join(OUT_DIR, "SG平滑.png"))
    plot_spectra(wn, y, X_mc,    "Spectra after Mean Centering",     os.path.join(OUT_DIR, "均值中心化.png"))
    plot_spectra(wn, y, X_sg_mc, "Spectra after SG smoothing + MC",  os.path.join(OUT_DIR, "SG平滑_均值中心化.png"))

    print("完成！输出已保存到：", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
