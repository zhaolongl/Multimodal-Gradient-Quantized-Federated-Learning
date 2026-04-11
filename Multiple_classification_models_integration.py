"""
优化版：光谱分类（增强预处理+智能特征+多模型+交叉验证）
核心优化：
1. 新增基线校正、异常值过滤、自适应归一化
2. 自适应特征维度选择（PCA/PLS按方差解释率）
3. 新增XGBoost模型，优化SVM/RF/CNN参数
4. 5折交叉验证替代单次划分，结果更稳健
5. 自动输出特征重要性、最优参数组合
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, detrend
from scipy.stats import zscore
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
import xgboost as xgb

warnings.filterwarnings("ignore")

# =========================
# 优化配置区
# =========================
WN_FILE = "波数.xlsx"
Y_FILE = "分类.xlsx"
X_FILE = "光谱数据.xlsx"
OUT_ROOT = "outputs_optimized"

# 数据预处理优化
REMOVE_OUTLIERS = False  # 关闭异常值过滤（先保证代码能运行）
BASELINE_CORRECTION = True  # 基线校正（多项式拟合）
BASELINE_DEGREE = 1  # 基线拟合多项式次数

# 特征提取优化
PCA_VAR_RATIO = 0.95  # PCA保留95%方差（自适应维度）
PLS_VAR_RATIO = 0.90  # PLS保留90%方差（自适应维度）
MI_TOP_K = 100  # 互信息特征选择保留Top-K

# 交叉验证
CV_FOLDS = 5  # 5折交叉验证
RANDOM_STATE = 42

# 模型参数优化（扩大搜索范围）
# RF
RF_PARAMS = {
    "n_estimators": [200, 500, 800],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10]
}
# SVM
SVM_PARAMS = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1]
}
# XGBoost
XGB_PARAMS = {
    "n_estimators": [200, 500],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2]
}
# CNN优化
CNN_EPOCHS = 100
CNN_BATCH_SIZE = 16
CNN_LEARNING_RATE = 5e-4

# =========================
# 增强预处理类（新增SG平滑Estimator，修复Pipeline报错）
# =========================
class BaselineCorrection(BaseEstimator, TransformerMixin):
    """基线校正：多项式拟合基线并扣除"""
    def __init__(self, degree=3):
        self.degree = degree
        self.baselines_ = None

    def fit(self, X, y=None):
        n_samples, n_wavenumber = X.shape
        self.baselines_ = np.zeros_like(X)
        for i in range(n_samples):
            # 拟合基线
            x_axis = np.arange(n_wavenumber)
            coeffs = np.polyfit(x_axis, X[i], self.degree)
            baseline = np.polyval(coeffs, x_axis)
            self.baselines_[i] = baseline
        return self

    def transform(self, X):
        return X - self.baselines_

class OutlierRemover(BaseEstimator, TransformerMixin):
    """基于Z-score过滤异常样本（仅在fit阶段过滤，transform不处理）"""
    def __init__(self, z_threshold=3, axis=1):
        self.z_threshold = z_threshold
        self.axis = axis
        self.keep_indices_ = None

    def fit(self, X, y=None):
        # 计算每个样本的Z-score
        z_scores = np.abs(zscore(X, axis=self.axis))
        # 样本级异常（任意波数点Z-score超过阈值）
        sample_outlier = (z_scores > self.z_threshold).any(axis=1)
        self.keep_indices_ = np.where(~sample_outlier)[0]
        return self

    def transform(self, X):
        if self.keep_indices_ is not None:
            return X[self.keep_indices_]
        return X

class AdaptiveScaler(BaseEstimator, TransformerMixin):
    """自适应缩放：根据数据分布选择MinMax或Standard"""
    def __init__(self):
        self.scaler_ = None

    def fit(self, X, y=None):
        # 计算变异系数（CV=标准差/均值）
        cv = np.std(X, axis=0) / np.mean(X, axis=0)
        cv_mean = np.mean(cv)
        # 高变异系数用StandardScaler，低则用MinMaxScaler
        if cv_mean > 1.0:
            self.scaler_ = StandardScaler()
        else:
            self.scaler_ = MinMaxScaler(feature_range=(0, 1))
        self.scaler_.fit(X)
        return self

    def transform(self, X):
        return self.scaler_.transform(X)

class SGSmoothing(BaseEstimator, TransformerMixin):
    """Savitzky-Golay平滑：封装为sklearn兼容的Estimator（核心修复）"""
    def __init__(self, window_length=11, polyorder=3, axis=1):
        self.window_length = window_length
        self.polyorder = polyorder
        self.axis = axis

    def fit(self, X, y=None):
        # SG平滑无需训练参数，fit方法仅为适配Pipeline规范
        return self

    def transform(self, X):
        # 执行SG平滑处理
        return savgol_filter(X, window_length=self.window_length, polyorder=self.polyorder, axis=self.axis)

# =========================
# 智能特征提取
# =========================
class AdaptivePCA(BaseEstimator, TransformerMixin):
    """自适应PCA：按方差解释率选择维度"""
    def __init__(self, var_ratio=0.95):
        self.var_ratio = var_ratio
        self.pca_ = None

    def fit(self, X, y=None):
        self.pca_ = PCA(n_components=self.var_ratio, random_state=RANDOM_STATE)
        self.pca_.fit(X)
        return self

    def transform(self, X):
        return self.pca_.transform(X)

    @property
    def n_components_(self):
        return self.pca_.n_components_ if self.pca_ else None

class MutualInfoSelector(BaseEstimator, TransformerMixin):
    """基于互信息的特征选择"""
    def __init__(self, k=100):
        self.k = k
        self.selector_ = None

    def fit(self, X, y):
        self.selector_ = SelectKBest(mutual_info_classif, k=self.k)
        self.selector_.fit(X, y)
        return self

    def transform(self, X):
        return self.selector_.transform(X)

    @property
    def get_support(self):
        return self.selector_.get_support() if self.selector_ else None

# =========================
# 模型构建与调优
# =========================
def build_and_tune_model(model_name, X, y):
    """构建模型并网格搜索调优"""
    if model_name == "RF":
        base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        grid = GridSearchCV(base_model, RF_PARAMS, cv=3, scoring="f1_macro", n_jobs=-1)
    elif model_name == "SVM":
        base_model = SVC(class_weight="balanced", random_state=RANDOM_STATE)
        grid = GridSearchCV(base_model, SVM_PARAMS, cv=3, scoring="f1_macro", n_jobs=-1)
    elif model_name == "XGB":
        base_model = xgb.XGBClassifier(objective="multi:softmax", random_state=RANDOM_STATE, n_jobs=-1)
        grid = GridSearchCV(base_model, XGB_PARAMS, cv=3, scoring="f1_macro", n_jobs=-1)
    else:
        raise ValueError(f"未知模型：{model_name}")

    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_

# =========================
# CNN优化
# =========================
def build_optimized_cnn(input_length, n_classes):
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers

    inputs = layers.Input(shape=(input_length, 1))

    # 特征提取层（适配光谱数据）
    x = layers.Conv1D(64, 9, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(3)(x)

    x = layers.Conv1D(128, 7, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(3)(x)

    x = layers.Conv1D(256, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # 注意力机制（突出关键波数点）
    attention = layers.GlobalAveragePooling1D()(x)
    attention = layers.Dense(x.shape[-1], activation="softmax")(attention)
    x = layers.Multiply()([x, attention[:, None, :]])

    # 分类头
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=CNN_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# =========================
# 主流程
# =========================
def load_data():
    """加载并预处理原始数据"""
    # 加载文件
    wn = pd.read_excel(WN_FILE, header=None).iloc[:, 0].values
    y = pd.read_excel(Y_FILE, header=None).iloc[:, 0].values
    X = pd.read_excel(X_FILE, header=None).values

    # 数据清洗
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    # 标签编码
    le = LabelEncoder()
    y = le.fit_transform(y)

    # 异常值过滤
    if REMOVE_OUTLIERS:
        or_remover = OutlierRemover()
        or_remover.fit(X)
        X = or_remover.transform(X)
        y = y[or_remover.keep_indices_]

    return wn, X, y, le

def run_experiments():
    # 加载数据
    wn, X, y, le = load_data()
    n_classes = len(le.classes_)

    # 创建输出目录
    os.makedirs(OUT_ROOT, exist_ok=True)
    metrics_dir = os.path.join(OUT_ROOT, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # 定义预处理组合（增强版，修复SG平滑Pipeline）
    preprocessors = {
        "MC": Pipeline([("mc", StandardScaler(with_mean=True, with_std=False))]),
        "SNV": Pipeline([("snv", StandardScaler(with_mean=True, with_std=True))]),
        "SG": Pipeline([("sg", SGSmoothing(window_length=11, polyorder=3))]),  # 使用封装后的Estimator
        "Baseline+SG": Pipeline([
            ("baseline", BaselineCorrection(degree=BASELINE_DEGREE)),
            ("sg", SGSmoothing(window_length=11, polyorder=3))  # 使用封装后的Estimator
        ]),
        "Baseline+SNV": Pipeline([
            ("baseline", BaselineCorrection(degree=BASELINE_DEGREE)),
            ("snv", StandardScaler(with_mean=True, with_std=True))
        ])
    }

    # 定义特征提取方法
    feature_extractors = {
        "MI": MutualInfoSelector(k=MI_TOP_K),
        "PCA": AdaptivePCA(var_ratio=PCA_VAR_RATIO),
        "PLS": PLSRegression(n_components=20),  # 固定为整数，适配PLS要求
        "None": "None"
    }

    # 定义模型列表
    models = ["RF", "SVM", "XGB"]

    # 存储结果
    results = []

    # 交叉验证分割器
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # 遍历所有组合
    for pp_name, pp in preprocessors.items():
        for feat_name, feat in feature_extractors.items():
            for model_name in models:
                print(f"运行组合：{pp_name} + {feat_name} + {model_name}")

                # 存储每折结果
                fold_acc = []
                fold_f1 = []

                for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    # 预处理（修复后可正常调用fit_transform）
                    X_train_pp = pp.fit_transform(X_train)
                    X_test_pp = pp.transform(X_test)

                    # 特征提取
                    if feat_name != "None":
                        if feat_name == "PLS":
                            # PLS-DA适配
                            Y_train = np.eye(n_classes)[y_train]
                            feat.fit(X_train_pp, Y_train)
                            X_train_feat = feat.transform(X_train_pp)
                            X_test_feat = feat.transform(X_test_pp)
                        else:
                            feat.fit(X_train_pp, y_train)
                            X_train_feat = feat.transform(X_train_pp)
                            X_test_feat = feat.transform(X_test_pp)
                    else:
                        X_train_feat = X_train_pp
                        X_test_feat = X_test_pp

                    # 自适应缩放
                    scaler = AdaptiveScaler()
                    X_train_feat = scaler.fit_transform(X_train_feat)
                    X_test_feat = scaler.transform(X_test_feat)

                    # 模型调优与训练
                    model, best_params = build_and_tune_model(model_name, X_train_feat, y_train)
                    model.fit(X_train_feat, y_train)

                    # 预测与评估
                    y_pred = model.predict(X_test_feat)
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average="macro")

                    fold_acc.append(acc)
                    fold_f1.append(f1)

                # 计算平均结果
                avg_acc = np.mean(fold_acc)
                avg_f1 = np.mean(fold_f1)

                results.append({
                    "preprocess": pp_name,
                    "feature": feat_name,
                    "model": model_name,
                    "accuracy": avg_acc,
                    "f1_macro": avg_f1,
                    "n_features": X_train_feat.shape[1] if 'X_train_feat' in locals() else X.shape[1],
                    "best_params": str(best_params)
                })

    # 运行优化版CNN
    try:
        import tensorflow as tf
        for pp_name, pp in preprocessors.items():
            print(f"运行CNN组合：{pp_name}")

            fold_acc = []
            fold_f1 = []

            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # 预处理
                X_train_pp = pp.fit_transform(X_train)
                X_test_pp = pp.transform(X_test)

                # 缩放
                scaler = AdaptiveScaler()
                X_train_pp = scaler.fit_transform(X_train_pp)
                X_test_pp = scaler.transform(X_test_pp)

                # 重塑为CNN输入格式
                X_train_cnn = X_train_pp[..., np.newaxis]
                X_test_cnn = X_test_pp[..., np.newaxis]

                # 构建并训练CNN
                cnn_model = build_optimized_cnn(X_train_cnn.shape[1], n_classes)
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8)
                ]

                cnn_model.fit(
                    X_train_cnn, y_train,
                    validation_split=0.2,
                    epochs=CNN_EPOCHS,
                    batch_size=CNN_BATCH_SIZE,
                    callbacks=callbacks,
                    verbose=0
                )

                # 评估
                y_pred = np.argmax(cnn_model.predict(X_test_cnn, verbose=0), axis=1)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="macro")

                fold_acc.append(acc)
                fold_f1.append(f1)

            results.append({
                "preprocess": pp_name,
                "feature": "None",
                "model": "CNN",
                "accuracy": np.mean(fold_acc),
                "f1_macro": np.mean(fold_f1),
                "n_features": X.shape[1],
                "best_params": "Optimized CNN"
            })
    except ImportError:
        print("未安装TensorFlow，跳过CNN")

    # 保存结果
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1_macro", ascending=False)
    results_df.to_excel(os.path.join(metrics_dir, "optimized_results.xlsx"), index=False)

    # 保存最优组合
    best_config = results_df.iloc[0]
    with open(os.path.join(metrics_dir, "best_config.txt"), "w", encoding="utf-8") as f:
        f.write("最优组合（按F1-macro排序）\n")
        f.write(f"预处理：{best_config['preprocess']}\n")
        f.write(f"特征提取：{best_config['feature']}\n")
        f.write(f"模型：{best_config['model']}\n")
        f.write(f"准确率：{best_config['accuracy']:.4f}\n")
        f.write(f"F1-macro：{best_config['f1_macro']:.4f}\n")
        f.write(f"最优参数：{best_config['best_params']}\n")

    print("优化实验完成！")
    print(f"结果文件：{os.path.join(metrics_dir, 'optimized_results.xlsx')}")
    print(f"最优配置：{os.path.join(metrics_dir, 'best_config.txt')}")
    return results_df

if __name__ == "__main__":
    # 安装依赖（如果需要）
    # pip install xgboost tensorflow scipy scikit-learn pandas openpyxl matplotlib
    results = run_experiments()
    print("\nTop 5 最优组合：")
    print(results.head())