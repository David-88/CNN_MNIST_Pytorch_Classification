# TASK:

## 实现一个手写数字自动识别系统。

ML：建议使用scikit-learn(sklearn)中提供的分类器来实现

数据集：统一采用mnist数据集MNIST手写数字数据集包含70000张手写数字图片。

这些数字是通过美国国家统计局的员工和美国高校的学生收集的。每张图片都是28x28的灰度图。



### 环境配置：

modelscope免费notebook:

[我的Notebook · 魔搭社区](https://modelscope.cn/my/mynotebook)

ubuntu22.04-py311-torch2.3.1-1.26.0 + CPU



# Machine learning Method

要实现一个手写数字自动识别系统，我们可以使用`scikit-learn`库中的分类器来训练和测试模型。

比较正常的想法：

- 读取数据（MNIST——读取有门道）

- 数据预处理
- 模型选择
- 模型调参
- 模型融合
- 模型评估

### **1. 环境准备**

确保已安装以下库：

```bash
pip install scikit-learn numpy matplotlib joblib
```

------

### **2. 加载MNIST数据集**

使用`idx2numpy`库加载MNIST数据集：

```bash
pip install idx2numpy
```

但是 `idx2numpy` 不支持直接读取 `.gz` 压缩文件。`.gz` 是 Gzip 压缩格式，而 `idx2numpy` 期望的是解压后的 `.ubyte` 文件。

需要先解压这些文件，然后再使用 `idx2numpy` 加载。

### **2.1 解压 `.gz` 文件**

使用 `gzip` 模块解压文件：

```python
import gzip
import shutil

# 解压函数
def decompress_gz(gz_file, output_file):
    with gzip.open(gz_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

# 解压所有文件
decompress_gz('MNIST_data/train-images-idx3-ubyte.gz', 'MNIST_data/train-images-idx3-ubyte')
decompress_gz('MNIST_data/train-labels-idx1-ubyte.gz', 'MNIST_data/train-labels-idx1-ubyte')
decompress_gz('MNIST_data/t10k-images-idx3-ubyte.gz', 'MNIST_data/t10k-images-idx3-ubyte')
decompress_gz('MNIST_data/t10k-labels-idx1-ubyte.gz', 'MNIST_data/t10k-labels-idx1-ubyte')
```

------

### **2.2 加载解压后的文件**

使用 `idx2numpy` 加载解压后的 `.ubyte` 文件：

```python
import idx2numpy

# 加载数据
train_images = idx2numpy.convert_from_file('MNIST_data/train-images-idx3-ubyte')
train_labels = idx2numpy.convert_from_file('MNIST_data/train-labels-idx1-ubyte')
test_images = idx2numpy.convert_from_file('MNIST_data/t10k-images-idx3-ubyte')
test_labels = idx2numpy.convert_from_file('MNIST_data/t10k-labels-idx1-ubyte')
```



### **3. 数据预处理**

#### **(1) 归一化**

将像素值缩放到`[0, 1]`：

```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

#### **(2) 标准化**

对数据进行标准化（均值为0，方差为1）：

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

#### **(3) 降维**

使用PCA（主成分分析）减少特征数量，保留95%的方差：

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95, random_state=42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
```



---

到这里，我们已经顺利完成了数据预处理、降维、归一化和标准化，接下来可以专注于 **XGBoost、LightGBM 和 CatBoost** 的调参与模型融合。



### **4. 选择模型**

避免使用简单模型（如SVM），选择以下高性能的树模型和集成方法：

- **随机森林（Random Forest）**
- **SVM**
- **KNN**
- **梯度提升树（Gradient Boosting）**

这些我不想用，我还是换成目前Kaggle上的常青藤：

- **XGBoost**
- **LightGBM**
- **CatBoost**



# EDA结束，下面是模型时间：

## **1. 模型训练**

### **(1) XGBoost**

```python
from xgboost import XGBClassifier

# 初始化模型
xgb_clf = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# 训练模型
xgb_clf.fit(X_train, y_train)
```

### **(2) LightGBM**

```python
from lightgbm import LGBMClassifier

# 初始化模型
lgbm_clf = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# 训练模型
lgbm_clf.fit(X_train, y_train)
```

**PS：如何是否真的在运行并查看进度？**

我们可以将 `LGBMClassifier` 中的 `verbose=0` 参数移除或设置为一个正数（例如 `verbose=1` 或 `verbose=100`）。设置为 `1` 通常会打印每一棵树的训练信息，这样我们就能清楚地看到训练进度了。

```
from lightgbm import LGBMClassifier

# 初始化模型，移除或更改 verbose 参数
lgbm_clf = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    # 移除 verbose=0，或者设置为 verbose=1 或更高
    # verbose=1
)

# 训练模型
print("Starting LightGBM training...") # 添加一个开始提示
lgbm_clf.fit(X_train, y_train)
print("LightGBM training finished.") # 添加一个结束提示
```

加上开始和结束提示以及调整 `verbose` 参数后，我们将能够更清楚地监控训练过程，并确认它是否确实在进行并最终完成。

训练 60,000 个样本、784个特征（经过 PCA 降维后可能特征数减少到 154 个左右，如输出所示）的模型需要一些时间，即使在 CPU 上也是如此。请耐心等待其完成。



### **(3) CatBoost**

```python
cat_clf = CatBoostClassifier(
    n_estimators=200,
    learning_rate=0.1,
    depth=5,
    subsample=0.8,
    rsm=0.8,
    random_state=42,
    verbose=0
)

```

------

## **2. 模型调参**

### **(1) 使用 GridSearchCV 进行调参**——我没用，但是可以换

```python
from sklearn.model_selection import GridSearchCV

# XGBoost 调参
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
xgb_grid = GridSearchCV(XGBClassifier(), xgb_param_grid, cv=3, n_jobs=-1)
xgb_grid.fit(X_train, y_train)
print(f"XGBoost 最佳参数: {xgb_grid.best_params_}")

# LightGBM 调参
lgbm_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
lgbm_grid = GridSearchCV(LGBMClassifier(), lgbm_param_grid, cv=3, n_jobs=-1)
lgbm_grid.fit(X_train, y_train)
print(f"LightGBM 最佳参数: {lgbm_grid.best_params_}")

# CatBoost 调参
cat_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
cat_grid = GridSearchCV(CatBoostClassifier(verbose=0), cat_param_grid, cv=3, n_jobs=-1)
cat_grid.fit(X_train, y_train)
print(f"CatBoost 最佳参数: {cat_grid.best_params_}")
```



### **(2) 使用 Optuna 进行自动调参**——我用了这个

好的！使用 Optuna 对 XGBoost、CatBoost 和 LightGBM 进行超参数调优是一个非常好的选择。Optuna 能够自动探索参数空间，找到能使模型在验证集上达到最佳性能的参数组合。

下面我将为我们展示如何使用 Optuna 分别对这三个模型进行调优。我们会定义一个目标函数，这个函数会接收 Optuna 提供的参数组合，构建并训练模型，然后在验证集上评估模型性能（例如使用交叉验证），最后返回性能指标（如准确率）。Optuna 会尝试最大化或最小化这个指标。

我们假设我们的数据已经被加载并分割为 `X_train`, `y_train`, `X_test`, `y_test` (或者我们会在 Optuna 内部使用交叉验证来评估参数)。

**准备工作：**

确保我们已经安装了必要的库：

```bash
pip install optuna xgboost lightgbm catboost scikit-learn numpy
```

**调优策略：**

1. **定义目标函数 (Objective Function)**: 这个函数是 Optuna 需要优化的核心。它接收一个 `trial` 对象，通过 `trial.suggest_*` 方法建议超参数值，构建模型，使用交叉验证在训练集上评估模型性能，然后返回评估指标（例如交叉验证的平均准确率）。
2. **创建 Study**: 使用 `optuna.create_study()` 创建一个研究对象。我们需要指定优化方向 (`direction='maximize'` 表示最大化指标，`'minimize'` 表示最小化指标)。
3. **运行优化**: 使用 `study.optimize()` 方法运行调优过程，指定运行的试验次数 (`n_trials`)。
4. **获取最佳参数**: 优化完成后，可以通过 `study.best_params` 获取找到的最佳参数组合。

------

### 1. XGBoost 调优示例

我们将对 XGBoost 的一些常用参数进行调优：`n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha` (L1 正则化), `reg_lambda` (L2 正则化)。

```python
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# 假设 X_train 和 y_train 已经准备好

def objective_xgb(trial):
    """Optuna 目标函数用于调优 XGBoost"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000), # 树的数量
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3), # 学习率
        'max_depth': trial.suggest_int('max_depth', 3, 10), # 树的最大深度
        'subsample': trial.suggest_float('subsample', 0.6, 1.0), # 样本采样比例
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), # 特征采样比例
        'gamma': trial.suggest_float('gamma', 0, 0.5), # 节点分裂所需的最小损失减少
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5), # L1 正则化项
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.5), # L2 正则化项
        'random_state': 42,
        'n_jobs': -1, # 使用所有可用的 CPU 核心
        'objective': 'multi:softmax', # 多分类任务
        'num_class': 10 # MNIST 有 10 个类别 (0-9)
    }

    model = XGBClassifier(**params)

    # 使用交叉验证评估模型，这里使用 3 折交叉验证，评估准确率
    # 可以根据任务类型和数据集大小调整 cv 参数和 scoring 指标
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)

    # Optuna 默认尝试最小化目标函数，所以返回负的准确率（或直接返回准确率并设置方向为 maximize）
    return score.mean() # Optuna 默认最小化，这里我们希望最大化准确率，所以要设置 study direction='maximize'

# 创建研究对象，方向为最大化（因为我们想最大化准确率）
study_xgb = optuna.create_study(direction='maximize')

# 运行优化，可以根据需要调整 n_trials（试验次数）
# 试验次数越多，找到最佳参数的概率越大，但耗时也越多
print("Starting XGBoost hyperparameter tuning with Optuna...")
study_xgb.optimize(objective_xgb, n_trials=50) # 运行 50 次试验

print("\nXGBoost 调优完成！")
print("最佳参数: ", study_xgb.best_params)
print("最佳准确率 (交叉验证): ", study_xgb.best_value)

# 使用找到的最佳参数训练最终模型
best_xgb_clf = XGBClassifier(**study_xgb.best_params, random_state=42, n_jobs=-1, objective='multi:softmax', num_class=10)
best_xgb_clf.fit(X_train, y_train)
```

------

### 2. LightGBM 调优示例

我们将对 LightGBM 的一些常用参数进行调优：`n_estimators`, `learning_rate`, `max_depth`, `num_leaves`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`。

```python
import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# 假设 X_train 和 y_train 已经准备好

def objective_lgbm(trial):
    """Optuna 目标函数用于调优 LightGBM"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000), # 树的数量
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3), # 学习率
        'max_depth': trial.suggest_int('max_depth', 3, 10), # 树的最大深度
        'num_leaves': trial.suggest_int('num_leaves', 8, 256), # 每棵树的最大叶子节点数
        'subsample': trial.suggest_float('subsample', 0.6, 1.0), # 样本采样比例
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), # 特征采样比例
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5), # L1 正则化项
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.5), # L2 正则化项
        'random_state': 42,
        'n_jobs': -1, # 使用所有可用的 CPU 核心
        'objective': 'multiclass', # 多分类任务
        'num_class': 10 # MNIST 有 10 个类别 (0-9)
    }

    model = LGBMClassifier(**params)

    # 使用交叉验证评估模型
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)

    return score.mean()

# 创建研究对象，方向为最大化
study_lgbm = optuna.create_study(direction='maximize')

# 运行优化
print("Starting LightGBM hyperparameter tuning with Optuna...")
study_lgbm.optimize(objective_lgbm, n_trials=50)

print("\nLightGBM 调优完成！")
print("最佳参数: ", study_lgbm.best_params)
print("最佳准确率 (交叉验证): ", study_lgbm.best_value)

# 使用找到的最佳参数训练最终模型
best_lgbm_clf = LGBMClassifier(**study_lgbm.best_params, random_state=42, n_jobs=-1, objective='multiclass', num_class=10)
best_lgbm_clf.fit(X_train, y_train)
```

------

### 3. CatBoost 调优示例

我们将对 CatBoost 的一些常用参数进行调优：`n_estimators` (`iterations`), `learning_rate`, `depth`, `subsample`, `rsm`, `l2_leaf_reg` (L2 正则化)，以及 `bootstrap_type` (这次我们知道需要设置为 'Bernoulli' 或 'MVS' 来使用 subsample)。

```python
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# 假设 X_train 和 y_train 已经准备好

def objective_cat(trial):
    """Optuna 目标函数用于调优 CatBoost"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000), # 树的数量
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3), # 学习率
        'depth': trial.suggest_int('depth', 3, 10), # 树的深度 (注意是 depth)
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bernoulli', 'MVS']), # 采样类型，只选择支持 subsample 的
        'subsample': trial.suggest_float('subsample', 0.6, 1.0), # 样本采样比例 (与 bootstrap_type='Bernoulli' 或 'MVS' 配合)
        'rsm': trial.suggest_float('rsm', 0.6, 1.0), # 特征采样比例 (相当于 colsample_bytree)
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10), # L2 正则化项
        'random_state': 42,
        'verbose': 0, # 隐藏 CatBoost 的训练日志
        'eval_metric': 'Accuracy', # 用于评估的指标
        'objective': 'MultiClass' # 多分类任务
    }

    # CatBoost 的 subsample 只在 bootstrap_type 是 Bernoulli 或 MVS 时有效
    # 虽然我们在 suggest_categorical 里已经限制了，但在组合参数时依然要注意
    if params['bootstrap_type'] == 'Poisson':
        del params['subsample'] # Poisson 不支持 subsample

    model = CatBoostClassifier(**params)

    # 使用交叉验证评估模型
    # 注意：CatBoost 的 cross_val_score 可能有自己的实现，但使用 sklearn 的通常也可以
    # 如果数据量大，或者想用 CatBoost 特有的 early stopping，可以考虑使用 CatBoost 的 cv 函数
    # 这里为了和前面保持一致，我们继续使用 sklearn 的 cross_val_score
    try:
        score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
        return score.mean()
    except Exception as e:
        # 有时候 CatBoost 在某些参数组合下可能会出错，这里可以捕获异常并返回一个极差的值
        print(f"Error during CatBoost trial: {e}")
        return 0.0 # 返回一个很低的准确率，让 Optuna 避免这个参数组合

# 创建研究对象，方向为最大化
study_cat = optuna.create_study(direction='maximize')

# 运行优化
print("Starting CatBoost hyperparameter tuning with Optuna...")
study_cat.optimize(objective_cat, n_trials=50)

print("\nCatBoost 调优完成！")
print("最佳参数: ", study_cat.best_params)
print("最佳准确率 (交叉验证): ", study_cat.best_value)

# 使用找到的最佳参数训练最终模型
# 注意：从 Optuna 得到的参数字典可以直接用于初始化 CatBoostClassifier
best_cat_clf = CatBoostClassifier(**study_cat.best_params, random_state=42, verbose=0, eval_metric='Accuracy', objective='MultiClass')
best_cat_clf.fit(X_train, y_train)
```

------

**注意事项和优化建议：**

1. **试验次数 (`n_trials`)**：`n_trials=50` 只是一个示例。对于复杂的模型和大型数据集，可能需要数百甚至数千次试验才能找到更好的参数。调优需要时间和计算资源。
2. **参数范围**：示例中的参数范围是基于经验设置的，我们可以根据模型的初始性能和我们对参数的理解来调整搜索范围。例如，如果初始学习率 `0.1` 表现不错，可以围绕它设置更窄的范围。
3. **交叉验证 (`cv`)**：交叉验证的折数 (`cv`) 越大，评估结果越稳定，但计算成本越高。常用的值是 3、5 或 10。
4. **评估指标 (`scoring`)**：对于分类任务，`'accuracy'` 是一个基本指标。根据我们的具体问题（例如类别不平衡），可能需要使用 `'f1_macro'`, `'roc_auc_ovr'` 等更合适的指标。
5. **并行计算 (`n_jobs`)**：在 `cross_val_score` 和模型初始化时设置 `n_jobs=-1` 可以利用所有 CPU 核心加速计算。
6. **Early Stopping**: 在 Optuna 目标函数内部的 `fit` 方法中，可以引入 Early Stopping。例如，如果我们有单独的验证集 `X_val`, `y_val`，可以在 fit 中设置 `eval_set=[(X_val, y_val)]` 和 `early_stopping_rounds`。或者在 `cross_val_score` 内部，可以使用 CatBoost 或 LightGBM 自带的交叉验证函数，它们通常更方便集成 Early Stopping。
7. **参数类型**: 注意不同模型参数的类型（整数、浮点数、类别）以及 CatBoost 中一些参数名的区别 (`depth` vs `max_depth`, `rsm` vs `colsample_bytree`, `iterations` vs `n_estimators`）。
8. **CatBoost 的 `subsample` 和 `bootstrap_type`**：再次强调，CatBoost 的 `subsample` 参数只有在 `bootstrap_type` 设置为 `'Bernoulli'` 或 `'MVS'` 时才有效。在 Optuna 中，可以通过 `suggest_categorical` 来限制 `bootstrap_type` 的选择范围。

通过运行这些调优脚本，我们将能够找到每个模型的最佳超参数，然后在模型融合时使用这些优化后的模型实例，从而最大化整体性能！



## 模型融合

在使用 Optuna 找到 XGBoost、LightGBM 和 CatBoost 的最佳参数后

下一步就是使用这些调优后的最佳模型实例来进行模型融合。

### VotingClassifier:

模型融合（如使用 `VotingClassifier`）的目的是结合不同模型的优势，通常能获得比任何单一模型更好的泛化性能和准确率。

我们将使用 `sklearn.ensemble.VotingClassifier`，并继续使用软投票 (`voting='soft'`)，因为它在分类任务中通常效果较好，会结合每个模型的预测概率。

**步骤：**

1. **运行 Optuna 调优**: 先运行前面提供的 Optuna 调优代码，获取每个模型的最佳参数组合 (`study_xgb.best_params`, `study_lgbm.best_params`, `study_cat.best_params`)。
2. **实例化最佳模型**: 使用 Optuna 找到的最佳参数来创建每个模型的实例。
3. **创建 VotingClassifier**: 将这些最佳模型实例放入一个列表中，然后创建 `VotingClassifier`。
4. **训练 VotingClassifier**: 使用整个训练集 (`X_train`, `y_train`) 训练 `VotingClassifier`。
5. **评估融合模型**: 在测试集 (`X_test`, `y_test`) 上评估融合模型的性能。

**代码示例：**

```python
# 假设我们已经运行了前面的 Optuna 调优代码
# 并且已经有了 study_xgb, study_lgbm, study_cat 的结果
# 尤其是它们的 study.best_params 属性

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# 假设 X_train, y_train, X_test, y_test 已经准备好

# ========================================================================
# 第 1 步：确保我们已经运行了 Optuna 调优部分，并获取了最佳参数
# 这里我们使用 Optuna 找到的 study_*.best_params
# 如果我们还没有运行 Optuna 或者只是想测试融合框架，
# 可以先用一些手动设定的参数代替 study_*.best_params 进行演示
# 例如:
# best_xgb_params = {'n_estimators': 300, 'learning_rate': 0.1, 'max_depth': 7, 'subsample': 0.9, 'colsample_bytree': 0.8, 'gamma': 0.1, 'reg_alpha': 0.2, 'reg_lambda': 0.3}
# best_lgbm_params = {'n_estimators': 250, 'learning_rate': 0.08, 'max_depth': 6, 'num_leaves': 64, 'subsample': 0.85, 'colsample_bytree': 0.9, 'reg_alpha': 0.1, 'reg_lambda': 0.2}
# best_cat_params = {'n_estimators': 280, 'learning_rate': 0.09, 'depth': 7, 'bootstrap_type': 'Bernoulli', 'subsample': 0.8, 'rsm': 0.85, 'l2_leaf_reg': 5}
# ========================================================================

# 从 Optuna study 中获取最佳参数
# ！！！请确保我们在运行此段代码前，先成功运行了上面的 Optuna 调优代码 ！！！
best_xgb_params = study_xgb.best_params
best_lgbm_params = study_lgbm.best_params
best_cat_params = study_cat.best_params

# CatBoost 需要一些额外的固定参数 (如 objective, eval_metric)
# 这些参数通常在调优时就固定下来，或者在构建最终模型时明确指定
# 这里我们沿用多分类任务的设置
cat_extra_params = {
    'objective': 'MultiClass',
    'eval_metric': 'Accuracy',
    'verbose': 0, # 最终模型也通常关闭日志
    'random_state': 42 # 也保持随机种子一致
}
# 将 Optuna 找到的 CatBoost 参数与额外的固定参数合并
# 如果有重叠，Optuna 的参数会覆盖这里的
final_best_cat_params = {**best_cat_params, **cat_extra_params}


# 第 2 步：使用最佳参数实例化每个模型
print("使用最佳参数实例化模型...")
best_xgb_clf = XGBClassifier(**best_xgb_params, random_state=42, n_jobs=-1, objective='multi:softmax', num_class=10)
best_lgbm_clf = LGBMClassifier(**best_lgbm_params, random_state=42, n_jobs=-1, objective='multiclass', num_class=10)
best_cat_clf = CatBoostClassifier(**final_best_cat_params) # CatBoost 参数已包含额外的固定参数

# 第 3 步：创建 VotingClassifier
print("创建 VotingClassifier...")
voting_clf = VotingClassifier(estimators=[
    ('xgb', best_xgb_clf),
    ('lgbm', best_lgbm_clf),
    ('cat', best_cat_clf)
], voting='soft', n_jobs=-1) # 使用软投票，利用所有核心并行训练基模型

# 第 4 步：训练 VotingClassifier
print("开始训练 VotingClassifier...")
# VotingClassifier 会依次训练列表中的每个模型，然后根据这些模型进行投票
# n_jobs=-1 在这里可以加速并行训练基模型
voting_clf.fit(X_train, y_train)
print("VotingClassifier 训练完成！")

# 第 5 步：评估融合模型
print("\n评估融合模型性能...")
y_pred = voting_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"融合模型 Accuracy: {accuracy:.4f}")

# 打印详细评估报告
print("\n融合模型 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n融合模型 Classification Report:\n", classification_report(y_test, y_pred))
```

**代码解释：**

1. 我们首先假设我们已经成功运行了 Optuna 调优过程，并从 `study_xgb`, `study_lgbm`, `study_cat` 中获取了 `.best_params`。
2. 我们使用这些最佳参数（以及一些固定的任务参数如 `objective`, `num_class`, `random_state`, `n_jobs` 等）来实例化 `XGBClassifier`, `LGBMClassifier`, `CatBoostClassifier` 对象。
3. 将这些实例化后的最佳模型对象作为元组（`('模型名称', 模型实例)`）放入一个列表中，然后传给 `VotingClassifier` 的 `estimators` 参数。
4. 设置 `voting='soft'` 表示进行软投票，即使用各模型的预测概率进行加权平均（如果权重相同，就是简单平均）。
5. `n_jobs=-1` 在 `VotingClassifier` 中可以让它并行地训练 `estimators` 列表中的模型，从而加快训练速度。
6. 最后，我们在测试集上进行预测并评估融合模型的性能。

运行这段代码，我们将得到使用 Optuna 优化后的 XGBoost, LightGBM, CatBoost 进行融合后的最终模型性能指标。通常，这个融合模型的性能会比任何一个单独的最佳模型都要好一些！

**甚至在目前结构化文本数据中这3者的融合结果都强于单个深度学习模型**



### Stacking:

是的，使用 **Stacking (堆叠)** 通常被认为是比简单的投票（Voting）**更强大**的一种模型融合技术，因为它不仅结合了不同模型的预测，还使用一个**元模型 (Meta-model)** 来学习如何最佳地组合这些预测。

**模型融合方式回顾：**

1. **平均/投票 (Averaging/Voting)**:
   - 软投票 (`voting='soft'`)：直接对各基模型的预测概率进行平均（或加权平均）。
   - 硬投票 (`voting='hard'`)：对各基模型的预测类别进行投票，选择得票最多的类别。
   - 优点：实现简单，计算量相对较小。
   - 缺点：组合方式是固定的（平均），无法学习基模型预测之间的复杂关系。
2. **Stacking (堆叠)**:
   - **两阶段过程**:
     - **第一阶段 (Base Models)**: 训练多个不同的基模型（如我们的 XGBoost, LightGBM, CatBoost）。为了避免信息泄露（即基模型在用于生成元特征的数据上进行了训练和预测），通常使用交叉验证或单独的验证集来生成基模型在训练数据上的预测。这些预测结果将作为第二阶段的输入。
     - **第二阶段 (Meta-model)**: 训练一个元模型。这个元模型以第一阶段基模型在训练数据上的预测结果（通常是概率）作为新的特征，原始训练数据的真实标签作为目标变量进行训练。元模型的任务是学习如何最好地结合基模型的预测来做出最终决策。
   - 优点：元模型可以学习基模型预测之间的非线性关系，从而找到更优的组合方式，通常能获得更好的性能。
   - 缺点：实现相对复杂，需要小心处理数据划分以防止信息泄露（尤其是在生成元特征时），训练时间更长（需要训练所有基模型和元模型），更容易过拟合（如果元模型过于复杂）。

**为什么说 Stacking 通常更好？**

因为 Stacking 引入了一个学习层 (元模型)，它能够识别不同基模型在哪些情况下表现更好或更差，并据此调整组合方式。而简单的投票只是机械地平均或统计。Stacking 允许模型之间的“对话”和学习，从而可能挖掘出更深层次的组合规律。

**使用 `sklearn.ensemble.StackingClassifier` 进行 Stacking：**

Scikit-learn 提供了 `StackingClassifier`，它简化了 Stacking 的实现过程，尤其是处理了生成元特征时的交叉验证部分。

我们需要提供：

1. `estimators`: 一组基模型。
2. `final_estimator`: 元模型。常用的元模型有 Logistic Regression, Ridge Classifier, 或甚至一个小的树模型（如 LightGBM 或 CatBoost），但通常建议选择相对简单的元模型以避免对元特征层过度拟合。Logistic Regression 是一个非常稳健且常用的选择。
3. `cv`: 用于生成元特征时的交叉验证折数。

**代码示例 (使用 Optuna 调优后的最佳模型)：**

```python
# 假设我们已经运行了前面的 Optuna 调优代码
# 并且已经有了 study_xgb, study_lgbm, study_cat 的结果
# 以及 X_train, y_train, X_test, y_test 数据

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression # 选用 Logistic Regression 作为元模型
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ========================================================================
# 第 1 步：确保我们已经运行了 Optuna 调优部分，并获取了最佳参数
# ！！！请确保我们在运行此段代码前，先成功运行了上面的 Optuna 调优代码 ！！！
# ========================================================================

# 从 Optuna study 中获取最佳参数
best_xgb_params = study_xgb.best_params
best_lgbm_params = study_lgbm.best_params
best_cat_params = study_cat.best_params

# CatBoost 需要一些额外的固定参数
cat_extra_params = {
    'objective': 'MultiClass',
    'eval_metric': 'Accuracy',
    'verbose': 0,
    'random_state': 42
}
final_best_cat_params = {**best_cat_params, **cat_extra_params}


# 第 2 步：使用最佳参数实例化基模型 (Base Estimators)
print("使用最佳参数实例化基模型...")
# 注意：StackingClassifier 会在内部使用 clone() 复制这些模型进行交叉验证
# 所以这里的 n_jobs=-1 可以确保基模型训练时是并行的
base_estimators = [
    ('xgb', XGBClassifier(**best_xgb_params, random_state=42, n_jobs=-1, objective='multi:softmax', num_class=10)),
    ('lgbm', LGBMClassifier(**best_lgbm_params, random_state=42, n_jobs=-1, objective='multiclass', num_class=10)),
    ('cat', CatBoostClassifier(**final_best_cat_params))
]

# 第 3 步：选择并实例化元模型 (Final Estimator)
# Logistic Regression 是一个不错的选择，因为它简单且不易过拟合元特征
# 对于多分类，LogisiticRegression 默认就是 OVR 或 Multinomial，对概率处理得很好
final_estimator = LogisticRegression(solver='liblinear', multi_class='auto', random_state=42, n_jobs=-1)
# 或者可以尝试其他简单的模型，比如 RidgeClassifier, Perceptron 等
# final_estimator = RidgeClassifier(random_state=42)

# 第 4 步：创建 StackingClassifier
# cv 参数指定用于生成元特征的交叉验证折数
print("创建 StackingClassifier...")
stacking_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=final_estimator,
    cv=5, # 使用 5 折交叉验证生成元特征
    n_jobs=-1 # 如果基模型不支持内部并行（但这里的都支持），这个参数可以并行化基模型的训练
)

# 第 5 步：训练 StackingClassifier
print("开始训练 StackingClassifier...")
# StackingClassifier 的 fit 方法会完成以下工作：
# 1. 使用指定的 cv 对训练数据进行划分
# 2. 在每个折叠上训练基模型
# 3. 使用训练好的基模型预测其他折叠的数据，生成元特征
# 4. 使用所有折叠生成的元特征和原始标签训练 final_estimator
# 5. 最后，在整个训练集上重新训练一遍基模型，用于对新数据进行预测
stacking_clf.fit(X_train, y_train)
print("StackingClassifier 训练完成！")

# 第 6 步：评估 Stacking 融合模型
print("\n评估 Stacking 融合模型性能...")
y_pred_stacking = stacking_clf.predict(X_test)
accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
print(f"Stacking 融合模型 Accuracy: {accuracy_stacking:.4f}")

# 打印详细评估报告
print("\nStacking 融合模型 Confusion Matrix:\n", confusion_matrix(y_test, y_pred_stacking))
print("\nStacking 融合模型 Classification Report:\n", classification_report(y_test, y_pred_stacking))
```

**重要提示：**

- **数据划分:** `StackingClassifier` 内部使用 `cv` 参数帮我们处理了生成元特征时的数据划分，这是防止信息泄露的关键。我们不需要手动创建 OOF (Out-of-Fold) 预测。
- **元模型选择:** 尝试不同的 `final_estimator`，但通常从 Logistic Regression 这样简单的模型开始是个好主意，因为它不易过拟合元特征。
- **计算成本:** Stacking 比简单的 Voting 需要更多的计算时间，因为它涉及交叉验证和训练一个额外的元模型。

总的来说，Stacking 通过引入学习层，能够更智能地组合基模型的预测，通常在性能上优于简单的投票。如果计算资源和时间允许，并且我们希望进一步提升模型性能，Stacking 是一个非常值得尝试的强大技术。





## 保存模型

### **(1) 保存融合模型**

```python
import joblib

# 保存 VotingClassifier
joblib.dump(voting_clf, 'mnist_voting_model.pkl')

# 保存 StackingClassifier
joblib.dump(stacking_clf, 'mnist_stacking_model.pkl')
```

### **(2) 加载模型**

```python
# 加载 VotingClassifier
loaded_model = joblib.load('mnist_voting_model.pkl')
```





完整代码见：mnist_classification.ipynb