# CASIA-HWDB-1.1 手写字符识别

这是一个面向 AutoDL GPU 的 PyTorch 单字符中文手写识别系统。默认采用论文
HCCR-CNN9Layer：`96x96` 输入、7 个卷积层、PReLU、1024 维全连接层和 3926 类输出。

## 当前最佳模型

当前已验证最佳模型为 HWDB1.0 兼容数据扩充实验 D1：

```text
模型：HCCR-CNN9Layer
输入：96x96
类别：3926
起始权重：outputs/full/best.pt
训练数据：844,792 个 HWDB1.1 样本 + 1,303,957 个兼容 HWDB1.0 样本
固定验证集：93,887 个 HWDB1.1 样本
最佳轮次：30
固定验证 Top-1：95.5521%
固定验证 Top-5：99.2139%
HWDB1.1 官方 Test Top-1：96.6592%
HWDB1.1 官方 Test Top-5：99.5299%
HWDB1.1 官方 Test Loss：0.20338
HWDB1.0 共有类别 Test Top-1：97.5525%
checkpoint：outputs/data-expansion/d1-warm/best.pt
```

原始 HWDB1.1 基线的官方 Test Top-1 为 `96.2609%`。D1 提高了约 `0.3983`
个百分点；两个数据集的 Test 指标独立报告，不合并计算。checkpoint 位于 AutoDL
训练输出目录，不纳入 Git。

完整的四轮训练配置、数据规模和同口径指标对比见 [`MODEL_COMPARISON.md`](MODEL_COMPARISON.md)。

## 数据布局

训练集和测试集必须放在项目目录内，目录名保持不变：

```text
项目目录/Gnt1.1TrainPart1  # 120 个 GNT 文件
项目目录/Gnt1.1TrainPart2  # 120 个 GNT 文件
项目目录/Gnt1.1Test        # 60 个 GNT 文件
项目目录/Gnt1.0TrainPart1  # 112 个 GNT 文件，可选扩充数据
项目目录/Gnt1.0TrainPart2  # 112 个 GNT 文件，可选扩充数据
项目目录/Gnt1.0TrainPart3  # 112 个 GNT 文件，可选扩充数据
项目目录/Gnt1.0Test        # 84 个 GNT 文件，仅作外部测试
项目目录/competition-gnt   # ICDAR-2013 比赛测试集，60 个 GNT 文件
```

程序通过 `Path(__file__).resolve().parent` 自动定位项目目录，不依赖 `D:\`、`F:\` 或
AutoDL 的固定 `/root` 路径。训练集只来自两个 TrainPart，Test 只用于最终评估。

## AutoDL 环境

推荐使用 PyTorch 2.8 / Python 3.12 / CUDA 12.8 基础镜像：

```bash
python -m pip install -e ".[dev]"
```

需要 ONNX 导出时再安装部署依赖：

```bash
python -m pip install -e ".[deploy]"
```

如果源码和 GNT 数据在 `/root/autodl-tmp/手写字符识别`，不需要修改配置，直接进入该目录运行命令。

## 建立索引

```bash
python main.py index
```

首次运行会扫描 300 个 GNT 文件并生成 `cache/train.npz`、`cache/test.npz`。索引只保存文件路径、
偏移、尺寸和标签，不生成上百万张中间图片。

默认 `--data-profile hwdb11` 完全保持原基线。兼容扩充索引使用 HWDB1.1 的 3926 类
顺序，过滤 HWDB1.0 独有类别，并保存到独立缓存：

```bash
python main.py index --data-profile hwdb10_11_shared --rebuild-index
```

该命令不会覆盖 `cache/train.npz`，扩充索引保存为
`cache/train_hwdb10_11_shared.npz`。

## 训练

默认使用 HCCR-CNN9Layer、`96x96`、3926 类、AdamW、5 轮学习率热身、
Cosine 学习率、EMA 权重平均、AMP 和断点恢复：

```bash
python main.py train --device cuda --batch-size 128 --num-workers 8 --epochs 60
```

首次训练会创建 `cache/validation_split.json`。它使用训练目录编号和相对路径记录固定验证文件，
后续实验即使更换训练随机种子也使用同一验证集。

安全边距和温和弹性增强：

```bash
python main.py train --device cuda --epochs 60 --patience 60 \
  --preprocess-profile margin_v1 --augmentation-profile gentle_elastic
```

从原始九层模型向残差注意力九层模型迁移权重：

```bash
python main.py train --device cuda --model hccr_cnn9_ra \
  --warm-start-from outputs/full/best.pt --learning-rate 1e-4 \
  --warmup-epochs 2 --epochs 30 --patience 30
```

`--resume`、`--finetune-from`、`--warm-start-from` 三者互斥：前者完整恢复训练状态，
第二个只加载同结构推理权重，第三个将原始九层权重迁移到增强九层模型。

论文风格 SGD 对照实验：

```bash
python main.py train --device cuda --model hccr_cnn9 --image-size 96 --recipe paper
```

输出位于 `outputs/`：`best.pt`、`last.pt`、`history.json`。

## HWDB1.0 兼容扩充训练

扩充模式同时使用 HWDB1.1 的训练部分和 HWDB1.0 三个 TrainPart 的全部兼容样本；
固定的 93,887 个 HWDB1.1 验证样本继续只用于验证。首次扩充训练必须从原始最佳模型
重新建立优化器：

```bash
python main.py train \
  --data-profile hwdb10_11_shared \
  --model hccr_cnn9 \
  --finetune-from outputs/full/best.pt \
  --device cuda --epochs 30 --patience 30 \
  --learning-rate 1e-4 --warmup-epochs 2 \
  --sampling-strategy natural \
  --output-dir outputs/data-expansion/d1-warm
```

`--sampling-strategy class-balanced` 使用逆类别频率采样并把最大倍率限制为 2.0，
用于保护只存在于 HWDB1.1 的稀有类别。训练中断后可以使用扩充实验自身的
`last.pt` 配合 `--resume` 恢复；原始 HWDB1.1 基线应使用 `--finetune-from`。

## 评估和预测

```bash
python main.py evaluate --checkpoint outputs/best.pt --device cuda
python main.py evaluate --checkpoint outputs/best.pt --test-profile hwdb10_shared --device cuda
python main.py evaluate --checkpoint outputs/best.pt --test-profile icdar2013 --device cuda
python main.py predict --checkpoint outputs/best.pt --input path/to/your-image.jpg --top-k 5
```

`icdar2013` 使用论文对应的 ICDAR-2013 Offline HCCR Competition 测试集，
索引单独保存为 `cache/test_icdar2013.npz`，不会覆盖 HWDB1.1 的 `cache/test.npz`，
也不会进入训练或固定验证集。

错误分析默认使用固定验证集，不会访问 Test：

```bash
python main.py analyze --checkpoint outputs/best.pt --device cuda \
  --output-dir outputs/best-analysis --max-errors 100
```

报告包含类别准确率、主要混淆对、高置信错误和对应原始字符图片。只有最终候选模型才使用
`--split test`。完整实验顺序见 `EXPERIMENTS.md`。

## 压缩

压缩顺序为 GSLRE 空间低秩分解、ADW 渐进剪枝、可移植权重量化。每一阶段都会独立保存：

```bash
python main.py compress \
  --checkpoint outputs/best.pt \
  --stages gslre adw quantize \
  --device cuda \
  --rank-ratio 0.5 \
  --sparsity 0.3 \
  --clusters 256 \
  --finetune-epochs 3
```

产物位于 `outputs/compressed/`：

```text
gslre.pt
adw.pt
int8.pt
compression_report.json
```

GSLRE 将 `3x3` 卷积分解为 `3x1 -> 1x3`，并使用 SVD 初始化后再进行监督微调。ADW 使用固定
mask 保持剪枝连接为零。INT8 文件保存 `uint8` 权重索引和量化中心，加载时自动重建模型。

## 导出和基准测试

```bash
python main.py benchmark --checkpoint outputs/compressed/int8.pt --device cuda
python main.py export --checkpoint outputs/compressed/int8.pt --output-dir outputs/exported
python main.py export --checkpoint outputs/compressed/int8.pt --output-dir outputs/exported --onnx
```

`benchmark.json` 会记录延迟、吞吐、参数量和稀疏率。导出目录包含 TorchScript 和 `fp16.pt`；ONNX
需要镜像中安装对应依赖。TensorRT 引擎可在 AutoDL 上基于导出的 ONNX 继续构建。

## 检查

```bash
python -m pytest -q
python -m ruff check .
python -m compileall -q .
```

本地测试使用合成 GNT 数据，不会自动训练项目内的完整数据集。完整训练和压缩应在 AutoDL GPU 上执行。
