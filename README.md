# CASIA-HWDB-1.1 手写字符识别

这是一个面向 AutoDL GPU 的 PyTorch 单字符中文手写识别系统。默认采用论文
HCCR-CNN9Layer：`96x96` 输入、7 个卷积层、PReLU、1024 维全连接层和 3926 类输出。

## 数据布局

训练集和测试集必须放在项目目录内，目录名保持不变：

```text
项目目录/Gnt1.1TrainPart1  # 120 个 GNT 文件
项目目录/Gnt1.1TrainPart2  # 120 个 GNT 文件
项目目录/Gnt1.1Test        # 60 个 GNT 文件
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

## 评估和预测

```bash
python main.py evaluate --checkpoint outputs/best.pt --device cuda
python main.py predict --checkpoint outputs/best.pt --input sample-pics/5.jpg --top-k 5
```

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
