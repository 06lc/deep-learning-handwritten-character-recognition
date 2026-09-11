# CASIA-HWDB-1.1 手写字符识别

这是一个基于 Python 3.13 和 PyTorch 2.9+ 的单字符中文手写识别系统。系统直接按需读取 `.gnt` 文件，不需要把约百万张样本转换成 JPEG。

## 数据路径

默认路径已经写入 `config.py`：

```text
F:\浏览器\Gnt1.1TrainPart1
F:\浏览器\Gnt1.1TrainPart2
F:\浏览器\Gnt1.1Test
```

训练集按 GNT 文件划分训练集和验证集，测试集始终保持独立。首次运行会在以下目录生成压缩索引：

```text
D:\Python+Ai\03_机器学习\手写字符识别\cache
```

HWDB-1.1 的默认类别数为 3926，包括 3755 个汉字和 171 个数字/符号。

## 安装

使用项目现有虚拟环境：

```powershell
D:\Python+Ai\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

GPU 版本的 PyTorch 请按照目标 CUDA 版本安装对应 wheel，再安装本项目依赖。

## 使用

先建立索引并检查文件：

```powershell
D:\Python+Ai\.venv\Scripts\python.exe "D:\Python+Ai\03_机器学习\手写字符识别\main.py" index
```

开始训练：

```powershell
D:\Python+Ai\.venv\Scripts\python.exe "D:\Python+Ai\03_机器学习\手写字符识别\main.py" train
```

默认配置包含 64×64 输入、AdamW、Cosine 学习率、AMP 和早停。训练输出默认写入：

```text
D:\Python+Ai\03_机器学习\手写字符识别\outputs
```

从已有 checkpoint 继续训练：

```powershell
D:\Python+Ai\.venv\Scripts\python.exe "D:\Python+Ai\03_机器学习\手写字符识别\main.py" train `
  --resume "D:\Python+Ai\03_机器学习\手写字符识别\outputs\last.pt"
```

在独立测试集上评估：

```powershell
D:\Python+Ai\.venv\Scripts\python.exe "D:\Python+Ai\03_机器学习\手写字符识别\main.py" evaluate `
  --checkpoint "D:\Python+Ai\03_机器学习\手写字符识别\outputs\best.pt"
```

预测单张图片或目录：

```powershell
D:\Python+Ai\.venv\Scripts\python.exe "D:\Python+Ai\03_机器学习\手写字符识别\main.py" predict `
  --checkpoint "D:\Python+Ai\03_机器学习\手写字符识别\outputs\best.pt" `
  --input "D:\Python+Ai\03_机器学习\手写字符识别\sample-pics\5.jpg" `
  --top-k 5
```

训练输出包括 `best.pt`、`last.pt`、`history.json`；评估输出包括 `metrics.json`。

## 开发检查

```powershell
D:\Python+Ai\.venv\Scripts\python.exe -m pytest -q
D:\Python+Ai\.venv\Scripts\python.exe -m ruff check .
D:\Python+Ai\.venv\Scripts\python.exe -m compileall -q .
```
