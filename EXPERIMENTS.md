# 单模型准确率实验手册

所有命令均在 `/root/autodl-tmp/hwdb-recognizer` 执行。保留
`outputs/full/best.pt` 作为 Top-1 96.2609% 的基线，实验期间只比较固定验证集。

## 0. 上传和烟测

上传本次变更的源码与测试后执行：

```bash
cd /root/autodl-tmp/hwdb-recognizer
mkdir -p outputs/accuracy
python -m pytest -q
python main.py train --device cuda --epochs 1 --warmup-epochs 0 \
  --max-train-batches 1 --max-eval-batches 1 \
  --output-dir outputs/accuracy/smoke
```

## 1. E1：训练策略

```bash
screen -dmS hwdb-e1 bash -lc 'cd /root/autodl-tmp/hwdb-recognizer && python -u main.py train --device cuda --model hccr_cnn9 --epochs 60 --patience 60 --batch-size 128 --num-workers 8 --learning-rate 3e-4 --warmup-epochs 5 --min-learning-rate 1e-6 --ema-decay 0.9999 --preprocess-profile legacy --augmentation-profile legacy --output-dir outputs/accuracy/e1-schedule > outputs/accuracy/e1-schedule.log 2>&1'
```

## 2. E2：安全边距与温和增强

```bash
screen -dmS hwdb-e2 bash -lc 'cd /root/autodl-tmp/hwdb-recognizer && python -u main.py train --device cuda --model hccr_cnn9 --epochs 60 --patience 60 --batch-size 128 --num-workers 8 --learning-rate 3e-4 --warmup-epochs 5 --min-learning-rate 1e-6 --ema-decay 0.9999 --preprocess-profile margin_v1 --augmentation-profile gentle_elastic --output-dir outputs/accuracy/e2-preprocess > outputs/accuracy/e2-preprocess.log 2>&1'
```

比较 E1/E2 的 `history.json`，后续实验使用验证 Top-1 更高的预处理组合。

## 3. E3：增强九层模型迁移训练

下面默认 E2 获胜；若 E1 更高，将两个 profile 改回 `legacy`：

```bash
screen -dmS hwdb-e3 bash -lc 'cd /root/autodl-tmp/hwdb-recognizer && python -u main.py train --device cuda --model hccr_cnn9_ra --warm-start-from outputs/full/best.pt --epochs 30 --patience 30 --batch-size 128 --num-workers 8 --learning-rate 1e-4 --warmup-epochs 2 --min-learning-rate 1e-6 --ema-decay 0.9999 --preprocess-profile margin_v1 --augmentation-profile gentle_elastic --output-dir outputs/accuracy/e3-ra-warm > outputs/accuracy/e3-ra-warm.log 2>&1'
```

## 4. E4：增强九层模型从头训练

以下命令同样假设 E2 获胜；若 E1 更高，将两个 profile 改回 `legacy`：

```bash
screen -dmS hwdb-e4 bash -lc 'cd /root/autodl-tmp/hwdb-recognizer && python -u main.py train --device cuda --model hccr_cnn9_ra --epochs 60 --patience 60 --batch-size 128 --num-workers 8 --learning-rate 3e-4 --warmup-epochs 5 --min-learning-rate 1e-6 --ema-decay 0.9999 --preprocess-profile margin_v1 --augmentation-profile gentle_elastic --output-dir outputs/accuracy/e4-ra-scratch > outputs/accuracy/e4-ra-scratch.log 2>&1'
```

## 5. 复验与最终测试

把 E1-E4 的最佳命令原样复制，仅增加 `--seed 2026`，将 screen 名称改为
`hwdb-e5`，并更换输出目录和日志路径为 `outputs/accuracy/e5-confirm` 与
`outputs/accuracy/e5-confirm.log`。两次固定验证结果分别达到 95.0332%，且平均达到
95.0832% 后，选择验证结果较高的 `best.pt` 执行一次最终测试：

```bash
python main.py evaluate --checkpoint outputs/accuracy/EXPERIMENT_NAME/best.pt \
  --device cuda --batch-size 128 --num-workers 8
```

若未通过验证门槛，第六次实验使用 `hccr_cnn9_ra_wide` 从头训练，其余参数与 E4 相同。
宽版仍未通过时结束本阶段，继续保留 `outputs/full/best.pt` 为正式模型。

## 6. HWDB1.0 兼容扩充实验

等待 E1-E5 全部结束后再上传覆盖服务器源码。先构建不影响基线缓存的扩充索引：

```bash
python main.py index --data-profile hwdb10_11_shared --rebuild-index
```

单批次烟测：

```bash
python main.py train --data-profile hwdb10_11_shared \
  --model hccr_cnn9 --finetune-from outputs/full/best.pt \
  --device cuda --epochs 1 --warmup-epochs 0 --batch-size 128 --num-workers 8 \
  --max-train-batches 1 --max-eval-batches 1 \
  --output-dir outputs/data-expansion/smoke
```

D1 使用全部兼容的 HWDB1.0 Train 样本和 HWDB1.1 训练部分：

```bash
mkdir -p outputs/data-expansion
screen -dmS hwdb-d1 bash -lc 'cd /root/autodl-tmp/hwdb-recognizer && python -u main.py train --data-profile hwdb10_11_shared --model hccr_cnn9 --finetune-from outputs/full/best.pt --device cuda --epochs 30 --patience 30 --batch-size 128 --num-workers 8 --learning-rate 1e-4 --warmup-epochs 2 --min-learning-rate 1e-6 --weight-decay 1e-4 --ema-decay 0.9999 --preprocess-profile legacy --augmentation-profile legacy --sampling-strategy natural --split-seed 42 --seed 42 --output-dir outputs/data-expansion/d1-warm > outputs/data-expansion/d1-warm.log 2>&1'
```

D1 固定验证 Top-1 达到 95.0332% 后，原样运行 D2，只把 `--seed` 改为 `2026`、
输出目录改为 `outputs/data-expansion/d2-confirm`。最终候选默认只评估 HWDB1.1 Test；
HWDB1.0 共享类别测试必须单独执行，两个指标不合并：

```bash
python main.py evaluate --checkpoint outputs/data-expansion/最佳实验/best.pt \
  --test-profile hwdb11 --device cuda --batch-size 128 --num-workers 8
python main.py evaluate --checkpoint outputs/data-expansion/最佳实验/best.pt \
  --test-profile hwdb10_shared --device cuda --batch-size 128 --num-workers 8
```
