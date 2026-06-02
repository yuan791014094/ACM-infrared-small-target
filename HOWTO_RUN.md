## 项目运行总指南 (HOWTO_RUN)

本文件是这个工作空间的**完整操作手册**：项目里有什么、哪些要跑、按什么顺序跑。
环境为 `acm` 虚拟环境（GPU / CUDA 12.1，RTX 4060 Laptop）。
**所有命令都在项目根目录 `D:\Desktop\doc\research\ACM` 下执行。**

配套文档：
- `SETUP.md` — 虚拟环境与依赖安装
- `IMPROVEMENTS.md` — 我提出的三项改进（ACM++）说明
- `INFERENCE.md` — 单张现实图片检测 + 图片该怎么拍
- 本文件 — 训练 / 评估 / 复现的总流程

---

### 0. 项目结构

```
ACM/
├── data/sirst/              SIRST 数据集（已就绪）
│   ├── images/              427 张红外图 (*.png)
│   ├── masks/               像素级标注 (*_pixels0.png)
│   └── idx_427/             train/val/test 划分 (*.txt)
├── src/
│   ├── models/
│   │   ├── acm.py           ACM 模块 + 消融变体 + 改进#1(空间门)
│   │   ├── backbone.py      ResNet-20 主干（可切下采样方案）
│   │   ├── networks.py      ACM-FPN / ACM-U-Net + 改进#2(深监督)
│   │   └── __init__.py      build_model 入口
│   ├── dataset.py           SIRST 数据加载（灰度化/缩放/归一化/翻转增强）
│   ├── metrics.py           Soft-IoU 损失、Focal、HybridLoss(改进#3)、IoU/nIoU
│   ├── train.py             训练（支持续训、改进开关）
│   ├── evaluate.py          test 集评估 + 可视化
│   ├── run_all.py           一键跑全套论文复现 + 消融
│   └── infer.py             单张图片检测（改进前后对比）
├── runs/                    训练产物（权重、history.json）
├── results/                 评估指标、可视化、单图检测结果
├── SETUP.md / IMPROVEMENTS.md / INFERENCE.md / HOWTO_RUN.md
└── requirements.txt
```

---

### 1. 准备与自检

```bash
conda activate acm
cd /d/Desktop/doc/research/ACM
```

确认 GPU 可用：

```bash
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望: cuda True NVIDIA GeForce RTX 4060 Laptop GPU
```

确认数据集就绪（无需再下载）：train 256 / val 85 / test 86，共 427 张。

```bash
python -c "import os;[print(s, sum(1 for _ in open(f'data/sirst/idx_427/{s}.txt'))) for s in ['train','val','test']]"
```

---

### 2. 论文配置（脚本默认值）

| 项 | 值 | 来源 |
|---|---|---|
| 输入分辨率 | 480×480 | 论文 Table 1 |
| 损失 | Soft-IoU | 论文 Sec 5.1 |
| 优化器 | Nesterov SGD | 论文 Sec 5.1 |
| 学习率 | 0.05（cosine 衰减） | 论文 Sec 5.1 |
| batch size | 8 | 论文 Sec 5.1 |
| epochs | 300 | 论文 Sec 5.1 |
| 权重初始化 | He (Kaiming) | 论文 Sec 5.1 |
| backbone | ResNet-20，下采样仅 /4 | 论文 Table 1 |

以上都是脚本默认值，直接跑即复现论文。

---

### 3. 该跑哪些？(总览)

整个项目要产出三类结果，建议**按顺序**跑：

| 阶段 | 目的 | 命令入口 | 大致耗时 |
|---|---|---|---|
| **A. 论文复现 + 消融** | 复现 ACM 主结果，验证非对称调制 / 下采样消融 | `run_all.py --preset full` | ~7.5h（6 个模型） |
| **B. 改进模型 (ACM++)** | 训练开启三项改进的模型，用于改进前后对比 | `train.py --improved` | ~1.5h（1 个模型） |
| **C. 单图检测** | 用一张现实图片对比改进前 vs 改进后 | `infer.py` | 秒级 |

> 时间紧可只跑关键项：A 阶段用 `--preset main`（2 个模型，~2.5h）+ B 阶段，
> 即可拿到"论文主结果 + 改进前后对比"两条主线。完整消融可后补。

---

### 4. 阶段 A：论文复现 + 消融（run_all.py）

`run_all.py` 自动「训练 → 在 test 集评估 → 汇总指标到 results/」。

#### 4.1 主结果（ACM-FPN + ACM-U-Net，对应论文 Table 3）

```bash
python src/run_all.py --preset main --epochs 300
```

- 训练 2 个模型，合计 **约 2.5 小时**。
- 产物：
  - `runs/acm_fpn/best.pth`、`runs/acm_unet/best.pth`
  - `results/acm_fpn/test_metrics.json`、`results/acm_unet/test_metrics.json`
  - `results/summary_main.json`（汇总 IoU / nIoU）
  - `results/<tag>/vis/*.png`（预测可视化）

#### 4.2 调制方案消融（论文 Table 2，FPN host）

```bash
python src/run_all.py --preset ablation --epochs 300
```

训练 `acm` / `topdown_local` / `bilocal` / `biglobal` 四种调制（acm 复用主结果，实际新增 3 个，约 4h）。
验证核心论点：**非对称调制 (ACM) > 双向同构 (BiLocal/BiGlobal) > 单向 (TopDownLocal)**。

#### 4.3 下采样方案消融（论文 Table 2）

```bash
python src/run_all.py --preset downsample --epochs 300
```

对比 `adjusted`（论文方案 /4）vs `regular`（标准 ResNet /16）。新增 1 个模型，约 1.5h。

#### 4.4 一次跑全（去重后 6 个模型）

```bash
python src/run_all.py --preset full --epochs 300
```

合计约 **7.5 小时**。跑完 `results/summary_full.json` 即为全部指标。

各 preset 实际训练的模型（去重后）：

| preset | 训练的模型(tag) |
|---|---|
| main | `acm_fpn`, `acm_unet` |
| ablation | `acm_fpn`, `acm_fpn_topdown_local`, `acm_fpn_bilocal`, `acm_fpn_biglobal` |
| downsample | `acm_fpn`, `acm_fpn_regular` |
| full | 以上去重 = 6 个 |

> 提速：加 `--epochs 100`（GPU 上常已接近收敛，指标略低于论文）。

---

### 5. 阶段 B：训练改进模型 ACM++（重点）

`run_all.py` 只跑论文配置；**改进模型要单独用 `--improved` 训练**（三项改进见 `IMPROVEMENTS.md`）。

```bash
# 改进前（论文 ACM）：阶段 A 已经训过 runs/acm_fpn/best.pth，无需重训
# 改进后（ACM++）：一次开启全部三项改进
python src/train.py --net acm_fpn --improved --epochs 300 --batch-size 8 --lr 0.05 --base-size 480
```

- 产物目录自动带 `_pp` 后缀：`runs/acm_fpn_pp/best.pth`、`last.pth`、`history.json`。
- `--improved` = `--spatial-gate` + `--deep-supervision` + `--hybrid-loss` 三个开关同时打开。
  也可单独开某一项做消融，例如只开混合损失：`--hybrid-loss`。

评估改进模型：

```bash
python src/evaluate.py --net acm_fpn --ckpt runs/acm_fpn_pp/best.pth --split test \
    --vis 12 --metrics-out results/acm_fpn_pp/test_metrics.json \
    --vis-dir results/acm_fpn_pp/vis
```

这样 `runs/acm_fpn`（改进前）与 `runs/acm_fpn_pp`（改进后）就成对齐全，可用于阶段 C 对比。

---

### 6. 阶段 C：单张现实图片检测（改进前 vs 改进后）

详见 `INFERENCE.md`（含"图片该拍成什么样"）。最常用的对比命令：

```bash
python src/infer.py --image my_shot.png --net acm_fpn \
    --ckpt-before runs/acm_fpn/best.pth \
    --ckpt-after  runs/acm_fpn_pp/best.pth \
    --out results/compare_my_shot.png
```

产物 `results/compare_my_shot.png`：2 行（改进前 / 改进后）× 4 列（输入 / 热力图 / 二值掩膜 / 检测框）。

---

### 7. 手动训练 / 评估（细粒度控制）

```bash
# 单独训练某个配置
python src/train.py --net acm_fpn --epochs 300 --batch-size 8 --lr 0.05 --base-size 480
python src/train.py --net acm_unet --epochs 300
python src/train.py --net acm_fpn --fusion bilocal    --epochs 300
python src/train.py --net acm_fpn --downsample regular --epochs 300

# 单独评估某个 checkpoint
python src/evaluate.py --net acm_fpn --ckpt runs/acm_fpn/best.pth --split test --vis 12
```

- `--fusion`：`acm`(默认) / `topdown_global` / `topdown_local` / `bilocal` / `biglobal`
- `--downsample`：`adjusted`(默认) / `regular`
- `--improved` / `--spatial-gate` / `--deep-supervision` / `--hybrid-loss`：改进开关（默认全关 = 论文原版）

训练产物按配置自动命名，如 `runs/acm_fpn_bilocal/`、`runs/acm_fpn_regular/`、`runs/acm_fpn_pp/`。

---

### 8. 续训（中途保存，可接着跑）

训练**每个 epoch**都会保存完整状态到 `runs/<tag>/last.pth`（含 optimizer、scheduler、epoch、best_niou）。
中断后接着跑：

```bash
# 自动从该 run 目录的 last.pth 续训
python src/train.py --net acm_fpn --epochs 300 --resume auto

# 或指定具体 checkpoint
python src/train.py --net acm_fpn --epochs 300 --resume runs/acm_fpn/last.pth
```

续训会恢复学习率调度和最优记录，从中断的下一个 epoch 继续，不会从头再来。
`best.pth` 始终保存**验证集 nIoU 最优**的权重。

---

### 9. 训练过程监控

每个 run 目录下 `history.json` 记录逐 epoch 的 train/val loss、IoU、nIoU。终端实时打印：

```
ep 100/300 | tr_loss 0.28 tr_IoU 0.71 | val_IoU 0.70 val_nIoU 0.72 | 15.0s
```

---

### 10. 跑完之后 → 出实验报告

把以下产物告诉我（已生成即可），我据此写中文实验报告并用 `/make-pdf` 生成 PDF（不是用 Python 生成）：

- `results/summary_*.json`（指标汇总）
- `results/*/test_metrics.json`（含 `acm_fpn_pp` 改进模型）
- `runs/*/history.json`（训练曲线）
- `results/*/vis/*.png`（可视化样例）
- `results/compare_my_shot.png`（单图改进前后对比）

报告将涵盖：论文复现结果对比、消融实验、三项改进(ACM++)的效果、单张现实图片的检测对比。

---

### 11. 常见问题

- **显存不足 (CUDA out of memory)**：`--batch-size` 调到 4 或 2，或 `--base-size` 调到 256。
- **训练太慢**：先 `--epochs 100` 跑通主结果，确认指标合理后再跑完整 300。
- **想重跑某配置**：删除对应 `runs/<tag>/` 目录即可（注意别误删已训好的）。
- **改进模型在哪**：必须手动 `--improved` 训练，目录是 `runs/acm_fpn_pp/`；`run_all.py` 不含改进模型。
- **infer.py 漏检 / 误检**：调 `--thresh`（漏检调低、误检调高）和 `--min-area`，详见 `INFERENCE.md`。

