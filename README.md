# ACM: 红外小目标检测 (复现 + 改进)

复现 **WACV 2020** 论文 *"Asymmetric Contextual Modulation for Infrared Small Target Detection"* (ACM)，
并在其基础上提出三项轻量改进 **ACM++**，附带单张现实图片的检测框架。

> 数据集：SIRST（427 张红外图，像素级标注）。
> 环境：Python 3.9 + PyTorch 2.5.1 + CUDA 12.1（RTX 4060 Laptop 验证）。

---

## 核心方法

ACM 的核心是**非对称上下文调制** (Eq. 5)：

```
Z = G(Y) ⊗ X  +  L(X) ⊗ Y
```

- `G(Y)`：自顶向下的**全局通道注意力**（高层语义 → 调制低层细节）
- `L(X)`：自底向上的**点式通道注意力**（低层细节 → 调制高层语义）

两条路径**不对称**，分别解决"高层语义定位"与"低层细节保留"，再融合。
主干为下采样受限的 ResNet-20（总降采样仅 /4，保住小目标），
分别接到 **ACM-FPN** 与 **ACM-U-Net** 两种结构。

---

## 我的改进：ACM++

三项改进**全部可开关**，不开 = 论文原版，开 = ACM++，对比公平。详见 [IMPROVEMENTS.md](IMPROVEMENTS.md)。

| # | 改进 | 解决的问题 | 开销 |
|---|---|---|---|
| 1 | **空间注意力门** (CBAM 式) | ACM 全是通道注意力，缺显式空间定位 | +98 参数/模块 |
| 2 | **深监督** (辅助分割头) | 小目标深层梯度弱 | 一个 1×1 卷积 |
| 3 | **混合损失** (Soft-IoU + Focal) | 极端类别不平衡、暗弱目标 | 0 参数 |

> 三项合计 ACM-FPN 仅 **+213 参数**（278,689 → 278,902，< 0.08%）。

---

## 快速开始

详细步骤见 [HOWTO_RUN.md](HOWTO_RUN.md)，环境安装见 [SETUP.md](SETUP.md)。

```bash
# 1. 装环境（torch 用 cu121 官方源，其余清华源）见 SETUP.md
# 2. 复现 + 消融（6 个模型，自动训练→评估→汇总）
python src/run_all.py --preset full --epochs 300

# 3. 训练改进模型 ACM++
python src/train.py --net acm_fpn --improved --epochs 300

# 4. 单张现实图片：改进前 vs 改进后
python src/infer.py --image my_shot.png \
    --ckpt-before runs/acm_fpn/best.pth \
    --ckpt-after  runs/acm_fpn_pp/best.pth
```

时间紧可用 `--epochs 100` 或 `--preset main`（仅主结果）。

---

## 项目结构

```
ACM/
├── data/sirst/          SIRST 数据集（自行下载，见 SETUP.md）
├── src/
│   ├── models/
│   │   ├── acm.py        ACM 模块 + 消融变体 + 改进#1(空间门)
│   │   ├── backbone.py   ResNet-20 主干（可切下采样方案）
│   │   └── networks.py   ACM-FPN / ACM-U-Net + 改进#2(深监督)
│   ├── dataset.py        SIRST 数据加载
│   ├── metrics.py        Soft-IoU 损失、Focal、HybridLoss(改进#3)、IoU/nIoU
│   ├── train.py          训练（支持续训、改进开关）
│   ├── evaluate.py       test 集评估 + 可视化
│   ├── run_all.py        一键复现 + 消融
│   └── infer.py          单张图片检测（改进前后对比）
├── runs/                 训练产物（权重，不进库）
├── results/              指标 / 可视化 / 单图检测结果
├── SETUP.md              环境安装
├── HOWTO_RUN.md          完整运行指南（该跑什么、什么顺序）
├── IMPROVEMENTS.md       三项改进说明
└── INFERENCE.md          单图检测 + 现实图片拍摄要求
```

---

## 实验结果

> 训练完成后填入实测数值（test 集，nIoU 为论文主指标 Eq. 1）。

### 主结果

| 模型 | 配置 | IoU | nIoU |
|---|---|---|---|
| ACM-FPN | 论文 ACM | _待填_ | _待填_ |
| ACM-U-Net | 论文 ACM | _待填_ | _待填_ |
| ACM-FPN | **ACM++（改进后）** | _待填_ | _待填_ |

### 消融：调制方案（ACM-FPN host，对应论文 Table 2）

| 方案 | 说明 | nIoU |
|---|---|---|
| TopDownLocal | 单向 | _待填_ |
| BiLocal | 双向局部 | _待填_ |
| BiGlobal | 双向全局 | _待填_ |
| **ACM** | **非对称（本文）** | _待填_ |

### 消融：下采样方案

| 方案 | 总降采样 | nIoU |
|---|---|---|
| adjusted（论文） | /4 | _待填_ |
| regular（标准 ResNet） | /16 | _待填_ |

---

## 引用

```bibtex
@inproceedings{dai2021asymmetric,
  title={Asymmetric contextual modulation for infrared small target detection},
  author={Dai, Yimian and Wu, Yiquan and Zhou, Fei and Barnard, Kobus},
  booktitle={WACV},
  year={2021}
}
```
