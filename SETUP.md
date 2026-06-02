# 环境配置指南 (SETUP)

本项目复现论文 **ACM: Asymmetric Contextual Modulation for Infrared Small Target Detection (WACV 2020)**。

下面的命令请**你自己在终端执行**。torch 相关使用**清华源**。

---

## 1. 创建虚拟环境（推荐 conda）

```bash
# 在工作目录 D:\Desktop\doc\research\ACM 下执行
conda create -n acm python=3.9 -y
conda activate acm
```

如果你想用 venv 而不是 conda：

```bash
python -m venv .venv
# Windows (bash):
source .venv/Scripts/activate
# Windows (cmd):
# .venv\Scripts\activate.bat
```

---

## 2. 安装 PyTorch（清华源，GPU / CUDA 12.1）

本机为 **NVIDIA RTX 4060 Laptop GPU**，安装 CUDA 12.1 版：

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

> 上面用的是 PyTorch 官方 CUDA 轮子源。若只想用清华 PyPI 源（CPU 版，不推荐用于训练）：
> ```bash
> pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```
>
> 说明：本复现代码的 backbone 是按论文 Table 1 **手写**的 ResNet-20，
> **不依赖 torchvision**；装它只是与文档保持一致，可选。

---

## 3. 安装其余依赖（清华源）

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 4. 验证安装

```bash
python -c "import torch, numpy, cv2, matplotlib; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

输出包含 `torch 2.5.1+cu121 ... cuda True` 即说明 GPU 环境就绪。

---

## 5. 数据集

SIRST 数据集已放在 `data/sirst/` 下：

```
data/sirst/
  images.zip          # 红外原图（PNG）
  masks.zip           # 像素级分割标注（PNG）
  idx_427/            # 官方 427 张划分
    train.txt         # 254 张训练
    val.txt           # 83  张验证
    test.txt          # 84  张测试
    trainval.txt
    trainvaltest.txt
```

数据预处理脚本会自动解压 zip。若需手动解压：

```bash
cd data/sirst
unzip -o images.zip
unzip -o masks.zip
```

---

## 6. 运行复现

详细的训练 / 评估 / 消融命令见 **`HOWTO_RUN.md`**。最简单的开始方式：

```bash
conda activate acm
# 主结果：训练 ACM-FPN + ACM-U-Net（论文配置 480px / 300 epoch）
python src/run_all.py --preset main --epochs 300
```

> 本机为 RTX 4060 Laptop GPU，480px 下约 15 秒/epoch，单网络 300 epoch ≈ 75 分钟。
