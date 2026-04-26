# SD1.5 LoRA 训练与推理

一个基于本地 Stable Diffusion 1.5 的 LoRA 训练与推理项目，支持：

- 本地 SD1.5 推理
- 基于 UNet 注意力层的 LoRA 训练
- 导出 `.safetensors` 格式 LoRA
- 推理时加载 LoRA
- 快速 smoke test
- 在 ComfyUI 中加载训练结果

本项目主要用于训练Lora模型用于comfyui中生图，打造自己的角色模型，作为学习扩散模型所用

## 项目内容

- 训练代码与推理代码分离
- 支持命令行运行，也支持 VSCode 直接点击运行 `main.py`
- 支持本地模型下载脚本
- 支持角色/风格类小数据集 LoRA 训练
- 支持将输出的 LoRA 放入 ComfyUI 使用

## 仓库说明

本仓库包含：

- 训练代码
- 推理代码
- 环境依赖文件
- 使用说明文档

本仓库不包含：

- SD1.5 基础模型权重
- 私有训练数据集
- 默认训练输出结果

也就是说，使用前你需要自己：

1. 安装依赖
2. 下载 SD1.5 基础模型
3. 准备训练数据
4. 训练 LoRA
5. 推理或放入 ComfyUI 使用

---

## 项目结构

```text
.
├─ src/                     # 推理相关代码
│  ├─ main.py
│  ├─ pipeline.py
│  ├─ clip.py
│  ├─ unet.py
│  ├─ vae.py
│  ├─ ddim_scheduler.py
│  ├─ lora.py
│  └─ comfyui.py
├─ train/                   # 训练相关代码
│  ├─ main.py
│  ├─ train_lora.py
│  ├─ dataset.py
│  ├─ lora.py
│  ├─ save_lora.py
│  └─ config.py
├─ data/                   
├─ models/                  
├─ outputs/                 
├─ download_models.py
├─ test_pipeline.py
├─ test_train_lora.py
├─ requirements.txt
├─ requirements-cpu.txt
├─ requirements-cu121.txt
├─ .gitignore
├─ LICENSE
└─ README.md
```

---

## Python 版本

推荐：

- Python `3.10.6`

---

## 环境安装

本项目将公共依赖和 PyTorch 运行时拆开，方便不同环境一次性安装。

### CPU 环境

```bash
pip install -r requirements-cpu.txt
```

### CUDA 12.1 环境

```bash
pip install -r requirements-cu121.txt
```

### 只安装公共依赖

如果你已经自行安装好了 `torch` 和 `torchvision`，可以只安装项目公共依赖：

```bash
pip install -r requirements.txt
```

---

## 下载 SD1.5 基础模型

运行：

```bash
python download_models.py
```

模型会下载到：

```text
./models/stable-diffusion-v1-5/
```

下载完成后目录结构应包含：

```text
models/stable-diffusion-v1-5/
├─ tokenizer/
├─ text_encoder/
├─ unet/
├─ vae/
└─ scheduler/
```

脚本会自动检查关键文件是否完整。

---

## 数据集格式

LoRA 训练使用图文配对数据。

支持图片格式：

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`

支持两种数据组织方式。

### 方式 A：分目录存放

```text
data/
├─ images/
│  ├─ 001.jpg
│  ├─ 002.jpg
│  └─ ...
└─ captions/
   ├─ 001.txt
   ├─ 002.txt
   └─ ...
```

### 方式 B：平铺存放

```text
data/
├─ 001.jpg
├─ 001.txt
├─ 002.jpg
├─ 002.txt
└─ ...
```

要求：

- 每张图片必须有对应的 `.txt` 标注文件
- 图片和文本文件名必须同名，仅后缀不同

示例 caption：

```text
leisai, purple hair, green eyes, smile, anime girl
```

如果你的目标是让角色特征更明显，建议在 caption 中稳定保留：

- 角色名或触发词
- 发色
- 瞳色
- 标志性配饰
- 关键服装元素

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements-cu121.txt
```

### 2. 下载模型

```bash
python download_models.py
```

### 3. 快速测试训练链路

```bash
python test_train_lora.py
```

### 4. 快速测试推理链路

```bash
python test_pipeline.py
```

### 5. 正式训练

```bash
python -m train.main
```

### 6. 正式推理

```bash
python -m src.main
```

---

## 训练方式

训练入口：

```bash
python -m train.main
```

也支持在直接运行：

```text
train/main.py
```

### 双模式说明

`train/main.py` 支持两种使用方式：

1. 直接运行文件  
   使用 `DEFAULTS` 中的默认训练参数

2. 命令行运行  
   使用命令行参数覆盖默认配置

### 可在 `train/main.py` 中直接修改的默认参数

例如：

- `model_path`
- `data_dir`
- `output_dir`
- `num_epochs`
- `learning_rate`
- `gradient_accumulation_steps`
- `lora_rank`
- `lora_alpha`

### 训练命令示例

```bash
python -m train.main \
  --model_path ./models/stable-diffusion-v1-5 \
  --data_dir ./data \
  --output_dir ./outputs/lora_full \
  --num_epochs 15 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 16
```

### 常用训练参数

- `--model_path`：本地 SD1.5 模型路径
- `--data_dir`：训练数据目录
- `--output_dir`：LoRA 输出目录
- `--num_epochs`：训练轮数
- `--batch_size`：批大小
- `--gradient_accumulation_steps`：梯度累积步数
- `--learning_rate`：学习率
- `--lora_rank`：LoRA rank
- `--lora_alpha`：LoRA alpha
- `--device`：`cpu` 或 `cuda`
- `--mixed_precision`：`no` / `fp16` / `bf16`

### 训练输出

训练完成后默认输出：

```text
outputs/lora_full/final_lora.safetensors
```

同时会保存：

```text
outputs/lora_full/final_lora_trainer_state.pt
```

如果启用了中间 checkpoint 保存，还会生成：

- `step-xx.safetensors`
- `step-xx_trainer_state.pt`

---

## 推理方式

推理入口：

```bash
python -m src.main
```

也支持直接运行：

```text
src/main.py
```

### 双模式说明

`src/main.py` 支持两种使用方式：

1. 直接运行文件  
   使用 `DEFAULTS` 中的默认推理参数

2. 命令行运行  
   命令行模式默认不加载 LoRA，除非显式指定 `--use_lora` 或传入 `--lora_path`

### 可在 `src/main.py` 中直接修改的默认参数

例如：

- 正向提示词
- 负向提示词
- 是否启用 LoRA
- LoRA 路径
- 图像尺寸
- 采样步数
- CFG scale
- 输出路径

### 不加载 LoRA 的推理示例

```bash
python -m src.main \
  --prompt "anime girl, purple hair, green eyes, smile, best quality" \
  --negative_prompt "low quality, blurry, bad anatomy" \
  --output output.png
```

### 加载 LoRA 的推理示例

```bash
python -m src.main \
  --prompt "leisai, purple hair, green eyes, smile, anime girl, best quality" \
  --negative_prompt "realistic, low quality, worst quality, bad anatomy" \
  --use_lora \
  --lora_path ./outputs/lora_full/final_lora.safetensors \
  --lora_strength 1.0 \
  --output output_lora.png \
  --verbose
```

### 常用推理参数

- `--prompt`：正向提示词
- `--negative_prompt`：负向提示词
- `--model_path`：模型路径
- `--use_lora`：启用 LoRA
- `--no_lora`：关闭 LoRA
- `--lora_path`：LoRA 路径
- `--lora_strength`：LoRA 强度
- `--height`：图像高度
- `--width`：图像宽度
- `--steps`：采样步数
- `--guidance_scale`：CFG scale
- `--seed`：随机种子
- `--output`：输出文件路径

---

## Smoke Test

这两个测试脚本的目标是快速验证“主流程能否跑通”，不是做严格 benchmark。

### 训练 smoke test

```bash
python test_train_lora.py
```

作用：

- 检查模型路径
- 检查数据目录
- 跑一轮最小 LoRA 训练
- 检查 `final_lora.safetensors` 是否成功生成

测试参数可以直接在文件顶部修改：

```text
test_train_lora.py
```

### 推理 smoke test

```bash
python test_pipeline.py
```

作用：

- 检查模型路径
- 可选加载本地 LoRA
- 执行一次完整推理
- 检查输出图片是否生成

是否启用 LoRA 可以直接在文件顶部修改：

```python
USE_LORA = True
```

或：

```python
USE_LORA = False
```

---

## 在 ComfyUI 中使用

训练好的 LoRA 会导出为 `.safetensors`，可以直接给 ComfyUI 使用。

通常流程：

1. 训练得到 LoRA
2. 将文件复制到：

```text
ComfyUI/models/loras/
```

3. 在 ComfyUI 中使用 `Load LoRA` 节点加载

项目中也包含一个简单的自定义节点入口：

```text
src/comfyui.py
```

---

## License

本项目使用 [MIT License](./LICENSE)。
