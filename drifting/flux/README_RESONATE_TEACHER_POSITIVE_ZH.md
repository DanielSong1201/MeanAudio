# Resonate Teacher Positive 生成与训练使用说明

本文说明如何在单张 NVIDIA RTX 4090 上，使用官方 Resonate-GRPO 为
AudioCaps 的每条训练 prompt 生成 teacher positive，并立即转换成原有
Flux/TFD 训练脚本可以直接读取的 MeanAudio 16 kHz VAE latent。

## 1. 目录布局

服务器上的两个仓库必须处于同一级目录：

```text
workspace/
├── MeanAudio/
└── Resonate/
```

后续命令均从 MeanAudio 根目录运行：

```bash
cd /path/to/workspace/MeanAudio
git switch drifting
git pull --ff-only origin drifting
```

脚本默认通过 `../Resonate` 访问官方 Resonate 仓库。如果实际位置不同，
可以设置 `RESONATE_ROOT=/absolute/path/to/Resonate`。

## 2. 创建环境

建议创建独立的 Python 3.11 环境，并安装与服务器驱动匹配的 PyTorch。
本项目已有流程使用 PyTorch 2.5.1 和 CUDA 12.4 wheel：

```bash
conda create -n meanaudio-resonate-positive python=3.11 -y
conda activate meanaudio-resonate-positive

python -m pip install --upgrade pip wheel
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install -e ../Resonate
python -m pip install -e .
python -m pip check
```

建议将 Hugging Face 缓存放在持久化磁盘上：

```bash
export HF_HOME=/path/to/persistent-cache/huggingface
```

检查 CUDA 和两个项目能否导入：

```bash
python - <<'PY'
import torch
import meanaudio
import resonate

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("meanaudio:", meanaudio.__file__)
print("resonate:", resonate.__file__)
PY
```

## 3. 所需文件

MeanAudio 侧需要：

```text
MeanAudio/data/audiocaps/train-memmap.tsv
MeanAudio/weights/v1-16.pth
MeanAudio/sets/latent_mean.pt
MeanAudio/sets/latent_std.pt
```

Resonate 侧需要：

```text
Resonate/weights/Resonate_GRPO.pth
Resonate/weights/v1-44.pth
Resonate/weights/bigvgan_v2_44khz_128band_512x/
Resonate/sets/latent_mean_44k.pt
Resonate/sets/latent_std_44k.pt
```

启动器默认设置 `AUTO_DOWNLOAD=1`。当 Resonate 资产缺失时，会调用
Hugging Face 的 `snapshot_download()`，把 `AndreasXi/resonate` 下载到
`../Resonate/weights/`。MeanAudio 的 `v1-16.pth` 不属于这次自动下载，
需要提前放入 `MeanAudio/weights/`。

## 4. 生成流程和输出格式

对于每一条 prompt，脚本依次执行：

1. 使用 Resonate-GRPO 生成三条 44.1 kHz 音频。
2. 仅为当前 prompt 写入临时 FLAC。
3. 重新读取 FLAC，转成单声道并重采样至 16 kHz。
4. 裁剪或补零到 MeanAudio VAE 所需长度。
5. 使用 `weights/v1-16.pth` 提取 VAE posterior mean。
6. 使用 `sets/latent_mean.pt` 和 `sets/latent_std.pt` 归一化。
7. 原子写入并校验 `<dataset_index>.npz`。
8. 立即删除当前 prompt 的临时 FLAC。

默认输出目录：

```text
data/audiocaps/train-teacher-positives-resonate-grpo-25step-cfg4.5/
```

输出结构：

```text
train-teacher-positives-resonate-grpo-25step-cfg4.5/
├── config.json
├── complete.json
├── 0.npz
├── 1.npz
├── 2.npz
└── ...
```

每个 NPZ 的核心字段是：

```text
latents_normalized: [3, 312, latent_dim], float16
item_index:          AudioCaps manifest 行号
item_id:             AudioCaps 样本 ID
seeds:               三条 positive 对应的确定性随机种子
```

`latents_normalized` 的名称、形状和归一化空间与
`drifting/flux/train.py` 的 `AudioCapsNpzDataset` 一致。

## 5. 先执行单条 smoke test

使用单独目录，避免测试结果与完整数据混合：

```bash
CUDA_VISIBLE_DEVICES=0 \
LIMIT=1 \
OUTPUT_DIR=data/audiocaps/resonate-teacher-positive-smoke \
bash drifting/scripts/flux/build_resonate_teacher_positive_bank_1gpu.sh
```

成功时应看到 `tqdm` 进度到达 `1/1`，并生成：

```text
data/audiocaps/resonate-teacher-positive-smoke/0.npz
data/audiocaps/resonate-teacher-positive-smoke/config.json
data/audiocaps/resonate-teacher-positive-smoke/subset_complete_*.json
```

`LIMIT=1` 不是完整数据，因此不会写入可供训练使用的
`complete.json`。

## 6. 生成完整 teacher-positive bank

### 多卡并行生成（推荐）

启动器会根据 `CUDA_VISIBLE_DEVICES` 中的编号数量自动决定 worker 数量。
例如使用四张卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
AUTO_DOWNLOAD=0 \
bash drifting/scripts/flux/build_resonate_teacher_positive_bank_4gpu.sh
```

也可以通过通用名称启动：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
AUTO_DOWNLOAD=0 \
bash drifting/scripts/flux/build_resonate_teacher_positive_bank_multigpu.sh
```

四个 worker 处理互不重叠的 prompt：

```text
GPU 0: selected_rows[0::4]
GPU 1: selected_rows[1::4]
GPU 2: selected_rows[2::4]
GPU 3: selected_rows[3::4]
```

所有 worker 将 NPZ 写入同一个 `OUTPUT_DIR`，但索引互不重叠。rank 0
显示唯一的全局进度条：

```text
teacher positives generated (4 GPUs): 1250/49838
```

全局进度通过输出目录中的小型共享进度文件汇总，不调用
`torch.distributed.init_process_group()`，也不执行 NCCL barrier、all-reduce
或 gather。不同 prompt 推理耗时不一致时，不会因为某个 rank 未及时进入
collective 而触发 NCCL timeout。`torchrun` 只负责启动和监管这些独立进程。

例如只使用物理 GPU 1 和 GPU 3：

```bash
CUDA_VISIBLE_DEVICES=1,3 \
AUTO_DOWNLOAD=0 \
bash drifting/scripts/flux/build_resonate_teacher_positive_bank_multigpu.sh
```

此时自动启动两个 worker。进程内的 `cuda:0` 对应物理 GPU 1，`cuda:1`
对应物理 GPU 3；两个 worker 分别处理 `selected_rows[0::2]` 和
`selected_rows[1::2]`。进度条会显示 `teacher positives generated (2 GPUs)`。

通常不需要设置 `NPROC_PER_NODE`。如果显式设置，其值必须等于
`CUDA_VISIBLE_DEVICES` 中的 GPU 数量，例如：

```bash
CUDA_VISIBLE_DEVICES=2,5 NPROC_PER_NODE=2 \
bash drifting/scripts/flux/build_resonate_teacher_positive_bank_multigpu.sh
```

首次测试建议让每张卡至少处理一条 prompt：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
AUTO_DOWNLOAD=0 \
LIMIT=8 \
OUTPUT_DIR=data/audiocaps/resonate-teacher-positive-smoke-4gpu \
bash drifting/scripts/flux/build_resonate_teacher_positive_bank_4gpu.sh
```

只有 rank 0 会执行 Resonate 资产检查或自动下载、写入 `config.json`、等待
全部 worker 完成、全量校验 NPZ，并最终写入 `complete.json`。其他 rank
通过文件标记等待这些一次性操作，默认协调等待上限为720分钟，可通过
`COORDINATION_TIMEOUT_MINUTES` 调整。该超时是文件轮询超时，不是 NCCL
超时。

每次启动会生成唯一的 `BANK_RUN_ID`，协调状态保存在：

```text
<OUTPUT_DIR>/.bank-build-progress/<BANK_RUN_ID>/
```

任务失败后直接重新运行即可；有效 NPZ 仍会被各个 worker 分片扫描并复用。

### 单卡生成

```bash
CUDA_VISIBLE_DEVICES=0 \
bash drifting/scripts/flux/build_resonate_teacher_positive_bank_1gpu.sh
```

默认生成配置为：

```text
Resonate checkpoint:       Resonate_GRPO.pth
positives per prompt:      3
Flow-Matching steps:       25
CFG strength:              4.5
duration:                  10 seconds
generation dtype:          bfloat16
output latent dtype:       float16
```

整体进度由 `tqdm` 显示，例如：

```text
teacher positives generated: 1250/49838
```

只有完整 manifest 中的每个 NPZ 均通过检查后，脚本才会写入：

```text
data/audiocaps/train-teacher-positives-resonate-grpo-25step-cfg4.5/complete.json
```

## 7. 中断恢复

中断后直接重新运行完整生成命令即可：

```bash
CUDA_VISIBLE_DEVICES=0 \
bash drifting/scripts/flux/build_resonate_teacher_positive_bank_1gpu.sh
```

恢复扫描不会只判断 NPZ 是否存在，还会检查：

- `latents_normalized` 字段；
- positive 数量和 latent 形状；
- FP16 dtype；
- NaN/Inf；
- manifest 索引；
- AudioCaps ID。

有效条目会计入 `tqdm` 的初始完成数量并直接跳过。损坏或不完整的条目
会重新生成。异常退出遗留的 `.temporary-flac/` 文件会在下次启动时清理。

不要在恢复下载前删除：

```text
Resonate/weights/.cache/huggingface/
$HF_HOME/
```

前者保存 `snapshot_download(local_dir=...)` 的本地元数据，后者通常保存
Flan-T5 等 Transformers 模型缓存。

## 8. 在原训练脚本中使用

生成完成后，可以直接使用原来的单卡训练启动器：

```bash
CUDA_VISIBLE_DEVICES=0 \
EXP_ID=flux_resonate_positive \
SAMPLES_PER_CONDITION=4 \
TEACHER_POSITIVE_COUNT=3 \
TEACHER_POSITIVE_DIR=data/audiocaps/train-teacher-positives-resonate-grpo-25step-cfg4.5 \
bash drifting/scripts/flux/train_flux_1x4090.sh
```

这里必须满足：

```text
SAMPLES_PER_CONDITION = 1 条真实 AudioCaps positive + 3 条 Resonate positive
                      = 4
TEACHER_POSITIVE_COUNT = 3
```

多卡训练同样只需要将上述三个变量传给对应的原训练启动器，例如：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
SAMPLES_PER_CONDITION=4 \
TEACHER_POSITIVE_COUNT=3 \
TEACHER_POSITIVE_DIR=data/audiocaps/train-teacher-positives-resonate-grpo-25step-cfg4.5 \
bash drifting/scripts/flux/train_flux_2x4090.sh
```

训练启动时会检查 `complete.json`，因此 smoke test 或未完成的 subset
不能被误当作完整 teacher-positive bank。

## 9. 常用覆盖参数

```text
PYTHON                    Python 可执行文件
RESONATE_ROOT             Resonate 仓库路径，默认 ../Resonate
MANIFEST                  AudioCaps 训练 manifest
OUTPUT_DIR                生成结果目录
TEACHER_POSITIVE_DIR      OUTPUT_DIR 未设置时也可用它指定结果目录
CHECKPOINT                Resonate checkpoint
TARGET_VAE_WEIGHTS        MeanAudio 16 kHz VAE
TARGET_LATENT_MEAN        MeanAudio latent mean
TARGET_LATENT_STD         MeanAudio latent std
POSITIVES_PER_CONDITION   每条 prompt 的 positive 数量
NUM_STEPS                 Resonate 推理步数
CFG_STRENGTH              Resonate CFG
DURATION                  音频时长
BASE_SEED                 确定性基础种子
START_INDEX               起始 manifest 行号，包含
END_INDEX                 结束 manifest 行号，不包含
LIMIT                     最多处理多少条 prompt
AUTO_DOWNLOAD             1=缺失时下载 Resonate 权重，0=禁止下载
OVERWRITE                 1=重新生成选定条目
FULL_PRECISION            1=使用 float32 生成
DRY_RUN                   1=只检查路径和任务范围
PROGRESS_POLL_INTERVAL    多卡全局进度轮询间隔，默认1秒
COORDINATION_TIMEOUT_MINUTES 多卡文件协调等待上限，默认720分钟
NPROC_PER_NODE            可选；默认由CUDA_VISIBLE_DEVICES自动推导，显式设置时必须一致
CUDA_VISIBLE_DEVICES      逗号分隔的物理GPU编号，例如1,3；数量决定worker数
BANK_RUN_ID               本次协调命名空间，默认自动生成
MASTER_ADDR               torchrun单机rendezvous地址，默认127.0.0.1
MASTER_PORT               torchrun单机rendezvous端口，默认根据启动器PID生成
```

例如处理 `[10000, 20000)` 范围：

```bash
CUDA_VISIBLE_DEVICES=0 \
START_INDEX=10000 \
END_INDEX=20000 \
bash drifting/scripts/flux/build_resonate_teacher_positive_bank_1gpu.sh
```

分段运行只写 subset 完成标记。所有分段结束后，再运行一次不带范围参数的
完整命令；它会跳过已经有效的 NPZ，并在全量检查后生成 `complete.json`。

## 10. Hugging Face 下载逻辑说明

当前代码先检查以下路径是否存在：

```text
Resonate_GRPO.pth
v1-44.pth
bigvgan_v2_44khz_128band_512x/
latent_mean_44k.pt
latent_std_44k.pt
```

如果全部存在，不会调用 `snapshot_download()`。如果有路径缺失，并且
`AUTO_DOWNLOAD=1`，脚本调用：

```python
snapshot_download(
    repo_id="AndreasXi/resonate",
    local_dir="../Resonate/weights",
)
```

下载完成后再次检查上述路径。如果仍有缺失，脚本会停止，不会进入模型
加载。`snapshot_download` 默认不会强制重新下载已经完成的文件，并会在
`Resonate/weights/.cache/huggingface/` 保存本地元数据，因此正常的下载
中断可以通过重新运行脚本继续补齐。

当前预检查的边界是：它主要使用 `Path.exists()`，没有在进入下载前比较
官方文件大小或哈希。非零但截断的 checkpoint，以及内容不完整但目录已经
存在的 BigVGAN，可能先通过路径检查，随后在 `torch.load()` 或 BigVGAN
初始化时才报错。遇到这种情况，应保留 Hugging Face 缓存并重新执行官方
下载，或删除明确损坏的单个目标文件后重新运行；不要删除整个权重目录。
