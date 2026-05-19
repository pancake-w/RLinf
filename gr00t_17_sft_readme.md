# GR00T 1.7 SFT 全流程实操手册（训练 / 转换 / Eval）

这份文档基于你当前仓库和实操过程整理，目标是：

1. 用 [`libero_sft_gr00t_16`](./examples/sft/config/libero_sft_gr00t_16.yaml) 训练 SFT。
2. 将 SFT checkpoint 转为 HF 格式目录。
3. 在不中断（或尽量少影响）训练的情况下做低显存 Eval 并录视频。

---

## 目录

- [0. 为什么要“先转换再评估”](#0-为什么要先转换再评估)
- [1. 关键文件与路径索引](#1-关键文件与路径索引)
- [2. 环境变量与会话准备](#2-环境变量与会话准备)
- [3. 数据集下载与 v3.0 转换](#3-数据集下载与-v30-转换)
- [4. SFT 训练](#4-sft-训练)
- [5. SFT checkpoint 转 HF 目录](#5-sft-checkpoint-转-hf-目录)
- [6. 边训练边 Eval（低显存 + 录视频）](#6-边训练边-eval低显存--录视频)
- [7. 你应该去改哪些文件](#7-你应该去改哪些文件)
- [8. 常见报错速查](#8-常见报错速查)
- [9. SFT 转 RL Eval 会不会有损失](#9-sft-转-rl-eval-会不会有损失)
- [10. 推荐执行顺序（Checklist）](#10-推荐执行顺序checklist)

---

## 0. 为什么要“先转换再评估”

在这个仓库里，**embodied SFT worker 不支持直接 eval**：

- 代码文件：[`rlinf/workers/sft/fsdp_vla_sft_worker.py`](./rlinf/workers/sft/fsdp_vla_sft_worker.py)
- 关键行为：`get_eval_model_output()` 直接 `NotImplementedError`

因此，如果你要做环境中的策略评估（[`eval_embodied_agent.py`](./examples/embodiment/eval_embodied_agent.py)），需要走 rollout worker 的模型接口。这个接口期望的模型形态与 SFT 训练 ckpt 直接加载路径不同，所以要先用转换脚本将 checkpoint 变成可作为 `model_path` 直接加载的 HF 目录。

---

## 1. 关键文件与路径索引

### 1.1 代码/配置入口

- SFT 训练入口：[`examples/sft/train_vla_sft.py`](./examples/sft/train_vla_sft.py)
- SFT 主配置：[`examples/sft/config/libero_sft_gr00t_16.yaml`](./examples/sft/config/libero_sft_gr00t_16.yaml)
- SFT 模型配置：[`examples/sft/config/model/gr00t_16.yaml`](./examples/sft/config/model/gr00t_16.yaml)
- Embodied Eval 入口：[`examples/embodiment/eval_embodied_agent.py`](./examples/embodiment/eval_embodied_agent.py)
- Embodied Eval 配置：[`examples/embodiment/config/libero_spatial_ppo_gr00t_16.yaml`](./examples/embodiment/config/libero_spatial_ppo_gr00t_16.yaml)
- Embodied GR00T 模型配置：[`examples/embodiment/config/model/gr00t_16.yaml`](./examples/embodiment/config/model/gr00t_16.yaml)
- 转换脚本：[`rlinf/utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf.py`](./rlinf/utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf.py)
- 转换默认配置：[`rlinf/utils/ckpt_convertor/fsdp_convertor/config/fsdp_model_convertor.yaml`](./rlinf/utils/ckpt_convertor/fsdp_convertor/config/fsdp_model_convertor.yaml)

### 1.2 你的实际目录（建议）

- 仓库根目录：`/mnt/public/weibingwen/Documents/RLinf`
- 训练日志根目录：`/mnt/public/weibingwen/Documents/RLinf/logs/bingwen_running_train`
- 示例 checkpoint（step 16000）：  
  `/mnt/public/weibingwen/Documents/RLinf/logs/bingwen_running_train/gr00t_16_sft_libero/checkpoints/global_step_16000/actor/model_state_dict/full_weights.pt`
- 示例转换输出目录：  
  `/mnt/public/weibingwen/Documents/RLinf/logs/bingwen_running_train/gr00t_16_sft_libero/hf_step16000`

---

## 2. 环境变量与会话准备

每个新终端建议先执行一次：

```bash
export REPO_PATH=/mnt/public/weibingwen/Documents/RLinf
export EMBODIED_PATH=$REPO_PATH/examples/embodiment
export PYTHONPATH=$REPO_PATH:$PYTHONPATH
```

训练 SFT 时再加：

```bash
export GR00T_SFT_MODEL_PATH=$REPO_PATH/.codex_tmp/GR00T-N1.7-3B
export GR00T_SFT_DATASET_PATH=$REPO_PATH/.codex_tmp/Gr00t_16-libero-Spatial-dataset
```

Eval（转换后）时再加：

```bash
export GR00T_MODEL_PATH=$REPO_PATH/logs/bingwen_running_train/gr00t_16_sft_libero/hf_step16000
```

---

## 3. 数据集下载与 v3.0 转换

> 结论先说：很多下载源是旧版 LeRobot 格式（每 episode 一个视频），建议统一转成 v3.0（chunk/file + episode 时间窗索引）再训练。

### 3.1 参考文档链接

- 中文文档（GR00T SFT）：[`docs/source-zh/rst_source/examples/embodied/sft_gr00t.rst`](./docs/source-zh/rst_source/examples/embodied/sft_gr00t.rst)
- 英文文档（GR00T SFT）：[`docs/source-en/rst_source/examples/embodied/sft_gr00t.rst`](./docs/source-en/rst_source/examples/embodied/sft_gr00t.rst)

### 3.2 下载示例（HF Hub）

```bash
export REPO_PATH=/mnt/public/weibingwen/Documents/RLinf
cd $REPO_PATH/.codex_tmp

# 你可以按实际 repo_id 调整
hf download IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
  --repo-type dataset \
  --local-dir Gr00t_16-libero-Spatial-dataset_old
```

### 3.3 转换到 LeRobot v3.0

```bash
export REPO_PATH=/mnt/public/weibingwen/Documents/RLinf

# 推荐把 old 目录复制/重命名到目标 repo-id 名后再转
# 例如目标目录名：Gr00t_16-libero-Spatial-dataset
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
  --repo-id=Gr00t_16-libero-Spatial-dataset \
  --root=$REPO_PATH/.codex_tmp \
  --push-to-hub=false \
  --force-conversion
```

### 3.4 转换后快速校验（建议）

```bash
python - <<'PY'
import json, pandas as pd
root="/mnt/public/weibingwen/Documents/RLinf/.codex_tmp/Gr00t_16-libero-Spatial-dataset"
with open(f"{root}/meta/info.json") as f:
    info=json.load(f)
print("episodes:", info["total_episodes"], "frames:", info["total_frames"])
ep=pd.read_parquet(f"{root}/meta/episodes/chunk-000/file-000.parquet")
print("episode rows:", len(ep), "sum length:", int(ep["length"].sum()))
PY
```

---

## 4. SFT 训练

### 4.1 推荐命令（从 `examples/sft` 目录启动）

```bash
cd /mnt/public/weibingwen/Documents/RLinf/examples/sft

python train_vla_sft.py \
  --config-path config \
  --config-name libero_sft_gr00t_16 \
  runner.logger.log_path=/mnt/public/weibingwen/Documents/RLinf/logs/bingwen_running_train \
  +data.num_workers=8
```

### 4.2 说明

- `+data.num_workers=8` 必须带 `+`（原配置里没该键）。
- `--config-path config` 需要你当前目录是 `examples/sft`，否则会触发 Hydra 相对路径拼接问题。
- 训练产物目录通常是：

```text
logs/bingwen_running_train/gr00t_16_sft_libero/
  checkpoints/global_step_xxx/actor/model_state_dict/full_weights.pt
```

---

## 5. SFT checkpoint 转 HF 目录

### 5.1 固定 step 16000 的转换命令（可直接用）

```bash
export REPO_PATH=/mnt/public/weibingwen/Documents/RLinf
export CKPT_PATH=$REPO_PATH/logs/bingwen_running_train/gr00t_16_sft_libero/checkpoints/global_step_16000/actor/model_state_dict/full_weights.pt
export SAVE_PATH=$REPO_PATH/logs/bingwen_running_train/gr00t_16_sft_libero/hf_step16000

PYTHONPATH=$REPO_PATH:$PYTHONPATH \
$REPO_PATH/.venv/bin/python \
  -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf \
  --config-path $REPO_PATH/rlinf/utils/ckpt_convertor/fsdp_convertor/config \
  --config-name fsdp_model_convertor \
  convertor.ckpt_path=$CKPT_PATH \
  convertor.save_path=$SAVE_PATH \
  ++model.model_type=gr00t_1_6_sft \
  ++model.model_path=$REPO_PATH/.codex_tmp/GR00T-N1.7-3B \
  ++model.embodiment_tag=libero_panda \
  ++model.denoising_steps=4 \
  ++model.num_action_chunks=16 \
  ++model.obs_converter_type=libero \
  ++model.is_lora=false \
  ++model.rl_head_config.add_value_head=true \
  ++model.rl_head_config.disable_dropout=true
```

### 5.2 关键注意事项

- 不要写成 `++model.rl_head_config={}`，后续代码会读取 `add_value_head`，会报缺键。
- 小心粘贴 typo：例如 `...false_type=...`、`...{}e=...`。
- 转换成功后，目录应包含 `model-*.safetensors`、`model.safetensors.index.json`、`config.json`。

---

## 6. 边训练边 Eval（低显存 + 录视频）

### 6.1 最稳命令（不依赖历史变量）

```bash
PYTHONPATH=/mnt/public/weibingwen/Documents/RLinf:$PYTHONPATH \
EMBODIED_PATH=/mnt/public/weibingwen/Documents/RLinf/examples/embodiment \
GR00T_MODEL_PATH=/mnt/public/weibingwen/Documents/RLinf/logs/bingwen_running_train/gr00t_16_sft_libero/hf_step16000 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
/mnt/public/weibingwen/Documents/RLinf/.venv/bin/python \
  /mnt/public/weibingwen/Documents/RLinf/examples/embodiment/eval_embodied_agent.py \
  --config-path /mnt/public/weibingwen/Documents/RLinf/examples/embodiment/config \
  --config-name libero_spatial_ppo_gr00t_16 \
  runner.logger.log_path=/mnt/public/weibingwen/Documents/RLinf/logs/bingwen_eval_video_8env_default_test \
  runner.logger.experiment_name=eval_video_step16000_lowmem \
  env.eval.total_num_envs=16 \
  algorithm.eval_rollout_epoch=100 \
  env.eval.auto_reset=False \
  env.eval.max_episode_steps=240 \
  env.eval.max_steps_per_rollout_epoch=240 \
  env.eval.use_fixed_reset_state_ids=True \
  env.eval.video_cfg.save_video=True \
  env.eval.video_cfg.info_on_video=True
```

### 6.2 参数解释（低显存策略）

- `env.eval.total_num_envs=1`：并发环境最小，尽量减少和训练进程争抢显存。
- `algorithm.eval_rollout_epoch=500`：用轮数补总轨迹量（接近全量评估）。
- 视频输出路径：`runner.logger.log_path/video/eval`

### 6.3 重要：不要再传 `runner.ckpt_path=full_weights.pt`

转换后评估应使用 `GR00T_MODEL_PATH=<hf_step目录>`。  
把 SFT 的 `full_weights.pt` 直接传给 eval worker 会导致 key mismatch。

---

## 7. 你应该去改哪些文件

### 7.1 SFT 训练相关

- 主配置：[`examples/sft/config/libero_sft_gr00t_16.yaml`](./examples/sft/config/libero_sft_gr00t_16.yaml)
- 模型子配置：[`examples/sft/config/model/gr00t_16.yaml`](./examples/sft/config/model/gr00t_16.yaml)
- 训练入口：[`examples/sft/train_vla_sft.py`](./examples/sft/train_vla_sft.py)

### 7.2 Eval 相关

- Eval 入口：[`examples/embodiment/eval_embodied_agent.py`](./examples/embodiment/eval_embodied_agent.py)
- Eval 主配置：[`examples/embodiment/config/libero_spatial_ppo_gr00t_16.yaml`](./examples/embodiment/config/libero_spatial_ppo_gr00t_16.yaml)
- Eval 模型子配置：[`examples/embodiment/config/model/gr00t_16.yaml`](./examples/embodiment/config/model/gr00t_16.yaml)

### 7.3 转换相关

- 转换入口：[`rlinf/utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf.py`](./rlinf/utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf.py)
- 转换默认配置：[`rlinf/utils/ckpt_convertor/fsdp_convertor/config/fsdp_model_convertor.yaml`](./rlinf/utils/ckpt_convertor/fsdp_convertor/config/fsdp_model_convertor.yaml)

---

## 8. 常见报错速查

### 8.1 `Environment variable 'EMBODIED_PATH' not found`

- 原因：eval 配置的 Hydra searchpath 依赖 `EMBODIED_PATH`。
- 处理：

```bash
export EMBODIED_PATH=/mnt/public/weibingwen/Documents/RLinf/examples/embodiment
```

### 8.2 `HFValidationError ... repo id ... ''`

- 原因：`GR00T_MODEL_PATH` 或 `runner.logger.log_path` 为空（变量没设置）。
- 处理：优先用“写死绝对路径”的命令；新终端重新 export。

### 8.3 `Missing/Unexpected keys in state_dict`

- 原因：把 SFT `full_weights.pt` 直接用于 eval 的 `runner.ckpt_path`。
- 处理：先转换成 HF 目录，再设 `GR00T_MODEL_PATH=<转换目录>`。

### 8.4 `Found multiple active Ray instances`

- 原因：机器上有多个 Ray 实例。
- 处理（任选其一）：
  1. 显式设定 `RAY_ADDRESS=<你要连的地址>`。
  2. 全停后重启：
     ```bash
     /mnt/public/weibingwen/Documents/RLinf/.venv/bin/ray stop --force
     unset RAY_ADDRESS
     ```

> 备注：如果你已经验证“不关 Ray 也能跑”，建议固定 `RAY_ADDRESS`，避免漂移到另一实例。

---

## 9. SFT 转 RL Eval 会不会有损失

工程上，“转换”本质是：

1. 读取训练好的权重；
2. 加载到模型；
3. 以 HF 目录（safetensors）重新保存。

正常情况下不应引入额外训练损失。  
若转换后效果下降，优先排查：

1. `num_action_chunks`、`denoising_steps`、`obs_converter_type` 是否和训练一致；
2. 是否误用了 `runner.ckpt_path` 直接加载；
3. Eval 采样参数与环境配置是否一致；
4. 评估时是否被训练进程严重抢占显存/算力。

---

## 10. 推荐执行顺序（Checklist）

1. 设置环境变量（至少 `REPO_PATH` / `EMBODIED_PATH` / `PYTHONPATH`）。
2. 准备数据集（必要时做 v3.0 转换）。
3. 启动 SFT 训练。
4. 选一个 step 的 `full_weights.pt` 执行转换。
5. 确认 `hf_stepXXXX` 目录中存在 `model-*.safetensors` 与 `config.json`。
6. 用 `GR00T_MODEL_PATH=<hf_stepXXXX>` 进行 eval（低显存建议先 `total_num_envs=1`）。
7. 查看指标和视频输出。

---

如果你希望把这一套做成“一键脚本”（自动取最新 step、自动转换、自动评估、自动归档日志），可以新增一个脚本例如：`tmp/run_gr00t_sft_eval.sh`（当前未创建，仅建议）。
