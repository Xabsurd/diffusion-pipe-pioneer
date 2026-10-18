# 维护分支

分支名：`diffusion-pipe-pioneer`。

GitHub：<https://github.com/Xabsurd/diffusion-pipe-pioneer>

这个分支在上游 [tdrussell/diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) 的 `main` 之上，多了上游还没有的训练功能。新模型尽量放在单独文件里，共享文件只加一小段挂钩，这样合并上游时冲突面小。

## 相对上游多出来的功能

- Qwen-Image-2.1 图片编辑。`control_path` 是一张参考图，`control_paths` 是多张。文本编码器会记下每张参考图插入文本的位置，DiT 把参考图 latent 嵌进这个位置，损失只算目标图。
- FLUX 3 Klein。`type = 'flux3'`。文生图，以及最多 10 张参考图。参考图留在 latent 里，不进文本编码器。
- MOSS-Image。`type = 'moss_image'`。目前只有文生图。
- Qwen-Image 和 Flux 2 也可以用 `control_paths` 做多参考图。

示例配置：

- `examples/qwen_image21_edit_dataset.toml`
- `examples/flux3_klein.toml`
- `examples/moss_image.toml`
- `examples/flux_kontext_dataset.toml` 里有 `control_paths` 的注释

LoRA 仍然按 ComfyUI 的格式保存，键名带 `diffusion_model.` 前缀。

## 训练时的限制

- Qwen-Image-2.1 编辑必须 `uncond_fraction = 0`。那条全局空 caption 没有参考图，也没有 `image_slots`。
- Qwen-Image-2.1 和 FLUX 3 同一次训练里，每个数据目录的参考图数量必须相同。文生图就是 0。不要把文生图和编辑混在同一次运行里，也不要混用不同的参考图数量。DeepSpeed 流水线在一次运行里要求阶段之间的张量个数固定。
- 改了 caption 或参考图之后，删掉对应缓存，或加上 `--regenerate_cache`。Qwen-Image-2.1 编辑的旧缓存没有 `image_slots`，会直接报错。
- 参考图会被裁到目标图的尺寸桶。宽高比差很多时，画面会被切掉。
- Qwen-Image-2.1 的文本编码器看的是 `edit_vlm_resolution`（默认 1024，设为 0 则用原图并对齐到 32 的倍数）。VAE 用的是目标图的尺寸桶。这两边的尺寸不需要相同。
- Qwen-Image-2.1 是 CFG 蒸馏模型。这个分支没有加去蒸馏适配器。需要的话继续用原来的 `merge_adapters`。不加的话，推理时要用 CFG。
- FLUX 3 要训练 Klein Base。带 `ref_time_in` 的 step-distilled 权重会被拒绝。Base 上训出来的 LoRA 之后可以挂到蒸馏模型上。
- FLUX 3 的 `shift` 是乘数形式，默认 `6.93`。ComfyUI 采样配置里写的是 `mu = ln(6.93)`，不要把那个数直接抄过来。
- MOSS-Image 训练 Base，不要拿 SFT 权重当 Base 用。`shift = 5`。这个版本没有编辑路径，写了 `control_path` 或 `control_paths` 会报错。
- FLUX 3 的文本编码器是 Qwen3-VL-4B，MOSS-Image 的是 Qwen3.5-9B（bf16 大约 19GB）。先缓存文本嵌入，训练开始后文本编码器会被卸掉。
- Flux 2 的参考图 latent 只在缓存时做一次 `process_in`。以前 `prepare_inputs` 里还有一次缩放，但它依赖一个并不存在的 `model_patcher`，那条路径实际上跑不起来。

## ComfyUI 子模块

这个分支把 `submodules/ComfyUI` 钉在 `c06bf5f569d268b80076e5f44126f87cdf835869`（2026-10-17，`Bump comfyui-frontend-package to 1.55.18`）。上游当时的钉点还没有 Flux 3 和 MOSS-Image，所以这里故意超前。

这个提交需要：

- `comfy-aimdo==0.5.6`
- `comfy-kitchen==0.2.38`

`requirements.txt` 里已经按这个钉死。旧的 0.4.x / 0.2.16 导入不了这份 ComfyUI。

子模块比上游新了一个月左右，旧模型的 Comfy 接口也可能变。合并上游之后，至少跑一次 `python -m unittest test.test_maintained_models`。

## 和上游同步

远程名按仓库现况：`upstream` 是 tdrussell/diffusion-pipe，`origin` 是你自己的仓库。

```bash
git fetch upstream
git merge upstream/main
git -c safe.directory="$(pwd)/submodules/ComfyUI" submodule update --init submodules/ComfyUI
```

或者：

```bash
bash tools/sync_upstream.sh
```

脚本只做 fetch、merge、以及在 merge 成功后更新 ComfyUI 子模块。它不自动解决冲突。

子模块指针冲突时：

- 上游的 ComfyUI 更旧，或者还没有 `comfy/ldm/flux3` 和 `comfy/ldm/moss_image`：保留本分支的指针。
- 上游的指针比本分支新，并且仍然包含这两个模型：采用上游指针，然后重新跑测试。旧模型的调用签名如果变了，要跟着改 `models/qwen_image21.py`、`models/flux3.py`、`models/moss_image.py`。

解决指针之后：

```bash
git add submodules/ComfyUI
git -c safe.directory="$(pwd)/submodules/ComfyUI" submodule update --init submodules/ComfyUI
```

不要改 git config。这个工作树如果报 `dubious ownership`，每次调用子模块 git 时加上 `git -c safe.directory=...`，脚本里已经这么写了。

合并之后看一眼 `README.md` 的 Recent changes。那里有一条指向本文的记录，上游如果也改了同一段，冲突会落在这个小地方，而不是散在模型文件里。
