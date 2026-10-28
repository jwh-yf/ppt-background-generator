# pptbg · 会留白的演示稿背景生成器

输入一句中文，得到一张 1920×1080 的 PPT 背景：画面里预先留出一块**安全区**，
标题正文压上去仍然清楚，并且可以直接导出成能改字的 `.pptx`。

三个学科方向：**高等数学 / 机械制图 / 人生哲学与科学方法论**。


## 最快上手（双击，不用命令行）

Windows 用户：双击仓库根目录的 **`一键运行.bat`**。

它会自动：找 Python 3.10–3.12 → 建虚拟环境 → 装依赖（清华镜像）→ 起服务 → 打开浏览器。
不需要懂命令行，也不需要 `activate`。

> **前提：机器上要有 Python 3.10 / 3.11 / 3.12。**
> 3.13 及以上不行——torch 目前没有对应的安装包，会报
> `Could not find a version that satisfies the requirement torch==2.9.1 (from versions: none)`。
> 没有的话先装 3.12，安装时勾 `Add python.exe to PATH`，三条路任选：
>
> - 一条命令：`winget install --id Python.Python.3.12 -e`
> - 国内镜像直链：<https://mirrors.huaweicloud.com/python/3.12.10/python-3.12.10-amd64.exe>
> - 官方：<https://www.python.org/downloads/release/python-31210/>

其他方式（手动命令行、无显卡说明、常见报错）见 `VERIFY.md`。

---

## 它和「让 AI 画一张背景」差在哪

| | 通用文生图 | 本项目 |
|---|---|---|
| 画面 | 细节丰富、对比强烈 | 大面积低对比，结构性留白 |
| 知道字放哪吗 | 不知道 | 每张图带 `safe_zones` 坐标（title / body / footer） |
| 留白可度量吗 | 没有这个量 | 用安全区局部亮度标准差 `contrast_std` 量化，阈值 26 |
| 能被约束吗 | 只能反复改提示词 | 推理后按坐标自适应柔化，超标区域压下去 |
| 能直接用吗 | 还要自己排版 | 网页里现场压字预览 → 导出 `.pptx`（文字可编辑） |

三条结论有实测支撑，见 [`docs/技术报告_v3.md`](docs/技术报告_v3.md)：同一 seed、同一条流水线，
加 LoRA 与不加相比，安全区对比度 **6.43 → 16.05**，版式合规率 **100% → 75%**。

---

## 预览

| 高等数学 | 机械制图 | 人生哲学与科学方法论 |
|---|---|---|
| ![](assets/preview/dish_math_1771755870_00) | ![](assets/preview/dish_mech_2127867448_00) | ![](assets/preview/dish_philo_0769358218_00) |

> 上面三张是本仓库自带的讲解图（含中文标注），也可以直接生成后自己看。

---

## 三层结构，缺一层都不成立

| 层 | 做什么 | 为什么必须有 |
|---|---|---|
| **数据合成** | 脚本渲染 600 张 1920×1080 背景，每张同时产出安全区元数据与训练 caption | 图库图没有安全区标注；自己合成才能让「留白」成为可学习、可度量的量，同时彻底规避素材版权 |
| **LoRA 微调** | SD 1.5 只训注意力投影层（rank 16，可训参数 0.37%） | 让模型学会「场景语义 → 影调 + 留白」的映射；单卡 20 分钟，权重 12 MB |
| **构图约束** | 推理后按预设版式算安全区，超标区域柔化；网页叠真实文字预览 | 模型只是「倾向于」留白，这一步把倾向变成约束 |

服务端是标准库 `http.server` 单文件后端 + 原生 HTML/JS，不引入 Gradio / Streamlit，
依赖面收窄到 diffusers 栈本身。

---

## 快速开始

需要 Python 3.10+；有 NVIDIA 显卡（≥8 GB 显存）效果最好，**没有显卡也能跑**（自动降级到程序化后端）。

```bash
pip install -r requirements.txt
python tools/fetch_model.py     # 下 SD 1.5 权重（约 3.1 GB，默认走 hf-mirror 镜像）
python app/server.py            # 浏览器打开 http://127.0.0.1:8760
```

跑一遍端到端自检（页面 / 生成 / 下载 / 导出 pptx / 负路径，共 7 项）：

```bash
python app/smoke_test.py
```

---

## 从零复现数据集与权重

数据集**不进仓库**（330 MB 图片 + 300 MB 训练产物），但它完全由 seed 决定，下面四条命令可以逐字节复现：

```bash
# 1) 母图 288 张（纯 Pillow/numpy，CPU，约 1 分钟）
python data/make_placeholder_assets.py --out data/assets_v2/raw --per-kind 36 --size 1600 --seed 7
python data/asset_manifest.py scan --raw data/assets_v2/raw --out data/assets_v2/manifest.jsonl

# 2) 600 张背景 + 安全区元数据（约 7 分钟）
python data/make_backgrounds.py --assets data/assets_v2/manifest.jsonl --assets_root data/assets_v2/raw \
       --out data/dataset_v4 --count 600 --seed 20261016

# 3) 体检 + 切分
python data/validate_dataset.py --records data/dataset_v4/records.jsonl --images-root data/dataset_v4/images --out data/dataset_v4/report
python train/prepare_dataset.py --records data/dataset_v4/records.jsonl --out data/splits_v4

# 4) 训练 LoRA（RTX 5070 Laptop，1000 步约 22 分钟）
python train/train_lora.py --config configs/lora_v5.yaml
```

---

## 目录结构

```
app/        网页后端与前端、pptx 导出、端到端自检
data/       数据集合成器（母图 / 版式 / 学科符号 / 校验）
train/      LoRA 训练循环、数据集切分、配置
eval/       五后端对照评测（40 条需求 × 5 后端）
tools/      取权重、取 ffmpeg、烧字幕、md→pdf 等小工具
docs/       技术报告、设计说明、开源清单、视频脚本
configs/    训练配置（lora_v1 / lora_v5）
```

---

## 评测

固定 40 条中文需求、同一 seed 基线，五个后端（LoRA / 无 LoRA / 程序化 / 程序化无约束 / 噪声）
跑同一条推理流水线，共 200 张。指标、失败样本、指标口径的三次修正都写在
[`docs/技术报告_v3.md`](docs/技术报告_v3.md) 第 6 节。

---

## 许可证（分三类，不是一个 License 打天下）

| 产出 | 许可证 | 说明 |
|---|---|---|
| 源代码 | MIT | 见 [`LICENSE`](LICENSE) |
| 数据集（生成脚本产出的图 + 标注） | CC BY 4.0 | 署名「pptbg 团队」即可商用、改写 |
| LoRA 适配器权重 | CreativeML OpenRAIL-M | 是 SD 1.5 的衍生作品，必须沿用基座条款 |

**基座权重不随本仓库分发**，用 `tools/fetch_model.py` 自行获取；
镜像 `hf-mirror.com` 的使用情况在 [`docs/开源及第三方资源使用清单.md`](docs/开源及第三方资源使用清单.md) 中如实披露。

---

## 已知限制

1. 合成数据的母图是程序化纹理，视觉复杂度低于真实摄影；对真实素材的泛化未验证。
2. 学科符号是**规则**画的（确定性代码），不是模型学出来的。模型的智能体现在色场与留白。
3. `relevance` 指标偏向规则渲染，不能用来论证「模型优于规则」。
4. 单张 8.5 秒（RTX 5070 Laptop，28 步），比纯规则慢约 9 倍。
5. 没有做人工盲评。脚本指标不能代替人眼。

完整清单见技术报告第 10 节。

---

## 依赖与致谢

PyTorch、diffusers、transformers、peft、Pillow、numpy、python-pptx（可选）。
每一个依赖的版本、许可证、义务与证据都在 [`docs/开源及第三方资源使用清单.md`](docs/开源及第三方资源使用清单.md)。
