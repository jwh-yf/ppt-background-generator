# 验收说明：怎么确认这个仓库能跑

这份文档写给第一次拿到本仓库的人。跟着做，每做完一步都有「应该看到什么」。
全程约 15 分钟，其中 5–10 分钟是装依赖。

不用显卡也能验完——没有 CUDA 时程序自动走程序化兜底，界面与导出功能完全一样，
只是背景图来自规则渲染而不是模型。

---

## 0. 准备

- Windows 10/11（macOS/Linux 也能跑，把下面的 `python` 换成 `python3`）
- **Python 3.10 / 3.11 / 3.12**（3.13 及以上不行，见下面的红色提示）
- 磁盘留 5 GB（装依赖约 1.5 GB；如果下载基座权重再加 3.1 GB）
- 网络：装依赖建议用国内镜像（见下）

验证 Python 是否就绪：

```bat
python --version
```

应该输出 `Python 3.10.x` / `3.11.x` / `3.12.x`。

> **如果输出的是 3.13 或更新的版本，先别往下走。**
> torch（PyTorch）目前没有 3.13+ 的 Windows 安装包，装依赖会失败在
> `Could not find a version that satisfies the requirement torch==2.9.1`。
> 这不是网络问题、也不是镜像问题，是解释器版本太新。
>
> 装 3.12（**别从 python.org 首页下**，首页默认给最新版；而且国内访问 python.org
> 经常 DNS 解析失败）。三条路任选一条，安装时都勾 `Add python.exe to PATH`：
>
> 1. **winget（最省事，Win10 1809+ 自带）**：
>
>    ```bat
>    winget install --id Python.Python.3.12 -e
>    ```
>
> 2. **国内镜像直链**（都是 3.12.10 官方安装包，26.9 MB，实测可达）：
>
>    - 华为云：<https://mirrors.huaweicloud.com/python/3.12.10/python-3.12.10-amd64.exe>
>    - 阿里云：<https://mirrors.aliyun.com/python-release/windows/python-3.12.10-amd64.exe>
>    - 中科大：<https://mirrors.ustc.edu.cn/python/3.12.10/python-3.12.10-amd64.exe>
>
> 3. **官方直链**（网络正常时）：<https://www.python.org/downloads/release/python-31210/>
>
> 装完在 cmd 里验一下，能打印版本号就对了：
>
> ```bat
> py -3.12 --version
> ```

---

## 1. 拿到代码（两种方式任选）

**方式 A：下载压缩包（不用装 git）**

打开仓库页 → 绿色按钮 `Code` → `Download ZIP` → 解压到任意目录。

**方式 B：git clone**

```bat
git clone https://github.com/jwh-yf/ppt-background-generator.git
cd ppt-background-generator
```

---

## 2. 装依赖

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**应该看到**：最后一屏是 `Successfully installed diffusers-0.36.0 ... python-pptx-...`。

> 有 RTX 50 系显卡时，torch 要用 CUDA 12.8 索引，`requirements.txt` 顶部写了那一行，先单独装它再装其余。

⚠️ 如果 `pip install` 在 `torch` 上卡住或报 SSL 错误：先单独装 torch，再装其余。

---

## 3. 启动

```bat
python app\server.py --port 8763
```

或者用附带的脚本（会自动开浏览器）：

```powershell
.\tools\run_local.ps1
```

**应该看到**：

```
已加载：程序化兜底（未找到权重）        ← 没下权重时
或
已加载：diffusers/cuda，LoRA：...      ← 下过权重且有显卡时
服务已启动：http://127.0.0.1:8763/
```

浏览器打开 <http://127.0.0.1:8763/>。

---

## 4. 六个验收点

按顺序点一遍，每一条都给出「通过长什么样」。

### ① 页面能打开

左上角标题是「演示稿背景生成器」，右上角有后端徽标（`程序化兜底` 或 `Diffusers + LoRA`）。

✅ 通过：能看到 3 张学科卡片（高等数学 / 机械制图 / 人生哲学与科学方法论），
下面是折叠的「更多风格」。

### ② 能生成一张

1. 选「高等数学」
2. 「内容主题」填 `导数与微分`
3. 「标题」填 `高等数学期末复习`
4. 点「生成背景」，等 9 秒左右（无显卡时更慢，30–60 秒）

✅ 通过：右侧出现一张 1920×1080 背景，左上角有网格与坐标轴，
「文字预览」视图里标题真的压在图上、看得清。

### ③ 三个视图都在

点右上角 `文字预览` / `纯背景` / `安全区`。

✅ 通过：安全区视图里有一圈虚线框，右下角显示对比度数值（比如 `5.2 / 8.4`），
数字越低说明压字区越干净。

### ④ 能导出 pptx

点 `导出 .pptx`，等几秒。

✅ 通过：浏览器下载一个 `.pptx`。用 PowerPoint 打开，是 2 页，
文字是**真正的文本框**，双击能改。

### ⑤ 能一键出整套

点左下角「一次生成整套 PPT（8 页）」。

✅ 通过：等 1 分钟左右，下载的 pptx 是 8 页，封面/目录/正文/尾页版式不同但配色统一。

### ⑥ 六个可调项在

在「高级参数」里，或直接看主面板：

✅ 通过：能看到「配色」（色点 + 随机）、「柔焦」、「磨砂强度」、「磨砂形状」、
「学科元素」、「符号质感」六个控件，改动后重新生成，出图确实变了。

---

## 5. 想看到模型效果（可选，需要 3.1 GB 下载）

```bat
python tools\fetch_model.py sd15
```

**应该看到**：进度条走完，最后打印「校验通过」，权重落在 `tools/hf/stable-diffusion-v1-5/`。

重启服务，右上角徽标应变成 `Diffusers + LoRA`（无显卡时是 `Diffusers`）。
此时出图的色调更收敛、留白更干净——这就是技术报告里 `6.43 vs 16.05` 的差别来源。

> 如果下载卡住：脚本会自动回退到 `hf-mirror.com` 镜像；也可以手动设
> `set HF_ENDPOINT=https://hf-mirror.com` 再跑。

---

## 6. 自检脚本（一条命令验完后端）

```bat
python app\smoke_test.py
```

**应该看到**：

```
[ok] T1 页面: HTTP 200，... 字符，关键元素齐全
[ok] T2 健康检查
[ok] T3 预设清单: 9 个预设 / 比例 ['16:9', '4:3', '9:16']
[ok] T4 生成: 2 张 / ...s
[ok] T5 图片下载
[ok] T6 导出 pptx
[ok] T7 负路径
通过 7 / 失败 0
```

七项全绿就是后端完全正常。这个脚本会自己起一个临时服务、跑完自动关掉，
不需要先启动 `server.py`。

---

## 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `python` 不是内部或外部命令 | 没加到 PATH | 重装 Python 时勾选 Add to PATH；或用 `py -3` 代替 `python` |
| `ERROR: Could not find a version that satisfies the requirement torch==2.9.1 (from versions: none)` | **Python 版本 ≥3.13**，torch 没有对应的安装包 | 装 Python 3.12（见「0. 准备」里的直链），删掉 `.venv` 重来 |
| `'锘緻echo' 不是内部或外部命令` | 用了带 BOM 的旧版 `.bat` | 重新下载最新 ZIP；新版 `一键运行.bat` 已改为无 BOM |
| `pip install` 卡在 torch | 默认源慢 | 加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`，或先单独装 CPU 版 torch |
| 启动报 `No module named 'diffusers'` | 没进虚拟环境 | 用 `.venv\Scripts\python.exe app\server.py` 直接跑，不必 activate |
| 页面能开但点生成一直转圈 | 无显卡 + 首次加载慢 | 等 60 秒；或先跑 `python tools/fetch_model.py sd15` |
| 导出 pptx 报错 | 缺 python-pptx | `pip install python-pptx` |
| 端口被占用 | 8763 有别的程序 | `python app\server.py --port 8764` |
| 权重下载 SSL 失败 | 直连 huggingface.co 不稳 | 用镜像：`set HF_ENDPOINT=https://hf-mirror.com` |

---

## 验完请告诉我（可选）

如果哪一步和上面写的不一样，把**那一步的完整报错**发我，我改文档。
开源作品的文档质量就看这个——别人踩过的坑应该写进文档，而不是让人再踩一遍。

---

## 附：如果提示「禁止运行脚本」

Windows 默认策略禁止运行 `.ps1`，所以你运行 `.venv\Scripts\activate` 或
`run_local.ps1` 时会看到：

```
无法加载文件 ...\Activate.ps1，因为在此系统上禁止运行脚本。
```

有两种解法：

**解法一（推荐）：双击 `一键运行.bat`**

仓库根目录有个 `一键运行.bat`，双击它就行——`.bat` 不受执行策略限制，
它会自动建虚拟环境、装依赖、起服务、开浏览器。**不需要先 activate。**

**解法二：不激活，直接用虚拟环境里的 python**

```bat
.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
.venv\Scripts\python.exe app\server.py --port 8763
```

这样绕开了 activate 脚本，同样能跑。

如果你确实想用 `activate`：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\activate
```

`-Scope Process` 只影响当前窗口，关掉就恢复，不会改系统设置。

---

## 附：命令里的空格是怎么来的

如果报错长这样：

```
ERROR: Invalid requirement: 'tuna.'
ERROR: Invalid requirement: 'requirements.'
```

说明命令里的 URL 或文件名被插入了空格（复制粘贴时换行被当成空格，
或输入法自动加了空格）。**不要手打长 URL**，改用 `一键运行.bat`，
或者用解法二那两行较短的命令。

同理，`powershel1` 是数字 `1` 代替了字母 `l`——这类错误肉眼很难发现，
所以尽量复制完整命令，不要手动输入。
