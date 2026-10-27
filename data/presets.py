"""风格预设库 —— 需求 → prompt → 视觉配方的唯一事实来源。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, List, Tuple



@dataclass(frozen=True)
class Tone:
    """影调参数。所有取值都在合成时固定，不使用随机数，保证可复现。"""

    key: str
    name_zh: str
    name_en: str
    curve: Tuple[Tuple[float, float], ...]
    saturation: float
    tint: Tuple[int, int, int]
    tint_strength: float
    ink_hex: str
    accent_hex: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["curve"] = [list(p) for p in self.curve]
        d["tint"] = list(self.tint)
        return d


@dataclass(frozen=True)
class Preset:
    """一个风格预设 = 语义标签 + 影调 + 版式候选 + 关键词。"""

    key: str
    name_zh: str
    name_en: str
    scenario_zh: str
    keywords_en: List[str]
    tone: str
    layouts: Tuple[str, ...]
    photo_roles: Tuple[str, ...]
    texture_weight: float


TONES: Dict[str, Tone] = {
    "academic_blue": Tone(
        key="academic_blue",
        name_zh="学术蓝",
        name_en="academic blue",
        curve=((0.0, 0.06), (0.25, 0.22), (0.5, 0.52), (0.75, 0.80), (1.0, 0.97)),
        saturation=0.78,
        tint=(28, 58, 110),
        tint_strength=0.30,
        ink_hex="#0F2B46",
        accent_hex="#2E6FBE",
    ),
    "clean_white": Tone(
        key="clean_white",
        name_zh="极简白",
        name_en="clean white",
        curve=((0.0, 0.30), (0.25, 0.55), (0.5, 0.76), (0.75, 0.90), (1.0, 0.99)),
        saturation=0.35,
        tint=(245, 246, 248),
        tint_strength=0.42,
        ink_hex="#1A1A1A",
        accent_hex="#8C8C8C",
    ),
    "warm_terra": Tone(
        key="warm_terra",
        name_zh="暖陶土",
        name_en="warm terracotta",
        curve=((0.0, 0.08), (0.25, 0.26), (0.5, 0.55), (0.75, 0.79), (1.0, 0.95)),
        saturation=0.92,
        tint=(168, 92, 62),
        tint_strength=0.24,
        ink_hex="#43220F",
        accent_hex="#C1663C",
    ),
    "forest_deep": Tone(
        key="forest_deep",
        name_zh="深林绿",
        name_en="deep forest green",
        curve=((0.0, 0.04), (0.25, 0.17), (0.5, 0.42), (0.75, 0.70), (1.0, 0.92)),
        saturation=0.82,
        tint=(26, 74, 58),
        tint_strength=0.31,
        ink_hex="#0E2A20",
        accent_hex="#3E8E6B",
    ),
    "violet_tech": Tone(
        key="violet_tech",
        name_zh="科技紫",
        name_en="violet tech",
        curve=((0.0, 0.03), (0.25, 0.14), (0.5, 0.40), (0.75, 0.72), (1.0, 1.0)),
        saturation=0.95,
        tint=(72, 44, 128),
        tint_strength=0.34,
        ink_hex="#1B0E3A",
        accent_hex="#7A5AF8",
    ),
    "ink_mono": Tone(
        key="ink_mono",
        name_zh="水墨灰",
        name_en="ink monochrome",
        curve=((0.0, 0.05), (0.25, 0.20), (0.5, 0.48), (0.75, 0.76), (1.0, 0.96)),
        saturation=0.0,
        tint=(60, 60, 64),
        tint_strength=0.16,
        ink_hex="#1C1C1E",
        accent_hex="#5A5A5F",
    ),
    "dish_math": Tone(
        key="dish_math",
        name_zh="高数黑板蓝",
        name_en="mathematics chalkboard blue",
        curve=((0.0, 0.05), (0.25, 0.18), (0.5, 0.44), (0.75, 0.72), (1.0, 0.93)),
        saturation=0.70,
        tint=(24, 46, 82),
        tint_strength=0.34,
        ink_hex="#0B2239",
        accent_hex="#4C8FD6",
    ),
    "dish_mech": Tone(
        key="dish_mech",
        name_zh="机械制图蓝晒",
        name_en="drafting blueprint",
        curve=((0.0, 0.10), (0.25, 0.30), (0.5, 0.55), (0.75, 0.78), (1.0, 0.94)),
        saturation=0.72,
        tint=(30, 72, 122),
        tint_strength=0.36,
        ink_hex="#0E2E52",
        accent_hex="#2F6FA8",
    ),
    "dish_philo": Tone(
        key="dish_philo",
        name_zh="宣纸墨",
        name_en="rice paper ink",
        curve=((0.0, 0.22), (0.25, 0.48), (0.5, 0.72), (0.75, 0.88), (1.0, 0.98)),
        saturation=0.20,
        tint=(238, 233, 222),
        tint_strength=0.40,
        ink_hex="#2A2622",
        accent_hex="#8A7B66",
    ),
}


TINT_VARIANTS: Dict[str, List[Dict[str, Any]]] = {
    "dish_math": [
        {"key": "chalkblue", "name_zh": "黑板蓝", "tint": (24, 46, 82),
         "ink_hex": "#0B2239", "accent_hex": "#4C8FD6"},
        {"key": "inkgreen", "name_zh": "深墨绿", "tint": (22, 54, 44),
         "ink_hex": "#0A241C", "accent_hex": "#57B08A"},
        {"key": "nightviolet", "name_zh": "夜紫", "tint": (40, 30, 66),
         "ink_hex": "#150E28", "accent_hex": "#9B7BE0"},
        {"key": "graphite", "name_zh": "石墨灰", "tint": (54, 58, 64),
         "ink_hex": "#1B1E22", "accent_hex": "#8FB0D8"},
        {"key": "sepia", "name_zh": "深褐", "tint": (58, 42, 32),
         "ink_hex": "#20150E", "accent_hex": "#D19A5E"},
    ],
    "dish_mech": [
        {"key": "cyanotype", "name_zh": "蓝晒", "tint": (30, 72, 122),
         "ink_hex": "#0E2E52", "accent_hex": "#2F6FA8"},
        {"key": "teal", "name_zh": "深青", "tint": (20, 68, 74),
         "ink_hex": "#062A2E", "accent_hex": "#3FA8A0"},
        {"key": "graphite", "name_zh": "石墨", "tint": (58, 60, 64),
         "ink_hex": "#1C1E22", "accent_hex": "#9AA3AD"},
        {"key": "cream", "name_zh": "米黄图纸", "tint": (228, 218, 192),
         "ink_hex": "#3A3325", "accent_hex": "#A98B4E"},
        {"key": "oxblood", "name_zh": "暗红", "tint": (92, 42, 38),
         "ink_hex": "#2E1110", "accent_hex": "#C4655A"},
    ],
    "dish_philo": [
        {"key": "rice", "name_zh": "宣纸", "tint": (238, 233, 222),
         "ink_hex": "#2A2622", "accent_hex": "#8A7B66"},
        {"key": "fogblue", "name_zh": "浅灰蓝", "tint": (226, 231, 236),
         "ink_hex": "#232A31", "accent_hex": "#6E8AA6"},
        {"key": "ochre", "name_zh": "淡赭", "tint": (234, 222, 199),
         "ink_hex": "#35291B", "accent_hex": "#A9803F"},
        {"key": "moss", "name_zh": "苔绿纸", "tint": (228, 232, 220),
         "ink_hex": "#262B22", "accent_hex": "#6E8055"},
        {"key": "sand", "name_zh": "暖沙", "tint": (238, 228, 208),
         "ink_hex": "#332A1B", "accent_hex": "#B08A4A"},
    ],
}


def variant_catalog() -> Dict[str, List[Dict[str, Any]]]:
    """网页端用：只暴露 key / 中文名 / RGB，内部字段不外泄。"""
    return {
        pk: [
            {"key": v["key"], "name_zh": v["name_zh"], "rgb": list(v["tint"])}
            for v in vs
        ]
        for pk, vs in TINT_VARIANTS.items()
    }


def resolve_tone(preset_key: str, variant_key: str = "") -> Tone:
    """取该预设的基调，并按 variant_key 覆盖 tint / ink / accent。"""
    tone = TONES[PRESETS[preset_key].tone]
    for v in TINT_VARIANTS.get(preset_key, []):
        if v["key"] == variant_key:
            return replace(
                tone,
                tint=v["tint"],
                ink_hex=v["ink_hex"],
                accent_hex=v["accent_hex"],
                name_zh=f"{tone.name_zh}·{v['name_zh']}",
            )
    return tone


def variant_keys(preset_key: str) -> List[str]:
    """随机配色时从这里抽，不包含基调本身。"""
    return [v["key"] for v in TINT_VARIANTS.get(preset_key, [])]


PRESETS: Dict[str, Preset] = {
    # 只能学到色调场，网格、尺寸箭头、墨弧这类符号化图形必须由代码画。
    "dish_math": Preset(
        key="dish_math",
        name_zh="高等数学",
        name_en="advanced mathematics",
        scenario_zh="高数、线代、概率论等数学类课程",
        keywords_en=[
            "dark chalkboard powerpoint background",
            "faint coordinate grid paper",
            "single elegant function curve in accent colour",
            "large quiet area for formulas and text",
            "no text, no letters, no watermark",
        ],
        tone="dish_math",
        layouts=("vignette_plain", "center_band", "top_band"),
        photo_roles=("texture", "any"),
        texture_weight=0.12,
    ),
    "dish_mech": Preset(
        key="dish_mech",
        name_zh="机械制图",
        name_en="mechanical drafting",
        scenario_zh="机械制图、工程图学、CAD 与设计类课程",
        keywords_en=[
            "engineering blueprint powerpoint background",
            "thin technical line frame",
            "dimension arrows and leader lines",
            "subtle hatching in one corner",
            "wide clean drawing area",
            "no text, no letters, no watermark",
        ],
        tone="dish_mech",
        layouts=("vignette_plain", "center_band", "top_band"),
        photo_roles=("texture", "any"),
        texture_weight=0.10,
    ),
    "dish_philo": Preset(
        key="dish_philo",
        name_zh="人生哲学与科学方法论",
        name_en="philosophy and scientific method",
        scenario_zh="思政、哲学、科学方法论、通识与读书报告",
        keywords_en=[
            "warm rice paper powerpoint background",
            "one single ink brush arc",
            "very large empty space",
            "calm restrained scholarly atmosphere",
            "no text, no letters, no watermark",
        ],
        tone="dish_philo",
        layouts=("vignette_plain", "center_band", "top_band"),
        photo_roles=("texture", "any"),
        texture_weight=0.30,
    ),
    "academic_blue": Preset(
        key="academic_blue",
        name_zh="学术汇报蓝",
        name_en="academic report blue",
        scenario_zh="课程报告、毕业论文答辩、项目汇报",
        keywords_en=[
            "clean powerpoint background",
            "muted academic blue tone",
            "large empty area reserved for text",
            "soft out-of-focus photographic texture",
            "subtle frosted glass panel",
            "no text, no letters, no watermark",
        ],
        tone="academic_blue",
        layouts=("left_panel", "right_panel", "center_band", "corner_soft", "top_band"),
        photo_roles=("photo", "texture", "any"),
        texture_weight=0.22,
    ),
    "clean_white": Preset(
        key="clean_white",
        name_zh="极简白",
        name_en="minimal white",
        scenario_zh="产品介绍、商业计划、简洁答辩",
        keywords_en=[
            "minimal light powerpoint background",
            "near-white base with one soft accent",
            "very large clean empty area",
            "extremely low visual noise",
            "thin geometric accent line",
            "no text, no letters, no watermark",
        ],
        tone="clean_white",
        layouts=("corner_soft", "right_panel", "center_band", "left_panel", "corner_card"),
        photo_roles=("texture", "any"),
        texture_weight=0.10,
    ),
    "warm_terra": Preset(
        key="warm_terra",
        name_zh="暖陶土人文",
        name_en="warm terracotta humanistic",
        scenario_zh="人文社科、校园文化、公益宣讲",
        keywords_en=[
            "warm terracotta powerpoint background",
            "humanistic earthy tone",
            "paper-like grain texture",
            "left third occupied by soft abstract shape",
            "generous empty space on the right for text",
            "no text, no letters, no watermark",
        ],
        tone="warm_terra",
        layouts=("left_panel", "right_panel", "corner_soft", "corner_card"),
        photo_roles=("photo", "texture", "any"),
        texture_weight=0.30,
    ),
    "forest_deep": Preset(
        key="forest_deep",
        name_zh="深林绿环保",
        name_en="deep forest environmental",
        scenario_zh="环保生态、农业、生命科学",
        keywords_en=[
            "deep green environmental powerpoint background",
            "natural botanical tonality",
            "soft focus organic shapes",
            "wide clean text area with good contrast",
            "no text, no letters, no watermark",
        ],
        tone="forest_deep",
        layouts=("left_panel", "right_panel", "center_band", "corner_soft", "top_band"),
        photo_roles=("photo", "any"),
        texture_weight=0.20,
    ),
    "violet_tech": Preset(
        key="violet_tech",
        name_zh="科技紫",
        name_en="violet technology",
        scenario_zh="算法竞赛、技术分享、产品发布",
        keywords_en=[
            "dark violet technology powerpoint background",
            "subtle diagonal light streaks",
            "deep gradient falloff toward edges",
            "large dark empty area for light-colored text",
            "no text, no letters, no watermark",
        ],
        tone="violet_tech",
        layouts=("center_band", "top_band", "left_panel", "right_panel", "vignette_plain"),
        photo_roles=("texture", "any"),
        texture_weight=0.26,
    ),
    "ink_mono": Preset(
        key="ink_mono",
        name_zh="水墨灰",
        name_en="ink monochrome",
        scenario_zh="文学艺术、总结回顾、正式场合",
        keywords_en=[
            "monochrome ink wash powerpoint background",
            "grayscale paper texture with soft bleed",
            "asymmetric composition, lots of negative space",
            "restrained and formal",
            "no text, no letters, no watermark",
        ],
        tone="ink_mono",
        layouts=("corner_soft", "left_panel", "right_panel", "vignette_plain"),
        photo_roles=("texture", "photo", "any"),
        texture_weight=0.34,
    ),
}


LAYOUTS: Dict[str, Dict[str, Any]] = {
    # 每种版式定义：面板位置、面板占比、以及由此推导出的安全区语义。
    "left_panel": {
        "desc_zh": "左侧竖条面板，右侧整片留白",
        "panel_side": "left",
        "panel_width_ratio": 0.38,
        "body_side": "right",
    },
    "right_panel": {
        "desc_zh": "右侧竖条面板，左侧整片留白",
        "panel_side": "right",
        "panel_width_ratio": 0.34,
        "body_side": "left",
    },
    "corner_card": {
        "desc_zh": "单个圆角卡片压在角落",
        "panel_side": "left",
        "panel_width_ratio": 0.30,
        "panel_height_ratio": 0.46,
        "panel_anchor": "corner",
        "body_side": "right",
    },
    "top_band": {
        "desc_zh": "顶部横向色带，下方大留白",
        "panel_side": "top",
        "panel_width_ratio": 0.0,
        "panel_height_ratio": 0.30,
        "body_side": "bottom",
    },
    "center_band": {
        "desc_zh": "中央横向色带，上下留白",
        "panel_side": "none",
        "panel_width_ratio": 0.0,
        "body_side": "bottom",
    },
    "corner_soft": {
        "desc_zh": "单角柔和色块，其余大面积留白",
        "panel_side": "left",
        "panel_width_ratio": 0.26,
        "body_side": "right",
    },
    "vignette_plain": {
        "desc_zh": "整体压暗 + 暗角，无实色面板",
        "panel_side": "none",
        "panel_width_ratio": 0.0,
        "body_side": "center",
    },
}

ASPECTS: Dict[str, Tuple[int, int]] = {
    "16:9": (1920, 1080),
    "4:3": (1600, 1200),
    "9:16": (1080, 1920),
}


def preset_choices_zh() -> List[Dict[str, str]]:
    """给网页端下拉框用的中文选项列表。"""
    return [
        {
            "key": p.key,
            "label": f"{p.name_zh}｜{p.scenario_zh}",
            "accent": TONES[p.tone].accent_hex,
        }
        for p in PRESETS.values()
    ]


def compose_prompt(preset_key: str, subject_en: str, extra: str = "") -> str:
    """把预设与用户输入的主题词拼成一条完整 prompt。"""
    p = PRESETS[preset_key]
    subject = to_ascii_hint(subject_en)
    parts = [subject] if subject else []
    parts.extend(p.keywords_en)
    if extra.strip():
        parts.append(extra.strip())
    parts.append("high resolution, 16:9")
    text = ", ".join(x for x in parts if x)
    return _trim_to_budget(text, keep_tail=len(p.keywords_en))


ZH_HINTS = {
    "毕业": "graduation", "答辩": "thesis defense", "论文": "academic paper",
    "学术": "academic", "研究": "research", "课题": "research project",
    "课程": "coursework", "期末": "final term", "作业": "assignment",
    "机器学习": "machine learning", "人工智能": "artificial intelligence",
    "算法": "algorithm", "竞赛": "competition", "比赛": "contest",
    "数据": "data", "分析": "analysis", "模型": "model",
    "乡村": "rural village", "支教": "volunteer teaching", "公益": "public welfare",
    "环保": "environmental protection", "雨水": "rainwater", "回收": "recycling",
    "系统": "system", "工程": "engineering", "城市": "urban city",
    "设计": "design", "建筑": "architecture", "医疗": "medical",
    "健康": "health", "金融": "finance", "商业": "business",
    "创业": "startup", "计划": "plan", "总结": "annual summary",
    "汇报": "report", "分享": "sharing session", "技术": "technology",
    "产品": "product", "发布": "launch", "团队": "team",
    "教育": "education", "教学": "teaching", "学生": "student",
    "科研": "scientific research", "实验": "experiment", "创新": "innovation",
}


def to_ascii_hint(text: str) -> str:
    """把用户输入压成 CLIP 能理解的一段英文提示词。"""
    text = (text or "").strip()
    if not text:
        return ""
    out: list = []
    buf = ""
    i = 0
    keys = sorted(ZH_HINTS, key=len, reverse=True)
    while i < len(text):
        hit = None
        for k in keys:
            if text.startswith(k, i):
                hit = k
                break
        if hit:
            if buf.strip():
                out.append(buf.strip())
            buf = ""
            out.append(ZH_HINTS[hit])
            i += len(hit)
            continue
        ch = text[i]
        buf += ch if ord(ch) < 128 else " "
        i += 1
    if buf.strip():
        out.append(buf.strip())
    seen, uniq = set(), []
    for w in out:
        w = " ".join(w.split())
        if not w or w.lower() in seen:
            continue
        seen.add(w.lower())
        uniq.append(w)
    return ", ".join(uniq)


def _approx_tokens(text: str) -> int:
    """粗估 CLIP token 数：英文按 4 字符 1 token，中文按 1 字 1 token 偏保守估。
    
    必须能在只装了 Pillow/numpy的轻量环境下跑起来。"""
    n = 0.0
    for ch in text:
        n += 1.0 if ch >= "\u4e00" else 0.25
    return int(n + 0.999)


def _trim_to_budget(text: str, keep_tail: int, budget: int = 74) -> str:
    """把 prompt 裁进 CLIP 的 77 token 窗口，且保证风格词不被截掉。"""
    if _approx_tokens(text) <= budget:
        return text
    parts = [p.strip() for p in text.split(",") if p.strip()]
    protected = set()
    for p in parts[-(keep_tail + 1):]:
        protected.add(p.lower())
    dropped = 0
    while _approx_tokens(", ".join(parts)) > budget and dropped < len(parts):
        for idx, p in enumerate(parts):
            if p.lower() not in protected:
                parts.pop(idx)
                dropped += 1
                break
        else:
            break
    return ", ".join(parts)


if __name__ == "__main__":
    assert all(p.tone in TONES for p in PRESETS.values()), "预设引用了未定义的色调"
    assert all(l in LAYOUTS for p in PRESETS.values() for l in p.layouts), "预设引用了未定义的版式"
    for key, p in PRESETS.items():
        print(f"{key:14s} {p.name_zh:8s} tone={p.tone:14s} layouts={','.join(p.layouts)}")
    print()
    print(compose_prompt("academic_blue", "a lecture hall", ""))
