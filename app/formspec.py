"""AI 生成 PPT 背景 —— 需求输入表单。

为什么要有这一层：本作品的主张是把模糊的一句话需求，翻译成可控的视觉配方。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SCENARIOS: List[Dict[str, str]] = [
    {"key": "math", "zh": "高等数学", "preset": "dish_math",
     "hint": "坐标纸 · 函数曲线", "group": "学科"},
    {"key": "mech", "zh": "机械制图", "preset": "dish_mech",
     "hint": "图框 · 尺寸标注", "group": "学科"},
    {"key": "philo", "zh": "人生哲学与科学方法论", "preset": "dish_philo",
     "hint": "宣纸 · 一笔墨弧", "group": "学科"},
    {"key": "academic", "zh": "课程报告 / 论文答辩", "preset": "academic_blue",
     "hint": "严谨，字多", "group": "行业风格"},
    {"key": "startup", "zh": "商业计划 / 产品发布", "preset": "clean_white",
     "hint": "极简，留白多", "group": "行业风格"},
    {"key": "culture", "zh": "人文社科 / 校园公益", "preset": "warm_terra",
     "hint": "有温度，纸质感", "group": "行业风格"},
    {"key": "green", "zh": "环保 / 农业 / 生命科学", "preset": "forest_deep",
     "hint": "自然，绿色调", "group": "行业风格"},
    {"key": "tech", "zh": "算法竞赛 / 技术分享", "preset": "violet_tech",
     "hint": "暗底，科技感", "group": "行业风格"},
    {"key": "formal", "zh": "总结回顾 / 正式场合", "preset": "ink_mono",
     "hint": "克制，灰阶", "group": "行业风格"},
]


def scenes_by_key() -> Dict[str, Dict[str, str]]:
    return {s["key"]: s for s in SCENARIOS}


def default_form_values() -> Dict[str, Any]:
    return {
        "scenario": SCENARIOS[0]["key"],
        "subject": "",
        "aspect": "16:9",
        "slides": 1,
        "title": "",
        "with_pptx": True,
        "steps": 28,
        "guidance": 6.5,
        "seed": -1,
        "strict_safe_zone": True,
    }


PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>演示稿背景生成器 · 一句话生成能压住字的背景</title>
<style>
  :root{
    --bg:#f5f6f8; --card:#ffffff; --line:#e4e7ee; --line2:#d3d8e3;
    --ink:#171a21; --dim:#6b7280; --mute:#9aa1ae;
    --accent:#4f46e5; --accent2:#eef0ff; --ok:#0f9d58; --warn:#b26b00; --err:#d93025;
    --shadow:0 1px 2px rgba(16,24,40,.06),0 8px 24px rgba(16,24,40,.06);
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:14px/1.6 "Segoe UI","Microsoft YaHei",system-ui,sans-serif;
       -webkit-font-smoothing:antialiased}
  header{display:flex;align-items:center;gap:14px;flex-wrap:wrap;
         padding:14px 28px;background:var(--card);border-bottom:1px solid var(--line)}
  .brand{display:flex;align-items:center;gap:10px}
  .dot{width:10px;height:10px;border-radius:3px;background:var(--accent);
       box-shadow:10px 0 0 -3px var(--accent2),0 0 0 4px var(--accent2)}
  header h1{font-size:15px;margin:0;font-weight:650;letter-spacing:.2px}
  header .sub{color:var(--dim);font-size:12.5px}
  .badge{margin-left:auto;font-size:12px;color:var(--dim);background:#f2f4f8;
         border:1px solid var(--line);border-radius:100px;padding:4px 11px;white-space:nowrap}
  .badge b{color:var(--ink);font-weight:600}
  .badge.warn{color:var(--warn);background:#fff8ec;border-color:#f5e0bb}

  main{display:grid;grid-template-columns:392px minmax(0,1fr);gap:20px;
       padding:20px 28px 32px;align-items:start;max-width:1440px;margin:0 auto}
  @media (max-width:1060px){main{grid-template-columns:1fr}}
  .panel{background:var(--card);border:1px solid var(--line);border-radius:14px;
         box-shadow:var(--shadow)}
  .form{padding:20px 20px 18px}
  .out{padding:16px 16px 14px;min-height:320px}

  .step{display:flex;align-items:center;gap:8px;font-size:13px;font-weight:600;
        margin:16px 0 8px}
  .step:first-child{margin-top:0}
  .step i{font-style:normal;width:18px;height:18px;border-radius:50%;background:var(--accent2);
          color:var(--accent);font-size:11px;font-weight:700;display:grid;place-items:center}
  .step em{font-style:normal;font-weight:400;color:var(--mute);font-size:12px}

  .chips{display:grid;grid-template-columns:1fr 1fr;gap:7px}
  /* 更多风格折叠区：默认收起，只在需要时展开，不抢学科卡的注意力 */
  .morewrap{margin-top:9px;border-top:1px dashed #e3e6ec;padding-top:8px}
  .morewrap summary{cursor:pointer;font-size:12px;color:#8a8f9a;list-style:none;
    padding:2px 0;user-select:none}
  .morewrap summary::-webkit-details-marker{display:none}
  .morewrap summary::before{content:"▸ ";font-size:11px}
  .morewrap[open] summary::before{content:"▾ "}
  .morewrap summary:hover{color:#4f46e5}
  .morewrap .chips{margin-top:8px}
  /* 配色圆点：只对学科预设出现。随机那个用斜纹区分，避免被当成第 6 种颜色 */
  .tones{display:flex;flex-wrap:wrap;gap:8px}
  .tone{width:34px;height:34px;border-radius:50%;border:2px solid transparent;
        cursor:pointer;position:relative;box-shadow:0 0 0 1px rgba(0,0,0,.10) inset}
  .tone.on{border-color:var(--accent);box-shadow:0 0 0 3px rgba(79,70,229,.16)}
  .tone.rand{border-radius:9px;
        background:repeating-linear-gradient(45deg,#d8dbe3 0 4px,#f2f3f7 4px 8px)}
  .tone i{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
        font-size:11px;font-style:normal;color:#5b6070}
  .chip{border:1px solid var(--line);border-radius:10px;padding:9px 10px;cursor:pointer;
        background:#fbfcfe;transition:border-color .12s,background .12s}
  .chip:hover{border-color:var(--line2)}
  .chip.on{border-color:var(--accent);background:var(--accent2)}
  .chip b{display:block;font-size:12.5px;font-weight:600;line-height:1.35}
  .chip span{display:block;color:var(--mute);font-size:11px;margin-top:1px}

  input[type=text],input[type=number],select{width:100%;background:#fbfcfe;
    border:1px solid var(--line);border-radius:9px;color:var(--ink);
    padding:9px 11px;font:inherit;outline:none;transition:border-color .12s}
  input[type=text]:focus,select:focus,input[type=number]:focus{border-color:var(--accent);
    background:#fff;box-shadow:0 0 0 3px rgba(79,70,229,.10)}

  .opts{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:16px}
  .opts label{font-size:12px;color:var(--dim);display:block;margin-bottom:4px}
  .switch{display:flex;align-items:center;gap:7px;font-size:12.5px;color:var(--dim);
          margin-top:14px;cursor:pointer}
  .switch input{width:15px;height:15px;accent-color:var(--accent)}
  details.adv{margin-top:12px;border-top:1px dashed var(--line);padding-top:10px}
  details.adv summary{cursor:pointer;font-size:12.5px;color:var(--dim);
                      list-style:none;user-select:none}
  details.adv summary::-webkit-details-marker{display:none}
  details.adv summary:before{content:"▸ ";color:var(--mute)}
  details.adv[open] summary:before{content:"▾ "}
  .adv .grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:10px}
  .adv .grid label{font-size:11px;color:var(--mute);display:block;margin-bottom:3px}

  button.primary{margin-top:18px;width:100%;padding:12px;border:0;border-radius:10px;
    background:var(--accent);color:#fff;font:inherit;font-weight:650;cursor:pointer;
    letter-spacing:.3px;transition:filter .12s}
  button.primary:hover{filter:brightness(1.07)}
  button.primary:disabled{opacity:.55;cursor:progress}
  .tip{color:var(--mute);font-size:11.5px;margin:10px 2px 0;line-height:1.55}

  .outhead{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px}
  .seg{display:flex;background:#f2f4f8;border:1px solid var(--line);border-radius:9px;padding:2px}
  .seg button{border:0;background:transparent;color:var(--dim);font:inherit;font-size:12.5px;
    padding:6px 12px;border-radius:7px;cursor:pointer}
  .seg button.on{background:#fff;color:var(--ink);font-weight:600;
    box-shadow:0 1px 2px rgba(16,24,40,.10)}
  .acts{margin-left:auto;display:flex;gap:8px}
  .acts a{font-size:12.5px;text-decoration:none;border:1px solid var(--line);
    background:#fbfcfe;color:var(--ink);border-radius:9px;padding:7px 12px;cursor:pointer}
  .acts a:hover{border-color:var(--line2);background:#fff}
  .acts a.acc{background:var(--accent);border-color:var(--accent);color:#fff}
  .acts a.acc:hover{filter:brightness(1.07)}
  .acts a[hidden]{display:none}

  .status{font-size:12.5px;color:var(--dim);background:#f8f9fb;border:1px solid var(--line);
    border-radius:10px;padding:9px 12px;margin-bottom:12px;white-space:pre-wrap}
  .status.err{color:var(--err);background:#fdf1f0;border-color:#f6cfcb}
  .status.ok{color:#0b6b3f;background:#f1fbf5;border-color:#c8e9d6}

  .stage{position:relative;border-radius:12px;overflow:hidden;background:#eef0f4;
    border:1px solid var(--line);display:none}
  .stage.on{display:block}
  .stage img{display:block;width:100%;height:auto}
  .olayer{position:absolute;inset:0}
  .txtbox{position:absolute;overflow:hidden;line-height:1.45;font-weight:700;
    letter-spacing:.5px;white-space:pre-wrap;word-break:break-word}
  .txtbox.body{font-weight:400;letter-spacing:.2px;line-height:1.6}
  .zonebox{position:absolute;border:1.5px dashed rgba(255,255,255,.9);border-radius:4px;
    font-size:10.5px;color:#fff;text-shadow:0 1px 3px #000;padding:2px 5px;
    background:rgba(0,0,0,.18)}
  .zonebox.bad{border-color:#ff8f8f;color:#ffd9d9}

  .empty{display:grid;place-items:center;height:300px;color:var(--mute);font-size:13px;
    border:1.5px dashed var(--line2);border-radius:12px;background:#fafbfd;text-align:center;
    padding:20px;line-height:1.9}
  .metrics{display:flex;gap:8px;flex-wrap:wrap;margin-top:11px;font-size:11.5px;color:var(--dim)}
  .metrics span{background:#f2f4f8;border:1px solid var(--line);border-radius:7px;padding:3px 8px}
  .metrics span b{color:var(--ink);font-weight:600}
  .thumbs{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
  .thumbs img{width:86px;height:auto;border-radius:7px;border:2px solid transparent;
    cursor:pointer;display:block}
  .thumbs img.on{border-color:var(--accent)}
  .acts button.ghost{background:#fff;color:var(--ink);border:1.5px solid var(--line2);
    border-radius:9px;padding:6px 12px;font-size:12.5px;font-family:inherit;cursor:pointer}
  .acts button.ghost:hover{border-color:var(--accent);color:var(--accent)}
  .acts button.ghost:disabled{opacity:.55;cursor:default;color:var(--mute);border-color:var(--line)}
  footer{color:var(--mute);font-size:11.5px;text-align:center;padding:0 20px 22px}
</style>
</head>
<body>
<header>
  <div class="brand"><span class="dot"></span>
    <h1>演示稿背景生成器</h1>
    <span class="sub">一句话说清场合 → 得到能压住字的 1920×1080 背景 → 直接导出可编辑 .pptx</span>
  </div>
  <span class="badge" id="health">正在连接…</span>
</header>

<main>
  <section class="panel form">
    <div class="step"><i>1</i>这份 PPT 是什么场合</div>
    <div class="chips" id="scenarios"></div>
    <details class="morewrap" id="morewrap" hidden>
      <summary>更多风格 · <span id="morecount">0</span> 种（行业模板取向，保留兼容）</summary>
      <div class="chips" id="scenarios-more"></div>
    </details>

    <div id="tonewrap" hidden>
      <div class="step"><i>2</i>配色<em>学科专用</em></div>
      <div class="tones" id="tones"></div>
    </div>

    <div id="blurwrap" hidden class="opts">
      <div><label>柔焦（整张纸的明暗起伏）</label>
        <select id="soft">
          <option value="0">关</option>
          <option value="0.35">弱</option>
          <option value="0.65" selected>中</option>
          <option value="1">强</option>
        </select></div>
      <div><label>磨砂块（只在空白处）</label>
        <select id="frost">
          <option value="0" selected>关</option>
          <option value="0.35">弱</option>
          <option value="0.65">中</option>
          <option value="1">强</option>
        </select></div>
      <div><label>磨砂形状</label>
        <select id="fshape">
          <option value="square" selected>方形卡</option>
          <option value="ellipse">椭圆雾</option>
          <option value="mix">随机</option>
        </select></div>
      <div><label>学科元素</label>
        <select id="richness">
          <option value="sparse">精简（只留骨架）</option>
          <option value="normal" selected>标准（+1 件）</option>
          <option value="rich">丰富（+2 件）</option>
        </select></div>
      <div><label>符号质感</label>
        <select id="style">
          <option value="brush">模型笔触（带纸纹）</option>
          <option value="vector">纯矢量（最锐利）</option>
        </select></div>
    </div>

    <div class="step"><i>2</i>内容主题<em>可留空</em></div>
    <input type="text" id="subject" placeholder="例如：校园垃圾分类调研 / 大模型推理加速">

    <div class="step"><i>3</i>标题<em>会真实压在预览图上</em></div>
    <input type="text" id="title" placeholder="例如：校园垃圾分类调研报告">

    <div class="opts">
      <div><label>画面比例</label>
        <select id="aspect">
          <option value="16:9">16:9 · 1920×1080</option>
          <option value="4:3">4:3 · 1600×1200</option>
          <option value="9:16">9:16 · 1080×1920</option>
        </select></div>
      <div><label>生成张数</label>
        <select id="count">
          <option value="1" selected>1 张</option>
          <option value="2">2 张</option>
          <option value="4">4 张</option>
        </select></div>
    </div>

    <label class="switch"><input type="checkbox" id="strict" checked>
      安全区约束（不达标的地方自动压暗）</label>

    <details class="adv"><summary>高级参数</summary>
      <div class="grid">
        <div><label>采样步数</label><input type="number" id="steps" value="28" min="8" max="60"></div>
        <div><label>引导强度</label><input type="number" id="guidance" value="6.5" step="0.5" min="1" max="15"></div>
        <div><label>种子（-1 随机）</label><input type="number" id="seed" value="-1"></div>
      </div>
    </details>

    <button class="primary" id="go">生成背景</button>
    <button class="deckbtn" id="deck">一次生成整套 PPT（8 页）</button>
    <style>.deckbtn{width:100%;margin-top:10px;padding:12px 14px;border-radius:10px;border:1px solid #c7cbe0;background:#fff;color:#2b2f45;font-size:14px;font-weight:600;cursor:pointer}.deckbtn:hover{border-color:#4f46e5;color:#4f46e5}.deckbtn:disabled{opacity:.55;cursor:default}</style>
    <p class="tip">安全区是系统判定的可以放心放字的矩形区域，按版式推导、用局部亮度标准差校验。
      文字预览里的字号与导出的 .pptx 完全一致（封面标题 40pt / 正文 17pt），
      所以这一屏看到的就是导出后的样子。</p>
  </section>

  <section class="panel out">
    <div class="outhead">
      <div class="seg" id="viewseg">
        <button data-v="text">文字预览</button>
        <button data-v="bg">纯背景</button>
        <button data-v="zone">安全区</button>
      </div>
      <div class="acts">
        <button id="regen" class="ghost" hidden title="换一个种子重画当前这一张，其他几张不动">重画这一张</button>
        <a id="dl" hidden>下载背景</a>
        <a id="pptx" class="acc" hidden>导出 .pptx</a>
      </div>
    </div>
    <div class="status" id="status">选择场合与标题后，点生成背景。</div>
    <div class="empty" id="empty">还没有生成结果<br><span style="font-size:12px">
      一张约 8 秒；本次开机第一次生成会先加载基座模型，约 20 秒</span></div>
    <div class="stage" id="stage">
      <img id="img" alt="">
      <div class="olayer" id="olayer"></div>
    </div>
    <div class="thumbs" id="thumbs"></div>
    <div class="metrics" id="metrics"></div>
  </section>
</main>

<footer>背景图由自训 LoRA 生成，安全区坐标与图像同一次渲染产出｜无第三方图片素材</footer>

<script>
const $ = (id) => document.getElementById(id);
const BODY_SAMPLE = "把这一页要讲的三件事写在这里，看看字压上去还清不清楚。";

let scenarios = [], presets = {}, variants = {}, images = [], idx = 0, view = "text", job = null, timer = null;
let toneSel = "";   // "" = 随机；否则是配色 key

/** 学科预设才显示配色 / 柔焦 / 磨砂这三组控件。 */
function syncTones(){
  const k = document.querySelector('.chip.on')?.dataset.k;
  const sc = scenarios.find(x => x.key === k) || scenarios[0] || {};
  const pk = sc.preset || '';
  const isDish = pk.startsWith('dish_');
  $('tonewrap').hidden = !isDish;
  $('blurwrap').hidden = !isDish;
  if(!isDish) return;
  const list = variants[pk] || [];
  const dots = [`<div class="tone rand ${toneSel === '' ? 'on' : ''}" data-v="" title="随机（同一 seed 仍然可复现）"><i>随机</i></div>`]
    .concat(list.map(v => {
      const [r,g,b] = v.rgb;
      return `<div class="tone ${toneSel === v.key ? 'on' : ''}" data-v="${v.key}" title="${v.name_zh}"
        style="background:rgb(${r},${g},${b})"></div>`;
    }));
  $('tones').innerHTML = dots.join('');
  document.querySelectorAll('#tones .tone').forEach(el => el.onclick = () => {
    toneSel = el.dataset.v;
    document.querySelectorAll('#tones .tone').forEach(x => x.classList.remove('on'));
    el.classList.add('on');
  });
}

async function boot(){
  try{
    const s = await (await fetch('api/scenarios')).json();
    scenarios = s.scenarios; presets = s.presets || {}; variants = s.variants || {};
    // 分组标题：学科在前、行业风格在后。分组只是视觉分隔，点哪张卡都等价。
    // 主界面只出学科三个；原来的六个行业风格收进更多风格折叠区，
    // 保留但不抢焦点（评测数字是在那六个上跑的，不能删）。
    const chip = sc => `<div class="chip" data-k="${sc.key}"><b>${sc.zh}</b><span>${sc.hint}</span></div>`;
    const main = scenarios.filter(sc => (sc.group || '学科') === '学科');
    const more = scenarios.filter(sc => (sc.group || '学科') !== '学科');
    $('scenarios').innerHTML = main.map(chip).join('');
    $('scenarios-more').innerHTML = more.map(chip).join('');
    $('morewrap').hidden = more.length === 0;
    $('morecount').textContent = String(more.length);
    document.querySelectorAll('.chip').forEach(el => el.onclick = () => {
      document.querySelectorAll('.chip').forEach(x => x.classList.remove('on'));
      el.classList.add('on');
      syncTones();
    });
    document.querySelector('.chip').classList.add('on');
    syncTones();
    const h = await (await fetch('api/health')).json();
    const be = String(h.backend || '');
    let txt;
    if(be.startsWith('diffusers')) txt = `生成后端 <b>Diffusers + LoRA</b>`;
    else if(be.startsWith('procedural')) txt = `生成后端 <b>程序化兜底</b>（未加载模型）`;
    else txt = `生成后端 <b>未加载</b>`;
    if(h.lora && be.startsWith('diffusers')) txt += ` · 已挂载自身 LoRA`;
    if(!h.ready) txt += ` · 首次生成时加载模型，约 20 秒`;
    $('health').innerHTML = txt;
    if(!h.ready) $('health').classList.add('warn');
  }catch(e){ $('health').textContent = '连接失败：' + e.message; }
}

function sel(k){ const e = document.querySelector(`.chip[data-k="${k}"]`); return e ? e.dataset.k : null; }

$('go').onclick = async () => {
  const k = document.querySelector('.chip.on')?.dataset.k;
  const sc = scenarios.find(x => x.key === k) || scenarios[0];
  const body = {
    preset: sc.preset,
    subject: $('subject').value.trim(),
    title: $('title').value.trim(),
    aspect: $('aspect').value,
    count: parseInt($('count').value, 10) || 1,
    steps: parseInt($('steps').value, 10) || 28,
    guidance: parseFloat($('guidance').value) || 6.5,
    seed: parseInt($('seed').value, 10),
    enforce_safe_zone: $('strict').checked
  };
  if(String(sc.preset || '').startsWith('dish_')){
    body.tone_variant = toneSel;              // "" = 随机
    body.tone_random = toneSel === '';
    body.soft_focus = parseFloat($('soft').value) || 0;
    body.frost = parseFloat($('frost').value) || 0;
    body.frost_shape = $('fshape').value || 'square';
    body.motif_richness = $('richness').value || 'normal';
    body.symbol_style = $('style').value || 'brush';
  }
  const t0 = Date.now();
  $('go').disabled = true; job = null;
  $('dl').hidden = true; $('pptx').hidden = true;
  setStatus('正在生成…', 'on');
  timer = setInterval(() => {
    setStatus(`正在生成… 已用 ${((Date.now()-t0)/1000).toFixed(0)} 秒`, 'on');
  }, 1000);
  try{
    const r = await fetch('api/generate', {method:'POST', headers:{'Content-Type':'application/json'},
                                          body: JSON.stringify(body)});
    const j = await r.json();
    if(!r.ok || j.ok === false) throw new Error(j.detail || j.error || r.statusText);
    images = j.images || []; idx = 0; job = j;
    show(0);
    setStatus(`完成：${images.length} 张，用时 ${((Date.now()-t0)/1000).toFixed(1)} 秒（后端 ${j.backend}）。`
              + (j.warnings?.length ? `\n提示：${j.warnings.join('；')}` : ''), 'ok');
  }catch(e){
    setStatus('出错：' + e.message, 'err');
  }finally{
    clearInterval(timer); $('go').disabled = false;
  }
};

function setStatus(t, cls){
  const el = $('status'); el.textContent = t;
  el.className = 'status' + (cls === 'ok' ? ' ok' : cls === 'err' ? ' err' : '');
}

function show(i){
  idx = i; const it = images[i]; if(!it) return;
  $('empty').style.display = 'none'; $('stage').classList.add('on');
  const im = $('img');
  im.onload = () => { draw(); };
  im.src = it.url;
  $('dl').hidden = false; $('dl').href = it.url; $('dl').setAttribute('download', `${it.id}.jpg`);
  $('pptx').hidden = false; $('regen').hidden = false;
  const zs = it.safe_zones || [];
  const std = zs.filter(z => z.region === 'title' || z.region === 'body')
                .map(z => z.contrast_std.toFixed(1)).join(' / ');
  $('metrics').innerHTML =
    `<span>尺寸 <b>${it.width}×${it.height}</b></span>` +
    `<span>预设 <b>${it.preset_zh || it.preset}</b></span>` +
    `<span>安全区对比度 <b>${std || '-'}</b>（越低越干净）</span>` +
    `<span>种子 <b>${it.seed}</b></span>` +
    `<span>耗时 <b>${it.seconds}s</b></span>`;
  $('thumbs').innerHTML = images.length > 1 ? images.map((x, n) =>
    `<img src="${x.url}" data-n="${n}" class="${n===i?'on':''}" alt="">`).join('') : '';
  document.querySelectorAll('#thumbs img').forEach(el => el.onclick = () => show(+el.dataset.n));
}

$('viewseg').onclick = (e) => {
  const b = e.target.closest('button'); if(!b) return;
  view = b.dataset.v;
  document.querySelectorAll('#viewseg button').forEach(x => x.classList.toggle('on', x === b));
  draw();
};

// 重画当前这一张：换一个种子、只重发这一张，结果写回同一个任务目录，
// 所以屏幕上看到的和导出 pptx 时取到的始终是同一份。
$('regen').onclick = async () => {
  if(!job || !images.length) return;
  const btn = $('regen');
  btn.disabled = true;
  const at = idx;
  setStatus('正在重画这一张…', 'on');
  try{
    const r = await fetch('api/regen', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({job_id: job.job_id, index: at})});
    const j = await r.json();
    if(!r.ok || j.ok === false) throw new Error(j.detail || j.error || r.statusText);
    images[at] = j.image; idx = at;
    show(at);
    setStatus(`已重画第 ${at+1} 张，新种子 ${j.image.seed}，用时 ${j.elapsed} 秒。`
              + '其他几张没动；导出 pptx 会自动用新图。', 'ok');
  }catch(e){
    setStatus('重画失败：' + e.message, 'err');
  }finally{
    btn.disabled = false;
  }
};

function lumOfZone(imgEl, z){
  try{
    const c = document.createElement('canvas');
    c.width = imgEl.naturalWidth; c.height = imgEl.naturalHeight;
    const g = c.getContext('2d'); g.drawImage(imgEl, 0, 0);
    const d = g.getImageData(z.x, z.y, Math.max(1, z.w), Math.max(1, z.h)).data;
    let s = 0, n = 0;
    for(let p = 0; p < d.length; p += 4*17){
      s += 0.299*d[p] + 0.587*d[p+1] + 0.114*d[p+2]; n++;
    }
    return n ? s/n : 128;
  }catch(e){ return 128; }
}

function draw(){
  const im = $('img'), layer = $('olayer');
  if(!im.naturalWidth || !images[idx]) return;
  const it = images[idx], zs = it.safe_zones || [];
  layer.innerHTML = '';
  if(view === 'zone'){
    zs.forEach(z => {
      const d = document.createElement('div');
      d.className = 'zonebox' + (z.contrast_std > 26 ? ' bad' : '');
      d.style.left = (z.x/z.width*100)+'%'; d.style.top = (z.y/z.height*100)+'%';
      d.style.width = (z.w/z.width*100)+'%'; d.style.height = (z.h/z.height*100)+'%';
      d.textContent = `${z.region} ${z.contrast_std.toFixed(1)}`;
      layer.appendChild(d);
    });
    return;
  }
  if(view !== 'text') return;
  const W = im.clientWidth, H = im.clientHeight;
  const scale = im.clientWidth / it.width;          // 1 图像素 = 多少 CSS 像素
  const pad = Math.min(it.width, it.height) * 0.02; // 与 pptx_exporter 的内边距一致
  const titleTxt = $('title').value.trim() || '在这里写下这一页的标题';
  zs.filter(z => ['title','body','footer'].includes(z.region)).forEach(z => {
    const txt = z.region === 'title' ? titleTxt : z.region === 'body' ? BODY_SAMPLE : '';
    if(!txt) return;
    const box = document.createElement('div');
    box.className = 'txtbox' + (z.region === 'title' ? ' title' : ' body');
    box.style.left = ((z.x + pad) / it.width * 100) + '%';
    box.style.top = ((z.y + pad) / it.height * 100) + '%';
    box.style.width = ((z.w - 2*pad) / it.width * 100) + '%';
    box.style.height = ((z.h - 2*pad) / it.height * 100) + '%';
    box.style.lineHeight = 1.25;
    box.textContent = txt;
    const lum = lumOfZone(im, z);
    box.style.color = lum < 140 ? '#ffffff' : '#14161c';
    box.style.textShadow = lum < 140 ? '0 2px 10px rgba(0,0,0,.45)' : '0 1px 8px rgba(255,255,255,.55)';
    layer.appendChild(box);
    // 字号与导出的 .pptx 完全一致：导出时 1 图像素 = 1/96 英寸 = 0.75 pt，
    // 所以 pt → 图像素是 ×4/3。只有放不下时才缩，缩到 6 成就不再缩。
    const pt = z.region === 'title' ? 40 : 17;
    fit(box, pt * (4/3) * scale, z.region === 'title');
  });
}

function fit(box, start, isTitle){
  let size = Math.max(9, Math.round(start));
  box.style.fontSize = size + 'px';
  const floor = Math.max(9, Math.round(start * 0.6));
  let guard = 0;
  while(size > floor && guard++ < 80 &&
        (box.scrollHeight > box.clientHeight + 1 || box.scrollWidth > box.clientWidth + 1)){
    size -= 1; box.style.fontSize = size + 'px';
  }
}

$('pptx').onclick = async () => {
  if(!job || !job.job_id) return;
  $('pptx').textContent = '正在打包…';
  try{
    const r = await fetch('api/export', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({job_id: job.job_id, topic: $('title').value.trim() || '演示主题',
                            slides: images.length})});
    const j = await r.json();
    if(!r.ok || j.ok === false) throw new Error(j.detail || j.error || r.statusText);
    const a = document.createElement('a');
    a.href = j.pptx_url; a.download = ''; document.body.appendChild(a); a.click(); a.remove();
    setStatus(`已导出 .pptx：${j.pptx_url}\n${j.size_kb} KB，背景铺满、标题与正文文本框可直接改字。`, 'ok');
  }catch(e){ setStatus('导出失败：' + e.message, 'err'); }
  finally{ $('pptx').textContent = '导出 .pptx'; }
};

// 一次生成整套 PPT：只要 3 张背景（循环复用），文字由服务端 plan_slides 规划。
// 这样"整套"的时间成本是可控的，而且几页之间的底色天然一致。
$('deck').onclick = async () => {
  // 注意：卡片上的 data-k 是场景 key（math / mech / philo…），
  // 服务端要的是预设 key（dish_math / dish_mech…）。必须查表转换，
  // 直接把 data-k 发给服务端就会得到 400未知预设。
  const k0 = document.querySelector('.chip.on')?.dataset.k;
  const sc0 = scenarios.find(x => x.key === k0) || scenarios[0] || {};
  const preset = sc0.preset || 'dish_math';
  $('deck').disabled = true;
  let n = 0;
  const t0 = Date.now();
  const tick = setInterval(() => {
    n = Math.round((Date.now() - t0) / 1000);
    setStatus(`正在生成整套 PPT（3 张底图 + 8 页排版）… 已用 ${n} 秒`);
  }, 1000);
  try{
    const r = await fetch('api/deck', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({preset, subject: $('subject').value.trim(),
                            title: $('title').value.trim(),
                            aspect: $('aspect').value || '16:9', slides: 8,
      tone_variant: String(preset||'').startsWith('dish_') ? toneSel : '',
      tone_random: String(preset||'').startsWith('dish_') && toneSel === '',
      soft_focus: String(preset||'').startsWith('dish_') ? (parseFloat($('soft').value)||0) : 0,
      frost: String(preset||'').startsWith('dish_') ? (parseFloat($('frost').value)||0) : 0,
    frost_shape: String(preset||'').startsWith('dish_') ? ($('fshape').value||'square') : 'square',
    motif_richness: String(preset||'').startsWith('dish_') ? ($('richness').value||'normal') : 'normal',
    symbol_style: String(preset||'').startsWith('dish_') ? ($('style').value||'brush') : 'vector'})});
    const j = await r.json();
    if(!r.ok || j.ok === false) throw new Error(j.detail || j.error || r.statusText);
    const a = document.createElement('a');
    a.href = j.pptx_url; a.download = ''; document.body.appendChild(a); a.click(); a.remove();
    setStatus(`整套 PPT 已生成：${j.slides && j.slides.slides || j.requested_slides} 页，${j.size_kb} KB，用时 ${n} 秒。\n封面/目录/正文/尾页文字已排好，背景与安全区一致，打开即可改字。`, 'ok');
  }catch(e){ setStatus('整套生成失败：' + e.message, 'err'); }
  finally{ clearInterval(tick); $('deck').disabled = false; }
};

$('title').oninput = () => { if(view === 'text' && images.length) draw(); };
window.addEventListener('resize', () => { if(images.length) draw(); });
document.querySelector('#viewseg button[data-v="text"]').classList.add('on');
boot();
</script>
</body>
</html>
"""


def render_page(title: str = "演示稿背景生成器") -> str:
    return PAGE_HTML.replace("$title", title)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
    from presets import PRESETS  # noqa: E402

    for s in SCENARIOS:
        assert s["preset"] in PRESETS, f"{s['key']} 指向了不存在的预设 {s['preset']}"
    html = render_page()
    assert "api/generate" in html and "安全区" in html and "导出" in html
    print(f"表单页自检通过：{len(SCENARIOS)} 个场景，HTML {len(html)} 字符")
    for s in SCENARIOS:
        print(f"  {s['key']:10s} {s['zh']:24s} -> {s['preset']}")
