#!/usr/bin/env python3
"""
liltrAIner Dashboard — Real-time training monitor.
Run: python dashboard.py
Open: http://localhost:8888
"""

import json
import subprocess
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

BASE = Path(__file__).parent
LOG_FILE = BASE / "experiment_log.jsonl"
BEST_FILE = BASE / "best_config.json"
STATUS_FILE = BASE / "status.json"
AGENT_LOG = Path("/tmp/liltrainer_agent.log")

app = FastAPI(title="liltrAIner Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_start = time.time()


def _read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default or {}


def _read_jsonl(path):
    results = []
    try:
        for line in open(path):
            line = line.strip()
            if line:
                results.append(json.loads(line))
    except Exception:
        pass
    return results


def _read_agent_log():
    if not AGENT_LOG.exists():
        return []
    try:
        lines = AGENT_LOG.read_text().split('\n')
        seen = set()
        out = []
        for line in lines:
            c = line.strip()
            if not c:
                continue
            # Skip terminal noise
            if any(c.startswith(ch) for ch in '\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f'):
                continue
            if c.startswith('context:'):
                continue
            # Deduplicate short lines
            key = c[:60]
            if key in seen and len(c) < 100:
                continue
            seen.add(key)
            out.append(c)
        return out[-300:]
    except Exception:
        return []


def _detect_phase():
    for name in ["train.py", "eval.py"]:
        try:
            r = subprocess.run(["pgrep", "-fl", name], capture_output=True, text=True, timeout=2)
            if name in r.stdout:
                return "training" if "train" in name else "evaluating"
        except Exception:
            pass
    for name in ["claude", "kimi", "codex"]:
        try:
            r = subprocess.run(["pgrep", "-fl", name], capture_output=True, text=True, timeout=2)
            if name in r.stdout:
                return "thinking"
        except Exception:
            pass
    return "idle"


def _git_log():
    try:
        r = subprocess.run(["git", "log", "--oneline", "-20"], capture_output=True, text=True, cwd=str(BASE), timeout=3)
        return [l for l in r.stdout.strip().split('\n') if l]
    except Exception:
        return []


@app.get("/api/status")
async def api_status():
    experiments = _read_jsonl(LOG_FILE)
    # Deduplicate by ID
    by_id = {}
    for e in experiments:
        eid = e.get("id", 0)
        if eid not in by_id or e.get("score", 0) > by_id[eid].get("score", 0):
            by_id[eid] = e
    experiments = sorted(by_id.values(), key=lambda e: e.get("id", 0))

    best = _read_json(BEST_FILE, {"score": 0})
    for e in experiments:
        if e.get("success") and e.get("score", 0) > best.get("score", 0):
            best = {"score": e["score"], "config": e.get("config")}

    return {
        "phase": _detect_phase(),
        "best": best,
        "experiments": experiments,
        "commits": _git_log(),
        "agent_log": _read_agent_log(),
        "uptime": time.time() - _start,
        "status": _read_json(STATUS_FILE),
    }


@app.get("/")
async def index():
    return HTMLResponse(HTML)


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>liltrAIner</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#09090b;--card:#111113;--border:#1c1c1f;--teal:#0d9488;--green:#22c55e;--red:#ef4444;--yellow:#eab308;--purple:#8b5cf6;--text:#e4e4e7;--dim:#52525b;--mono:'SF Mono',Menlo,monospace}
body{background:var(--bg);color:var(--text);font-family:-apple-system,system-ui,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}

/* Header */
.hdr{padding:12px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px;flex-shrink:0}
.hdr h1{font-size:15px;font-weight:600;letter-spacing:-0.3px}
.hdr h1 span{color:var(--teal)}
.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.dot.training{background:var(--teal);box-shadow:0 0 8px var(--teal);animation:glow 1.5s infinite}
.dot.evaluating{background:var(--yellow);box-shadow:0 0 8px var(--yellow);animation:glow 1s infinite}
.dot.thinking{background:var(--purple);box-shadow:0 0 8px var(--purple);animation:glow 2s infinite}
.dot.idle{background:var(--dim)}
@keyframes glow{0%,100%{opacity:1}50%{opacity:.3}}
.badge{font-size:9px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;padding:2px 7px;border-radius:3px}
.badge.training{background:rgba(13,148,136,.15);color:var(--teal)}
.badge.evaluating{background:rgba(234,179,8,.15);color:var(--yellow)}
.badge.thinking{background:rgba(139,92,246,.15);color:var(--purple)}
.badge.idle{background:rgba(82,82,91,.15);color:var(--dim)}
.best{margin-left:auto;font:700 22px var(--mono);color:var(--green)}
.best small{display:block;font:400 10px var(--mono);color:var(--dim);text-align:right}

/* Stats */
.stats{display:flex;border-bottom:1px solid var(--border);flex-shrink:0}
.stat{flex:1;padding:8px 14px;border-right:1px solid var(--border)}
.stat:last-child{border:none}
.stat-k{font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:.5px}
.stat-v{font:700 18px var(--mono);margin-top:1px}

/* Layout */
.main{flex:1;display:grid;grid-template-columns:1fr 1fr;overflow:hidden}
.col{display:flex;flex-direction:column;overflow:hidden}
.col:first-child{border-right:1px solid var(--border)}
.panel-hdr{padding:8px 14px;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:var(--dim);border-bottom:1px solid var(--border);flex-shrink:0;display:flex;align-items:center;gap:6px}
.panel-hdr .n{background:var(--border);padding:0 5px;border-radius:3px;font-size:9px}
.panel{flex:1;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border) transparent}

/* Feed */
.feed{padding:6px 10px;font:11px/1.8 var(--mono)}
.fl{padding:1px 0;display:flex;gap:6px}
.fl-t{color:var(--dim);font-size:9px;flex-shrink:0;min-width:38px}
.fl-c{flex:1;word-break:break-word}
.fl.tool .fl-c{color:var(--teal)}
.fl.think .fl-c{color:var(--purple);font-style:italic}
.fl.result .fl-c{color:var(--green);font-weight:600}
.fl.error .fl-c{color:var(--red)}

/* Chart */
.chart-wrap{height:140px;padding:10px 12px;border-bottom:1px solid var(--border);flex-shrink:0}
canvas{width:100%;height:100%}

/* Table */
.tbl{width:100%;font:11px var(--mono);border-collapse:collapse}
.tbl th{text-align:left;padding:5px 8px;color:var(--dim);font-size:9px;text-transform:uppercase;letter-spacing:.3px;border-bottom:1px solid var(--border);position:sticky;top:0;background:var(--bg)}
.tbl td{padding:4px 8px;border-bottom:1px solid rgba(255,255,255,.02)}
.tbl .best-row{background:rgba(13,148,136,.05)}
.g{color:var(--green)}.r{color:var(--red)}.y{color:var(--yellow)}

/* Commits */
.cm{padding:3px 12px;font:10px var(--mono);color:var(--dim);border-bottom:1px solid rgba(255,255,255,.02)}
.cm b{color:var(--yellow);font-weight:400;margin-right:5px}
.cm span{color:var(--text)}

@media(max-width:768px){.main{grid-template-columns:1fr;grid-template-rows:1fr 1fr}}
</style>
</head>
<body>
<div class="hdr">
  <div class="dot idle" id="dot"></div>
  <h1>lil<span>trAI</span>ner</h1>
  <div class="badge idle" id="badge">idle</div>
  <div class="best" id="best">--<small id="round"></small></div>
</div>
<div class="stats" id="stats"></div>
<div class="main">
  <div class="col">
    <div class="panel-hdr">Agent Feed <span class="n" id="fn">0</span></div>
    <div class="panel" id="fp"><div class="feed" id="feed"></div></div>
  </div>
  <div class="col">
    <div class="panel-hdr">Score</div>
    <div class="chart-wrap"><canvas id="cv"></canvas></div>
    <div class="panel-hdr">Experiments <span class="n" id="en">0</span></div>
    <div class="panel" id="ep"><table class="tbl" id="et"><thead><tr><th>#</th><th>Score</th><th>Loss</th><th>Config</th><th></th></tr></thead><tbody></tbody></table></div>
    <div class="panel-hdr">Commits <span class="n" id="cn">0</span></div>
    <div class="panel" style="max-height:120px" id="cp"></div>
  </div>
</div>
<script>
let pll=0,asc=true;
const esc=s=>(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

async function go(){
  try{
    const d=await(await fetch('/api/status')).json();
    hdr(d);bar(d);feed(d);chart(d);exps(d);commits(d);
  }catch(e){}
}

function hdr(d){
  const p=d.phase||'idle';
  document.getElementById('dot').className='dot '+p;
  const b=document.getElementById('badge');b.className='badge '+p;b.textContent=p;
  const bs=d.best?.score||0;
  document.getElementById('best').innerHTML=(bs*100).toFixed(1)+'%';
  const rs=(d.experiments||[]).filter(e=>e.success&&e.score).map(e=>e.score);
  const rb=rs.length?Math.max(...rs):0;
  document.getElementById('best').innerHTML+='<small>best this round: '+(rb*100).toFixed(1)+'%</small>';
}

function bar(d){
  const e=d.experiments||[],ok=e.filter(x=>x.success).length,fl=e.filter(x=>!x.success).length;
  const u=d.uptime||0,h=Math.floor(u/3600),m=Math.floor(u%3600/60);
  document.getElementById('stats').innerHTML=
    '<div class="stat"><div class="stat-k">Experiments</div><div class="stat-v" style="color:var(--teal)">'+ok+'</div></div>'+
    '<div class="stat"><div class="stat-k">Failed</div><div class="stat-v">'+fl+'</div></div>'+
    '<div class="stat"><div class="stat-k">Best Config</div><div class="stat-v" style="font-size:11px;color:var(--dim)">'+(d.best?.config?JSON.stringify(d.best.config).slice(0,30):'—')+'</div></div>'+
    '<div class="stat"><div class="stat-k">Uptime</div><div class="stat-v">'+(h?h+'h ':'')+m+'m</div></div>';
}

function feed(d){
  const li=d.agent_log||[];
  if(li.length===pll)return;
  pll=li.length;
  const f=document.getElementById('feed');
  let h='';
  li.forEach((l,i)=>{
    if(!l.trim())return;
    let c='dim';const s=l.trim();
    if(s.startsWith('\u2022')&&s.includes('Used'))c='tool';
    else if(s.match(/score|PASS|BEST|Score:|result/i))c='result';
    else if(s.match(/FAIL|Error|error|failed|timed out/i))c='error';
    else if(s.startsWith('\u2022')||s.startsWith('Started')||s.match(/\[(starting|completed|running)\]/))c='think';
    const age=(li.length-i)*2;
    const t=new Date(Date.now()-age*1000);
    const ts=String(t.getHours()).padStart(2,'0')+':'+String(t.getMinutes()).padStart(2,'0')+':'+String(t.getSeconds()).padStart(2,'0');
    h+='<div class="fl '+c+'"><span class="fl-t">'+ts+'</span><span class="fl-c">'+esc(s)+'</span></div>';
  });
  f.innerHTML=h;
  document.getElementById('fn').textContent=li.length;
  if(asc){document.getElementById('fp').scrollTop=document.getElementById('fp').scrollHeight}
}

function chart(d){
  const pts=(d.experiments||[]).filter(e=>e.success&&e.score).map(e=>e.score*100);
  const cv=document.getElementById('cv'),cx=cv.getContext('2d');
  const dp=devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;
  cv.width=w*dp;cv.height=h*dp;cx.scale(dp,dp);cx.clearRect(0,0,w,h);
  if(pts.length<1){cx.fillStyle='#333';cx.font='11px sans-serif';cx.fillText('Waiting for results...',w/2-55,h/2);return}
  const pad={t:12,r:8,b:16,l:36},cw=w-pad.l-pad.r,ch=h-pad.t-pad.b;
  const mn=Math.min(...pts)*.95,mx=Math.max(...pts)*1.02,rng=mx-mn||1;
  for(let i=0;i<=3;i++){const y=pad.t+ch-(i/3)*ch;cx.strokeStyle='#1a1a1a';cx.lineWidth=.5;cx.beginPath();cx.moveTo(pad.l,y);cx.lineTo(pad.l+cw,y);cx.stroke();cx.fillStyle='#444';cx.font='9px monospace';cx.textAlign='right';cx.fillText((mn+(i/3)*rng).toFixed(0)+'%',pad.l-3,y+3)}
  // Area
  cx.beginPath();pts.forEach((v,i)=>{const x=pad.l+(pts.length===1?cw/2:(i/(pts.length-1))*cw),y=pad.t+ch-((v-mn)/rng)*ch;i?cx.lineTo(x,y):cx.moveTo(x,y)});
  const lx=pad.l+(pts.length===1?cw/2:cw);cx.lineTo(lx,pad.t+ch);cx.lineTo(pad.l,pad.t+ch);cx.closePath();cx.fillStyle='rgba(13,148,136,.06)';cx.fill();
  // Line
  cx.beginPath();cx.strokeStyle='#0d9488';cx.lineWidth=2;cx.lineJoin='round';
  pts.forEach((v,i)=>{const x=pad.l+(pts.length===1?cw/2:(i/(pts.length-1))*cw),y=pad.t+ch-((v-mn)/rng)*ch;i?cx.lineTo(x,y):cx.moveTo(x,y)});cx.stroke();
  // Best dot
  const bv=Math.max(...pts),bi=pts.lastIndexOf(bv);
  const bx=pad.l+(pts.length===1?cw/2:(bi/(pts.length-1))*cw),by=pad.t+ch-((bv-mn)/rng)*ch;
  cx.beginPath();cx.arc(bx,by,4,0,Math.PI*2);cx.fillStyle='#22c55e';cx.fill();
  cx.fillStyle='#22c55e';cx.font='bold 9px monospace';cx.textAlign='center';cx.fillText(bv.toFixed(1)+'%',bx,by-8);
}

function exps(d){
  const ex=[...(d.experiments||[])].reverse(),bs=d.best?.score||0;
  document.getElementById('en').textContent=ex.length;
  const tb=document.querySelector('#et tbody');
  if(!ex.length){tb.innerHTML='<tr><td colspan="5" style="color:var(--dim);text-align:center;padding:16px">No experiments yet</td></tr>';return}
  tb.innerHTML=ex.slice(0,40).map(e=>{
    const ib=e.score===bs&&e.score>0,cfg=e.config||{};
    const desc=cfg.desc||cfg.change||Object.entries(cfg).filter(([k])=>k!=='desc'&&k!=='change').map(([k,v])=>k+'='+v).join(' ');
    return'<tr class="'+(ib?'best-row':'')+'"><td>'+e.id+'</td><td class="'+(e.success?'g':'r')+'">'+(e.success?((e.score||0)*100).toFixed(1)+'%':'—')+'</td><td style="color:var(--dim)">'+(e.train_loss!=null?e.train_loss.toFixed(3):'—')+'</td><td style="color:var(--dim);max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="'+esc(desc)+'">'+esc(desc.slice(0,40))+'</td><td class="'+(ib?'g':'')+'">'+(ib?'BEST':'')+'</td></tr>'
  }).join('');
}

function commits(d){
  const c=d.commits||[];
  document.getElementById('cn').textContent=c.length;
  document.getElementById('cp').innerHTML=c.map(x=>{const[s,...m]=x.split(' ');return'<div class="cm"><b>'+s+'</b><span>'+esc(m.join(' '))+'</span></div>'}).join('');
}

document.getElementById('fp')?.addEventListener('scroll',function(){asc=this.scrollTop+this.clientHeight>=this.scrollHeight-30});
go();setInterval(go,3000);
</script>
</body>
</html>"""

if __name__ == "__main__":
    print("\n  liltrAIner Dashboard: http://localhost:8888\n")
    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="warning")
