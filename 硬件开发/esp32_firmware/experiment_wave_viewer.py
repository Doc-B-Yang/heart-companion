"""Local-only real sensor display. RAM capture restores existing firmware afterward."""
import csv
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / 'diagnostics' / 'experiment-waveforms'
FOLDER.mkdir(exist_ok=True)
STAMP = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
CSV_PATH = FOLDER / ('signals-' + STAMP + '.csv')
LOG_PATH = FOLDER / ('viewer-' + STAMP + '.log')
lock = threading.Lock()
state = dict(status='正在连接 COM6、初始化传感器…', ppg=[], eda=[], imu=[],
             elapsed=0, count=0, errors={}, updated=0, finished=False,
             csv=str(CSV_PATH), evidence='', result={})

def ingest(row, writer, output):
    with lock:
        kind = row.get('type')
        if kind == 'stage':
            state['status'] = '真实数据采集中 · 最长 10 分钟后自动结束'
        if kind == 'raw':
            state['elapsed'] = row['elapsed_ms'] / 1000
            for channel in ('ppg', 'eda', 'imu'):
                samples = row.get(channel, [])
                state[channel].extend(samples)
                state[channel] = state[channel][-12000:]
                for sample in samples:
                    writer.writerow([channel] + sample)
            state['count'] += len(row.get('ppg', []))
            state['updated'] = time.time()
            for fault in row.get('fifo_events', []):
                state['errors']['fifo_events'] = state['errors'].get('fifo_events', 0) + 1
            output.flush()
        elif kind == 'result':
            state['result'] = row
        elif kind == 'host_result':
            state['finished'] = True
            state['status'] = ('采集结束，已恢复原固件 · 显示最后一段实测数据' if row.get('startup_seen') and row.get('production_packet_seen') else '采集结束，固件恢复尚未确认')

def capture():
    cmd = [sys.executable, '-u', str(ROOT / 'run_bounded_raw_diagnostic.py'),
           '--port', 'COM6', '--worn', '--single-range', '--duration', '600',
           '--label', 'experiment-live', '--out-dir', str(FOLDER)]
    with CSV_PATH.open('x', newline='', encoding='utf-8-sig') as output, LOG_PATH.open('x', encoding='utf-8') as log:
        writer = csv.writer(output)
        writer.writerow(['channel', 'segment_or_device_ms', 'device_ms_or_code_or_accel_g', 'red_or_voltage_V_or_gyro_dps', 'ir_or_tilt_deg', 'motion_blocked'])
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8', errors='replace')
        evidence = None
        position = 0
        pending = ''
        def tail():
            nonlocal position, pending
            if evidence is None or not evidence.exists():
                return
            with evidence.open(encoding='utf-8') as data:
                data.seek(position)
                pending += data.read()
                position = data.tell()
            while '\n' in pending:
                line, pending = pending.split('\n', 1)
                if line.strip():
                    ingest(json.loads(line), writer, output)
        events = []
        event_lock = threading.Lock()
        def reader():
            for line in proc.stdout:
                with event_lock:
                    events.append(line)
        threading.Thread(target=reader, daemon=True).start()
        while True:
            with event_lock:
                batch = events[:]
                events.clear()
            for line in batch:
                log.write(line); log.flush()
                print(line, end='', flush=True)
                if line.startswith('EVIDENCE '):
                    evidence = Path(line[len('EVIDENCE '):].strip())
                    with lock:
                        state['evidence'] = str(evidence)
                if 'HOST_ERROR' in line or 'SUPERVISOR_ABORT' in line:
                    with lock:
                        state['status'] = line.strip()
            tail()
            if proc.poll() is not None:
                tail()
                with lock:
                    state['finished'] = True
                    if proc.returncode and not state['count']:
                        state['status'] = '串口或初始化失败，请查看采集日志；没有生成模拟数据'
                break
            time.sleep(.1)

HTML = r'''<!doctype html><meta charset="utf-8"><title>PPG / GSR 实验波形</title>
<style>body{margin:24px;background:#101827;color:#e6eef7;font:16px system-ui}h1{font-size:24px;margin-bottom:10px}.row{display:flex;gap:12px;flex-wrap:wrap}.tag{padding:10px;background:#223149;border-radius:8px}section{background:#172339;padding:14px;margin-top:14px;border-radius:12px}h2{font-size:17px;margin:0 0 8px}canvas{width:100%;height:175px}small{color:#a8bbd0}button,a{color:#d7ecff}button{background:#304462;border:0;padding:10px;border-radius:6px}#paths{white-space:pre-wrap;font-size:12px}</style>
<h1>PPG · GSR · 运动 实验波形</h1><div id="status">正在等待真实设备数据…</div>
<p><small>USB 真实采集 · 原始数据始终记录 · 页面滤波仅帮助观察，不改变保存的数据。缺口不连线。</small></p>
<div class="row"><span class="tag" id="elapsed"></span><span class="tag" id="n"></span><span class="tag" id="gsr"></span><span class="tag" id="fault"></span><button id="pause">暂停画面</button><a href="/download">下载本次原始 CSV</a></div>
<section><h2>红外 PPG 原始波形（ADC 计数） · 最近 8 秒</h2><canvas id="ir"></canvas></section>
<section><h2>红外 PPG 去直流波形（ADC 计数，仅显示处理） · 最近 8 秒</h2><canvas id="ac"></canvas></section>
<section><h2>红光 PPG 原始波形（ADC 计数） · 最近 8 秒</h2><canvas id="red"></canvas></section>
<section><h2>GSR 原始输出电压（V） · 最近 30 秒</h2><canvas id="eda"></canvas></section>
<section><h2>MPU 运动：加速度偏差（g） · 最近 30 秒</h2><canvas id="imu"></canvas></section>
<p><small>PPG 时间来自 FIFO 顺序与名义 100 Hz；GSR 使用转换时间。波形显示不代表情绪分类已经验证。</small></p><div id="paths"></div>
<script>
let paused=false;document.getElementById('pause').onclick=()=>{paused=!paused;document.getElementById('pause').textContent=paused?'继续画面':'暂停画面'};
function draw(id,pts,seconds,gap){const c=document.getElementById(id),r=c.getBoundingClientRect();c.width=r.width*devicePixelRatio;c.height=175*devicePixelRatio;const x=c.getContext('2d');x.scale(devicePixelRatio,devicePixelRatio);let w=r.width,h=175;x.clearRect(0,0,w,h);if(!pts.length){x.fillStyle='#abc';x.fillText('等待采样…',65,80);return}let end=pts.at(-1)[0],start=Math.max(0,end-seconds*1000);pts=pts.filter(p=>p[0]>=start);let lo=Math.min(...pts.map(p=>p[1])),hi=Math.max(...pts.map(p=>p[1]));let margin=(hi-lo)*.1||1;lo-=margin;hi+=margin;const X=t=>55+(t-start)/Math.max(1000,end-start)*(w-75),Y=y=>145-(y-lo)/(hi-lo)*130;x.font='11px system-ui';for(let i=0;i<5;i++){let v=lo+(hi-lo)*i/4;x.strokeStyle='#2b3c55';x.beginPath();x.moveTo(55,Y(v));x.lineTo(w-20,Y(v));x.stroke();x.fillStyle='#a8bbd0';x.fillText(v.toFixed(id==='eda'||id==='imu'?3:0),2,Y(v)+3)}x.fillText((start/1000).toFixed(1)+' s',55,169);x.fillText((end/1000).toFixed(1)+' s',w-75,169);x.strokeStyle=id==='eda'?'#ffa657':'#51d6c8';x.lineWidth=1.4;x.beginPath();let prev=null;for(let p of pts){if(!prev||p[0]-prev[0]>gap||p[2]!==prev[2])x.moveTo(X(p[0]),Y(p[1]));else x.lineTo(X(p[0]),Y(p[1]));prev=p}x.stroke()}
async function poll(){try{let d=await(await fetch('/data')).json();document.getElementById('status').textContent=d.status+(!d.finished&&d.updated&&Date.now()/1000-d.updated>3?' · 数据已停止更新':'');document.getElementById('elapsed').textContent='设备采集 '+d.elapsed.toFixed(1)+' 秒';document.getElementById('n').textContent='PPG '+d.count+' 点';document.getElementById('gsr').textContent='GSR '+(d.eda.length?d.eda.at(-1)[2].toFixed(4)+' V':'等待');document.getElementById('fault').textContent='FIFO 事件 '+(d.errors.fifo_events||0);document.getElementById('paths').textContent='原始文件：'+d.evidence+'\nCSV：'+d.csv;if(!paused){draw('ir',d.ppg.map(p=>[p[1],p[3],p[0]]),8,35);draw('red',d.ppg.map(p=>[p[1],p[2],p[0]]),8,35);let dc=null,prev=null,ac=[];for(let p of d.ppg){if(!prev||p[0]!==prev[0]||p[1]-prev[1]>35)dc=p[3];dc+=.02*(p[3]-dc);ac.push([p[1],p[3]-dc,p[0]]);prev=p}draw('ac',ac,8,35);draw('eda',d.eda.map(p=>[p[0],p[2],0]),30,150);draw('imu',d.imu.map(p=>[p[0],p[1],0]),30,450)}}catch(e){document.getElementById('status').textContent='显示服务连接中断：'+e.message}setTimeout(poll,250)}poll();
</script>'''

class Handler(BaseHTTPRequestHandler):
    def end_headers(self):
        # Only explicitly configured web origins can read local biosignals.
        origin = self.headers.get('Origin')
        allowed = [x.strip() for x in os.getenv('WEB_ALLOWED_ORIGINS', '').split(',') if x.strip()]
        if origin in allowed:
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Vary', 'Origin')
        super().end_headers()

    def do_GET(self):
        if self.path == '/data':
            with lock:
                body = json.dumps(state, ensure_ascii=False).encode()
            content_type = 'application/json; charset=utf-8'
        elif self.path == '/download':
            body = CSV_PATH.read_bytes() if CSV_PATH.exists() else b''
            content_type = 'text/csv; charset=utf-8'
        elif self.path == '/':
            body = HTML.encode()
            content_type = 'text/html; charset=utf-8'
        else:
            self.send_error(404); return
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        if self.path == '/download':
            self.send_header('Content-Disposition', 'attachment; filename="signals.csv"')
        self.end_headers(); self.wfile.write(body)
    def log_message(self, *args):
        pass

if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', 8011), Handler)
    threading.Thread(target=capture, daemon=True).start()
    print('VIEWER http://127.0.0.1:8011/', flush=True)
    server.serve_forever()
