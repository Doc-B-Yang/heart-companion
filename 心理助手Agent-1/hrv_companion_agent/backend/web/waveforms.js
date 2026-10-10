import {BleSignals} from './ble_signals.js';
const $=id=>document.getElementById(id);
const ble=new BleSignals();
export function ingestBlePacket(packet){if(ble.add(packet))updateDisplay();}
export function resetBleSignals(){ble.clear();updateDisplay();}
const read=key=>{try{return localStorage.getItem(key);}catch{return null;}};
const save=(key,value)=>{try{localStorage.setItem(key,value);}catch{}};
let apiBase=read('heart-api')??(location.hostname.endsWith('github.io')?'http://127.0.0.1:8000':''),waveBase=read('heart-wave')||'http://127.0.0.1:8011',paused=false,latest=null;
export const connectionBase=()=>apiBase;
export function normalizeBase(value){
  if(!value.trim())return '';
  const url=new URL(value);
  if(!['http:','https:'].includes(url.protocol)||url.username||url.password||url.search||url.hash||url.pathname!=='/')throw Error('请输入 http/https 服务地址，不含密码、路径或参数。');
  if(url.protocol==='http:'&&!['localhost','127.0.0.1','[::1]'].includes(url.hostname)&&location.protocol==='https:')throw Error('远程后端请使用 HTTPS；HTTP 仅支持本机回环地址。');
  return url.origin;
}
export function ppgAC(rows){
  let dc=null,previous=null;
  return rows.filter(p=>p.length>=4&&p.slice(0,4).every(Number.isFinite)).map(p=>{
    if(!previous||p[0]!==previous[0]||p[1]-previous[1]>35||p[1]<=previous[1])dc=p[3];
    dc+=.02*(p[3]-dc);previous=p;return [p[1],p[3]-dc,p[0]];
  });
}
function draw(id,points,seconds,gap){
  const c=$(id),w=c.getBoundingClientRect().width,h=170,dpr=devicePixelRatio||1;
  c.width=w*dpr;c.height=h*dpr;const x=c.getContext('2d');x.scale(dpr,dpr);x.font='11px system-ui';
  if(!points.length){x.fillStyle='#70857c';x.fillText('等待真实设备采样',20,85);return;}
  const end=points.at(-1)[0],start=Math.max(0,end-seconds*1000);points=points.filter(p=>p[0]>=start&&p.every(Number.isFinite));
  if(!points.length)return;
  let low=Infinity,high=-Infinity;for(const p of points){low=Math.min(low,p[1]);high=Math.max(high,p[1]);}
  const margin=(high-low)*.1||.001;low-=margin;high+=margin;
  const X=t=>52+(t-start)/Math.max(1000,end-start)*(w-65),Y=v=>140-(v-low)/(high-low)*125;
  for(let i=0;i<4;i++){const v=low+(high-low)*i/3;x.strokeStyle='#e4ebe5';x.beginPath();x.moveTo(52,Y(v));x.lineTo(w-10,Y(v));x.stroke();x.fillStyle='#70857c';x.fillText(v.toFixed(id==='gsrWave'?3:0),2,Y(v)+3);}
  x.fillText((start/1000).toFixed(1)+' s',52,165);x.fillText((end/1000).toFixed(1)+' s',w-55,165);
  x.strokeStyle=id==='gsrWave'?'#bd815a':'#347960';x.lineWidth=1.6;x.beginPath();let previous=null;
  for(const p of points){if(!previous||p[0]<=previous[0]||p[0]-previous[0]>gap||p[2]!==previous[2])x.moveTo(X(p[0]),Y(p[1]));else x.lineTo(X(p[0]),Y(p[1]));previous=p;}x.stroke();
}
function usbFresh(){return !!latest?.updated&&!latest.finished&&Date.now()/1000-latest.updated<3;}
function render(){
  if(ble.fresh()){
    if(usbFresh())draw('ppgWave',ppgAC(latest.ppg||[]),8,35);
    else{
    draw('ppgWave',[],8,35);
    const c=$('ppgWave'),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);x.fillStyle='#70857c';x.font='13px system-ui';
    x.fillText('BLE 已收到 PPG 指标；当前固件不发送原始 PPG 波形。',20,70);
    x.fillText('原始 PPG 曲线请使用 USB 波形采集。',20,100);
    }
    draw('gsrWave',ble.eda.map(p=>[p[0],p[1],0]),30,150);
  }else{draw('ppgWave',ppgAC(latest?.ppg||[]),8,35);draw('gsrWave',(latest?.eda||[]).filter(p=>p.length>=3&&p.slice(0,3).every(Number.isFinite)).map(p=>[p[0],p[2],0]),30,150);}
}
function updateDisplay(){
  if(ble.fresh()){
    const p=ble.packet,format=v=>Number.isFinite(v)?v.toFixed(3):'未知';
    $('waveStatus').textContent=usbFresh()?'USB PPG + BLE GSR 实时波形':'BLE 真实数据接收中 · GSR 实时波形';
    $('waveCount').textContent=(usbFresh()?`USB PPG ${latest.count} 点 · `:'')+`已接收 ${ble.count} 包 · 最新序号 ${p.seq} · GSR 显示 ${ble.eda.length} 点`;
    $('waveMotion').textContent=`PPG 质量 ${format(p.sig_quality)} · PI ${format(p.pi)}% · ${p.moving===false?'静止':p.moving===true?'运动':'运动状态未知'}`;
    $('downloadWave').textContent='下载 BLE 接收数据';$('downloadWave').disabled=false;
  }else{
    const d=latest,fresh=!!d?.updated&&!d.finished&&Date.now()/1000-d.updated<3;
    $('waveStatus').textContent=fresh?'USB 真实信号接收中':d?.finished?'USB 采集已结束 · 历史波形':ble.packet?'BLE 数据已停止 · 可下载已收数据':'波形服务未连接';
    $('waveCount').textContent=d?'USB PPG '+(d.count||0)+' 点 · '+Number(d.elapsed||0).toFixed(1)+' 秒':'等待采样';
    const motion=d?.imu?.at(-1);$('waveMotion').textContent=motion?'运动偏差 '+Number(motion[1]).toFixed(3)+' g':'连接 BLE 可直接显示 GSR；原始 PPG 需启动 USB 波形服务';
    $('downloadWave').textContent=ble.records.length?'下载 BLE 接收数据':'下载 USB 原始 CSV';
    $('downloadWave').disabled=!(ble.records.length||d?.count||d?.eda?.length);
  }
  if(!paused)render();
}
export function startWaves(){
  $('apiBase').value=apiBase;$('waveBase').value=waveBase;
  $('saveConnection').onclick=()=>{try{const a=normalizeBase($('apiBase').value),w=normalizeBase($('waveBase').value);apiBase=a;waveBase=w;save('heart-api',a);save('heart-wave',w);latest=null;render();$('connectionHint').textContent='地址已保存，正在连接。';}catch(e){$('connectionHint').textContent=e.message;}};
  $('pauseWave').onclick=()=>{paused=!paused;$('pauseWave').textContent=paused?'继续画面':'暂停画面';if(!paused)render();};
  $('downloadWave').onclick=async()=>{try{let blob,name;if(ble.records.length){blob=new Blob([ble.jsonl()],{type:'application/x-ndjson'});name='ble-signals.jsonl';}else{const r=await fetch(waveBase+'/download',{signal:AbortSignal.timeout(8000)});if(!r.ok)throw Error('下载失败');blob=await r.blob();name='signals.csv';}const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){$('waveStatus').textContent=e.message;}};
  async function poll(){
    const source=waveBase;
    try{const r=await fetch(source+'/data',{signal:AbortSignal.timeout(4000),cache:'no-store'});if(!r.ok)throw Error('波形服务未响应');const d=await r.json();if(source!==waveBase)return;
      if(!Array.isArray(d.ppg)||!Array.isArray(d.eda))throw Error('波形格式不匹配');latest=d;
      updateDisplay();
    }catch(e){if(source!==waveBase)return;latest=null;updateDisplay();}
    finally{setTimeout(poll,500);}
  }
  render();poll();window.addEventListener('resize',()=>{if(!paused)render();});
}
