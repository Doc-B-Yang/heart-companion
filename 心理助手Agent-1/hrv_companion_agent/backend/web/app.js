import {FrameAssembler} from './ble_frames.js';
import {connectionBase, startWaves, ingestBlePacket, resetBleSignals} from './waveforms.js';
const SERVICE='8e7c0001-7f8a-4a2c-9f81-4b9e764d0001',DATA='8e7c0002-7f8a-4a2c-9f81-4b9e764d0001';
const $=id=>document.getElementById(id);
let device=null,characteristic=null,lastPacket=0,lastState=null,queue=[],posting=false,assembler=new FrameAssembler(),labelUntil=0;
const uid=()=>$('userId').value.trim()||'user001';
async function api(path,data){const r=await fetch(connectionBase()+'/api/'+path,{method:data?'POST':'GET',headers:{'Content-Type':'application/json'},body:data?JSON.stringify(data):undefined,signal:AbortSignal.timeout(8000)});const body=await r.json();if(!r.ok)throw Error(body.detail||'请求失败');return body;}
function showState(state){
  lastState=state;
  $('emotion').textContent=state?.emotion||'等待数据';
  $('emotionReason').textContent=state?.reason||(state?.validated?'当前为模型估计，请与你自己的感受核对。':'连接传感器后，先积累稳定的数据。');
  const q=state?.quality||{};
  $('quality').textContent=q.quality_ok?'分析窗口检查通过':state?.sensor&&state.layer!=='stale'?'已采集 · 分析窗口尚未合格':'等待采集数据';
  $('qualityDot').className='dot '+(q.quality_ok?'good':'bad');
  $('qualityReason').textContent=q.reason||'只有通过质量检查的数据才会参与分类。';
  const channels=q.channels||{};
  $('channelDiagnostics').textContent=state?.layer==='stale'?'数据已过期，等待新数据。':
    ['ppg','gsr','imu'].filter(key=>channels[key]).map(key=>{
      const c=channels[key],name={ppg:'PPG',gsr:'GSR',imu:'运动'}[key];
      let numbers='';
      if(key==='ppg') numbers=[Number.isFinite(c.sig_quality)?`质量 ${c.sig_quality.toFixed(3)}`:'',Number.isFinite(c.pi_pct)?`PI ${c.pi_pct.toFixed(3)}%`:''].filter(Boolean).join('，');
      if(key==='gsr'&&Number.isFinite(c.min_v)&&Number.isFinite(c.max_v)) numbers=`${c.min_v.toFixed(4)}–${c.max_v.toFixed(4)} V`;
      return name+'：'+(c.packet_ok?'当前包检查通过':(c.reasons||[]).join('、'))+(numbers?`（${numbers}）`:'');
    }).join('；')||'各传感器诊断等待数据。';
  $('ppgState').textContent=q.ppg_ok?'窗口合格':'积累 / 待检查';
  $('edaState').textContent=q.eda_ok?'窗口合格':'积累 / 待检查';
  $('beatCount').textContent=q.n_beats!=null?`${q.n_beats} 拍 · ${(q.window_span_sec||0).toFixed(1)} 秒`:'—';
  $('modelState').textContent=state?.model_available?'已启用 · 需持续验证':'尚无已验证模型';
  const c=state?.calibration||{};
  $('confirmContact').disabled=state?.simulation!==false||state?.layer==='stale';
  $('captureBaseline').disabled=!q.quality_ok||!c.contact_ready||c.baseline_ready;
  $('calibrationStatus').textContent=c.baseline_ready?'本次静息基线已保存。':!c.contact_confirmed?'请确认电极已佩戴。':c.settle_remaining_sec>0?`电极稳定等待中，还需约${Math.ceil(c.settle_remaining_sec)}秒。`:'继续保持静息，信号稳定后可保存基线。';
  $('saveLabel').disabled=!q.quality_ok||!c.baseline_ready||state?.simulation!==false||Date.now()<labelUntil;
}
async function calibration(action){const user=uid();try{await api('sensor/calibration',{user_id:user,action});if(uid()===user){labelUntil=0;await poll(false);}}catch(e){$('calibrationStatus').textContent=e.message;}}
$('confirmContact').onclick=()=>calibration('confirm_contact');
$('captureBaseline').onclick=()=>calibration('capture_baseline');
async function drain(){if(posting)return;posting=true;try{while(queue.length){const next=queue.shift();if(Date.now()-next.received>2000)continue;try{const result=await api('sensor/raw',{user_id:next.user,packet:next.packet});if(uid()===next.user){showState(result.state);$('serverStatus').textContent='后端已连接';}}catch(e){$('serverStatus').textContent='后端连接失败';$('qualityReason').textContent=e.message;queue=[];}}}finally{posting=false;}}
function notification(event){const packet=assembler.add(event.target.value);if(!packet)return;lastPacket=Date.now();ingestBlePacket(packet);$('connectionStatus').textContent='BLE 数据接收中';if(queue.length>=3)queue.shift();queue.push({packet,user:uid(),received:lastPacket});drain();}
function disconnected(){queue=[];lastPacket=0;assembler.reset();$('connectBle').disabled=false;$('disconnectBle').disabled=true;$('connectionStatus').textContent='蓝牙已断开';showState(null);}
$('connectBle').onclick=async()=>{if(!navigator.bluetooth){$('connectionStatus').textContent='请使用 Chrome/Edge 或电脑网关';return;}try{$('connectBle').disabled=true;device=await navigator.bluetooth.requestDevice({filters:[{services:[SERVICE]}],optionalServices:[SERVICE]});device.addEventListener('gattserverdisconnected',disconnected);const server=await device.gatt.connect();const service=await server.getPrimaryService(SERVICE);characteristic=await service.getCharacteristic(DATA);await characteristic.startNotifications();characteristic.addEventListener('characteristicvaluechanged',notification);$('disconnectBle').disabled=false;$('connectionStatus').textContent='蓝牙已连接，等待数据';}catch(e){disconnected();$('connectionStatus').textContent=e.name==='NotFoundError'?'未选择设备':e.message;}};
$('disconnectBle').onclick=()=>{if(device?.gatt.connected)device.gatt.disconnect();disconnected();};
$('userId').onchange=()=>{queue=[];lastState=null;labelUntil=0;resetBleSignals();showState(null);};
$('saveLabel').onclick=async()=>{try{$('saveLabel').disabled=true;const rating=id=>$(id).value===''?null:Number($(id).value);const r=await api('dataset/label',{user_id:uid(),label:$('selfLabel').value,arousal_score:rating('arousalScore'),valence_score:rating('valenceScore')});labelUntil=Date.now()+r.next_label_after_sec*1000;$('arousalScore').value='';$('valenceScore').value='';$('labelStatus').textContent=`已保存「${r.label}」及本次自评，请继续稳定采集60秒后重新评分。`;}catch(e){$('labelStatus').textContent=e.message;showState(lastState);}};
function bubble(text,role){const element=document.createElement('div');element.className='bubble '+role;element.textContent=text;$('messages').appendChild(element);$('messages').scrollTop=$('messages').scrollHeight;}
$('chatForm').onsubmit=async event=>{event.preventDefault();const message=$('message').value.trim();if(!message)return;bubble(message,'user');$('message').value='';$('sendMessage').disabled=true;$('chatStatus').textContent='正在回应…';try{const r=await api('chat/message',{user_id:uid(),message});bubble(r.response,'assistant');$('chatStatus').textContent='以你的表达为准，生理结果仅在可靠时用于调整陪伴。';}catch(e){$('chatStatus').textContent='发送失败：'+e.message;}finally{$('sendMessage').disabled=false;}};
async function poll(schedule=true){const user=uid();try{const state=await api('emotion/state?user_id='+encodeURIComponent(user));if(uid()!==user)return;showState(state);$('serverStatus').textContent='后端已连接';if(!device?.gatt.connected){$('connectionStatus').textContent=state.layer&&state.layer!=='stale'?'电脑网关已接入':'未连接设备';}else if(lastPacket&&Date.now()-lastPacket>5000){$('connectionStatus').textContent='BLE 数据已停，请重新连接';}}catch{$('serverStatus').textContent='后端离线';showState(null);}finally{if(schedule)setTimeout(poll,1000);}}
poll();
startWaves();
