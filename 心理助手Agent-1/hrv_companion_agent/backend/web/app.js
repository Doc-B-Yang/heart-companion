import {FrameAssembler} from './ble_frames.js';
import {connectionBase, startWaves, ingestBlePacket, resetBleSignals} from './waveforms.js';
import {SpeechInput} from './speech_input.js';
import {AffectTracker, renderAffect} from './affect_view.js';
const SERVICE='8e7c0001-7f8a-4a2c-9f81-4b9e764d0001',DATA='8e7c0002-7f8a-4a2c-9f81-4b9e764d0001';
const $=id=>document.getElementById(id);
let device=null,characteristic=null,lastPacket=0,lastState=null,queue=[],posting=false,assembler=new FrameAssembler(),labelUntil=0;
let labelSaving=false;
let userEpoch=0,chatSending=false,analyzing=false;
const affectTracker=new AffectTracker(),speechSources=new Set();
const uid=()=>$('userId').value.trim()||'user001';
const speech=new SpeechInput({Recognition:window.SpeechRecognition||window.webkitSpeechRecognition,inputs:[
  {field:$('message'),button:$('chatVoice'),cancel:$('chatVoiceCancel'),status:$('chatVoiceStatus')},
  {field:$('selfNote'),button:$('noteVoice'),cancel:$('noteVoiceCancel'),status:$('noteVoiceStatus')},
],onCommit:id=>speechSources.add(id)});
for(const id of ['message','selfNote']) $(id).addEventListener('input',()=>{if(!$(id).value)speechSources.delete(id);});
window.addEventListener('pagehide',()=>speech.cancel());
function applyAffect(context){if(affectTracker.update(context))renderAffect(affectTracker.context,$);}
function requestStillCurrent(user,epoch){return uid()===user && userEpoch===epoch;}
async function api(path,data){let r;const timeout=path.startsWith('chat/')||path.startsWith('emotion/valence')?90000:8000;try{r=await fetch(connectionBase()+'/api/'+path,{method:data?'POST':'GET',headers:data?{'Content-Type':'application/json'}:{},body:data?JSON.stringify(data):undefined,signal:AbortSignal.timeout(timeout)});}catch(e){throw Error(e.name==='TimeoutError'?'后端响应超时，输入已保留；请稍后重试。':'无法连接本机后端：请启动电脑服务，并在 Chrome 网站设置中允许本地网络访问。输入已保留。');}let body;try{body=await r.json();}catch{throw Error('后端未返回有效结果，输入已保留。');}if(!r.ok)throw Error(typeof body.detail==='string'?body.detail:'请求失败，请检查输入或后端状态。');return body;}
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
  $('confirmContact').disabled=false;
  $('captureBaseline').disabled=!q.quality_ok||!c.contact_ready||c.baseline_ready;
  $('calibrationStatus').textContent=c.baseline_ready?'本次静息基线已保存。':!c.contact_confirmed?'请确认电极已佩戴。':c.settle_remaining_sec>0?`电极稳定等待中，还需约${Math.ceil(c.settle_remaining_sec)}秒。`:'继续保持静息，信号稳定后可保存基线。';
  $('saveLabel').disabled=labelSaving;
}
async function calibration(action){const user=uid(),epoch=userEpoch;try{await api('sensor/calibration',{user_id:user,action});if(requestStillCurrent(user,epoch)){labelUntil=0;await poll(false);}}catch(e){if(requestStillCurrent(user,epoch))$('calibrationStatus').textContent=e.message;}}
$('confirmContact').onclick=()=>calibration('confirm_contact');
$('captureBaseline').onclick=()=>calibration('capture_baseline');
async function drain(){if(posting)return;posting=true;try{while(queue.length){const next=queue.shift();if(Date.now()-next.received>2000)continue;try{const result=await api('sensor/raw',{user_id:next.user,packet:next.packet});if(uid()===next.user){showState(result.state);$('serverStatus').textContent='后端已连接';}}catch(e){$('serverStatus').textContent='后端连接失败';$('qualityReason').textContent=e.message;queue=[];}}}finally{posting=false;}}
function notification(event){const packet=assembler.add(event.target.value);if(!packet)return;lastPacket=Date.now();ingestBlePacket(packet);$('connectionStatus').textContent='BLE 数据接收中';if(queue.length>=3)queue.shift();queue.push({packet,user:uid(),received:lastPacket});drain();}
function disconnected(){queue=[];lastPacket=0;assembler.reset();$('connectBle').disabled=false;$('disconnectBle').disabled=true;$('connectionStatus').textContent='蓝牙已断开';showState(null);}
$('connectBle').onclick=async()=>{if(!navigator.bluetooth){$('connectionStatus').textContent='请使用 Chrome/Edge 或电脑网关';return;}try{$('connectBle').disabled=true;device=await navigator.bluetooth.requestDevice({filters:[{services:[SERVICE]}],optionalServices:[SERVICE]});device.addEventListener('gattserverdisconnected',disconnected);const server=await device.gatt.connect();const service=await server.getPrimaryService(SERVICE);characteristic=await service.getCharacteristic(DATA);await characteristic.startNotifications();characteristic.addEventListener('characteristicvaluechanged',notification);$('disconnectBle').disabled=false;$('connectionStatus').textContent='蓝牙已连接，等待数据';}catch(e){disconnected();$('connectionStatus').textContent=e.name==='NotFoundError'?'未选择设备':e.message;}};
$('disconnectBle').onclick=()=>{if(device?.gatt.connected)device.gatt.disconnect();disconnected();};
$('userId').onchange=()=>{userEpoch++;speech.cancel();speechSources.clear();queue=[];lastState=null;labelUntil=0;resetBleSignals();showState(null);affectTracker.reset();renderAffect(null,$);$('analysisStatus').textContent='';$('labelStatus').textContent='参与者已切换，可记录新的自评。';$('chatStatus').textContent='参与者已切换。';$('message').value='';$('selfNote').value='';$('arousalScore').value='';$('valenceScore').value='';$('messages').replaceChildren();bubble('今天有什么想说的吗？不用急着给自己的感受下定义。','assistant');};
$('saveLabel').onclick=async()=>{if(labelSaving)return;if(speech.session?.input.field.id==='selfNote'){$('labelStatus').textContent='请先停止语音识别，确认转写文字后再保存。';return;}const user=uid(),epoch=userEpoch;const rating=id=>$(id).value===''?null:Number($(id).value);const submitted={user_id:user,label:$('selfLabel').value,note:$('selfNote').value,arousal_score:rating('arousalScore'),valence_score:rating('valenceScore')};labelSaving=true;$('saveLabel').disabled=true;try{const r=await api('dataset/self-report',submitted);if(!requestStillCurrent(user,epoch))return;applyAffect(r.affect_context);if(rating('arousalScore')===submitted.arousal_score)$('arousalScore').value='';if(rating('valenceScore')===submitted.valence_score)$('valenceScore').value='';if($('selfNote').value===submitted.note){$('selfNote').value='';speechSources.delete('selfNote');}$('labelStatus').textContent=r.training_sample_saved?`已保存「${r.label}」、评分及文字，并关联合格训练窗口。`:`已保存「${r.label}」、评分及文字。此次未关联训练窗口：${r.training_reason}`;}catch(e){if(requestStillCurrent(user,epoch))$('labelStatus').textContent='保存失败，输入已保留：'+e.message;}finally{labelSaving=false;$('saveLabel').disabled=false;}};
$('analyzeNote').onclick=async()=>{if(analyzing)return;if(speech.session?.input.field.id==='selfNote'){$('analysisStatus').textContent='请先停止语音识别，确认转写文字后再分析。';return;}const text=$('selfNote').value.trim();if(!text){$('analysisStatus').textContent='请先写下或转写你的当前感受。';return;}const user=uid(),epoch=userEpoch;analyzing=true;$('analyzeNote').disabled=true;$('analysisStatus').textContent='正在分析语言线索…';try{const result=await api('emotion/valence',{user_id:user,text,input_source:speechSources.has('selfNote')?'speech_transcript':'text'});if(!requestStillCurrent(user,epoch))return;applyAffect(result.affect_context);$('analysisStatus').textContent='语言分析已完成；未保存为训练标签。';}catch(e){if(requestStillCurrent(user,epoch))$('analysisStatus').textContent='分析失败，文字已保留：'+e.message;}finally{analyzing=false;$('analyzeNote').disabled=false;}};
function bubble(text,role){const element=document.createElement('div');element.className='bubble '+role;element.textContent=text;$('messages').appendChild(element);$('messages').scrollTop=$('messages').scrollHeight;}
$('chatForm').onsubmit=async event=>{event.preventDefault();if(chatSending)return;if(speech.session?.input.field.id==='message'){$('chatStatus').textContent='请先停止语音识别，确认转写文字后再发送。';return;}const input=$('message').value,message=input.trim();if(!message)return;const user=uid(),epoch=userEpoch;chatSending=true;$('sendMessage').disabled=true;$('chatStatus').textContent='正在分析表达并回应…';try{const r=await api('chat/message',{user_id:user,message,input_source:speechSources.has('message')?'speech_transcript':'text'});if(!requestStillCurrent(user,epoch))return;bubble(message,'user');bubble(r.response,'assistant');applyAffect(r.affect_context);if($('message').value===input){$('message').value='';speechSources.delete('message');}$('chatStatus').textContent='已回应。自评优先，语言推测显示于效价面板；未作为训练标签。';}catch(e){if(requestStillCurrent(user,epoch))$('chatStatus').textContent='发送失败，输入已保留：'+e.message;}finally{chatSending=false;$('sendMessage').disabled=false;}};
async function poll(schedule=true){const user=uid(),epoch=userEpoch;try{const state=await api('emotion/state?user_id='+encodeURIComponent(user));if(!requestStillCurrent(user,epoch))return;showState(state);applyAffect(state.affect_context);$('serverStatus').textContent='后端已连接';if(!device?.gatt.connected){$('connectionStatus').textContent=state.layer&&state.layer!=='stale'?'电脑网关已接入':'未连接设备';}else if(lastPacket&&Date.now()-lastPacket>5000){$('connectionStatus').textContent='BLE 数据已停，请重新连接';}}catch(e){if(requestStillCurrent(user,epoch)){$('serverStatus').textContent='后端离线';showState(null);$('calibrationStatus').textContent=e.message;}}finally{if(schedule)setTimeout(poll,1000);}}
renderAffect(null,$);
setInterval(()=>renderAffect(affectTracker.context,$),1000);
poll();
startWaves();
