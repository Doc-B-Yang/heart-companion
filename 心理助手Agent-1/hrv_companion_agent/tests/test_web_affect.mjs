import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

const web=new URL('../backend/web/',import.meta.url);
const importSource=async name=>import('data:text/javascript;base64,'+Buffer.from(await readFile(new URL(name,web),'utf8')).toString('base64'));
const {SpeechInput,appendTranscript}=await importSource('speech_input.js');
const {AffectTracker,affectCoordinates,renderAffect}=await importSource('affect_view.js');

function element(id='',value='') {
  return {id,value,textContent:'',disabled:false,hidden:false,style:{},maxLength:5000,
    attributes:{},events:[],children:[],setAttribute(name,value){this.attributes[name]=value;},
    dispatchEvent(event){this.events.push(event.type);},focus(){this.focused=true;},
    addEventListener(){},appendChild(child){this.children.push(child);},replaceChildren(){this.children=[];}};
}
class Recognition {
  constructor(){Recognition.last=this;}
  start(){this.started=true;}
  stop(){this.stopped=true;this.onend?.();}
  abort(){this.aborted=true;this.onend?.();}
  result(text,final=true,index=0){const results=Array.from({length:index+1},()=>({0:{transcript:''},isFinal:false}));results[index]={0:{transcript:text},isFinal:final};this.onresult?.({results,resultIndex:index});}
}
function speechFixture(RecognitionClass=Recognition) {
  const inputs=['message','selfNote'].map(id=>({field:element(id,'原有文字'),button:element(),cancel:element(),status:element()}));
  const commits=[];
  const speech=new SpeechInput({Recognition:RecognitionClass,inputs,onCommit:id=>commits.push(id)});
  return {speech,inputs,commits};
}
test('speech appends final words once to latest editable input, no automatic send',()=>{
  const {speech,inputs,commits}=speechFixture();
  speech.start(inputs[0]);const recognition=Recognition.last;
  assert.equal(recognition.lang,'zh-CN');assert.equal(inputs[1].button.disabled,true);
  recognition.result('我现在很开心',false);
  assert.equal(inputs[0].field.value,'原有文字');
  inputs[0].field.value='录音时修改的文字';
  recognition.result('我现在很开心');recognition.result('我现在很开心');
  speech.stop();
  assert.equal(inputs[0].field.value,'录音时修改的文字 我现在很开心');
  assert.deepEqual(commits,['message']);assert.deepEqual(inputs[0].field.events,['input']);
  assert.equal(speech.session,null);assert.equal(inputs[1].button.disabled,false);
  assert.equal(recognition.onresult,null);
});
test('cancel, permissions failure and unavailable speech preserve text and clean controls',()=>{
  const fixture=speechFixture();fixture.speech.start(fixture.inputs[0]);
  const previousResult=Recognition.last.onresult;
  Recognition.last.result('不应写入');fixture.speech.cancel();
  previousResult({results:[{0:{transcript:'迟到的结果'},isFinal:true}]});
  assert.equal(fixture.inputs[0].field.value,'原有文字');assert.equal(Recognition.last.aborted,true);
  fixture.speech.start(fixture.inputs[1]);Recognition.last.onerror({error:'not-allowed'});
  assert.equal(fixture.speech.session,null);assert.match(fixture.inputs[1].status.textContent,/权限/);
  assert.equal(fixture.inputs[1].cancel.hidden,true);assert.equal(fixture.inputs[0].button.disabled,false);
  const unavailable=speechFixture(undefined);
  // Explicitly construct unsupported case, since the fixture default is a fake recognizer.
  const unsupported=new SpeechInput({Recognition:null,inputs:unavailable.inputs});
  assert.equal(unavailable.inputs[0].button.disabled,true);assert.equal(unsupported.session,null);
});
test('speech respects maxlength and empty transcription never erases typed input',()=>{
  assert.equal(appendTranscript('1234','六七八九',6),'1234 六');
  const {speech,inputs}=speechFixture();speech.start(inputs[0]);Recognition.last.onend();
  assert.equal(inputs[0].field.value,'原有文字');assert.match(inputs[0].status.textContent,/保留/);
});
const timestamp='2026-10-11T10:00:00+00:00',expiry='2026-10-11T10:05:00+00:00';
const axis=value=>({value,status:'available',source:'self_report',score_1_9:5,updated_at:timestamp,expires_at:expiry});
test('missing arousal, invalid numeric values and expired axes never become a neutral point',()=>{
  const now=Date.parse(timestamp);
  assert.deepEqual(affectCoordinates({valence:axis(.5)},now),{valence:.5,arousal:null,complete:false});
  assert.equal(affectCoordinates({valence:axis(NaN),arousal:axis(0)},now).complete,false);
  assert.equal(affectCoordinates({valence:axis(.5),arousal:axis(0)},Date.parse(expiry)).complete,false);
  assert.equal(affectCoordinates({valence:axis(.5),arousal:axis(0)},now).complete,true);
});
test('context revision rejects late polling and accepts expiry snapshot at same revision',()=>{
  const tracker=new AffectTracker();
  assert.equal(tracker.update({session_id:'sessionA',revision:4,updated_at:timestamp,valence:axis(.5)}),true);
  assert.equal(tracker.update({session_id:'sessionA',revision:3,updated_at:expiry,valence:axis(-.5)}),false);
  assert.equal(tracker.context.valence.value,.5);
  assert.equal(tracker.update({session_id:'sessionA',revision:4,updated_at:timestamp,valence:{...axis(.5),status:'unknown'}}),true);
  assert.equal(tracker.update({session_id:'sessionB',revision:0,updated_at:expiry}),true);
  assert.equal(tracker.update({session_id:'sessionA',revision:7,updated_at:'2026-10-11T10:06:00+00:00'}),false);
  tracker.reset();assert.equal(tracker.context,null);
});
test('source panel safely renders unknown reason and only plots complete coordinates',()=>{
  const nodes=new Map(),get=id=>{if(!nodes.has(id))nodes.set(id,element(id));return nodes.get(id);};
  renderAffect({valence:axis(.5),language_estimate:{status:'unknown',reason:'<img onerror=alert(1)>'}},get,Date.parse(timestamp));
  assert.equal(get('affectPoint').style.display,'none');assert.match(get('languageEstimate').textContent,/<img/);
  assert.equal(get('languageEstimate').innerHTML,undefined);
  renderAffect({valence:axis(.5),arousal:axis(-.5),conflict:true},get,Date.parse(timestamp));
  assert.equal(get('affectPoint').attributes.cx,219);assert.equal(get('affectPoint').attributes.cy,219);
  assert.match(get('valenceSource').textContent,/最近60秒/);assert.match(get('affectConflict').textContent,/优先/);
});

const turn=()=>new Promise(resolve=>setImmediate(resolve));
const defer=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
const response=body=>({ok:true,json:async()=>body});
async function appFixture(fetchImpl) {
  const nodes=new Map(),get=id=>{if(!nodes.has(id))nodes.set(id,element(id));return nodes.get(id);};
  get('userId').value='user001';get('arousalScore').value='';get('valenceScore').value='';
  const source=(await readFile(new URL('app.js',web),'utf8')).replace(/^import .*;\r?\n/gm,'');
  const context=vm.createContext({document:{getElementById:get,createElement:()=>element()},
    window:{addEventListener(){}},navigator:{},fetch:fetchImpl,AbortSignal:{timeout:ms=>({ms})},Event,Date,Number,Math,Map,Set,
    SpeechInput,AffectTracker,renderAffect,FrameAssembler:class {reset(){}},
    connectionBase:()=>'http://localhost',startWaves(){},resetBleSignals(){},ingestBlePacket(){},
    setTimeout(){},setInterval(){},console});
  vm.runInContext(source,context);await turn();return {get,context};
}
test('failed chat preserves text and late chat result cannot cross participant change',async()=>{
  const pending=defer();let rejectChat=true;
  const {get}=await appFixture(url=>url.includes('chat/message') ? rejectChat ? Promise.reject(new Error('offline')) : pending.promise : Promise.resolve(response({affect_context:{revision:0}})));
  get('message').value='我今天很开心';
  await get('chatForm').onsubmit({preventDefault(){}});
  assert.equal(get('message').value,'我今天很开心');assert.match(get('chatStatus').textContent,/已保留/);
  rejectChat=false;const submission=get('chatForm').onsubmit({preventDefault(){}});
  get('userId').value='user002';get('userId').onchange();
  pending.resolve(response({response:'旧用户回复',affect_context:{revision:8}}));await submission;
  assert.equal(get('messages').children.some(child=>child.textContent==='旧用户回复'),false);
  assert.equal(get('message').value,'');assert.equal(get('sendMessage').disabled,false);
});
test('AI requests use extended timeout and self-report saves avoid clearing new edits',async()=>{
  const pending=defer();const requests=[];
  const {get}=await appFixture((url,options)=>{requests.push({url,options});return url.includes('dataset/self-report')?pending.promise:Promise.resolve(response({response:'好的',affect_context:{revision:1}}));});
  get('selfNote').value='提交时的记录';get('valenceScore').value='7';
  const saving=get('saveLabel').onclick();get('selfNote').value='请求过程中修改的记录';
  pending.resolve(response({label:'愉悦',training_sample_saved:false,training_reason:'无合格窗口',affect_context:{revision:1}}));await saving;
  assert.equal(get('selfNote').value,'请求过程中修改的记录');assert.equal(get('valenceScore').value,'');
  get('message').value='感到开心';await get('chatForm').onsubmit({preventDefault(){}});
  const chat=requests.find(request=>request.url.includes('chat/message'));
  assert.equal(JSON.parse(chat.options.body).input_source,'text');assert.equal(chat.options.signal.ms,90000);
  assert.equal(requests.find(request=>request.url.includes('dataset/self-report')).options.signal.ms,8000);
  assert.equal(get('message').value,'');
});
