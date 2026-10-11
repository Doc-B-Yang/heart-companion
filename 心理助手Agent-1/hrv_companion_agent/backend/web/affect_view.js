const finiteAxis=value=>typeof value==='number' && Number.isFinite(value) && value>=-1 && value<=1;
const time=value=>typeof value==='string' ? Date.parse(value)||0 : 0;
export function currentAxis(axis,now=Date.now()) {
  if(!axis || axis.status!=='available' || !finiteAxis(axis.value) || (time(axis.expires_at) && time(axis.expires_at)<=now)) return null;
  return axis.value;
}
export function affectCoordinates(context,now=Date.now()) {
  const valence=currentAxis(context?.valence,now), arousal=currentAxis(context?.arousal,now);
  return {valence,arousal,complete:valence!==null && arousal!==null};
}
export class AffectTracker {
  constructor(){this.reset();}
  reset(){this.context=null;this.revision=-1;this.time=0;this.sessionId=null;this.oldSessions=new Set();}
  update(context){
    if(!context) return false;
    const revision=Number.isInteger(context.revision) ? context.revision : 0;
    const stamp=time(context.updated_at)||Math.max(time(context.valence?.updated_at),time(context.arousal?.updated_at),time(context.language_estimate?.updated_at));
    const sessionId=typeof context.session_id==='string' ? context.session_id : null;
    if(sessionId && this.oldSessions.has(sessionId)) return false;
    if(sessionId && this.sessionId && sessionId!==this.sessionId) {
      if(stamp<this.time) return false;
      this.oldSessions.add(this.sessionId);this.revision=-1;this.time=0;
    }
    // Older backends without session IDs can restart their per-user revision counter.
    if(!sessionId && revision<this.revision && stamp>this.time) this.revision=-1;
    if(revision<this.revision || (revision===this.revision && stamp<this.time)) return false;
    this.context=context;this.revision=revision;this.time=stamp;this.sessionId=sessionId;return true;
  }
}
function valueText(value){return value===null ? '未知' : (value>0?'+':'')+value.toFixed(2);}
function sourceText(axis){return axis?.source==='self_report' ? '最近60秒的自评' : axis?.source==='text_inference' ? '本条表达的当前感受推测' : '等待自评或语言表达';}
function stampText(axis){
  if(!time(axis?.updated_at)) return '';
  const updated=new Date(axis.updated_at).toLocaleTimeString('zh-CN',{hour12:false});
  const expiry=time(axis.expires_at) ? new Date(axis.expires_at).toLocaleTimeString('zh-CN',{hour12:false}) : '';
  return ` · 更新 ${updated}`+(expiry?`，有效至 ${expiry}`:'');
}
export function renderAffect(context,getElement,now=Date.now()) {
  const coordinates=affectCoordinates(context,now);
  for(const key of ['valence','arousal']) {
    const axis=context?.[key],value=coordinates[key];
    getElement(key+'Value').textContent=valueText(value)+(value!==null && Number.isInteger(axis?.score_1_9)?`（${axis.score_1_9}/9）`:'');
    getElement(key+'Source').textContent=value===null ? '暂未知；不会用中性值补齐' : sourceText(axis)+stampText(axis);
  }
  const point=getElement('affectPoint');
  point.hidden=!coordinates.complete;
  point.style.display=coordinates.complete?'':'none';
  if(coordinates.complete){point.setAttribute('cx',160+118*coordinates.valence);point.setAttribute('cy',160-118*coordinates.arousal);}
  getElement('affectSummary').textContent=coordinates.complete ? `效价 ${valueText(coordinates.valence)}，唤醒度 ${valueText(coordinates.arousal)}。` : coordinates.valence!==null ? '效价已有依据；补充唤醒度自评后才显示二维坐标。' : coordinates.arousal!==null ? '唤醒度已有自评；补充效价自评或分析感受文字后才显示二维坐标。' : '两条轴尚未知。自评可补充两轴，语言只能辅助估计效价。';
  getElement('affectConflict').textContent=context?.conflict && coordinates.valence!==null ? '自评与语言推测差异较大，也可能描述时段不同；坐标优先保留你的自评。' : '';
  const estimate=context?.language_estimate;
  const available=estimate?.status==='available' && finiteAxis(estimate.value) && (!time(estimate.expires_at) || time(estimate.expires_at)>now);
  getElement('languageEstimate').textContent=available ? `AI 语言推测效价 ${valueText(estimate.value)}（${estimate.input_source==='speech_transcript'?'语音转写':'文字'}）${stampText(estimate)}。这是语言线索估计，未经准确率验证。` : estimate?.reason ? '语言推测暂未知：'+String(estimate.reason).slice(0,300) : '语言推测等待你的表达；身体信号不会被强行补成效价。';
}
