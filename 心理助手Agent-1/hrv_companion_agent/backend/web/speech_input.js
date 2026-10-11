// Chrome's recognition service transcribes words; this module does not analyze voice tone.
export const speechErrorMessage = code => ({
  'not-allowed':'麦克风权限未允许。可在 Chrome 网站设置中允许，或继续文字输入。',
  'service-not-allowed':'浏览器语音识别服务不可用，请继续文字输入。',
  'audio-capture':'未找到可用麦克风，请检查麦克风连接。',
  'no-speech':'没有识别到语音，可重试或直接输入文字。',
  'network':'语音识别网络连接失败，可重试或直接输入文字。',
  'language-not-supported':'语音服务暂不支持中文，请继续文字输入。',
  'aborted':'本次语音输入已取消。',
}[code] || '语音识别未完成，请重试或继续文字输入。');

export function appendTranscript(value, transcript, maxLength) {
  const suffix = value && !/\s$/.test(value) ? ' ' : '';
  return (value + suffix + transcript.trim()).slice(0, maxLength > 0 ? maxLength : 5000);
}

export class SpeechInput {
  constructor({Recognition, inputs, onCommit=()=>{}}) {
    this.Recognition=Recognition;
    this.inputs=inputs;
    this.onCommit=onCommit;
    this.session=null;
    inputs.forEach(input=>{
      input.button.disabled=!Recognition;
      input.button.onclick=()=>this.session?.input===input ? this.stop() : this.start(input);
      input.cancel.onclick=()=>this.cancel();
      input.cancel.hidden=true;
      if(!Recognition) input.status.textContent='此浏览器不支持语音识别，请使用 Chrome 或继续文字输入。';
    });
  }
  start(input) {
    if(!this.Recognition || this.session) return;
    let recognition;
    try { recognition=new this.Recognition(); }
    catch { input.status.textContent='无法启动语音识别，请继续文字输入。'; return; }
    const session={recognition,input,finals:new Map(),failed:false};
    this.session=session;
    recognition.lang='zh-CN';
    recognition.continuous=false;
    recognition.interimResults=true;
    recognition.maxAlternatives=1;
    this.inputs.forEach(item=>{item.button.disabled=item!==input;});
    input.button.textContent='停止并写入';
    input.cancel.hidden=false;
    input.button.setAttribute('aria-pressed','true');
    input.status.textContent='正在听…结束后会写入输入框，可编辑后再发送。';
    recognition.onresult=event=>{
      if(this.session!==session || session.failed) return;
      const previews=[];
      for(let index=0; index<event.results.length; index++) {
        const result=event.results[index];
        const transcript=result[0]?.transcript?.trim()||'';
        if(result.isFinal) session.finals.set(index,transcript);
        previews.push(transcript);
      }
      input.status.textContent='识别预览：'+previews.join(' ').slice(0,300);
    };
    recognition.onerror=event=>{
      if(this.session!==session) return;
      session.failed=true;
      this.finish(session,false,speechErrorMessage(event.error));
      try { recognition.abort(); } catch {}
    };
    recognition.onend=()=>{
      if(this.session===session) this.finish(session,!session.failed);
    };
    try { recognition.start(); }
    catch { this.finish(session,false,'无法启动语音识别，请检查权限或继续文字输入。'); }
  }
  stop() {
    const session=this.session;
    if(!session) return;
    session.input.status.textContent='正在结束识别…';
    try { session.recognition.stop(); }
    catch { this.finish(session,true); }
  }
  cancel() {
    const session=this.session;
    if(!session) return;
    this.finish(session,false,'本次语音已取消，原有文字已保留。');
    try { session.recognition.abort(); } catch {}
  }
  finish(session,commit,message='') {
    if(this.session!==session) return;
    this.session=null;
    const {input}=session;
    const transcript=[...session.finals.entries()].sort((a,b)=>a[0]-b[0]).map(entry=>entry[1]).join(' ').trim();
    if(commit && transcript) {
      const before=input.field.value;
      const after=appendTranscript(before,transcript,input.field.maxLength);
      input.field.value=after;
      if(after!==before) {
        this.onCommit(input.field.id);
        input.field.dispatchEvent(new Event('input',{bubbles:true}));
      }
      input.status.textContent=after.length >= input.field.maxLength ? '已写入识别文字，达到字数上限，请检查后再发送。' : '识别文字已写入，可编辑后再发送；尚未发送。';
      input.field.focus();
    } else input.status.textContent=message||'没有识别到完整语句，原有文字已保留。';
    this.inputs.forEach(item=>{item.button.disabled=!this.Recognition;});
    input.button.textContent='语音输入';
    input.button.setAttribute('aria-pressed','false');
    input.cancel.hidden=true;
    session.recognition.onresult=null;
    session.recognition.onerror=null;
    session.recognition.onend=null;
  }
}
