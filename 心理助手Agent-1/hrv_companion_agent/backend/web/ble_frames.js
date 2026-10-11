// Shared framing contract: HC, uint16-le id, uint8 index, uint8 count, payload.
const PACKET_FIELDS=['schema','device_id','boot_id','seq','device_ms','segment_id','continuity','ppg','imu','motion_known','moving','motion','sig_quality','pi','sample_rate','expected_rate','ibi_events','ads','eda_samples','errors','nn','hr'];
export class FrameAssembler {
  constructor(timeoutMs=2500){this.timeoutMs=timeoutMs;this.dropped=0;this.reset();}
  reset(){this.id=null;this.count=0;this.parts=new Map();this.started=0;}
  add(input,now=performance.now()){
    const bytes=input instanceof Uint8Array?input:new Uint8Array(input.buffer,input.byteOffset,input.byteLength);
    if(bytes.length<7||bytes.length>244||bytes[0]!==72||bytes[1]!==67)return null;
    const id=bytes[2]|bytes[3]<<8,index=bytes[4],count=bytes[5];
    if(!count||index>=count)return null;
    if(this.id!==id||now-this.started>this.timeoutMs){if(this.parts.size)this.dropped++;this.reset();this.id=id;this.count=count;this.started=now;}
    if(count!==this.count){this.dropped++;this.reset();return null;}
    const payload=bytes.slice(6),old=this.parts.get(index);
    if(old&&(old.length!==payload.length||old.some((x,i)=>x!==payload[i]))){this.dropped++;this.reset();return null;}
    this.parts.set(index,payload);
    const size=[...this.parts.values()].reduce((a,b)=>a+b.length,0);
    if(size>8192){this.dropped++;this.reset();return null;}
    if(this.parts.size!==count)return null;
    const full=new Uint8Array(size);let offset=0;
    for(let i=0;i<count;i++){full.set(this.parts.get(i),offset);offset+=this.parts.get(i).length;}
    this.reset();
    try{let result=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(full));if(result?.hc===1){if(!Array.isArray(result.v)||![22,23].includes(result.v.length))throw Error('Invalid HC compact packet');const raw=result.v[22]||[];result=Object.fromEntries(PACKET_FIELDS.map((key,i)=>[key,result.v[i]]));if(!Number.isFinite(result.device_ms)||!Array.isArray(result.eda_samples))throw Error('Invalid HC sample clock');result.eda_samples=result.eda_samples.map(([t,v])=>{if(!Number.isFinite(t)||!Number.isFinite(v))throw Error('Invalid HC sample');return [t+result.device_ms,v];});if(!Array.isArray(raw)||raw.length>28)throw Error('Invalid HC PPG samples');result.ppg_samples=raw.map(row=>{if(!Array.isArray(row)||row.length!==3||!row.every(Number.isFinite))throw Error('Invalid HC PPG sample');const [t,ir,segment]=row;return [segment,t+result.device_ms,ir,ir,0];});}return result&&typeof result==='object'&&!Array.isArray(result)?result:null;}
    catch{this.dropped++;return null;}
  }
}
