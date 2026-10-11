// Display/export reception separately from eligibility for emotion analysis.
export class BleSignals {
  constructor(){this.clear();}
  clear(){this.key=null;this.seq=null;this.eda=[];this.ppg=[];this.records=[];this.packet=null;this.received=0;this.count=0;}
  add(packet,received=Date.now()){
    if(packet?.schema!==2||!Number.isInteger(packet.seq)||!Number.isFinite(packet.device_ms)||typeof packet.device_id!=='string'||typeof packet.boot_id!=='string')return false;
    const key=JSON.stringify([packet.device_id,packet.boot_id]);
    if(key!==this.key){this.clear();this.key=key;}
    if(this.seq!==null&&packet.seq<=this.seq)return false;
    this.seq=packet.seq;this.packet=packet;this.received=received;this.count++;
    this.records.push({received_unix:received/1000,packet});this.records=this.records.slice(-600);
    for(const row of Array.isArray(packet.eda_samples)?packet.eda_samples:[]){
      if(Array.isArray(row)&&row.length===2&&row.every(Number.isFinite)&&row[0]<=packet.device_ms&&(!this.eda.length||row[0]>this.eda.at(-1)[0]))this.eda.push([row[0],row[1]]);
    }
    this.eda=this.eda.filter(p=>p[0]>=packet.device_ms-30000).slice(-1000);
    for(const row of Array.isArray(packet.ppg_samples)?packet.ppg_samples:[]){
      if(Array.isArray(row)&&row.length>=4&&row.slice(0,4).every(Number.isFinite)&&row[1]<=packet.device_ms&&(!this.ppg.length||row[1]>this.ppg.at(-1)[1]))this.ppg.push(row);
    }
    this.ppg=this.ppg.filter(p=>p[1]>=packet.device_ms-10000).slice(-400);
    return true;
  }
  fresh(now=Date.now()){return !!this.packet&&now-this.received<5000;}
  jsonl(){return this.records.map(r=>JSON.stringify(r)).join('\n')+'\n';}
}
