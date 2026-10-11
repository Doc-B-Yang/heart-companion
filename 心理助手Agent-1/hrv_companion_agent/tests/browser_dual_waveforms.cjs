const {chromium}=require('playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const page=await browser.newPage();let payload=null;const errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/api/dataset/self-report',async route=>{payload=route.request().postDataJSON();await route.fulfill({json:{label:payload.label,training_sample_saved:false,training_reason:'浏览器测试：无硬件训练窗口'}});});
 await page.goto('http://127.0.0.1:8000');
 await page.waitForFunction(()=>document.getElementById('serverStatus').textContent==='后端已连接');
 assert(await page.locator('#confirmContact').isEnabled());
 await page.locator('#selfNote').fill('浏览器回归测试（不会存入参与者记录）');
 await page.locator('#arousalScore').selectOption('5');await page.locator('#valenceScore').selectOption('4');
 await page.locator('#saveLabel').click();await page.waitForFunction(()=>document.getElementById('labelStatus').textContent.startsWith('已保存'));
 assert.equal(payload.note,'浏览器回归测试（不会存入参与者记录）');assert.equal(payload.arousal_score,5);assert.equal(payload.valence_score,4);
 const result=await page.evaluate(async()=>{
  const {FrameAssembler}=await import('/static/ble_frames.js');
  const {ingestBlePacket}=await import('/static/waveforms.js');
  const values=[2,'qa-fixture','qa-boot',1,1000,3,false,true,true,true,false,0,0,0,100,100,[],true,[[0,1.4]],{ppg:0,imu:0,ads:0,fifo:0,clock:0},0,0,Array.from({length:25},(_,i)=>[-960+i*40,62000+Math.round(Math.sin(i)*50),3])];
  const data=new TextEncoder().encode(JSON.stringify({hc:1,v:values}));const count=Math.ceil(data.length/14);const assembler=new FrameAssembler();let packet;
  for(let i=0;i<count;i++){let p=data.slice(i*14,(i+1)*14),f=new Uint8Array(p.length+6);f.set([72,67,1,0,i,count]);f.set(p,6);packet=assembler.add(f);}
  ingestBlePacket(packet);return {samples:packet.ppg_samples.length,status:document.getElementById('waveStatus').textContent,ppgHasInk:document.getElementById('ppgWave').getContext('2d').getImageData(0,0,100,100).data.some(x=>x!==0)};
 });
 assert.equal(result.samples,25);assert.equal(result.status,'BLE PPG + GSR 实时波形');assert(result.ppgHasInk);assert.deepEqual(errors,[]);
 console.log(JSON.stringify({chrome:'passed',contact:'clickable',selfReport:'payload preserved; mocked save passed',dualWaveforms:result,pageErrors:errors}));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
