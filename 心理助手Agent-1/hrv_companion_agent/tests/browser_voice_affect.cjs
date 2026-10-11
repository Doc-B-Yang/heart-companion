// Browser smoke: fake recognition/API, no microphone, hardware, user records or provider calls.
const assert=require('node:assert/strict'),http=require('node:http'),fs=require('node:fs/promises'),path=require('node:path');
const {chromium}=require('playwright');
const web=path.resolve(__dirname,'../backend/web');
const assets=new Set(['index.html','style.css','refinements.css','app.js','waveforms.js','ble_frames.js','ble_signals.js','speech_input.js','affect_view.js']);
const server=http.createServer(async(req,res)=>{
  const name=req.url==='/'?'index.html':req.url?.replace(/^\/static\//,'');
  if(!assets.has(name)){res.writeHead(404);res.end();return;}
  try{res.writeHead(200,{'Content-Type':name.endsWith('.js')?'text/javascript':name.endsWith('.css')?'text/css':'text/html'});res.end(await fs.readFile(path.join(web,name)));}
  catch{res.writeHead(500);res.end();}
});
function axis(value,score,source='self_report'){
  return {value,score_1_9:score,status:'available',source,updated_at:new Date().toISOString(),expires_at:new Date(Date.now()+300000).toISOString()};
}
async function main(){
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  let browser;
  try{
    let executablePath;
    for(const candidate of [process.env.HEART_TEST_BROWSER,chromium.executablePath(),'C:/Program Files/Google/Chrome/Application/chrome.exe','C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'].filter(Boolean)){
      try{await fs.access(candidate);executablePath=candidate;break;}catch{}
    }
    if(!executablePath)throw Error('No installed Chromium browser. Set HEART_TEST_BROWSER to a Chrome/Edge executable or install Playwright Chromium.');
    browser=await chromium.launch({headless:true,executablePath});
    const page=await browser.newPage({viewport:{width:1280,height:900}}),errors=[];
    page.on('pageerror',error=>errors.push(error.message));
    let context={session_id:'browser-test',revision:0,updated_at:new Date().toISOString()},chatBody=null,saveBody=null,failChat=false;
    const fulfill=(route,body,status=200)=>route.fulfill({status,contentType:'application/json',body:JSON.stringify(body)});
    await page.route('**/api/**',async route=>{
      const endpoint=new URL(route.request().url()).pathname,data=route.request().method()==='POST'?route.request().postDataJSON():null;
      if(endpoint==='/api/emotion/state')return fulfill(route,{layer:'unknown',affect_context:context});
      if(endpoint==='/api/chat/message'){
        chatBody=data;if(failChat)return fulfill(route,{detail:'测试：后端不可用'},503);
        context={...context,revision:context.revision+1,updated_at:new Date().toISOString(),valence:axis(.5,null,'text_inference'),arousal:null};
        return fulfill(route,{response:'听起来你现在感觉愉快。',affect_context:context});
      }
      if(endpoint==='/api/dataset/self-report'){
        saveBody=data;context={...context,revision:context.revision+1,updated_at:new Date().toISOString(),valence:axis((data.valence_score-5)/4,data.valence_score),arousal:axis((data.arousal_score-5)/4,data.arousal_score)};
        return fulfill(route,{label:data.label,training_sample_saved:false,training_reason:'测试：无生理窗口',affect_context:context});
      }
      if(endpoint==='/api/emotion/valence')return fulfill(route,{user_id:data.user_id,affect_context:context});
      return fulfill(route,{});
    });
    await page.route('http://127.0.0.1:8011/**',route=>fulfill(route,{ppg:[],eda:[],count:0}));
    await page.addInitScript(()=>{
      class FakeRecognition{
        constructor(){window.__recognition=this;}
        start(){this.started=true;}
        stop(){this.onend?.();}
        abort(){this.onend?.();}
        finish(text){this.onresult?.({results:[{0:{transcript:text},isFinal:true}],resultIndex:0});this.onend?.();}
      }
      window.SpeechRecognition=FakeRecognition;
    });
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.waitForFunction(()=>document.querySelector('#serverStatus').textContent==='后端已连接');
    assert.equal(await page.locator('#affectPoint').isVisible(),false);
    await page.locator('#message').fill('今天');await page.locator('#chatVoice').click();
    await page.evaluate(()=>window.__recognition.finish('我感觉很开心'));
    assert.equal(await page.locator('#message').inputValue(),'今天 我感觉很开心');assert.equal(chatBody,null);
    await page.locator('#message').fill('今天 我感觉很开心，但有点紧张');await page.locator('#sendMessage').click();
    await page.waitForFunction(()=>document.querySelector('#chatStatus').textContent.startsWith('已回应'));
    assert.equal(chatBody.input_source,'speech_transcript');assert.equal(await page.locator('#affectPoint').isVisible(),false);
    assert.match(await page.locator('#valenceSource').textContent(),/当前感受推测/);
    await page.locator('#selfNote').fill('这条自评可编辑');await page.locator('#noteVoice').click();
    await page.evaluate(()=>window.__recognition.finish('有点疲惫但仍然开心'));
    assert.match(await page.locator('#selfNote').inputValue(),/这条自评可编辑/);
    await page.locator('#selfNote').fill('修改后的自评：比较开心，身体平静');
    await page.locator('#valenceScore').selectOption('7');await page.locator('#arousalScore').selectOption('3');await page.locator('#saveLabel').click();
    await page.waitForFunction(()=>document.querySelector('#labelStatus').textContent.startsWith('已保存'));
    assert.equal(saveBody.note,'修改后的自评：比较开心，身体平静');assert.equal(await page.locator('#affectPoint').isVisible(),true);
    assert.match(await page.locator('#valenceSource').textContent(),/自评/);
    assert.equal(Number(await page.locator('#affectPoint').getAttribute('cx')),219);assert.equal(Number(await page.locator('#affectPoint').getAttribute('cy')),219);
    failChat=true;await page.locator('#message').fill('失败时这条文字应保留');await page.locator('#sendMessage').click();
    await page.waitForFunction(()=>document.querySelector('#chatStatus').textContent.startsWith('发送失败'));
    assert.equal(await page.locator('#message').inputValue(),'失败时这条文字应保留');
    await page.locator('#chatVoice').click();await page.locator('#chatVoiceCancel').click();
    assert.equal(await page.locator('#message').inputValue(),'失败时这条文字应保留');
    await page.setViewportSize({width:390,height:844});await page.waitForTimeout(100);
    const overflow=await page.evaluate(()=>document.documentElement.scrollWidth-window.innerWidth);
    assert.ok(overflow<=1,`mobile page overflows by ${overflow}px`);assert.deepEqual(errors,[]);
    console.log('Headless Chromium smoke passed ('+executablePath+'): editable speech, explicit provenance, self-report save/source, partial/full VA point, failed-send preservation, cancel, mobile fit; no page errors.');
  }finally{if(browser)await browser.close();await new Promise(resolve=>server.close(resolve));}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
