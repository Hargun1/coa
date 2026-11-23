'use strict';
const $ = id => document.getElementById(id);
const names = {lru:'LRU',fifo:'FIFO',lfu:'LFU',srrip:'SRRIP',online:'Online',frozen:'Frozen',recency:'Recency'};
const order = Object.keys(names);
let paperData = null;
let activeData = null;
let demoMode = false;
const fixed = (value, digits=2) => Number(value).toFixed(digits);

function render() {
  if (!activeData) return;
  const workload = $('workload').value;
  const size = Number($('capacity').value);
  $('tier-size').textContent = `${size / 16} KiB`;
  const rows = activeData.summary.filter(r => r.workload === workload && r.l3_lines === size).sort((a,b)=>order.indexOf(a.policy)-order.indexOf(b.policy));
  if (!rows.length) {
    $('status').textContent = 'This selection is not in the current demo. Run an experiment or restore paper results.';
    for (const id of ['amat','reduction','hit-rate','runtime']) $(id).textContent='—';
    $('bars').replaceChildren(); $('policy-table').replaceChildren(); $('timeline').replaceChildren();
    return;
  }
  const online = rows.find(r=>r.policy==='online');
  if (!online) return;
  $('amat').textContent = fixed(online.amat_cycles);
  const reduction=online.amat_reduction_vs_lru_pct;
  $('reduction').textContent = `${reduction >= 0 ? '+' : ''}${fixed(reduction)}%`;
  $('reduction').className = reduction >= 0 ? 'positive' : 'negative';
  $('reduction-note').textContent = reduction >= 0 ? 'Lower modeled latency than LRU' : 'Higher modeled latency than LRU';
  $('hit-rate').textContent=`${fixed(100*online.cache_hit_ratio,1)}%`;
  $('runtime').textContent=`${fixed(online.runtime_ms,0)} ms`;
  const study=activeData.manifest.study;
  $('status').textContent = `${demoMode ? 'INTERACTIVE DEMO' : 'PAPER RESULTS'} · ${study.seeds.length} seed${study.seeds.length===1?'':'s'} · ${study.warmup.toLocaleString()} warmup + ${study.measured_requests.toLocaleString()} measured requests per run${demoMode ? '' : ' · H = 256'}`;
  const max=Math.max(...rows.map(r=>r.amat_cycles))*1.06;
  $('bars').innerHTML=rows.map(r=>`<div class="bar-row"><span class="bar-label">${names[r.policy]}</span><div class="bar-track"><div class="bar ${r.policy}" style="width:${r.amat_cycles/max*100}%"></div></div><span class="bar-value">${fixed(r.amat_cycles)}</span></div>`).join('');
  $('bars').setAttribute('aria-label',rows.map(r=>`${names[r.policy]} ${fixed(r.amat_cycles)} cycles`).join(', '));
  $('policy-table').innerHTML=rows.map(r=>`<tr class="${r.policy==='online'?'highlight':''}"><td>${names[r.policy]}</td><td>${fixed(r.amat_cycles)}</td><td>${fixed(r.cache_hit_ratio*100,1)}%</td><td>${fixed(r.fill_bytes/1024,1)}</td><td>${fixed(r.runtime_ms,1)}</td></tr>`).join('');
  drawTimeline(workload,size);
}

function drawTimeline(workload,size) {
  const series=['lru','srrip','online'].map(policy=>{
    const runs=activeData.runs.filter(r=>r.policy===policy && r.workload===workload && r.config.l3_lines===size);
    if(!runs.length) return {policy,points:[]};
    return {policy,points:runs[0].windows.map((w,i)=>({x:w.end,y:runs.reduce((sum,r)=>sum+r.windows[i].amat_cycles,0)/runs.length}))};
  });
  const all=series.flatMap(s=>s.points);
  if(!all.length){$('timeline').replaceChildren();return;}
  const W=900,H=255,L=49,R=20,T=17,B=38;
  const xmax=Math.max(...all.map(p=>p.x));
  const ymax=Math.max(10,Math.ceil(Math.max(...all.map(p=>p.y))/20)*20);
  const x=value=>L+value/xmax*(W-L-R);
  const y=value=>H-B-value/ymax*(H-T-B);
  const colors={online:'#176b53',lru:'#798b95',srrip:'#c87931'};
  let svg=`<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">`;
  for(let i=0;i<=4;i++){
    const value=ymax*i/4;
    svg+=`<line x1="${L}" y1="${y(value)}" x2="${W-R}" y2="${y(value)}" stroke="#e7ece7"/><text x="${L-10}" y="${y(value)+4}" text-anchor="end" fill="#647780" font-size="10">${fixed(value,0)}</text>`;
  }
  for(let i=0;i<=4;i++){
    const value=xmax*i/4;
    svg+=`<text x="${x(value)}" y="${H-14}" text-anchor="middle" fill="#647780" font-size="10">${fixed(value/1000,1)}k</text>`;
  }
  for(const s of series){
    svg+=`<polyline fill="none" stroke="${colors[s.policy]}" stroke-width="${s.policy==='online'?2.6:1.7}" points="${s.points.map(p=>`${x(p.x)},${y(p.y)}`).join(' ')}"/>`;
    for(const p of s.points) svg+=`<circle cx="${x(p.x)}" cy="${y(p.y)}" r="3" fill="${colors[s.policy]}"><title>${names[s.policy]}: ${fixed(p.y)} cycles at request ${p.x}</title></circle>`;
  }
  $('timeline').innerHTML=svg+'</svg>';
}

$('workload').addEventListener('change',render);
$('capacity').addEventListener('change',render);
$('restore').addEventListener('click',()=>{
  if(!paperData){$('status').textContent='No paper benchmark found. Run python -m adaptive_memory benchmark first.';return;}
  demoMode=false;activeData=paperData;$('horizon').value='256';render();
});
$('controls').addEventListener('submit',async event=>{
  event.preventDefault();
  const button=$('run'); button.disabled=true;button.textContent='Simulating…';
  $('status').textContent='Running seven policies locally on the same seeded trace…';
  try{
    const response=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json','X-Lab-Request':'1'},body:JSON.stringify({workload:$('workload').value,l3_lines:Number($('capacity').value),horizon:Number($('horizon').value),seed:Number($('seed').value),requests:15000,warmup:3000})});
    const data=await response.json();if(!response.ok)throw new Error(data.error || 'Experiment failed');
    activeData=data;demoMode=true;render();
  }catch(error){$('status').textContent=`Could not run experiment: ${error.message}`;}
  finally{button.disabled=false;button.innerHTML='Run experiment <span>↗</span>';}
});
fetch('/api/results').then(async response=>{
  if(!response.ok)throw new Error('Run the benchmark to populate saved results, or use Run experiment.');
  paperData=await response.json();activeData=paperData;render();
}).catch(error=>{$('status').textContent=error.message;});
