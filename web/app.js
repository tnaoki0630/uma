const state={tab:"win",result:null,payload:null};
const api=(window.UMA_CONFIG?.API_BASE_URL||"").replace(/\/$/,"");
const $=s=>document.querySelector(s);
const pct=n=>`${(n*100).toFixed(n<.01?2:1)}%`;
async function loadDemo(){
  const urls=api?[`${api}/api/demo`,"demo-race.json"]:["demo-race.json"];
  for(const url of urls){try{const response=await fetch(url);if(response.ok){state.payload=await response.json();break}}catch{}}
  if(!state.payload)throw new Error("デモデータを読み込めませんでした");
  const c=state.payload.conditions;$("#conditions").innerHTML=[c.venue,c.surface,`${c.distance_m}m`,c.track_condition].map(x=>`<span class="chip">${x}</span>`).join("");
}
function row(number,title,prob,detail,ev){return `<article class="result-card"><span class="number">${number}</span><div><div class="name">${title}</div><div class="detail">${detail}</div></div><div class="prob">${pct(prob)}${ev==null?"":`<div class="ev">期待回収率 ${(ev*100).toFixed(1)}%</div>`}</div><div class="meter"><i style="width:${Math.min(100,prob*300)}%"></i></div></article>`}
function render(){if(!state.result)return;let html="";if(state.tab==="win")html=state.result.entries.sort((a,b)=>b.win_probability-a.win_probability).map(x=>row(x.horse_number,x.name,x.win_probability,`${x.mean_speed_mps.toFixed(2)} ± ${x.std_speed_mps.toFixed(2)} m/s · ${x.model==="gp"?"GP":"補完"}`,x.expected_return)).join("");else html=state.result[state.tab].slice(0,20).map((x,i)=>row(i+1,x.selection.join(" – "),x.probability,x.decimal_odds?`オッズ ${x.decimal_odds.toFixed(1)}倍`:"券種別オッズ未入力",x.expected_return)).join("");$("#results").innerHTML=html;$("#warnings").innerHTML=state.result.warnings.map(x=>`<p>※ ${x}</p>`).join("")}
async function predict(){const button=$("#predictButton");button.disabled=true;button.textContent="計算中…";try{let result;if(api){const response=await fetch(`${api}/api/predict`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(state.payload)});if(!response.ok)throw new Error(await response.text());result=await response.json();$("#status").textContent="API接続中"}else{const response=await fetch("demo-result.json");if(!response.ok)throw new Error("静的デモ結果がありません");result=await response.json();$("#status").textContent="静的デモ"}state.result=result;render()}catch(error){$("#results").innerHTML=`<div class="empty">計算できませんでした。<br>${error.message}</div>`}finally{button.disabled=false;button.textContent="確率を計算"}}
document.querySelectorAll(".tabs button").forEach(button=>button.addEventListener("click",()=>{document.querySelector(".tabs .active").classList.remove("active");button.classList.add("active");state.tab=button.dataset.tab;render()}));$("#predictButton").addEventListener("click",predict);loadDemo().catch(error=>$("#results").textContent=error.message);
