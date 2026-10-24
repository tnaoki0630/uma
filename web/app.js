const state={tab:"win",catalog:null,result:null};
const api=(window.UMA_CONFIG?.API_BASE_URL||"").replace(/\/$/,"");
const $=selector=>document.querySelector(selector);
const pct=value=>`${(value*100).toFixed(value<.01?2:1)}%`;
function currentMode(){return state.catalog.modes[$("#basisSelect").value]}
function selectedRaces(){const mode=currentMode();const count=Math.max(1,Math.min(Number($("#validationCount").value)||1,mode.max_validation_races));return mode.races.slice(0,count)}
function currentRace(){return selectedRaces()[Number($("#raceSelect").value)||0]}

async function loadCatalog(){
  const urls=api?[`${api}/api/demo-catalog`,"demo-catalog.json"]:["demo-catalog.json"];
  for(const url of urls){try{const response=await fetch(url);if(response.ok){state.catalog=await response.json();break}}catch{}}
  if(!state.catalog)throw new Error("CSV検証データを読み込めませんでした");
  $("#status").textContent=api?"API接続中":"CSV静的デモ";updateMode();
}
function updateMode(){
  const mode=currentMode();const count=$("#validationCount");count.max=mode.max_validation_races;count.value=Math.min(Number(count.value)||3,mode.max_validation_races);
  $("#modelCopy").textContent=mode.basis==="horse"?"レース条件の基準速度に、各馬の過去走から推定した残差を重ねています。":"レース条件の基準速度に、各ジョッキーの過去騎乗から推定した残差を重ねています。出馬表には各馬の騎手名を表示します。";
  refreshRaceOptions();
}
function refreshRaceOptions(){
  const races=selectedRaces();const select=$("#raceSelect");const previous=Math.min(Number(select.value)||0,races.length-1);
  select.innerHTML=races.map((race,index)=>`<option value="${index}">${race.race_label}</option>`).join("");select.value=String(Math.max(0,previous));
  updateRaceHeader();renderEntryList();renderMetrics();state.result=null;$("#results").innerHTML='<div class="empty">「確率を計算」を押してください</div>';
}
function updateRaceHeader(){const race=currentRace();if(!race)return;const c=race.conditions;$("#conditions").innerHTML=[c.venue,c.surface,`${c.distance_m}m`,c.track_condition,`実勝馬 ${race.actual_winner_number}番`].map(value=>`<span class="chip">${value}</span>`).join("")}
function renderEntryList(){
  const race=currentRace();if(!race)return;
  const entries=[...race.entries].sort((a,b)=>a.horse_number-b.horse_number);
  $("#entryCount").textContent=`${entries.length}頭`;
  $("#entryList").innerHTML=entries.map(entry=>{
    const isActual=entry.horse_number===race.actual_winner_number;
    const odds=entry.decimal_odds==null?"—":`${entry.decimal_odds.toFixed(1)}倍`;
    const jockey=entry.jockey_name||"騎手情報なし";
    return `<article class="entry-row${isActual?" actual":""}"><span class="entry-number">${entry.horse_number}</span><div class="entry-name"><strong>${entry.name}</strong><span>${jockey}</span></div><div class="entry-history"><span class="entry-label">過去走</span><strong>${entry.history_count}走</strong></div><div class="entry-odds"><span class="entry-label">単勝</span><strong>${odds}</strong></div>${isActual?'<span class="winner-mark">1着</span>':""}</article>`;
  }).join("");
}
function renderMetrics(){
  const races=selectedRaces();const hit=races.filter(race=>race.predicted_winner_hit).length/races.length;const brier=races.reduce((sum,race)=>sum+race.brier_score,0)/races.length;const winnerRank=races.reduce((sum,race)=>sum+race.actual_winner_probability_rank,0)/races.length;
  $("#metrics").innerHTML=`<div class="metric"><span class="label">検証数</span><strong>${races.length}</strong></div><div class="metric"><span class="label">1位的中率</span><strong>${pct(hit)}</strong></div><div class="metric"><span class="label">平均Brier</span><strong>${brier.toFixed(3)}</strong></div>`;$("#metrics").title=`実勝馬の平均予測順位: ${winnerRank.toFixed(1)}位`;
}
function row(number,title,probability,detail,expectedReturn,isActual=false){return `<article class="result-card${isActual?" actual":""}"><span class="number">${number}</span><div><div class="name">${title}</div><div class="detail">${detail}${isActual?" · 実際の勝馬":""}</div></div><div class="prob">${pct(probability)}${expectedReturn==null?"":`<div class="ev">期待回収率 ${(expectedReturn*100).toFixed(1)}%</div>`}</div><div class="meter"><i style="width:${Math.min(100,probability*300)}%"></i></div></article>`}
function render(){
  if(!state.result)return;let html="";
  if(state.tab==="win")html=[...state.result.entries].sort((a,b)=>b.win_probability-a.win_probability).map(entry=>row(entry.horse_number,entry.name,entry.win_probability,`${entry.jockey_name?entry.jockey_name+" · ":""}${entry.mean_speed_mps.toFixed(2)} ± ${entry.std_speed_mps.toFixed(2)} m/s · ${entry.model==="gp"?"GP":"補完"}`,entry.expected_return,entry.horse_number===state.result.actual_winner_number)).join("");
  else html=state.result[state.tab].slice(0,20).map((item,index)=>row(index+1,item.selection.join(" – "),item.probability,item.decimal_odds?`オッズ ${item.decimal_odds.toFixed(1)}倍`:"券種別オッズ未入力",item.expected_return)).join("");
  $("#results").innerHTML=html;$("#warnings").innerHTML=[`出典: ${currentMode().source}（時系列末尾を検証に使用）`,...state.result.warnings].map(value=>`<p>※ ${value}</p>`).join("");
}
function predict(){const button=$("#predictButton");button.disabled=true;button.textContent="計算中…";state.result=currentRace();render();button.disabled=false;button.textContent="確率を計算"}
document.querySelectorAll(".tabs button").forEach(button=>button.addEventListener("click",()=>{document.querySelector(".tabs .active").classList.remove("active");button.classList.add("active");state.tab=button.dataset.tab;render()}));
$("#basisSelect").addEventListener("change",updateMode);$("#validationCount").addEventListener("change",refreshRaceOptions);$("#raceSelect").addEventListener("change",()=>{updateRaceHeader();renderEntryList();state.result=null;$("#results").innerHTML='<div class="empty">「確率を計算」を押してください</div>'});$("#predictButton").addEventListener("click",predict);
$("#entryToggle").addEventListener("click",()=>{const button=$("#entryToggle");const expanded=button.getAttribute("aria-expanded")==="true";button.setAttribute("aria-expanded",String(!expanded));button.textContent=expanded?"表示":"閉じる";$("#entryList").hidden=expanded});
loadCatalog().catch(error=>$("#results").textContent=error.message);
