function configurationComparison(run, paths) {
  const wrap = document.createElement("div"); wrap.className = "ranking";
  const label = document.createElement("label"); label.textContent = "Rank configurations by ";
  const select = document.createElement("select"); select.setAttribute("aria-label", "Configuration ranking metric");
  const metrics = [["excess", "Mean P&L above holding stock"], ["median", "Median P&L"], ["mean", "Mean P&L"], ["p05", "5th percentile P&L"], ["prob_loss", "Lowest loss probability"], ["median_drawdown", "Lowest median drawdown"], ["return", "Mean return on deployed capital"]];
  select.innerHTML = metrics.map(([key, name]) => `<option value="${key}">${name}</option>`).join(""); label.append(select);
  const host = document.createElement("div"); wrap.append(label, host);
  const groups = run.configurations || [];
  const flatten = (cfg) => {
    const result = Object.fromEntries(Object.entries(cfg).filter(([key, value]) => !["paths", "shares", "path_days", "put_days_left", "initial_dte"].includes(key) && ["string", "number"].includes(typeof value)));
    (cfg.tenors || []).forEach((t, i) => { result[`Call OTM ${i + 1}w`] = t.otm; result[`Contracts ${i + 1}w`] = t.contracts_at_100 ?? t.contracts; });
    (cfg.put_sleeves || []).forEach((s, i) => { result[`Put sleeve ${i + 1}`] = `${s.weight * 100}%:${s.dte}:${s.roll_at}`; });
    result.rules = (cfg.rules || []).join(",") || "all";
    result.put_ratio_grid = JSON.stringify(cfg.put_ratio_grid || []);
    return result;
  };
  const flattened = groups.map(g => flatten(g.config));
  const changed = [...new Set(flattened.flatMap(Object.keys))].filter(key => new Set(flattened.map(cfg => String(cfg[key]))).size > 1);
  const rows = groups.map((g, i) => {
    const members = paths.filter(p => p.configuration === g.id);
    const hold = members.length ? members.reduce((sum, p) => sum + (p.ending_spot - g.config.spot) * g.config.lots * 100, 0) / members.length : null;
    return { ...g.monte_carlo, id: g.id, paths: g.config.paths, excess: hold === null ? null : g.monte_carlo.mean - hold,
      return: g.book.deployed > 0 ? g.monte_carlo.mean / g.book.deployed : null,
      parameters: changed.map(key => `${key.replaceAll("_", " ")}: ${flattened[i][key]}`).join(" · ") || "Same settings" };
  });
  const draw = () => {
    const direction = ["prob_loss", "median_drawdown"].includes(select.value) ? 1 : -1;
    const ranked = rows.slice().sort((a, b) => a[select.value] === null ? 1 : b[select.value] === null ? -1 : direction * (a[select.value] - b[select.value]));
    host.replaceChildren(table(["Rank", "Configuration", "Changed parameters", "Paths", "Mean P&L", "Median", "5th", "Loss", "Drawdown", "Above stock hold", "Mean return"], ranked.map((g, i) => [String(i + 1), String(g.id), g.parameters, String(g.paths), money(g.mean), money(g.median), money(g.p05), pct(g.prob_loss), money(g.median_drawdown), g.excess === null ? "—" : money(g.excess), g.return === null ? "—" : pct(g.return)]), (index) => {
      const filter = document.querySelector('[aria-label="Path configuration"]');
      if (filter) { filter.value = String(ranked[index].id); filter.dispatchEvent(new Event("change")); filter.scrollIntoView?.({block:"center"}); }
    }));
  };
  select.addEventListener("change", draw); draw(); return wrap;
}

function replayComparison(analysis) {
  const section = document.createElement("section");
  section.innerHTML = '<h2>Paired strategy comparison</h2>';
  const note = document.createElement("p"); note.className = "note"; note.textContent = analysis.note;
  const rows = [["All saved paths", analysis.all_paths], ["Fitted path", analysis.training], ["Other saved paths", analysis.other_paths]];
  section.append(note, table(["Evaluation", "Paths", "Mean improvement", "Median improvement", "Paths improved", "Mean above stock hold"], rows.map(([label, stats]) => [label, String(stats.paths), stats.mean_improvement === null ? "—" : money(stats.mean_improvement), stats.median_improvement === null ? "—" : money(stats.median_improvement), stats.prob_improved === null ? "—" : pct(stats.prob_improved), stats.mean_excess_hold === null ? "—" : money(stats.mean_excess_hold)])));
  const link = document.createElement("a"); link.href = `#/run/${analysis.source_run}`; link.textContent = "Open original simulation"; section.append(link); return section;
}

async function showOptimizerForm(run, path) {
  const view = state.view; crumb.textContent = "Optimize saved path";
  app.innerHTML = '<h1>Optimize a saved path</h1><p class="lede">Loading the original strategy and stock-hold benchmark…</p>';
  const context = await get(`/api/optimizer/context/${run}/${path}`);
  if (view !== state.view) return;
  app.innerHTML = `<h1>Optimize path ${path}</h1><p class="lede">Search within your strategy bounds on these saved market prices. Market assumptions and position size stay fixed. The selected path is the fitting sample; apply a candidate to the full simulation to inspect the other paths.</p>`;
  app.append(facts([["Original P&L", money(context.originalPnl)], ["Stock-only P&L", money(context.stockHold)], ["Above holding stock", money(context.originalPnl - context.stockHold)]]));
  const form = document.createElement("form"); form.className = "form optimizer-form";
  form.innerHTML = `<label>Score<select name="objective"><option value="excess_hold">P&L above holding stock</option><option value="pnl">Final P&L</option><option value="risk_adjusted">Above stock hold minus drawdown penalty</option></select></label>
    <label>Drawdown penalty<input name="penalty" type="number" min="0" step="0.1" value="1"/></label>
    <label>Strategy candidates<select name="strategies"><option value="wheel,hold,stock">Wheel, hold opening book, and stock only</option><option value="wheel">Wheel rules only</option><option value="wheel,stock">Wheel and stock only</option></select></label>
    <label>Call policy candidates<select name="policies"><option value="fixed,adaptive">Fixed and adaptive</option><option value="fixed">Fixed only</option><option value="adaptive">Adaptive only</option></select></label>
    <div class="full"><h3>Parameter bounds</h3><p class="note">The original strategy is always included as a reference. Ranges combine with selected strategies, policies and rule switches. At most 1,000 candidates are evaluated.</p><div class="optimizer-ranges"></div><button type="button" class="ghost add-optimizer-range">Add parameter range</button></div>
    <fieldset class="full rule-toggles"><legend>Also search whether these rules should be on or off</legend>${context.rules.map(name => `<label class="check"><input type="checkbox" value="${name}"/>${escapeHtml(name.replaceAll("_", " "))}</label>`).join("")}</fieldset>
    <div class="launch-summary full" role="status">Calculating candidates…</div><button class="run" type="submit" disabled>Start optimizer</button>`;
  const chosen = (savedConfig().ranges || []).filter(r => context.parameters.some(p => p.key === r.parameter));
  const ranges = chosen.length ? chosen.map(r => ({...r})) : [{ parameter:"otm.0", from:0.01, to:0.08, step:0.01 }];
  const host = form.querySelector(".optimizer-ranges"), status = form.querySelector(".launch-summary"), launch = form.querySelector("button.run");
  const payload = () => ({ ranges, objective:form.elements.objective.value, drawdownPenalty:Number(form.elements.penalty.value), strategies:form.elements.strategies.value.split(","), policies:form.elements.policies.value.split(","), toggleRules:[...form.querySelectorAll('.rule-toggles input:checked')].map(i => i.value) });
  let timer, version = 0;
  const preview = () => {
    clearTimeout(timer); const current = ++version; launch.disabled = true;
    timer = setTimeout(async () => { if (!form.isConnected) return; try {
      const result = await postJson(`/api/optimizer/${run}/${path}/preview`, payload());
      if (current !== version || !form.isConnected) return;
      status.textContent = `${result.candidates} strategy candidates · identical saved market path`; status.classList.remove("invalid"); launch.disabled = false;
    } catch(error) { if (current === version && form.isConnected) {status.textContent=error.message;status.classList.add("invalid");} } }, 250);
  };
  const draw = () => {
    host.replaceChildren(); ranges.forEach((range, index) => {
      const row = document.createElement("div"); row.className = "range-row";
      row.innerHTML = `<label>Parameter<select>${context.parameters.map(p=>`<option value="${p.key}">${escapeHtml(p.label)}</option>`).join("")}</select></label>${[["from","Start"],["to","End"],["step","Increment"]].map(([key,label])=>`<label>${label}<input type="number" step="any" data-key="${key}" value="${escapeHtml(range[key])}"/></label>`).join("")}<button type="button" class="remove-range">Remove</button>`;
      row.querySelector("select").value = range.parameter;
      row.querySelector("select").addEventListener("change", e => {range.parameter=e.target.value;preview();});
      row.querySelectorAll("input").forEach(input=>input.addEventListener("input",()=>{range[input.dataset.key]=input.value;}));
      row.querySelector("button").addEventListener("click",()=>{ranges.splice(index,1);draw();preview();}); host.append(row);
    });
  };
  form.querySelector(".add-optimizer-range").addEventListener("click",()=>{const p=context.parameters.find(p=>!ranges.some(r=>r.parameter===p.key));if(p){ranges.push({parameter:p.key,from:p.min??0,to:p.min??0,step:p.integer?1:0.01});draw();preview();}});
  form.addEventListener("input",preview); form.addEventListener("change",preview);
  form.addEventListener("submit",async event=>{event.preventDefault();launch.disabled=true;try{await postJson(`/api/optimizer/${run}/${path}/start`,payload());location.hash="#/active";}catch(error){status.textContent=error.message;preview();}});
  draw(); app.append(form); preview();
}

async function showOptimizerResult(id) {
  const view=state.view; crumb.textContent="Optimizer results"; app.innerHTML='<p class="lede">Loading optimization…</p>';
  const result=await get(`/api/optimizations/${id}`); if(view!==state.view)return;
  app.innerHTML=`<h1>Optimizer · path ${result.training_path}</h1><p class="lede">${escapeHtml(result.note)} ${escapeHtml(result.price_precision)}.</p>`;
  app.append(facts([["Original replay P&L",money(result.baseline.replayed_pnl)],["Stock-only P&L",money(result.baseline.stock_hold.pnl)],["Candidates",String(result.candidates.length)],["Score",result.objective.replaceAll("_"," ")]]));
  const controls=document.createElement("div"); controls.className="optimizer-actions";
  controls.innerHTML=`<label>Candidate<select aria-label="Optimizer candidate">${result.candidates.map((c,i)=>`<option value="${c.id}">${i+1}. ${c.baseline ? "Original · " : ""}${escapeHtml(c.config.strategy)} · ${escapeHtml(c.config.callPolicy)} · ${money(c.excess_hold)} above stock hold</option>`).join("")}</select></label><label>Configuration name<input aria-label="Optimized configuration name" value="Optimized path ${result.training_path}"/></label><div class="exports"><button class="ghost save-candidate">Save configuration</button><button class="ghost use-candidate">Use for new simulations</button><button class="ghost replay-path">Replay selected path</button><button class="ghost replay-run">Apply to whole simulation</button></div>`;
  const status=document.createElement("p");status.className="status";status.setAttribute("role","status");
  const details=document.createElement("div"), chartHost=document.createElement("div");
  const select=controls.querySelector("select"); const candidate=()=>result.candidates.find(c=>c.id===select.value);
  const draw=()=>{
    const c=candidate(); details.replaceChildren(facts([["Candidate P&L",money(c.pnl)],["Above stock hold",money(c.excess_hold)],["Improvement over original",money(c.improvement)],["Maximum drawdown",money(c.max_dd)]]));
    const pre=document.createElement("pre");pre.className="configuration-data";pre.textContent=JSON.stringify(c.config,null,2);details.append(pre);
    chartHost.querySelectorAll(".chart-surface").forEach(disposeChart);
    chartHost.replaceChildren();if(c.id===result.winning_candidate)chartHost.append(chartCard("Best candidate on fitted path","Original strategy, candidate and stock hold on the same prices.",multiChart(result.plot,[{key:"original",label:"original",color:"#a33b32"},{key:"optimized",label:"candidate",color:"#5b4b8a"},{key:"stock_hold",label:"stock only",color:"#8a6a2f"}],{money:true})));
  };
  const action=(selector,work)=>controls.querySelector(selector).addEventListener("click",async event=>{event.target.disabled=true;try{await work();}catch(error){status.textContent=error.message;}finally{event.target.disabled=false;}});
  action(".save-candidate",async()=>{const saved=await postJson(`/api/optimizations/${id}/save`,{candidate:candidate().id,name:controls.querySelector("input").value});status.textContent=`Saved “${saved.name}”. It is available in the simulation configuration library.`;});
  action(".use-candidate",async()=>{const c=candidate();const keys=(await get(`/api/optimizer/context/${result.source_run}/${result.training_path}`)).strategyKeys;saveConfig({...savedConfig(),...Object.fromEntries(keys.map(key=>[key,c.config[key]])),sweepEnabled:false});location.hash="#/";});
  const replay=async(scope)=>{await postJson(`/api/optimizations/${id}/replay`,{candidate:candidate().id,scope});location.hash="#/active";};
  action(".replay-path",()=>replay("path"));action(".replay-run",()=>replay("run"));select.addEventListener("change",draw);
  app.append(controls,status,details,chartHost,table(["Rank","Strategy","Call policy","P&L","Above stock hold","Drawdown","Improvement"],result.candidates.map((c,i)=>[String(i+1),c.config.strategy,c.config.callPolicy,money(c.pnl),money(c.excess_hold),money(c.max_dd),money(c.improvement)]),(index)=>{select.value=result.candidates[index].id;draw();})); draw();
}
