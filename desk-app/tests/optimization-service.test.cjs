const test=require('node:test');const assert=require('node:assert/strict');
const {mkdtempSync,mkdirSync,writeFileSync,rmSync}=require('node:fs');const {tmpdir}=require('node:os');const {join}=require('node:path');
const {OptimizationService}=require('../dist/desk/optimization-service');
function setup(t){const root=mkdtempSync(join(tmpdir(),'desk-opt-service-'));t.after(()=>rmSync(root,{recursive:true,force:true}));
 const snapshot={lots:100,spot:59.22,iv:.71,paths:2,horizon_days:2,seed:13,put_dte:180,put_sleeves:[{weight:1,dte:60,roll_at:30}],rules:[],tenors:[.03,.05,.08,.10].map((otm,i)=>({days:5*(i+1),otm,contracts:10,contracts_at_100:10,tag:'w'+(i+1)}))};
 mkdirSync(join(root,'source'));writeFileSync(join(root,'source','summary.json'),JSON.stringify({config:snapshot}));
 writeFileSync(join(root,'source','paths.jsonl'),[0,1].map(path=>JSON.stringify({path,pnl:-100,days:[{price:59.22},{price:60},{price:61}]})).join('\n'));
 const calls=[];const queue={enqueueTask:(...args)=>{calls.push(args);return{job:'job'};}};return{root,calls,service:new OptimizationService(root,queue)};}

test('optimizer locks market inputs and position size while searching strategy bounds',async t=>{
 const {service,calls}=setup(t);const context=await service.context('source',0);
 assert.equal(context.stockHold,(61-59.22)*10000);
 assert.ok(context.parameters.some(p=>p.key==='otm.0'));assert.ok(context.parameters.some(p=>p.key==='callTrendSensitivity'));
 assert.ok(!context.parameters.some(p=>p.key==='iv'||p.key==='lots'));
 await assert.rejects(service.preview('source',0,{ranges:[{parameter:'iv',from:.5,to:1,step:.1}]}));
 const preview=await service.preview('source',0,{ranges:[{parameter:'otm.0',from:.03,to:.04,step:.01}]});assert.ok(preview.candidates>2);
 await service.start('source',0,{ranges:[],strategies:['wheel','stock'],policies:['fixed'],objective:'excess_hold'});
 assert.equal(calls[0][0],'optimization');assert.equal(calls[0][1].sourceRun,'source');assert.equal(calls[0][1].objective,'excess_hold');
 assert.ok(calls[0][1].candidates.every(c=>c.iv==='0.71'&&c.lots==='100'));
});

test('learned configurations persist and whole-run replay retains its source paths',async t=>{
 const {service,root,calls}=setup(t);const {base}=await service.context('source',0);
 mkdirSync(join(root,'optimizations','search'),{recursive:true});writeFileSync(join(root,'optimizations','search','result.json'),JSON.stringify({source_run:'source',training_path:0,candidates:[{id:'candidate-1',config:{...base,callPolicy:'adaptive'}}]}));
 const saved=service.save('search',{candidate:'candidate-1',name:'Adaptive winner'});
 assert.equal(saved.config.callPolicy,'adaptive');assert.equal(service.configurations()[0].name,'Adaptive winner');
 service.replay('search',{candidate:'candidate-1',scope:'run'});
 assert.equal(calls[0][0],'replay');assert.equal(calls[0][1].sourceRun,'source');assert.equal(calls[0][1].path,undefined);assert.equal(calls[0][1].trainingPath,0);assert.equal(calls[0][3],2);
});
