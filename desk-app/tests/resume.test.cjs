const test=require('node:test');const assert=require('node:assert/strict');
const {EventEmitter}=require('node:events');const {PassThrough}=require('node:stream');
const {mkdtempSync,rmSync,existsSync,mkdirSync,writeFileSync,readFileSync}=require('node:fs');
const {tmpdir}=require('node:os');const {join}=require('node:path');
const {SimulationQueue}=require('../dist/desk/simulation-queue');
const turn=()=>new Promise(r=>setImmediate(r));
function launcher(children,calls){return (...args)=>{calls.push(args);const child=new EventEmitter();child.stdout=new PassThrough();child.stderr=new PassThrough();child.stdin=new PassThrough();child.kill=()=>{};children.push(child);return child;};}

test('paused and failed jobs persist and resume with the same ID and checkpoint directory',async t=>{
 const root=mkdtempSync(join(tmpdir(),'desk-resume-'));t.after(()=>rmSync(root,{recursive:true,force:true}));
 const children=[],calls=[];const launch=launcher(children,calls);
 let queue=new SimulationQueue('/repo',root,launch,true);
 const start=queue.enqueue({paths:5,horizon:2});await turn();
 children[0].stdout.write('TQQQ_PROGRESS {"percent":40,"completed":2}\n');
 queue.pause(start.job);assert.ok(existsSync(join(root,'.jobs',start.job,'pause.requested')));
 children[0].emit('close',75);assert.equal(queue.get(start.job).status,'paused');
 queue=new SimulationQueue('/repo',root,launch,true);
 assert.equal(queue.get(start.job).completedPaths,2);
 queue.resume(start.job);await turn();
 assert.equal(queue.list().length,1);assert.equal(calls[1][2].env.TQQQ_CHECKPOINT_DIR,calls[0][2].env.TQQQ_CHECKPOINT_DIR);
 children[1].stdout.write('TQQQ_PROGRESS {"percent":10,"completed":1}\n');assert.equal(queue.get(start.job).completedPaths,2);
 children[1].stderr.write('temporary failure');children[1].emit('close',1);assert.equal(queue.get(start.job).status,'error');
 queue.resume(start.job);await turn();
 mkdirSync(join(root,'finished'));writeFileSync(join(root,'finished','summary.json'),'{}');
 children[2].stdout.write('TQQQ_PROGRESS {"percent":100,"completed":5,"run":"finished"}\n');children[2].emit('close',0);
 assert.equal(queue.get(start.job).status,'done');
 assert.equal(JSON.parse(readFileSync(join(root,'.jobs','jobs.json')))[0].status,'done');assert.equal(existsSync(join(root,'.jobs',start.job)),false);
});

test('server restart exposes interrupted running jobs as resumable, without starting them',async t=>{
 const root=mkdtempSync(join(tmpdir(),'desk-recover-'));t.after(()=>rmSync(root,{recursive:true,force:true}));
 const children=[],calls=[];const launch=launcher(children,calls);
 const before=new SimulationQueue('/repo',root,launch,true);const started=before.enqueue({paths:4});await turn();
 const after=new SimulationQueue('/repo',root,launch,true);await turn();assert.equal(after.get(started.job).status,'paused');assert.equal(children.length,1);
 after.resume(started.job);await turn();assert.equal(children.length,2);
});

test('graceful shutdown preserves a running job as paused without a spurious failure',async t=>{
 const root=mkdtempSync(join(tmpdir(),'desk-shutdown-'));t.after(()=>rmSync(root,{recursive:true,force:true}));
 const children=[],calls=[];const queue=new SimulationQueue('/repo',root,launcher(children,calls),true);
 const job=queue.enqueue({paths:3});await turn();queue.shutdown();children[0].emit('close',null);
 assert.equal(queue.get(job.job).status,'paused');assert.equal(queue.get(job.job).error,undefined);
 const restored=new SimulationQueue('/repo',root,launcher(children,calls),true);assert.equal(restored.get(job.job).status,'paused');
});
