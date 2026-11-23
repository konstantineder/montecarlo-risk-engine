const test = require('node:test');
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const { PassThrough } = require('node:stream');
const { mkdtempSync, mkdirSync, writeFileSync, rmSync } = require('node:fs');
const { tmpdir } = require('node:os');
const { join } = require('node:path');
const { planSimulations, commandArguments } = require('../dist/desk/simulation-plan');
const { SimulationQueue } = require('../dist/desk/simulation-queue');
const turn = () => new Promise((resolve) => setImmediate(resolve));
const range = (parameter, from, to, step) => ({ parameter, from, to, step });

test('decimal range includes all four values and changes only the selected tenor', () => {
  const plan = planSimulations({ ranges: [range('otm.0', 0.2, 0.5, 0.1)] });
  assert.equal(plan.total, 4);
  assert.equal(plan.totalPaths, 400);
  assert.deepEqual(plan.configurations.map((c) => c.otm), ['0.2,0.05,0.08,0.10', '0.3,0.05,0.08,0.10', '0.4,0.05,0.08,0.10', '0.5,0.05,0.08,0.10']);
});

test('ranges multiply across call contracts, model settings and rule settings', () => {
  const plan = planSimulations({ ranges: [range('contracts.3', 8, 10, 1), range('iv', 0.6, 0.7, 0.1), range('writeEvery', 4, 6, 2)] });
  assert.equal(plan.total, 12);
  assert.equal(new Set(plan.configurations.map((c) => `${c.contracts}/${c.iv}/${c.writeEvery}`)).size, 12);
  assert.ok(plan.configurations.every((c) => commandArguments(c).includes('--progress')));
});

test('supports put sleeve and put ratio components and unaligned endpoints', () => {
  const plan = planSimulations({ ranges: [range('puts.0.1', 50, 65, 10), range('putRatioGrid.1.1', 1.1, 1.3, 0.1)] });
  assert.equal(plan.total, 6);
  assert.deepEqual(plan.ranges[0].values, [50, 60]);
  assert.equal(plan.configurations[5].puts, '10:60:30,40:90:45,50:180:90');
  assert.equal(plan.configurations[5].putRatioGrid, '0:1,15:1.3,30:1.2,50:1.5');
});

test('canonicalizes valid whole numbers for Python integer CLI arguments', () => {
  const plan = planSimulations({ lots: '1e2', paths: '2.0', contracts: '12.0,10,8,8', puts: '100:60.0:30' });
  assert.equal(plan.configurations[0].lots, '100');
  assert.equal(plan.configurations[0].paths, '2');
  assert.equal(plan.configurations[0].contracts, '12,10,8,8');
  assert.equal(plan.configurations[0].puts, '100:60:30');
});

test('rejects bad ranges and invalid combinations before queuing any work', () => {
  for (const ranges of [
    [range('iv', 0.5, 0.8, 0)], [range('iv', 0.8, 0.5, 0.1)],
    [range('paths', 1, 3, 0.5)], [range('missing', 0, 1, 1)],
    [range('iv', 0.5, 0.8, 0.1), range('iv', 0.5, 0.8, 0.1)],
    [range('seed', 0, 1000, 1)], [range('puts.0.1', 20, 60, 20)],
    [range('puts.0.0', 10, 50, 10)], [range('iv', '', 1, 0.1)],
  ]) assert.throws(() => planSimulations({ ranges }));
  assert.throws(() => planSimulations({ contracts: '1,2,3' }));
  assert.throws(() => planSimulations({ paths: 0 }));
  assert.throws(() => planSimulations({ iv: 'Infinity' }));
});

test('queue streams split progress events, isolates results, and runs one job at a time', async (t) => {
  const root = mkdtempSync(join(tmpdir(), 'tqqq-queue-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const children = [];
  const calls = [];
  const launch = (...args) => {
    calls.push(args);
    const child = new EventEmitter();
    child.stdout = new PassThrough(); child.stderr = new PassThrough();
    children.push(child);
    return child;
  };
  const queue = new SimulationQueue('/repo', root, launch);
  const started = queue.enqueue({ paths: 2, ranges: [range('seed', 13, 14, 1)] });
  assert.equal(started.total, 2);
  await turn();
  assert.equal(children.length, 1);
  assert.equal(calls[0][2].env.RUNS_DIR, root);
  assert.equal(queue.get(started.job).status, 'running');
  children[0].stdout.write('TQQQ_PROG');
  children[0].stdout.write('RESS {"percent":47.5,"phase":"Simulating paths","completed":1}\n');
  assert.equal(queue.get(started.job).progress, 47.5);
  assert.equal(queue.get(started.job).completedPaths, 1);
  mkdirSync(join(root, 'own-result'));
  writeFileSync(join(root, 'own-result', 'summary.json'), '{}');
  mkdirSync(join(root, 'unrelated-newest'));
  writeFileSync(join(root, 'unrelated-newest', 'summary.json'), '{}');
  children[0].stdout.write('TQQQ_PROGRESS {"percent":100,"run":"own-result","completed":2}\n');
  children[0].emit('close', 0);
  await turn();
  assert.equal(queue.get(started.job).run, 'own-result');
  assert.equal(queue.get(started.job).progress, 100);
  assert.equal(queue.get(started.job).status, 'done');
  assert.equal(children.length, 2);
  children[1].stderr.write('Invalid configuration');
  children[1].emit('close', 1);
  assert.equal(queue.list()[0].status, 'error');
  assert.match(queue.list()[0].error, /Invalid configuration/);
});

test('a spawn failure does not stall the remaining queue', async () => {
  let attempts = 0;
  const queue = new SimulationQueue('/repo', '/runs', () => { attempts++; throw new Error('Python missing'); });
  queue.enqueue({ ranges: [range('seed', 1, 2, 1)] });
  await turn();
  assert.equal(attempts, 2);
  assert.ok(queue.list().every((job) => job.status === 'error'));
});
