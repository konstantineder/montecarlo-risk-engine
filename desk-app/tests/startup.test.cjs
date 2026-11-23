const test = require('node:test');
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const { once } = require('node:events');
const { mkdtempSync, mkdirSync, copyFileSync, writeFileSync, symlinkSync, rmSync, existsSync } = require('node:fs');
const { tmpdir } = require('node:os');
const { join } = require('node:path');

const serverCode = `
const http = require('node:http');
const net = require('node:net');
if (process.env.IGNORE_TERM === '1') process.on('SIGTERM', () => {});
const server = process.env.RAW_TCP === '1'
  ? net.createServer(socket => socket.end())
  : http.createServer((req, res) => {
      res.writeHead(200, {'content-type':'application/json'});
      res.end(JSON.stringify({pid:process.pid,runsDir:process.env.RUNS_DIR}));
    });
server.listen(Number(process.env.PORT || 0), '127.0.0.1', () => {
  console.log(JSON.stringify({port:server.address().port,pid:process.pid}));
});
`;
const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function fixture(t) {
  const root = mkdtempSync(join(tmpdir(), 'desk-startup-'));
  const children = [];
  mkdirSync(join(root, 'bin'));
  mkdirSync(join(root, 'dist'));
  mkdirSync(join(root, 'node_modules/@nestjs/core'), {recursive:true});
  mkdirSync(join(root, 'node_modules/.bin'));
  copyFileSync(join(__dirname, '../up.sh'), join(root, 'up.sh'));
  writeFileSync(join(root, 'dist/main.js'), serverCode);
  writeFileSync(join(root, 'node_modules/.bin/tsc'), '#!/bin/sh\nexit 0\n', {mode:0o755});
  writeFileSync(join(root, 'bin/npm'), '#!/bin/sh\nif [ "${BUILD_FAIL:-}" = 1 ]; then exit 9; fi\nprintf build >> build.log\n', {mode:0o755});
  symlinkSync(process.execPath, join(root, 'bin/node'));
  const env = {...process.env, PATH:join(root, 'bin')+':'+process.env.PATH};
  const launch = (command, args, extra = {}) => {
    const child = spawn(command, args, {cwd:root, env:{...env,...extra}, stdio:['ignore','pipe','pipe']});
    child.output = '';
    child.stdout.on('data', (chunk) => {child.output += chunk.toString();});
    child.stderr.on('data', (chunk) => {child.output += chunk.toString();});
    children.push(child);
    return child;
  };
  t.after(async () => {
    for (const child of children) {
      if (child.exitCode === null && child.signalCode === null) {
        child.kill('SIGKILL');
        await once(child, 'exit');
      }
    }
    rmSync(root, {recursive:true, force:true});
  });
  return {root, launch, up:(port, extra={})=>launch('bash',['up.sh'],{PORT:String(port),...extra})};
}

async function listener(f, extra = {}) {
  const child = f.launch(process.execPath, ['dist/main.js'], {PORT:'0', ...extra});
  const deadline = Date.now()+3000;
  while (Date.now()<deadline) {
    const line = child.output.split('\n').find((part)=>part.startsWith('{"port":'));
    if(line) return {...JSON.parse(line), child};
    if(child.exitCode!==null) break;
    await pause(10);
  }
  assert.fail('Test listener did not start: '+child.output);
}

async function health(port, pid, child) {
  const deadline = Date.now()+15000;
  while (Date.now()<deadline) {
    if(child && (child.exitCode!==null || child.signalCode!==null)) assert.fail('Startup exited: '+child.output);
    try {
      const response = await fetch(`http://127.0.0.1:${port}/health`, {signal:AbortSignal.timeout(300)});
      const body = await response.json();
      if(body.pid===pid) return body;
    } catch {}
    await pause(30);
  }
  assert.fail('Expected desk did not take over the port: '+(child?.output||''));
}

async function exited(child) {
  if(child.exitCode!==null || child.signalCode!==null) return [child.exitCode,child.signalCode];
  return once(child,'exit');
}

test('repeated startup replaces the port owner and leaves another port untouched', async (t) => {
  const f = fixture(t);
  const target = await listener(f);
  const other = await listener(f);
  const first = f.up(target.port, {RUNS_DIR:join(f.root, 'custom-runs')});
  const firstHealth = await health(target.port, first.pid, first);
  assert.equal(firstHealth.runsDir, join(f.root, 'custom-runs'));
  assert.equal((await exited(target.child))[1], 'SIGTERM');
  const second = f.up(target.port);
  await health(target.port, second.pid, second);
  assert.equal((await exited(first))[1], 'SIGTERM');
  await health(other.port, other.pid, other.child);
});

test('startup reclaims a listener that does not serve HTTP health', async (t) => {
  const f = fixture(t);
  const target = await listener(f, {RAW_TCP:'1'});
  const desk = f.up(target.port);
  await health(target.port, desk.pid, desk);
  assert.equal((await exited(target.child))[1], 'SIGTERM');
});

test('startup force-stops a listener that ignores TERM', async (t) => {
  const f = fixture(t);
  const target = await listener(f, {IGNORE_TERM:'1'});
  const desk = f.up(target.port);
  await health(target.port, desk.pid, desk);
  assert.equal((await exited(target.child))[1], 'SIGKILL');
  assert.match(desk.output, /force-stopping/);
});

test('a failed build leaves the existing port owner running', async (t) => {
  const f = fixture(t);
  const target = await listener(f);
  const attempt = f.up(target.port, {BUILD_FAIL:'1'});
  assert.equal((await exited(attempt))[0],9);
  await health(target.port,target.pid,target.child);
});

test('invalid port values fail before building', async (t) => {
  const f = fixture(t);
  for(const port of ['0','65536','3000junk']) {
    const attempt=f.up(port);
    assert.equal((await exited(attempt))[0],2);
    assert.match(attempt.output,/Invalid PORT/);
  }
  assert.equal(existsSync(join(f.root,'build.log')),false);
});
