const test = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const { JSDOM } = require('jsdom');
const { runInContext } = require('node:vm');
const { DEFAULTS, SCALARS, MAX_SIMULATIONS, planSimulations } = require('../dist/desk/simulation-plan');

async function waitFor(check) {
  const end = Date.now() + 2000;
  while (Date.now() < end) {
    if (check()) return;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  assert.ok(check(), 'Interface did not reach the expected state');
}

function setup(t, options = {}) {
  const html = readFileSync(join(__dirname, '../public/index.html'), 'utf8');
  const dom = new JSDOM(html, { url: `http://desk.test/${options.hash || ''}`, runScripts: 'outside-only' });
  t.after(() => dom.window.close());
  const server = { jobs: options.jobs || [], submitted: null };
  dom.window.fetch = async (url, init) => {
    let value;
    let status = 200;
    try {
      if (url === '/api/simulation/config') value = { defaults: DEFAULTS, scalars: SCALARS, maxSimulations: MAX_SIMULATIONS };
      else if (url === '/api/simulation/preview') value = planSimulations(JSON.parse(init.body));
      else if (url === '/api/rules') value = [{ name: 'harvest', says: 'Harvest short calls', drop: 0.08, keep: 0.5 }];
      else if (url === '/api/runs/combined') value = options.run;
      else if (url === '/api/runs/combined/paths') value = options.paths;
      else if (url === '/api/runs/combined/series') value = [];
      else if (url === '/api/runs') value = [];
      else if (url === '/api/jobs') value = server.jobs;
      else if (url === '/api/simulate') {
        server.submitted = JSON.parse(init.body);
        const plan = planSimulations(server.submitted);
        server.jobs = (plan.mode === "combined" ? [plan.configurations[0]] : plan.configurations).map((config, i) => ({ id: `job-${i}`, batch: 'batch', index: i + 1, batchSize: plan.total,
          config, mode: plan.mode, configurationCount: plan.mode === 'combined' ? plan.total : 1, configurations: plan.mode === 'combined' ? plan.configurations : undefined, variations: {}, status: i ? 'queued' : 'running', phase: i ? 'Queued' : 'Simulating paths', progress: i ? 0 : 47.5,
          completedPaths: i ? 0 : 1, totalPaths: plan.mode === 'combined' ? plan.totalPaths : Number(config.paths) }));
        value = { job: 'job-0', batch: 'batch', total: server.jobs.length, configurationCount: plan.total, totalPaths: plan.totalPaths, mode: plan.mode };
      } else throw new Error('Unexpected request: ' + url);
    } catch (error) { status = 400; value = { message: error.message }; }
    return { ok: status < 400, status, json: async () => value, text: async () => JSON.stringify(value) };
  };
  if (options.config) dom.window.localStorage.setItem('desk-config', JSON.stringify(options.config));
  for (const file of ['simulation-ui.js', 'app.js']) runInContext(readFileSync(join(__dirname, '../public', file), 'utf8'), dom.getInternalVMContext());
  const document = dom.window.document;
  const input = (element, value) => { element.value = value; element.dispatchEvent(new dom.window.Event('input', { bubbles: true })); };
  return { dom, document, server, input };
}

test('form previews a Cartesian sweep, launches tiles, and follows jobs across tabs and refreshes', async (t) => {
  const { dom, document, server, input } = setup(t);
  await waitFor(() => document.querySelector('button.run:not(:disabled)'));
  document.querySelector('details.advanced').open = true;
  const enabled = document.querySelector('#enable-ranges');
  enabled.checked = true;
  enabled.dispatchEvent(new dom.window.Event('change', { bubbles: true }));
  document.querySelector('.add-range').click();
  const first = document.querySelector('.range-row');
  input(first.querySelector('[data-range="from"]'), '0.2');
  input(first.querySelector('[data-range="to"]'), '0.5');
  input(first.querySelector('[data-range="step"]'), '0.1');
  await waitFor(() => document.querySelector('button.run').textContent === 'Run 4 simulations');
  document.querySelector('.add-range').click();
  const second = document.querySelectorAll('.range-row')[1];
  const select = second.querySelector('select');
  select.value = 'iv'; select.dispatchEvent(new dom.window.Event('change', { bubbles: true }));
  input(second.querySelector('[data-range="from"]'), '0.6');
  input(second.querySelector('[data-range="to"]'), '0.8');
  input(second.querySelector('[data-range="step"]'), '0.1');
  await waitFor(() => document.querySelector('button.run').textContent === 'Run 12 simulations');
  assert.match(document.querySelector('.launch-summary').textContent, /1,200 paths/);
  document.querySelector('form').dispatchEvent(new dom.window.Event('submit', { bubbles: true, cancelable: true }));
  await waitFor(() => document.querySelectorAll('.job').length === 12);
  assert.equal(server.submitted.ranges.length, 2);
  assert.match(document.querySelector('.job.running').textContent, /47.5%/);
  assert.equal(document.querySelector('.job.running progress').value, 47.5);
  dom.window.location.hash = '#/active';
  await waitFor(() => document.querySelector('h1')?.textContent === 'Active runs' && document.querySelectorAll('.job').length === 12);
  server.jobs[0] = { ...server.jobs[0], status: 'done', progress: 100, phase: 'Completed', run: 'own-result' };
  await document.querySelector('.batch').parentElement.refresh();
  assert.equal(document.querySelector('.job.done a').getAttribute('href'), '#/run/own-result');
  const saved = JSON.parse(dom.window.localStorage.getItem('desk-config'));
  assert.equal(saved.sweepEnabled, true);
  assert.equal(saved.ranges.length, 2);
  const refreshed = setup(t, { hash: '#/active', jobs: server.jobs, config: saved });
  await waitFor(() => refreshed.document.querySelectorAll('.job').length === 12);
  assert.match(refreshed.document.querySelector('.job.done').textContent, /Completed/);
});

test('saved advanced settings fill the form and invalid increments prevent launch', async (t) => {
  const config = { rate: '0.03', vol: '0.6', sweepEnabled: true, ranges: [{parameter:'contracts.0',from:10,to:12,step:1}] };
  const { dom, document, input } = setup(t, { hash: '#/config', config });
  await waitFor(() => document.querySelector('button.run:not(:disabled)'));
  assert.equal(document.querySelector('[name="rate"]').value, '0.03');
  assert.equal(document.querySelector('details.advanced').open, true);
  assert.match(document.querySelector('.launch-summary').textContent, /3 simulations/);
  input(document.querySelector('[data-range="step"]'), '0');
  await waitFor(() => document.querySelector('.launch-summary.invalid'));
  assert.equal(document.querySelector('button.run').disabled, true);
  input(document.querySelector('[data-range="step"]'), '1');
  await waitFor(() => !document.querySelector('button.run').disabled);
  document.querySelector('form').dispatchEvent(new dom.window.Event('submit', {bubbles:true,cancelable:true}));
  await waitFor(() => document.querySelector('.status')?.textContent.startsWith('Saved.'));
  assert.equal(JSON.parse(dom.window.localStorage.getItem('desk-config')).vol, '0.6');
});

test('histogram uses readable currency labels, the P&L axis, and gold gain bars', async (t) => {
  const { dom, document } = setup(t);
  await waitFor(() => document.querySelector('button.run:not(:disabled)'));
  const svg = dom.window.histogram([{pnl:-285429}, {pnl:195396}, {pnl:1000}]);
  assert.match(svg, /P&amp;L|P&L/);
  assert.match(svg, /−\$285k/);
  assert.match(svg, /#8a6a2f/);
  assert.doesNotMatch(svg, />day</);
  const constant = dom.window.histogram([{pnl:0}, {pnl:0}]);
  assert.doesNotMatch(constant, /NaN|Infinity/);
});


test('rule switches remain usable alongside advanced configuration', async (t) => {
  const { document, dom } = setup(t, { hash: '#/rules' });
  await waitFor(() => document.querySelector('.rule .switch'));
  document.querySelector('.rule .switch').click();
  assert.equal(document.querySelector('.rule .switch').textContent, 'OFF');
  assert.equal(dom.window.rulePayload().rules, 'none');
});


test('combined mode saves its choice, previews 600 paths, and renders one progress tile', async (t) => {
  const config = { sweepEnabled: true, sweepMode: 'combined', ranges: [{ parameter: 'otm.0', from: 0.03, to: 0.08, step: 0.01 }] };
  const { document, dom, server } = setup(t, { config });
  await waitFor(() => document.querySelector('button.run:not(:disabled)'));
  assert.equal(document.querySelector('[name="sweepMode"]').value, 'combined');
  assert.match(document.querySelector('.launch-summary').textContent, /1 combined simulation.*6 configurations.*600 paths/);
  assert.equal(document.querySelector('button.run').textContent, 'Run combined simulation');
  document.querySelector('form').dispatchEvent(new dom.window.Event('submit', { bubbles: true, cancelable: true }));
  await waitFor(() => document.querySelector('.job'));
  assert.equal(document.querySelectorAll('.job').length, 1);
  assert.match(document.querySelector('.job').textContent, /Combined simulation · 6 configurations/);
  assert.match(document.querySelector('.job').textContent, /1\/600 paths/);
  assert.equal(server.submitted.sweepMode, 'combined');
  assert.equal(JSON.parse(dom.window.localStorage.getItem('desk-config')).sweepMode, 'combined');
});


test('combined results show one P&L distribution and filter paths by configuration', async (t) => {
  const config = { lots: 100, paths: 1, horizon_days: 3, tenors: [{otm:0.03}, {otm:0.05}, {otm:0.08}, {otm:0.10}] };
  const book = { puts: [], calls: [], deployed: 1000, put_premium: 100, call_premium: 50, floor: 1000 };
  const mc = { paths: 2, median: 0, mean: 0, p05: -1, p95: 1, prob_loss: 0.5 };
  const run = { config: { ...config, mode: 'combined', configuration_count: 2, paths: 2, horizon_range: [3,3] },
    book, monte_carlo: mc, note: 'Pooled observations', ribbon: [],
    configurations: [1,2].map((id) => ({id, config, book, monte_carlo: mc, rules:[]})) };
  const paths = [0,1].map((path) => ({path, configuration:path+1, ending_spot:60, pnl:path?1:-1, min_spot:59, max_dd:2, ending_shares:10000, trades:0}));
  const { document, dom } = setup(t, { hash: '#/run/combined', run, paths });
  await waitFor(() => document.querySelector('[aria-label="Path configuration"]'));
  assert.match(document.querySelector('h1').textContent, /Combined simulation · 2 configurations/);
  assert.equal([...document.querySelectorAll('.chart-card h3')].filter((h) => h.textContent === 'P&L distribution').length, 1);
  const filter = document.querySelector('[aria-label="Path configuration"]');
  const pathTable = filter.nextElementSibling;
  assert.equal(pathTable.querySelectorAll('tbody tr').length, 2);
  filter.value = '2'; filter.dispatchEvent(new dom.window.Event('change', {bubbles:true}));
  assert.equal(pathTable.querySelectorAll('tbody tr').length, 1);
  assert.equal(pathTable.querySelectorAll('tbody td')[1].textContent, '2');
});
