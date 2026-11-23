import json
from dataclasses import replace
from pathlib import Path

import context
import numpy as np
import pytest

from tqqq_book import desk, store
from tqqq_book.checkpoint import SimulationPaused
from tqqq_book.combined import run_combined
from tqqq_book.optimizer import optimize, replay
from tqqq_book.policy import CallPolicy


def test_dynamic_call_policy_uses_only_observed_history_and_obeys_bounds():
    policy = CallPolicy('adaptive', 1, .5, .5, .01, .2)
    assert policy.otm(.03, .71, .71, .1, .2) == pytest.approx(.18)
    assert policy.otm(.03, 1.6, .71, 1, 1) == .2
    assert CallPolicy().otm(.03, 1.6, .71, 1, 1) == .03
    prefix = np.linspace(59.22, 70, 16)
    a = np.concatenate([prefix, np.linspace(70, 90, 15)])
    b = np.concatenate([prefix, np.linspace(70, 40, 15)])
    cfg = desk.BookConfig(paths=1, horizon_days=30, call_policy='adaptive', call_lookback=5)
    book = desk.build_book(cfg)
    left = desk._simulate(a, book, cfg)
    right = desk._simulate(b, book, cfg)
    assert [d['trades'] for d in left['days'][:16]] == [d['trades'] for d in right['days'][:16]]
    lean = desk._simulate(a, book, cfg, record_history=False)
    assert lean['pnl'] == left['pnl']
    assert lean['max_dd'] == left['max_dd']
    stock_cfg = replace(cfg, strategy='stock')
    stock = desk._simulate(a, desk.build_book(stock_cfg), stock_cfg)
    assert stock['pnl'] == pytest.approx((a[-1]-a[0])*10000)


def test_single_run_resumes_completed_paths_without_simulating_them_again(tmp_path, monkeypatch):
    checkpoint = tmp_path/'checkpoint'
    monkeypatch.setenv('TQQQ_CHECKPOINT_DIR', str(checkpoint))
    monkeypatch.setattr(store, 'RUNS', tmp_path/'runs')
    original = desk._simulate
    calls = []
    def simulate(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(desk, '_simulate', simulate)
    def progress(event):
        if event['phase']=='Simulating paths' and event['completed']==1:
            (checkpoint/'pause.requested').write_text('pause')
    cfg = desk.BookConfig(paths=3, horizon_days=3)
    with pytest.raises(SimulationPaused): desk.run(cfg, progress)
    assert len(calls)==1
    (checkpoint/'pause.requested').unlink()
    resumed = desk.run(cfg)
    assert len(calls)==3
    monkeypatch.delenv('TQQQ_CHECKPOINT_DIR')
    uninterrupted = desk.run(cfg)
    assert resumed['monte_carlo']==uninterrupted['monte_carlo']
    assert resumed['ribbon']==uninterrupted['ribbon']


def test_combined_resume_does_not_duplicate_already_saved_paths(tmp_path, monkeypatch):
    checkpoint = tmp_path/'checkpoint'; checkpoint.mkdir()
    monkeypatch.setenv('TQQQ_CHECKPOINT_DIR',str(checkpoint))
    monkeypatch.setattr(store,'RUNS',tmp_path/'runs')
    configs=[{'paths':2,'horizon':2,'seed':seed} for seed in (1,2)]
    def progress(event):
        if event['phase']=='Configuration 1/2 completed': (checkpoint/'pause.requested').write_text('pause')
    with pytest.raises(SimulationPaused): run_combined(configs,progress)
    (checkpoint/'pause.requested').unlink()
    result=run_combined(configs)
    paths=[json.loads(line) for line in Path(result['saved']['paths']).read_text().splitlines()]
    assert [p['path'] for p in paths]==[0,1,2,3]
    assert result['config']['paths']==4
    assert len(store.list_runs())==1


def test_optimizer_scores_against_stock_hold_and_replays_exact_source_paths(tmp_path, monkeypatch):
    import tqqq_book.optimizer as module
    monkeypatch.setattr(store,'RUNS',tmp_path/'runs')
    monkeypatch.setattr(module,'RUNS',tmp_path/'runs')
    original=desk.run(desk.BookConfig(paths=3,horizon_days=10))
    source=original['saved']['id']
    base={'lots':'100','horizon':'10','paths':'3','strategy':'wheel','otm':'0.03,0.05,0.08,0.1'}
    candidates=[base,{**base,'strategy':'stock'},{**base,'callPolicy':'adaptive','callTrendSensitivity':'1'}]
    module.optimize({'sourceRun':source,'path':0,'candidates':candidates,'objective':'excess_hold'},'search',lambda _:None)
    result=json.loads((module.RUNS/'optimizations/search/result.json').read_text())
    assert all(c['score']==c['pnl']-result['baseline']['stock_hold']['pnl'] for c in result['candidates'])
    stock=next(c for c in result['candidates'] if c['config']['strategy']=='stock')
    assert stock['excess_hold']==pytest.approx(0,abs=1e-8)
    replayed=module.replay({'sourceRun':source,'config':stock['config'],'trainingPath':0,'optimizationId':'search','candidateId':stock['id']},'replay',lambda _:None)
    assert replayed['analysis']['all_paths']['paths']==3
    assert replayed['analysis']['training']['paths']==1
    assert replayed['analysis']['other_paths']['paths']==2
    old=[json.loads(line) for line in Path(original['saved']['paths']).read_text().splitlines()]
    new=[json.loads(line) for line in Path(replayed['saved']['paths']).read_text().splitlines()]
    assert [p['prices'] for p in new]==[p['prices'] for p in old]
    assert all(p['pnl']==pytest.approx(p['stock_hold_pnl'],abs=1e-8) for p in new)


def test_optimizer_cli_dispatches_stdin_payload(tmp_path, monkeypatch):
    import io
    import sys
    import tqqq_book.optimizer as module
    called=[]
    monkeypatch.setattr(sys,'argv',['optimizer','--kind','optimization','--id','task'])
    monkeypatch.setattr(sys,'stdin',io.StringIO('{"sourceRun":"source"}'))
    monkeypatch.setattr(module,'optimize',lambda payload,task,progress:called.append((payload,task)))
    module.main()
    assert called==[({'sourceRun':'source'},'task')]
