"""Bounded strategy search and paired replay on saved market paths."""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tqqq_book.checkpoint import PathCheckpoint, atomic_json, check_pause, guarded, root
from tqqq_book.combined import _CombinedPaths
from tqqq_book.desk import BookConfig, CONTRACT, DT, _opening_trades, _simulate, _summary, build_book, config_from_mapping
from tqqq_book.store import RUNS, folder_for

STRATEGY_FIELDS = (
    'strategy', 'rules', 'tenors', 'put_sleeves', 'put_ratio_grid', 'harvest_drop', 'harvest_keep',
    'roll_fraction', 'crash_line', 'crash_down', 'roll_up_gap', 'roll_down_gap', 'roll_down_floor',
    'write_every', 'write_floor', 'stagger_min_spot_frac', 'reserve_shares', 'reenter_otm', 'reenter_tenor',
    'call_policy', 'call_vol_sensitivity', 'call_trend_sensitivity', 'call_drawdown_sensitivity',
    'call_otm_min', 'call_otm_max', 'call_lookback',
)


def saved_input(cfg):
    cfg = {**asdict(BookConfig()), **cfg}
    raw = {key: value for key, value in cfg.items() if not isinstance(value, (list, tuple, dict))}
    raw['put_sleeves'] = ','.join(f"{row['weight'] * 100:g}:{row['dte']}:{row['roll_at']}" for row in cfg['put_sleeves'])
    raw['otm'] = [t['otm'] for t in cfg['tenors']]
    raw['contracts'] = [t.get('contracts_at_100', round(t['contracts'] * 100 / cfg['lots'])) for t in cfg['tenors']]
    raw['rules'] = ','.join(cfg.get('rules', []))
    raw['put_ratio_grid'] = ','.join(f'{level*100:g}:{ratio:g}' for level, ratio in cfg.get('put_ratio_grid', ((0,1),(.15,1.1),(.3,1.2),(.5,1.5))))
    return raw


def source_path(run_id, index):
    with (folder_for(run_id) / 'paths.jsonl').open() as handle:
        for number, line in enumerate(handle):
            if number == index:
                path = json.loads(line)
                if path['path'] == index:
                    return path
    raise ValueError(f'No path {index}')


def prices_for(path):
    prices = np.array(path.get('prices') or [day['price'] for day in path['days']], dtype=float)
    if len(prices) < 2 or not np.all(np.isfinite(prices)) or np.min(prices) <= 0:
        raise ValueError('Saved path has invalid prices')
    return prices


def path_config(summary, path):
    return path.get('config') or next((g['config'] for g in summary.get('configurations', []) if g['id'] == path.get('configuration')), summary['config'])


def effective_config(source, candidate, prices):
    base = config_from_mapping(saved_input(source))
    chosen = config_from_mapping(candidate)
    return replace(base, **{key: getattr(chosen, key) for key in STRATEGY_FIELDS}, spot=float(prices[0]), horizon_days=len(prices)-1, paths=1)


def benchmark(prices, lots):
    shares = lots * CONTRACT
    return {'pnl': float(shares * (prices[-1]-prices[0])), 'max_dd': float(shares * np.max(np.maximum.accumulate(prices)-prices))}


def metadata(cfg, book):
    config = {**asdict(cfg), 'initial_dte':cfg.put_dte, 'path_days':cfg.horizon_days, 'put_days_left':max(0,cfg.put_dte-cfg.horizon_days),
        'tenors':[{**asdict(t),'contracts_at_100':t.contracts,'contracts':round(t.contracts*cfg.lots/100)} for t in cfg.tenors]}
    legs=[]
    for leg in book['legs']:
        legs.append({'tag':leg.tag,'side':leg.side,'qty':leg.qty,'strike':leg.strike,'dte':leg.expiry,'years':leg.expiry*DT,
            'premium':leg.entry,'otm':leg.otm,'days_left_at_horizon':max(0,leg.expiry-cfg.horizon_days),'priced_by':'BlackScholesModel analytic','kind':leg.kind})
    view={key:book[key] for key in ('deployed','put_premium','call_premium','floor')}
    view.update(model_gap_vs_fill=book['model_gap'],opening_trades=_opening_trades(book),puts=[l for l in legs if l['kind']=='put'],calls=[l for l in legs if l['kind']=='call'])
    from tqqq_book.strategy import catalogue
    return {'config':config,'book':view,'rules':catalogue(cfg),'vol':{'note':'Same saved market prices; candidate strategy rules are replayed.'}}


def optimize(payload, task_id, progress):
    summary=json.loads((folder_for(payload['sourceRun'])/'summary.json').read_text())
    path=source_path(payload['sourceRun'],int(payload['path']))
    prices=prices_for(path)
    source=path_config(summary,path)
    baseline_cfg=replace(config_from_mapping(saved_input(source)),spot=float(prices[0]),horizon_days=len(prices)-1,paths=1)
    baseline=_simulate(prices,build_book(baseline_cfg),baseline_cfg,record_history=False)
    hold=benchmark(prices,baseline_cfg.lots)
    candidates=payload['candidates']
    checkpoint=PathCheckpoint(root()/'candidates' if root() else None,payload)
    scored=[]
    objective=payload.get('objective','excess_hold')
    penalty=float(payload.get('drawdownPenalty',1))
    for index,candidate in enumerate(candidates):
        row=checkpoint.load(index)
        if row is None:
            cfg=effective_config(source,candidate,prices)
            stats=_simulate(prices,build_book(cfg),cfg,record_history=False)
            excess=stats['pnl']-hold['pnl']
            score=stats['pnl'] if objective=='pnl' else excess-penalty*stats['max_dd'] if objective=='risk_adjusted' else excess
            row={'id':f'candidate-{index+1}','baseline':index==0,'config':candidate,'score':score,'pnl':stats['pnl'],'excess_hold':excess,'max_dd':stats['max_dd'],
                 'improvement':stats['pnl']-baseline['pnl'],'assignments':stats['assignments'],'new_calls':stats['new_calls']}
            checkpoint.save(index,row)
        scored.append(row)
        progress({'phase':'Searching strategy candidates','percent':90*(index+1)/len(candidates),'completed':index+1,'total':len(candidates)})
    scored.sort(key=lambda row:row['score'],reverse=True)
    cfg=effective_config(source,scored[0]['config'],prices)
    winner=_simulate(prices,build_book(cfg),cfg,record_history=False)
    result={'id':task_id,'source_run':payload['sourceRun'],'training_path':payload['path'],'objective':objective,'drawdown_penalty':penalty,
        'baseline':{'stored_pnl':path['pnl'],'replayed_pnl':baseline['pnl'],'max_dd':baseline['max_dd'],'stock_hold':hold},
        'candidates':scored,'winning_candidate':scored[0]['id'],
        'price_precision':'Full saved prices' if path.get('prices') else 'Legacy saved daily prices rounded to four decimals',
        'plot':[{'day':i,'original':baseline['days'][i]['pnl'],'optimized':winner['days'][i]['pnl'],'stock_hold':float((price-prices[0])*cfg.lots*CONTRACT)} for i,price in enumerate(prices)],
        'note':'Candidates were fitted to this selected path. Replay on the other saved paths to evaluate whether the improvement extends beyond the fitted path.'}
    atomic_json(RUNS/'optimizations'/task_id/'result.json',result)
    progress({'phase':'Optimization completed','percent':100,'completed':len(candidates),'total':len(candidates),'optimization':task_id})


def replay(payload, task_id, progress):
    summary=json.loads((folder_for(payload['sourceRun'])/'summary.json').read_text())
    selected=payload.get('path')
    total=1 if selected is not None else summary['config']['paths']
    horizon=summary['config']['horizon_days']
    checkpoint=PathCheckpoint(root()/'replay-paths' if root() else None,payload)
    groups=[]; group_keys={}
    with TemporaryDirectory(prefix='tqqq-replay-') as temp:
        writer=_CombinedPaths(Path(temp),total,horizon)
        try:
            with (folder_for(payload['sourceRun'])/'paths.jsonl').open() as handle:
                index=0
                for line_number,line in enumerate(handle):
                    if selected is not None and line_number!=selected: continue
                    check_pause()
                    original=json.loads(line); prices=prices_for(original)
                    source=path_config(summary,original)
                    cfg=effective_config(source,payload['config'],prices)
                    key=json.dumps(asdict(cfg),sort_keys=True)
                    if key not in group_keys:
                        group_keys[key]=len(groups)+1
                        book=build_book(cfg)
                        group={'id':len(groups)+1,**metadata(cfg,book),'source_run':payload['sourceRun']}
                        group['config']['paths']=0; groups.append(group)
                    group=groups[group_keys[key]-1]
                    local=group['config']['paths']; group['config']['paths']+=1
                    row=checkpoint.load(index)
                    if row is None:
                        stats=_simulate(prices,build_book(cfg),cfg)
                        stats.pop('nav')
                        row={'path':original['path'],**stats,'config':metadata(cfg,build_book(cfg))['config'],'prices':prices.tolist(),'original_pnl':original['pnl'],
                            'stock_hold_pnl':benchmark(prices,cfg.lots)['pnl'],'improvement':stats['pnl']-original['pnl'],
                            'is_training_path':original['path']==payload.get('trainingPath')}
                        checkpoint.save(index,row)
                    acted={rule['name'] for day in row['days'] for rule in day.get('rules',[]) if rule.get('acted')}
                    for rule in group['rules']: rule['paths']=rule.get('paths',0)+int(rule['name'] in acted)
                    writer.add(row,group,local)
                    index+=1
                    progress({'phase':'Replaying the same saved market paths','percent':90*index/total,'completed':index,'total':total})
                    if selected is not None: break
            if index!=total: raise ValueError('Source simulation path count differs from its metadata')
            for group in groups:
                rows=[r for r in writer.rows if r['configuration']==group['id']]
                group['monte_carlo']=_summary(group['config']['strategy'],rows)
            def comparison(rows):
                return {'paths':len(rows),'mean_improvement':float(np.mean([r['improvement'] for r in rows])) if rows else None,
                    'median_improvement':float(np.median([r['improvement'] for r in rows])) if rows else None,
                    'prob_improved':float(np.mean([r['improvement']>0 for r in rows])) if rows else None,
                    'mean_excess_hold':float(np.mean([r['pnl']-r['stock_hold_pnl'] for r in rows])) if rows else None}
            analysis={'type':'strategy_replay','source_run':payload['sourceRun'],'optimization_id':payload.get('optimizationId'),
                'candidate_id':payload.get('candidateId'),'training_path':payload.get('trainingPath'),
                'all_paths':comparison(writer.rows),'training':comparison([r for r in writer.rows if r['is_training_path']]),
                'other_paths':comparison([r for r in writer.rows if not r['is_training_path']]),
                'note':'Identical saved prices and original market assumptions. Older files use four-decimal daily prices. The fitted path is reported separately from other saved paths.'}
            result=writer.save(groups,progress,analysis)
            return result
        finally: writer.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('optimization','replay'),required=True)
    parser.add_argument('--id',required=True)
    args=parser.parse_args(); payload=json.load(sys.stdin)
    def progress(event): print('TQQQ_PROGRESS '+json.dumps(event),flush=True)
    if args.kind=='optimization': optimize(payload,args.id,progress)
    else: replay(payload,args.id,progress)


if __name__=='__main__': guarded(main)
