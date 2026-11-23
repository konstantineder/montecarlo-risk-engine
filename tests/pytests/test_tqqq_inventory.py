import json
from dataclasses import asdict
from pathlib import Path

import context
import numpy as np
import pytest

from tqqq_book import desk
from tqqq_book.strategy import Order, Position, ReplaceAssigned


def position(shares, protected, short_puts=0, max_shares=10000):
    legs = [desk.Leg('put', 'long', 50, protected, 60, 1, 'hedge')]
    if short_puts:
        legs.append(desk.Leg('put', 'short', 45, short_puts, 10, 1, 'reenter'))
    return Position(1, 30, 50, 59.22, .71, shares, legs, [], False,
                    lambda _: 1, lambda *_: 1, max_shares=max_shares)


@pytest.mark.parametrize('shares,protected,short_puts,limit,expected', [
    (10000, 150, 0, 10000, 0),
    (5400, 150, 20, 10000, 26),
    (5000, 70, 30, 10000, 0),
    (9901, 150, 0, 10000, 0),
    (9801, 150, 0, 10000, 1),
    (200, 8, 1, 500, 2),
    (0, 100, 0, 10000, 100),
])
def test_reentry_obeys_inventory_and_hedge_limits(shares, protected, short_puts, limit, expected):
    orders = ReplaceAssigned().apply(position(shares, protected, short_puts, limit))
    assert sum(order.qty for order in orders) == expected


def assert_capped(result, limit):
    for day in result['days']:
        snapshots = [day['position']]
        snapshots += [step[side] for step in day['steps'] for side in ('before', 'after')]
        for rows in snapshots:
            shares = sum(row['qty'] for row in rows if row['name'] == 'share')
            puts = sum(row['qty'] for row in rows if row['name'] == 'put' and row['side'] == 'short')
            assert 0 <= shares + puts * 100 <= limit
        assert day['shares'] <= limit


@pytest.mark.parametrize('lots', [1, 17, 100])
def test_falling_market_hedges_cannot_expand_inventory(lots):
    cfg = desk.BookConfig(lots=lots, paths=1, horizon_days=30)
    result = desk._simulate(np.geomspace(cfg.spot, 15, 31), desk.build_book(cfg), cfg)
    assert_capped(result, lots * 100)
    assert result['ending_shares'] == lots * 100
    # Extra long puts remain allowed as hedges; they cannot create reentry room.
    assert any(sum(p['qty'] for p in d['position'] if p['name'] == 'put' and p['side'] == 'long') > lots
               for d in result['days'])
    assert not any(t['action'] == 'sell_put' for d in result['days'] for t in d['trades'])


def test_execution_caps_each_order_and_preserves_assignment_cash(monkeypatch):
    import tqqq_book.strategy as strategy

    class OversizedReentry:
        name = 'oversized_reentry'

        def apply(self, position):
            if position.day != 1:
                return []
            # Both orders see the same initial position. The desk must reserve
            # room as each is filled, and credit only contracts actually sold.
            return [Order(self.name, 'sell_put', 4, 50, 3, 'a'),
                    Order(self.name, 'sell_put', 999, 50, 4, 'b')]

    monkeypatch.setattr(strategy, 'rules_for', lambda _: [OversizedReentry()])
    cfg = desk.BookConfig(lots=5, paths=1, horizon_days=4, slippage=0)
    # Three opening calls deliver 300 shares; this frees exactly three puts.
    book = desk.build_book(cfg)
    book['legs'] = [desk.Leg('call', 'short', 60, 3, 1, 0, 'opening')]
    result = desk._simulate(np.array([59.22, 70, 49, 40, 35]), book, cfg)
    assert_capped(result, 500)
    sold = [t for d in result['days'] for t in d['trades'] if t['action'] == 'sell_put']
    assigned = [t for d in result['days'] for t in d['trades'] if t['action'] == 'assign_put']
    assert [t['qty'] for t in sold] == [3]
    assert [t['qty'] for t in assigned] == [3]
    assert result['ending_shares'] == 500
    premium = desk._fast_price(70, 50, 2 * desk.DT, cfg.rate,
                              desk._iv_at(70, cfg.spot, cfg.iv, cfg.iv_elasticity), 'put')
    expected_cash = book['cash'] + 300 * 60 + 300 * premium - 300 * 50
    assert result['pnl'] == pytest.approx(expected_cash + 500 * 35)


def test_execution_reserves_room_for_multiple_put_expiries(monkeypatch):
    import tqqq_book.strategy as strategy

    class Reentry:
        name = 'reentry'

        def apply(self, position):
            if position.day != 1:
                return []
            return [Order(self.name, 'sell_put', 1, 50, 3, 'a'),
                    Order(self.name, 'sell_put', 999, 50, 4, 'b')]

    monkeypatch.setattr(strategy, 'rules_for', lambda _: [Reentry()])
    cfg = desk.BookConfig(lots=5, paths=1, horizon_days=4)
    book = desk.build_book(cfg)
    book['legs'] = [desk.Leg('call', 'short', 60, 3, 1, 0, 'opening')]
    result = desk._simulate(np.array([59.22, 70, 49, 40, 35]), book, cfg)
    assert_capped(result, 500)
    assert [t['qty'] for d in result['days'] for t in d['trades'] if t['action'] == 'sell_put'] == [1, 2]
    assert result['ending_shares'] == 500


def test_simulation_rejects_an_opening_book_over_the_limit():
    cfg = desk.BookConfig(lots=1, paths=1, horizon_days=1)
    book = desk.build_book(cfg)
    book['legs'].append(desk.Leg('put', 'short', 50, 1, 1, 1, 'invalid'))
    with pytest.raises(ValueError, match='Inventory limit exceeded'):
        desk._simulate(np.array([cfg.spot, 45]), book, cfg)


def test_wheel_replaces_called_away_shares_without_growing_the_position():
    cfg = desk.BookConfig(paths=1, horizon_days=20, rules=('replace_assigned',),
                          tenors=(desk.Tenor(5, .03, 100, 'w1'),))
    prices = np.array([59.22] + [70] * 5 + [40] * 15)
    result = desk._simulate(prices, desk.build_book(cfg), cfg)
    assert_capped(result, 10000)
    trades = [t for day in result['days'] for t in day['trades']]
    assert sum(t['qty'] for t in trades if t['action'] == 'assign_call') == 100
    assert sum(t['qty'] for t in trades if t['action'] == 'sell_put') == 100
    assert sum(t['qty'] for t in trades if t['action'] == 'assign_put') == 100
    assert result['ending_shares'] == 10000


@pytest.mark.parametrize('style', ['wheel', 'hold', 'stock'])
def test_generated_paths_keep_inventory_capped(style):
    cfg = desk.BookConfig(lots=17, paths=3, horizon_days=75, strategy=style)
    for path in desk._paths(cfg, cfg.horizon_days):
        assert_capped(desk._simulate(path, desk.build_book(cfg), cfg), cfg.lots * 100)

@pytest.mark.parametrize('kind', ['', 'candidates', 'replay-paths'])
def test_legacy_checkpoints_are_retained_but_not_reused(tmp_path, kind):
    from tqqq_book.checkpoint import PathCheckpoint, atomic_json

    folder = tmp_path / kind
    checkpoint = PathCheckpoint(folder, {'lots': 100})
    legacy = folder / 'path-0.json'
    atomic_json(legacy, {'pnl': 999999, 'ending_shares': 171300})
    assert checkpoint.load(0) is None
    checkpoint.save(0, {'pnl': 100, 'ending_shares': 10000})
    assert PathCheckpoint(folder, {'lots': 100}).load(0)['ending_shares'] == 10000
    assert json.loads(legacy.read_text())['ending_shares'] == 171300


def test_resume_ignores_legacy_combined_spool_and_configuration_paths(tmp_path, monkeypatch):
    from tqqq_book import store
    from tqqq_book.checkpoint import PathCheckpoint, SimulationPaused, atomic_json, engine_folder
    from tqqq_book.combined import run_combined

    checkpoint = tmp_path / 'checkpoint'
    cfg = desk.BookConfig(paths=1, horizon_days=3)
    # Old files contain pre-fix values. A resume must recompute, then preserve
    # only corrected work on any subsequent pause/resume.
    PathCheckpoint(checkpoint / 'configuration-1', asdict(cfg))
    atomic_json(checkpoint / 'configuration-1' / 'path-0.json', {'ending_shares': 171300})
    legacy_spool = checkpoint / 'combined.paths.jsonl'
    legacy_spool.write_text('{"ending_shares":171300}\n')
    monkeypatch.setenv('TQQQ_CHECKPOINT_DIR', str(checkpoint))
    monkeypatch.setattr(store, 'RUNS', tmp_path / 'runs')
    calls = []
    simulate = desk._simulate

    def record(*args, **kwargs):
        calls.append(1)
        return simulate(*args, **kwargs)

    monkeypatch.setattr(desk, '_simulate', record)

    def pause(event):
        if event['phase'] == 'Configuration 1/2 completed':
            (checkpoint / 'pause.requested').touch()

    configs = [asdict(cfg), {**asdict(cfg), 'lots': 17}]
    # asdict uses structured sleeves/tenors; use the same input adapter as replay.
    from tqqq_book.optimizer import saved_input
    configs = [saved_input(config) for config in configs]
    with pytest.raises(SimulationPaused):
        run_combined(configs, pause)
    assert len(calls) == 1
    (checkpoint / 'pause.requested').unlink()
    result = run_combined(configs)
    assert len(calls) == 2
    rows = [json.loads(line) for line in Path(result['saved']['paths']).read_text().splitlines()]
    assert [row['path'] for row in rows] == [0, 1]
    for row in rows:
        assert_capped(row, row['config']['lots'] * 100)
    assert len((engine_folder(checkpoint) / 'combined.paths.jsonl').read_text().splitlines()) == 2
    assert json.loads(legacy_spool.read_text())['ending_shares'] == 171300
