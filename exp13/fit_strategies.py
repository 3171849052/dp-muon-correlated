import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'src')]
import argparse
from dataclasses import asdict
import jax.numpy as jnp
from jax_privacy.matrix_factorization import banded
from exp13.common import configuration, contract, output, table, write_json
from exp13.strategies import fit, save
from exp13.workloads import ema


def main(smoke=False):
    c, root = configuration(smoke), output(smoke)
    p = contract(c, smoke)
    dest = root/'strategies'; dest.mkdir(exist_ok=True)
    strategies = {}
    summaries = []
    for channel,beta in [('m',c.beta1),('v',c.beta2)]:
        for corrected in (False,True):
            name = f"C_{channel}_{'bc' if corrected else 'raw'}"
            strategy, metadata = fit(beta, c, p, corrected)
            save(dest/f'{name}.npz', strategy, metadata)
            strategies[name] = strategy
            summaries.append(dict(strategy=name, **metadata))
    rows = [dict(workload=f"H_{channel}_{'bc' if corrected else 'raw'}", strategy=name,
                 objective=float(jnp.mean(banded.per_query_error(strategy, A=ema(beta,corrected)))))
            for channel,beta in [('m',c.beta1),('v',c.beta2)] for corrected in (False,True)
            for name,strategy in strategies.items()]
    table(root/'cross_eval', rows)
    table(root/'strategy_summary', summaries)
    write_json(root/'contract.json', asdict(p))
    print(rows, flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true')
    main(parser.parse_args().smoke)
