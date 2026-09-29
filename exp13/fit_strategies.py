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
    for name, beta in [('C_m', c.beta1), ('C_v', c.beta2)]:
        strategy, metadata = fit(beta, c, p)
        save(dest/f'{name}.npz', strategy, metadata)
        strategies[name] = strategy
    rows = [dict(workload=moment, strategy=name,
                 objective=float(jnp.mean(banded.per_query_error(strategy, A=ema(beta)))))
            for moment,beta in [('m',c.beta1),('v',c.beta2)] for name,strategy in strategies.items()]
    gain = 1 - rows[3]['objective']/rows[2]['objective']
    table(root/'cross_eval', rows)
    write_json(root/'specialization_gain.json', dict(second_moment_specialization_gain=gain))
    write_json(root/'contract.json', asdict(p))
    print(rows, '\nsecond-moment specialization gain:', gain, flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true')
    main(parser.parse_args().smoke)
