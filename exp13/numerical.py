"""Stage 1 only: fit, cross-evaluate, collect once, replay and plot."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src')]
import argparse
import subprocess
import time
from exp13.common import ROOT, output, configuration, contract, config_hash, write_json


def main(smoke=False):
    start=time.monotonic()
    timings={}
    for script in ('fit_strategies.py','collect_trajectory.py','replay.py'):
        before=time.monotonic()
        subprocess.run([sys.executable,'-B',str(ROOT/'exp13'/script)]+(['--smoke'] if smoke else []),check=True,cwd=ROOT)
        timings[script]=time.monotonic()-before
    c=configuration(smoke)
    write_json(output(smoke)/'provenance.json',dict(config=vars(c),config_hash=config_hash(c,contract(c,smoke)),
        runtime_seconds=time.monotonic()-start,timings=timings))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true')
    main(p.parse_args().smoke)
