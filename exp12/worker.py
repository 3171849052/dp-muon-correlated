import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src')]
import argparse
import subprocess
from exp12.common import ROOT, jobs

def worker(gpu):
    for method,seed in jobs(gpu):
        subprocess.run([sys.executable,str(ROOT/'exp12/full_training.py'),'--method',method,'--seed',str(seed)],check=True,cwd=ROOT)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--gpu',type=int,required=True); worker(p.parse_args().gpu)
