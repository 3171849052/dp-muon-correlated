import os
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src')]
import argparse
import subprocess
from exp13.common import ROOT, jobs

def worker(gpu,smoke=False):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    for method,seed in jobs(gpu):
        subprocess.run([sys.executable,str(ROOT/'exp13/full_training.py'),'--method',method,'--seed',str(seed)]+(['--smoke'] if smoke else []),check=True,cwd=ROOT)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--gpu',type=int,required=True); p.add_argument('--smoke',action='store_true')
    a=p.parse_args(); worker(a.gpu,a.smoke)
