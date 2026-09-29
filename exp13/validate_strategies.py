import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src')]
import argparse
from exp13.common import configuration, contract, output
from exp13.strategies import validate
if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true')
    smoke=parser.parse_args().smoke
    c=configuration(smoke)
    validate(output(smoke),c,contract(c,smoke))
    print('Stage 1 artifacts and configuration verified.',flush=True)
