import os
os.environ['JAX_ENABLE_X64']='true'
os.environ['JAX_PLATFORMS']='cpu'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parents[2]/'src')]
import pytest
from exp12.common import configuration
from dp_muon.training.cifar10_experiment import derive_fixed_cycle_participation
@pytest.fixture
def c(): return configuration(True)
@pytest.fixture
def p(): return derive_fixed_cycle_participation(64,1,16)
