import yaml, sys
from pathlib import Path
# sweep_definition.py  (run on your laptop or login node)
import itertools, yaml, json, hashlib, textwrap
from pathlib import Path
import subprocess   # for submitting automatically
import math

#open baseline config for the transformation we want to perform
import copy    # built‑in


# ------PATHS--------------------------------------------------
import os
#print("project root ",Path(__file__).resolve().parent.parent.parent.parent)
PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent# Path(__file__).resolve().parent[1]          # …/main_folder

#ACTION: CHOOSE WHICH TRANSFORMATION TO OPTIMIZE FOR
#choose which config_file to use - comment out the other config_file lines
config_file="rep2_em.yaml"      #ONE-WAY REPEATER WITH TLEs
config_file="rep11_chi3.yaml"      #Step 1 of ONE-WAY REPEATER WITH CHI3-NONLINEARITIES
config_file="rep12_chi3.yaml"      #Step 2 of ONE-WAY REPEATER WITH CHI3-NONLINEARITIES
config_file="CZ_chi3.yaml"      #CZ GATE WITH CHI3-NONLINEARITIES
config_file="CZ_chi3_5m_qutrit.yaml"      #qubit-QUTRIT CZ GATE WITH CHI3-NONLINEARITIES
config_file="CZ_em.yaml"      #CZ GATE WITH TLEs
config_file="Tofolli_chi3.yaml" #THREE-QUBIT TOFFOLI GATE    
cfg_path   = (PROJECT_ROOT / "execute_run" / "config" / config_file)
### from here there is nothing more to edit ###

cfg_dict      = yaml.safe_load(Path(cfg_path).read_text())
print(cfg_dict)
dict_sys = cfg_dict["sysSpec"]
dict_para = cfg_dict["para"]
dict_amp = cfg_dict["amp"]
dict_opti = cfg_dict["opti"]
wandb_specs = cfg_dict["wandbSpec"]
print("RQPNN config loaded in run_RQPNN.py")

# ─── ②  Use the config ──────────────────────────────────────────────────────
from pathlib import Path
import sys
print("generic modules imported in run_RQPNN.py")
# ".." from the notebook’s directory → absolute path to the package root
SRC_ROOT = Path.cwd().resolve().parent.parent
print("SRC_ROOT:", SRC_ROOT)
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
print("SRC-root path added to sys.path in run_RQPNN.py")
import workflow.RQPNN_workflow as RQPNN_workflow
print("RQPNN workflow loaded in run_RQPNN.py")
if __name__ == "__main__":
    #instantiate workflow
    workflow = RQPNN_workflow.RQPNNWorkflow(
                            dict_sys=dict_sys,      #define the quantum system
                            dict_para_spec=dict_para,    #define the parameterization of the quantum system
                            dict_amp_spec=dict_amp,      #define amplitudes og control signals
                            dict_opti=dict_opti,    #define the optimization that we will perform and parameters to use
                            wandb_specs=wandb_specs          
                            )
    print("Workflow loaded. Now we will train")
    workflow.train()    #train RQPNN

