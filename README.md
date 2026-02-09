# Modeling and optimization of Recirculating Quantum Photonic Networks
This repository provides all the code necessary to reproduce the results in the paper "Recirculating Quantum Photonic Networks for Fast Deterministic Quantum Information Processing".

In src\results you also find the data files for the all numerical results and a notebook (inspect_results.ipynb) for plotting them.

In the following, we walk through how to get started reproducing the results of the paper.

## Paper
This code accompanies: *Recirculating Quantum Photonic Networks for Fast Deterministic Quantum Information Processing* (2026).  
arXiv: https://arxiv.org/abs/XXXX.XXXXX

**If you find the code or paper useful in your research, please consider citing the paper**

## Getting started
### Install virtual environment (locally)
Virtual environemt

1: create

- python -m venv venv_RQPN

2: activate (in Windows PowerShell) - only step we need after installment

- .\venv_RQPN\Scripts\Activate.ps1

3: Install requirements

- pip install -r requirements.txt

The code was maintained until 7th of February 2026. The requirements_freeze.txt file records the pinned versions of all Python dependencies used in the virtual environment on 7th of February 2026


### Run optimization
You perform an optimization using src/run/run_RQPNN_local.py. You can run the script in powershell using: 
- python .\src\run\run_RQPNN_local.py    

To choose a given run, consider lines 18-29 in the script. Here you can choose between the configuration files for all 7 results, included in the paper.

To reproduce a given result, you have to access the corresponding configuration file under execute_run\config. Modify the configuration file, so that the parameters match the specifications given in Supplemental Material of the paper. In each configuration file, the exact seed to generate the result is also written as a comment.

In the configuration file CZ_chi3.yaml we have commented to explain all the different variables of the configuration file.

## Structure of the code
Here we provide an overview of the structure of the code.
RQPNs

- 'execute_run:
    - config: contains configuration files to initialize optimization of: 
    dual-rail encoded gates: qubit-qutrit CZ gate processing 5 (CZ_chi3_5m_qutrit) or 4 modes (CZ_chi3_qutrit) using chi3-nonlinearities, qubit-qubit CZ gate using chi3-nonlinearities (CZ_chi3), qubit-qubit CZ gate with two-level emitters (TLEs) (CZ_em), three-qubit Toffoli gate
    one-way repeater implementations: correction of loss in mode 1 (rep1_chi3) or mode 2 (rep12_chi3) or either mode (rep2_chi3) using chi3-nonlinearities, correction of loss in either mode (rep2_em)
- 'venv_RQPNN: virtual environment to use (ignored by git)
- src: all code for modeling and optimizing RQPNs
    - rqpnn:
        - rqpnn_spec: dataclass for specifying the hardware of a rqpnn instance (tested in test_rqpnn_spec.py)
        - rqpnn_basis: generate the basis for the Hilbert space and a matrix for switching between various convinient basises (tested in test_rqpnn_basis.py through analytical tests for the one-way repeaters and CZ gates with a chi3-rqpn and a TLE-rqpn and the three-qubit Toffoli gate for a chi3-rqpn)
        - rqpnn_raising_ops: generate all raising operators for an architecture
        - rqpnn_ham_ops.py: generate the operator for each term of the RQPN hamiltonian (tested in test_ham_ops)
        - rqpnn_parameterization: generate physical parameters of the Hamiltonian from trainable and fixed parameters of our parameterization (tested in test_parameterization.py through an example and in test_rates_parameterization.py and test_matrix_parameterization.py through examples)
        - rqpnn_hamiltonian.py: build RQPN hamiltonian from physical parameters and operators (tested in test_ham.py)
        - rqpnn_exc_idx_basis.py: NOT RELEVANT ANYMORE
        - rqpnn_stategeneration.py: generate product states (tested in test_stategen through analytical tests for the one-way repeaters and CZ gates with a chi3-rqpn and a TLE-rqpn and the three-qubit Toffoli gate for a chi3-rqpn) and transform modes NOT RELEVANT ANYMORE (tested in test_modetrans_unitary.py)
    --- qip_task:
        --- qiptask_generate_params.py: generate amplitude and trainable parameters from configuration dictionaries (tested in test_qip_para_amp_dicts.py) OPTION: UPDATE TEST FOR CUSTOM 
        --- qiptask_init_target_states.py: generate initial and target states for ta variety of quantum information processing tasks (tested in test_init_target.py through examples)
        --- qiptask_evolve.py: evolve a photonic input to output using the RQPN in its recirculating configuration (tested in test_evolve.py by comparing to various analytical examples that test the whole RQPN hamiltonian for both possible nonlinearities: chi3 or TLE)
        --- qip_task_build_masked_optimizer.py: generate mask for which parameters to optimize, code written with AI and tested by running optimizations OPTION: CLEAN UP CODE SLIGHTLY
        --- qiptask_costfunc.py: generate costfunction (tested in test_qip_loss.py through various examples)
        --- qiptask_stepper.py: update parameters, tested by running optimizations and varifying that we minimize the cost function and obtain the desired qip-transformation at the end of the optimization
        --- qiptask_optimizer.py: optimize parameters with gradient-descent-type updates over several epochs,  tested  by running optimizations and varifying that we obtain the desired qip-transformation at the end of the optimization OPTION: CLEAN UP CODE SLIGHTLY
        --- qiptask_optimizer.py: class (using all functionalities) for optimizing the parameters of a given RQPN to achieve a desired transformation, tested  by running optimizations and verifying that we obtain the desired qip-transformation at the end of the optimization.
        --- qiptask_generateplots.py: for generating various plots based on the RQPN parameters - NOT USED FOR PLOTS IN PAPER
    --- plots: generate plots for final results
    --- workflow: contains a script for putting all aspects of the above code together to perform an optimzation run for a given configuration
    --- run: includes run_RQPNN_local for running an optimization locally 
--- tests: tests to verify that the code is correct - see source code overview to see what each script tests

requirements/_GPU/_HPC: requirements for virtual environment, depending on whether we use a GPU or CPU.
pyproject.toml: for enabling pytests

### testing code
The repository includes examples of tests of the code.
We use pytest:
- activate virtual environment: .\venv_RQPN\Scripts\Activate.ps1
- in powershell: python -m pip install -e . (ONLY FIRST TIME to import a package from different directories)
- in powershell: python -m pytest

