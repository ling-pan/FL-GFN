# Better Training of GFlowNets with Local Credit and Incomplete Trajectories

This repository is the implementation of [Better Training of GFlowNets with Local Credit and Incomplete Trajectories](https://openreview.net/pdf?id=beHp3L9KXc) in ICML 2023. This codebase is based on the open-source [gflownet](https://github.com/GFNOrg/gflownet) implementation, and please refer to that repo for more documentation.

## Citing

If you used this code in your research or found it helpful, please consider citing our paper:
```
@inproceedings{
	pan2023better,
	title={Better Training of GFlowNets with Local Credit and Incomplete Trajectories},
	author={Ling Pan and Nikolay Malkin and Dinghuai Zhang and Yoshua Bengio},
	booktitle={International Conference on Machine Learning},
	year={2023},
	url={https://openreview.net/forum?id=beHp3L9KXc}
}
```

## Requirements

### Grid
- python: 3.6
- torch: 1.3.0
- scipy: 1.5.4
- numpy: 1.19.5
- tdqm

### Molecule discovery
Please check the [gflownet](https://github.com/GFNOrg/gflownet) repo for more details about the environment

## Usage

Please follow the instructions below to replicate the results in the paper. 
- Grid
```
python gflownet.py --method <METHOD> --fl <FL_FLAG> --size <SIZE> --seed <SEED>
```
Specifiy METHOD=db_gfn with FL_FLAG=1 for FL-GFN.

- Molecule discovery
```
python gflownet.py --objective <OBJECTIVE> --fl <FL_FLAG> --run <RUN>
```
