# CS-MIA
This repository contains code for our paper: "CS-MIA: Membership inference attack based on prediction confidence series in federated learning".
***
## Code usage: 
### Prepare dataset
1. Please create a **"data"** folder in the same directory of the code to save the raw dataset.
2. For MNIST, CIFAR-10 and CIFAR-100, torchvision is used to automatically download the raw dataset to **"data"** when network is available.
3. As for Purchase100 and Texas100, please download the raw dataset derived by <a href="https://ieeexplore.ieee.org/abstract/document/7958568/">Shokri et al.</a> in advance and save them to **"data"**.
   We provide preprocessing scripts for both datasets in the dictionary **"common"**. Please execute the scripts as follows to generate Purchase100 and Texas100 dataset suitable for CS-MIA.
```
python3 preprocess_purchase.py
python3 preprocess_texas100.py
```
### Prepare configuration file
We provide a configuration file template **"config_template.yaml"** in the directory **"tests"** for running our code. You can refer to it to generate your own configuration file as needed. 

We also provide the configuration files of all experiments in our paper in the **"tests/final_config"** directory for reference.
### Run scripts
All experimental scripts are located in the dictionary **"tests"**. You can execute the scripts by providing the parameter **--config** which specified the configuration file path as follows.
1. To evaluate local CS-MIA
```angular2html
python3 local_mia_compare.py --config final_config/local_mia_compare/cifar10_alexnet_5.yaml
```
2. To evaluate passive global CS-MIA
```angular2html
python3 passive_global_mia_compare.py --config final_config/passive_global_mia_compare/cifar10_alexnet_5.yaml
```
3. To evaluate active global CS-MIA
```angular2html
python3 active_global_mia_compare.py --config final_config/active_global_mia_compare/cifar10_alexnet_5.yaml
```
4. To evaluate the impact of confidence metrics
```angular2html
python3 confidence_metrics_local_evaluation.py --config final_config/metric_evaluation/cifar10_alexnet_5.yaml
python3 confidence_metrics_global_evaluation.py --config final_config/metric_evaluation/cifar10_alexnet_5.yaml
```
5. To evaluate the impact of the number of clients
```angular2html
python3 client_number_impact.py --config final_config/other/client_number_impact_purchase100.yaml
```
6. To evaluate the impact of rounds
```angular2html
python3 rounds_impact.py --config final_config/other/rounds_impact_cifar100.yaml
```
7. To evaluate the impact of number of selected clients
```angular2html
python3 selected_clients_number_impact.py --config final_config/other/selected_clients_number_impact_purchase100.yaml
```
8. To evaluate the impact of non-IID data
```angular2html
python3 non_iid_impact.py --config final_config/non_iid_impact/local_mia/cifar10_alexnet_5.yaml
```

To evaluate the effectiveness of CS-MIA under defense, please also provide the parameter **--defense** as follows.
1. To evaluate CS-MIA under dropout defense
```angular2html
python3 defense_local_mia.py --config final_config/defense/local_mia_purchase100.yaml --defense dropout
python3 defense_passive_global_mia.py --config final_config/defense/passive_global_mia_purchase100.yaml --defense dropout
```
2. To evaluate CS-MIA under knowledge distillation defense
```angular2html
python3 defense_local_mia.py --config final_config/defense/local_mia_purchase100.yaml --defense know_dist
```
3. To evaluate global CS-MIA under secure aggregation defense
```angular2html
python3 defense_passive_global_mia.py --config final_config/defense/passive_global_mia_purchase100.yaml --defense sec_agg
```
4. To evaluate CS-MIA under differential privacy defense
```angular2html
python3 defense_local_mia.py --config final_config/defense/local_mia_purchase100.yaml --defense dp
```
5. To evaluate the impact of differential privacy epsilon
```angular2html
python3 dp_epsilon_impact.py --config final_config/defense/dp_epsilon_impact_purchase100.yaml
```
## Code architecture
```angular2html
.
├── common               # implementation of loading dataset, training and testing single model
├── dataset
├── federated            # implementation of federated learning, three defenses: differential privacy, knowledge distillation and secure aggregation
├── membership_inference # implementation of CS-MIA (local_cs.py, global_cs.py) and other membership inference attacks
├── model                # architecture of all target models
├── tests                # experimental scripts
│   ├── checkpoints      # to save model parameters during training
│   ├── data             # to save indices of training and testing dataset divided for each participant and for shadow model
│   ├── final_config     # to save configuration files of all experiments in our paper
│   ├── result           # to save the log output by experimental scripts
└── utils
```

***
## Citation
If you use this code, please cite the following paper: 
### <a href="https://www.sciencedirect.com/science/article/abs/pii/S2214212622000801">CS-MIA</a>
```
@article{GU2022103201,
title = {CS-MIA: Membership inference attack based on prediction confidence series in federated learning},
journal = {Journal of Information Security and Applications},
volume = {67},
pages = {103201},
year = {2022},
issn = {2214-2126},
doi = {https://doi.org/10.1016/j.jisa.2022.103201},
url = {https://www.sciencedirect.com/science/article/pii/S2214212622000801},
author = {Yuhao Gu and Yuebin Bai and Shubin Xu},
keywords = {Federated learning, Privacy leakage, Membership inference, Prediction confidence series, White-box attack},
abstract = {Federated learning (FL) is vulnerable to membership inference attacks even it is designed to protect users’ data during model training, as model parameters remember the information of training data. However, existing inference attacks against FL perform poorly in multi-participant scenarios. We propose CS-MIA, a novel membership inference based on prediction confidence series, posing a more critical privacy threat to FL. The inspirations of CS-MIA are the different prediction confidence of a model on training and testing data, and multiple versions of target models over rounds during FL. We use a neural network to learn individual features of confidence series on training and testing data for subsequent membership inference. We design inference algorithms for both local and global adversaries in FL. And we also design an active attack for global adversaries to extract more information. Our confidence-series-based membership inference outperforms most state-of-the-art attacks on various datasets in different scenarios, demonstrating the severe privacy leakage in FL.}
}

```