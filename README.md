![](https://raw.githubusercontent.com/forgivingsoldier/image/main/teat/DALL%C2%B7E%202023-12-21%2017.32.04%20-%20Design%20a%20sharp%20and%20professional%20logo%20for%20'SSL%20Graph'.%20The%20logo%20must%20clearly%20showcase%20the%20acronym%20'SSL'%20incorporated%20within%20a%20molecular%20structure%20compo.png)

# SSL Graph——基于Paddle的自监督图对比学习框架

SSL Graph（Self-Supervised Learning Graph）是首个基于[PGL [Paddle Graph Learning]](https://github.com/PaddlePaddle/PGL)和[Paddle](https://github.com/PaddlePaddle/Paddle)实现的开源的自监督图对比学习框架，集成了一些图对比学习的前沿模型 。

## News

<details>
<summary>
2023-12-22 release v0.1
</summary>
<br/>

We release the latest version v0.1.0

- Models:MVGRL,DGI,GRACE,GCA,CCA-SSG,GraphCL
- Dataset:Cora,Citeseer

</details>

## Key Features

- Easy-to-Use: SSL Graph provides easy-to-use interfaces for running experiments with the given models and dataset.
- Extensibility: User can define customized task/model/dataset to apply new models to new scenarios.
- Efficiency: The backend pgl provides efficient APIs.
- Innovativeness: The first self-supervised graph contrastive learning framework based on Paddle for graph neural networks.

## Get Started

### Requirements

SSL Graph needs the following requirements to be satisfied beforehand:

- paddlepaddle
- paddle-bfloat
- python>3.7
- PGL
- tqdm
- numpy
- wheel
- Pillow
- openssl

### Usage

**Step 1: Load datasets**

```python
dataset = load("cora")
```

In train.py, choose dataset by name, such as cora, pumbed, citeseer

**Step 2: Initialize the model and config**

```python
encoder = GCN(graph.node_feat["words"].shape[1], 512, 1, 512, 0.0) #choose an encoder
grace = Grace(encoders) #choose a model
data_loader = [(graph, graph.node_feat["words"])] #data_loader
```

In train.py, choose encoder and model

**Step 3: Train GSL model**

```python
#construct Trainer Class
train = Trainer(full_dataset=data_loader, dataset=dataset)
# config
train.setup_train_config(p_optim='Adam', p_lr=0.001, runs=10, p_epoch=300, weight_decay=0.0, batch_szie=256)
# start train and get scores
train.train_encoder(grace)
```

`p_optim`:  opitimizer to optimize encoder and classification

`p_lr`: learning rate of training the encoder

`runs`: number of trainings

`p_epoch`:  the number of times the entire training dataset

`weight_decay`: decay weight ratio

`batch_size`: the number of data samples

## Models

| Model                                                        | Node classification |
| ------------------------------------------------------------ | ------------------- |
| [DGI]([PetarV-/DGI: Deep Graph Infomax (https://arxiv.org/abs/1809.10341) (github.com)](https://github.com/PetarV-/DGI))[ICLR 2019] | :heavy_check_mark:  |
| [GRACE]([ipjohnson/Grace: Grace is a feature rich dependency injection container library (github.com)](https://github.com/ipjohnson/Grace))[ICML 2020 ] | :heavy_check_mark:  |
| [GCA]([Yaoyi-Li/GCA-Matting: Official repository for Natural Image Matting via Guided Contextual Attention (github.com)](https://github.com/Yaoyi-Li/GCA-Matting))[WWW 2021] | :heavy_check_mark:  |
| [MVGRL]([hengruizhang98/mvgrl: DGL Implementation of ICML 2020 Paper 'Contrastive Multi-View Representation Learning on Graphs' (github.com)](https://github.com/hengruizhang98/mvgrl))[ICML 2020] | :heavy_check_mark:  |
| [CCA-SSG]([hengruizhang98/CCA-SSG: Codes for 'From Canonical Correlation Analysis to Self-supervised Graph Neural Networks'. https://arxiv.org/abs/2106.12484 (github.com)](https://github.com/hengruizhang98/CCA-SSG))[NeurIPS 2021] | :heavy_check_mark:  |
| [GraphCL]([Shen-Lab/GraphCL: [NeurIPS 2020\] "Graph Contrastive Learning with Augmentations" by Yuning You, Tianlong Chen, Yongduo Sui, Ting Chen, Zhangyang Wang, Yang Shen (github.com)](https://github.com/Shen-Lab/GraphCL))[NeurIPS 2020] | :heavy_check_mark:  |

## Contributors
