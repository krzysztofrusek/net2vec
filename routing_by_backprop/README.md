# Fast Traffic Engineering by Gradient Descent with
Learned Differentiable Routing
#### Link to paper: [[here](https://arxiv.org/abs/2209.10380)]
#### Krzysztof Rusek, Paul Almasan, José Suárez-Varela, Piotr Chołda, Pere Barlet-Ros, Albert Cabellos-Aparicio

Contact: <krusek@agh.edu.pl>

## Abstract

Emerging applications such as the metaverse, telesurgery or cloud computing require increasingly complex operational demands on networks (e.g., ultra-reliable low latency). Likewise, the ever-faster traffic dynamics will demand network control mechanisms that can operate at short timescales (e.g., sub-minute). In this context, Traffic Engineering (TE) is a key component to efficiently control network traffic according to some performance goals (e.g., minimize network congestion).
This paper presents Routing By Backprop (RBB), a novel TE method based on Graph Neural Networks (GNN) and differentiable programming. Thanks to its internal GNN model, RBB builds an end-to-end differentiable function of the target TE problem (MinMaxLoad). This enables fast TE optimization via gradient descent. In our evaluation, we show the potential of RBB to optimize OSPF-based routing (≈25\% of improvement with respect to default OSPF configurations). Moreover, we test the potential of RBB as an initializer of computationally-intensive TE solvers. The experimental results show promising prospects for accelerating this type of solvers and achieving efficient online TE optimization. 

# Instructions to execute

This repository contains code used in the numerical experiments, and allow for reproductioin of our results.

## Train

First we need to train the model for appropriate large dataset.

```shell
python3 python/sp.py --train_graphs=10 --test_graphs=2 --checkpoint_steps=2
```

## Optimize

Once the model is trained, it can be used for network optimization.
Below is the example for nsf network topology.

```shell
	mkdir -p out/$@
	PYTHONPATH=baselines:${PYTHONPATH} python3 python/optimize_sp.py \
		--num_opt 5 \
		--report out/$@/opt.csv \
		--num_sgd 1 \
		--nopmap

```

**If you decide to apply the concepts presented or base on the provided code, please do refer our paper.**

```
@misc{https://doi.org/10.48550/arxiv.2209.10380,
  doi = {10.48550/ARXIV.2209.10380},
  
  url = {https://arxiv.org/abs/2209.10380},
  
  author = {Rusek, Krzysztof and Almasan, Paul and Suárez-Varela, José and Chołda, Piotr and Barlet-Ros, Pere and Cabellos-Aparicio, Albert},
  
  keywords = {Networking and Internet Architecture (cs.NI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Fast Traffic Engineering by Gradient Descent with Learned Differentiable Routing},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```

