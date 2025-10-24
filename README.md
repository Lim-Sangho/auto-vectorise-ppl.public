# Automatic Loop Vectorisation in PPLs

This repository contains the source code for reproducing the experiments in the paper [Optimising Probabilistic Programs for Efficient Probabilistic Inference Through Automatic Loop Vectorisation]()
by Sangho Lim, Hyoungjin Lim, [Wonyeol Lee](https://wonyeol.github.io/), [Xavier Rival](https://scholar.google.com/citations?user=YGy_zroAAAAJ&hl=en), and [Hongseok Yang](https://sites.google.com/view/hongseokyang/home).


# Dependencies
Our implementation relies heavily on the [Pyro PPL](https://github.com/pyro-ppl/pyro), a deep probabilistic programming library built on top of PyTorch.

All examples and experiments were conducted using **Python 3.8**, **Pyro 1.9.1**, and **PyTorch 2.4.1+cu118**. You can install these dependencies using either pip or a conda environment with the following commands:

```
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install pyro-ppl==1.9.1
```

```
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pyro-ppl==1.9.1
```


# Hidden Markov Model (HMM) Example
The following examples demonstrate **Hidden Markov Models (HMM)** implemented in Pyro, desinged to model a piano sequence dataset. This implementation is adapted from the offical [Pyro tutorial](https://pyro.ai/examples/hmm.html). The complete source code for our experiments based on this model is available in [experiment/benchmark/hmm_ord1.py](https://github.com/Lim-Sangho/auto-vectorise-ppl.public/blob/main/experiment/benchmark/hmm_ord1.py).

### 1. Baseline: Sequential Loops
`model_seq` serves as the baseline reference model. It provides a straightforward implementation of the HMM with 3-level sequential loops without using any of the vectorisation primitives supported by Pyro.


```python
import pyro

def model_seq(sequences, lengths, hidden_dim=16, is_guide=False):
    num_sequences, max_length, data_dim = sequences.shape
    probs_x = pyro.sample(
        "probs_x",
        dist.Dirichlet(0.9 * torch.eye(hidden_dim) + 0.1).to_event(1),
    )
    probs_y = pyro.sample(
        "probs_y",
        dist.Beta(0.1, 0.9).expand([hidden_dim, data_dim]).to_event(2),
    )

    if not is_guide:
        for i in range(num_sequences):
            x = 0
            for j in pyro.markov(range(max_length)):
                with poutine.mask(mask=(j < lengths[i])):
                    x = pyro.sample(
                        "x_{}_{}".format(i, j),
                        Categorical(logits=probs_x[x].log()),
                        infer={"enumerate": "parallel"}
                    )
                    for k in range(data_dim):
                        pyro.sample(
                            "y_{}_{}_{}".format(i, j, k),
                            Bernoulli(logits=probs_y[x, k].log()),
                            obs=sequences[i, j, k],
                        )
```

### 2. Pyro Vectorised Version
`model_vmarkov` implements the model using Pyro's `plate` and `vectorized_markov` primitives to vectorise the three nested loops.
These primitives enable efficient vectorisation of loops with no or short-range data dependencies.
The `vectorized_markov` primitive takes a `history` argument specifying the number of previous iterations each step depends on,
and requires explicit tensor shape adjustments (e.g., via slicing or `unsqueeze`) to ensure correct alignment.

```python
import pyro
from pyro.contrib.funsor import vectorized_markov, plate

def model_vmarkov(sequences, lengths, hidden_dim=16, is_guide=False):
    num_sequences, max_length, data_dim = sequences.shape
    probs_x = pyro.sample(
        "probs_x",
        dist.Dirichlet(0.9 * torch.eye(hidden_dim) + 0.1).to_event(1),
    )
    probs_y = pyro.sample(
        "probs_y",
        dist.Beta(0.1, 0.9).expand([hidden_dim, data_dim]).to_event(2),
    )

    if not is_guide:
        tones_plate = plate("tones", data_dim, dim=-1)
        with plate("sequences", num_sequences, dim=-3) as batch:
            batch = batch.unsqueeze(-1)
            x = 0
            for t in vectorized_markov(None, "lengths", max_length, dim=-2, history=1):
                with mask(mask=(t < lengths.unsqueeze(-1)).unsqueeze(-1)):
                    x = pyro.sample(
                        "x_{}".format(t),
                        Categorical(logits=probs_x[x].log()),
                        infer={"enumerate": "parallel"},
                    )
                    with tones_plate:
                        pyro.sample(
                            "y_{}".format(t),
                            Bernoulli(logits=probs_y[x.squeeze(-1)].log()),
                            obs=sequences[batch, t],
                        )
```

### Our Propsed Vectorisation Approach
`model_ours` implements our proposed **automatic vectorisation** method, which vectorises the three nested loops automatically.
It supports vectorisation of nested loops with arbitrary data dependency lengths without requiring a `history` argument.
This is achieved through speculative vectorisation and fixed-point checking in our algorithm.

Our method uses the `@vec.vectorize` decorator and the `vec.range`
constructor to parallelise loops while preserving their original sequential semantics.
It also introduces an additional argument `s` (of type `vec.State`) for managing variable reads and writes,
and an `Index` operator for `NaN`-mask-based indexing.
These additions are orthogonal to the model structure and independent of tensor shapes or dimensionality,
allowing users to avoid manual tensor manipulations.

```python
import pyro
import vectorized_loop as vec
from vectorized_loop.ops import Index, cat

@vec.vectorize
def model_ours(s: vec.State, sequences, lengths, hidden_dim=16, is_guide=False):
    num_sequences, max_length, data_dim = sequences.shape
    probs_x = pyro.sample(
        "probs_x",
        dist.Dirichlet(0.9 * torch.eye(hidden_dim) + 0.1).to_event(1),
    )
    probs_y = pyro.sample(
        "probs_y",
        dist.Beta(0.1, 0.9).expand([hidden_dim, data_dim]).to_event(2),
    )

    if not is_guide:
        for i in vec.range("sequences", num_sequences, vectorized=True):
            s.x = 0
            for j in vec.range("lengths", max_length, vectorized=True):
                with poutine.mask(mask=(j < lengths[i])):
                    s.x = pyro.sample(
                        "x",
                        Categorical(logits=Index(probs_x)[s.x].log()),
                        infer={"enumerate": "parallel"}
                    )
                    for k in vec.range("tones", data_dim, vectorized=True):
                        pyro.sample(
                            "y",
                            Bernoulli(logits=Index(probs_y)[s.x, k].log()),
                            obs=sequences[i, j, k],
                        )
```