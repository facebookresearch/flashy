# Flashy

![tests badge](https://github.com/facebookresearch/flashy/workflows/tests/badge.svg)
![linter badge](https://github.com/facebookresearch/flashy/workflows/linter/badge.svg)
![docs badge](https://github.com/facebookresearch/flashy/workflows/docs/badge.svg)



## Motivations

We noticed we reused the same structure over and over again in all of our research projects.
PyTorch-Lightning is vastly over engineered and due to its complexity does not allow
the same level of hackability. Flashy aims to be an alternative. We do not claim it will
fit all use cases, and our first goal is for it to fit ours. We aim at keeping the code
simple enough that you can just inherit and override behaviors, or even copy paste what you want
into your project.

## Definitions

At the core of Flashy is the *Solver*. The Solver takes care of 2 things only:
- logging metrics, to multiple backends (file logs, tensorboard or WanDB), with custom formatting,
- checkpointing and automatically tracking stateful part of the solver.

Beyond those core features, Flashy also provide distributed training utilities,
in particular alternatives to DistributedDataParallel, which can break with complex workflows,
along with simple wrappers around DataLoader to support distributed training.

Flashy is *epoch* based, which might sound outdated to some of you. Think of epochs not
as a single pass over your dataset, but as the atomic unit of time for workflow management.
Each epoch end is marked by a call to `flashy.BaseSolver.commit(save_checkpoint=True)`.

Each epoch is composed of a number of *stages*, for instance `train`, `valid`, `test` etc, and do not need
to be the same each time. Stages are a convenience to help with automatically reporting
metrics with appropriate metadata.


## Dependencies and installation

Flashy assume PyTorch is used along with [Dora][dora]. You could use it without PyTorch
with minor changes to `flashy/state.py`. Dora is builtin in a few places and shouldn't be too hard
to remove, although we warmly recommend using it. Flashy requires at least Python 3.8.

To install Flashy, run the following

```bash
# For the moment we recommend having bleeding edge versions of Dora and Submitit
pip install -U git+https://github.com/facebookincubator/submitit@main#egg=submitit
pip install -U git+https://git@github.com/facebookresearch/dora#egg=dora-search
# Now let's install Flashy!
pip install git+ssh://git@github.com/facebookresearch/flashy.git#egg=flashy
```

To install Flashy for development, you can clone this repository and run
```
make install
```

## Getting Started

We will assume you are using [Hydra][hydra]. You will need to be familiar with [Dora][dora].
Let's build a very basic project, called `basic`,
with the following structure:

```
basic/
  conf/
    config.yaml
  train.py
  __init__.py
```

This project is provided in the [examples](examples/) folder.
For [config.yaml](examples/basic/config.yaml), we can start with the basic:

```yaml
epochs: 10
lr: 0.1

dora:
  # Output folder for all the artifacts of an experiment.
  dir: /tmp/flashy_basic_${oc.env:USER}/outputs
```

`__init__.py` is just empty. [train.py](examples/basic/train.py) contains most of the logic:

```python
import torch
from dora import hydra_main
import flashy


class Solver(flashy.BaseSolver):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = torch.nn.Linear(32, 1)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.best_state = {}
        # register_stateful supports any attribute. On checkpoints loading,
        # it will try to use inplace method when possible (i.e. Modules, lists, dicts).
        self.register_stateful('model', 'optim', 'best_state')
        self.init_tensorboard()  # all metrics will be reported to stderr and tensorboard.

    def run(self):
        self.restore()  # load checkpoint
        for epoch in range(self.epoch, self.cfg.epochs):
            # Stages are used for automatic metric reporting to Dora, and it also
            # allows tuning how metrics are formatted.
            self.run_stage('train', self.train)
            # Commit will send the metrics to Dora and save checkpoints by default.
            self.commit(save_checkpoint=epoch % 2 == 1)

    def train(self):
        # this is super dumb, checkout `examples/cifar/solver.py` for more advance usage!
        x = torch.randn(4, 32)
        y = self.model(x)
        loss = y.abs().mean()
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        return {'loss': loss.item()}


@hydra_main(config_path='config', config_name='config', version_base='1.1')
def main(cfg):
    # Setup logging both to XP specific folder, and to stderr.
    flashy.setup_logging()
    # Initialize distributed training, no need to specify anything when using Dora.
    flashy.distrib.init()
    solver = Solver(cfg)
    solver.run()


if __name__ == '__main__':
    main()
```

From the folder containing `basic`, you can launch training with
```
dora -P basic run
dora run  # if no other package contains a train.py file in the current folder.
```


## Example

See [examples/cifar/solver.py](examples/cifar/solver.py) for a more advanced example,
with real training and distributed. When running examples from the `examples/` folder,
you must pass the package you want to run to Dora, as there are multiple possibilities:
```
dora -P [basic|cifar] run
```


## API

Checkout [Flashy API Documentation][api]

[api]: https://share.honu.io/flashy/docs/flashy/index.html
[dora]: https://github.com/facebookresearch/dora
[hydra]: https://github.com/facebookresearch/hydra


## Licence

Flashy is provided under the MIT license, which can be found in the [LICENSE](./LICENSE) file
in the root of the repository. Parts of `flashy.loggers.utils` were adapted from
PyTorch-Lightning, originally under the Apache 2.0 License, see [flashy/loggers/utils.py](flashy/loggers/utils.py)
for details.
