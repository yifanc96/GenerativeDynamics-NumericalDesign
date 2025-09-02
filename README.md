# GenerativeDynamics - NumericalDesign
Design of interpolation schedules, noise, and source terms in flow and diffusion-based generative models, with a focus on data distributions from science and engineering applications.

The Git Repository includes examples based on solutions to stochastic Allen-Cahn and Navier-Stokes equations. These examples involve training neural networks on datasets with wide-ranging Fourier spectra, making them numerically ill-conditioned. The neural network training codes require GPU computation, and the repository comes with provided datasets. Note that these codes will need modification to run on your system (specifically, updating data paths, save locations, and wandb locations).

We also present proof-of-concept examples using Gaussian measures and mixtures in high dimensions. These examples are based on explicit formulas and can be run directly on CPUs without modification.

All implementations adopt the stochastic interpolants framework.

#### Relevant papers
- [Lipschitz-Guided Design of Interpolation Schedules]()
- [Scale-Adaptive Design of Generative Flows]()
- Generative Diffusions from A Point Source and Follmer Processes
```
```

#### Dataset
- [Stochastic Navier-Stokes](https://zenodo.org/records/10939479)
- [Stochastic Allen-Cahn](https://zenodo.org/uploads/15708250)

#### Other helpful repository
For general codes of stochastic interpolants, please refer to [interpolant](https://github.com/interpolants). Specifically for probabilistic forecasting applications, please refer to [forecasting](https://github.com/interpolants/forecasting). I'll thank [Mark Goldstein](https://marikgoldstein.github.io/) for helping me in the initial phase of these codes.
