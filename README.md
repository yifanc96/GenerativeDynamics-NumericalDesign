# GenerativeDynamics - NumericalDesign
Design of interpolation schedules, noise, and source terms in flow and diffusion-based generative models, with a focus on data distributions from science and engineering applications.

The Git Repository includes examples based on solutions to stochastic Allen-Cahn and Navier-Stokes equations. These examples involve training neural networks on datasets with wide-ranging Fourier spectra, making them numerically ill-conditioned. The neural network training codes require GPU computation, and the repository comes with provided datasets. Note that these codes will need modification to run on your system (specifically, updating data paths, save locations, and wandb accounts).

We also present proof-of-concept examples using Gaussian measures and mixtures in high dimensions. These examples are based on explicit formulas and can be run directly on CPUs without modification.

All implementations adopt the stochastic interpolants framework.

#### Relevant papers
- [Lipschitz-Guided Design of Interpolation Schedules]()
- [Scale-Adaptive Design of Generative Flows]()
- [Design of Point Source and Optimization of Diffusion Coefficients to Yield Follmer Processes](https://openreview.net/pdf/9dc86834c15cdb6e583ef6154ec5fa6c51ecee8e.pdf)
```
```

#### Dataset
- [Stochastic Navier-Stokes](https://zenodo.org/records/10939479)
- [Stochastic Allen-Cahn](https://zenodo.org/uploads/15708250)

#### Other helpful repository
For general codes of stochastic interpolants, please refer to [interpolant](https://github.com/interpolants). Specifically for probabilistic forecasting applications, please refer to [forecasting](https://github.com/interpolants/forecasting). There, the point source is the current state, and the generative model forecasts the next state. In this repository, the point source is always zero, and the generative model forecasts the difference between the next and current states.

I'll thank [Mark Goldstein](https://marikgoldstein.github.io/) for helping me in the initial phase of the forecasting code, and [Mengjian Hua](https://scholar.google.com/citations?user=llRFiBEAAAAJ&hl=en) for preparing the stochastic Navier-Stokes dataset.
