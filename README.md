# GenerativeDynamics - NumericalDesign
Design of interpolation schedules, noises, and sources in flow and diffusion-based generative models with a focus on data distributions arising from science and engineering.

Examples include solutions to stochastic Allen-Cahn and Navier-Stokes equations (based on training neural networks; need to run on GPUs and we provide the data) that have a wide range of Fourier spectra and are thus numerically ill-conditioned. 

We also present proof-of-concept examples (based on explicit formulas; these examples can be run directly on CPUs) using Gaussian measures and mixtures in high dimensions.

We adopt the framework of stochastic interpolants in the codes.

#### Relevant papers
- Numerical design through min-Lip: check xxx
- Design of noise and schedules for ill-conditioned distributions: check xxx
- Generative diffusions from a point source: check xxx


#### Dataset
- [Stochastic Navier-Stokes](https://zenodo.org/records/10939479)
- [Stochastic Allen-Cahn](https://zenodo.org/uploads/15708250)

#### Other helpful repository
For general codes of stochastic interpolants, please refer to [interpolant](https://github.com/interpolants). Specifically for probabilistic forecasting applications, please refer to [forecasting](https://github.com/interpolants/forecasting).
