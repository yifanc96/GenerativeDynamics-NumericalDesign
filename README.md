# GenerativeDynamics - NumericalDesign
Design of interpolation schedules, noises, and sources in flow and diffusion-based generative models with a focus on data distributions arising from science and engineering.

Examples include solutions to stochastic Allen-Cahn and Navier-Stokes equations (these are based on training neural networks. The code needs to be run on GPUs. The data are provided) that have a wide range of Fourier spectra and are thus numerically ill-conditioned. 

We also present proof-of-concept examples (these are based on explicit formulas; the code can be run directly on CPUs) using Gaussian measures and mixtures in high dimensions.

We adopt the framework of stochastic interpolants in the codes.

#### Relevant papers
- [Lipschitz-Guided Design of Interpolation Schedules]()
- [Scale-Adaptive Design of Generative Flows]()
- [Generative Diffusions from A Point Source]()
```
@misc{chen2024probabilistic,
      title={Probabilistic Forecasting with Stochastic Interpolants and F\"ollmer Processes}, 
      author={Yifan Chen and Mark Goldstein and Mengjian Hua and Michael S. Albergo and Nicholas M. Boffi and Eric Vanden-Eijnden},
      year={2024},
      eprint={2403.13724},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

#### Dataset
- [Stochastic Navier-Stokes](https://zenodo.org/records/10939479)
- [Stochastic Allen-Cahn](https://zenodo.org/uploads/15708250)

#### Other helpful repository
For general codes of stochastic interpolants, please refer to [interpolant](https://github.com/interpolants). Specifically for probabilistic forecasting applications, please refer to [forecasting](https://github.com/interpolants/forecasting).
