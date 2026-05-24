# Align Your Steps (AYS): Reproduction and Extension with Reinforcement Learning

This project reproduces and extends Align Your Steps (AYS), a framework developed by [NVIDIA](https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/) for improving diffusion model sampling efficiency and quality.

Diffusion models have become a leading approach in image generation due to their ability to produce highly realistic outputs by progressively transforming noise into coherent images. However, a key limitation is their slow sampling speed, as they require many sequential neural network evaluations, making them less suitable for real-time applications.

While recent advances have improved solver efficiency, the design of sampling schedules—which control noise progression during generation—remains relatively underexplored. The AYS framework addresses this by using stochastic calculus to derive optimized, model- and dataset-specific sampling schedules, outperforming traditional handcrafted approaches.

In this project, we reproduce the AYS methodology and further extend it by integrating Reinforcement Learning (RL). Our extension introduces a mechanism that adapts sampling schedules dynamically based on text embedding features, enabling the model to adjust its generation strategy according to prompt complexity.

Inspired by recent work such as DPOK, which demonstrates the effectiveness of online RL for aligning diffusion models with human preferences, our approach explores how RL can further enhance both generation quality and adaptability in diffusion sampling.
