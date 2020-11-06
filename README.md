# ADADP

PyTorch implementation of the differentially private learning rate adaptive algorithm described in

A. Koskela and A. Honkela. "Learning Rate Adaptation for Differentially Private Learning." In: International Conference on Artificial Intelligence and Statistics. PMLR, 2020. p. 2465-2475. http://proceedings.mlr.press/v108/koskela20a/koskela20a.pdf

Usage, e.g.

python3 main_adadp.py --n_epochs=100 --tol=1.0 --noise_sigma=2.0 --batch_size=200

The code was developed with CUDA Version 10.1.105, PyTorch 1.4.0, torchvision 0.2.2, Python 3.6.9.

The current version of the CIFAR-10 experiments runs only with Cuda.

The MNIST experiments run also using CPU.
