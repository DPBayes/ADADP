# ADADP

Code for the differentially private learning rate adaptive algorithm described in

A. Koskela and A. Honkela. "Learning Rate Adaptation for Differentially Private Learning." In: International Conference on Artificial Intelligence and Statistics. PMLR, 2020. p. 2465-2475. http://proceedings.mlr.press/v108/koskela20a/koskela20a.pdf

The current version runs only with GPU.

Usage, e.g.

python3 main_adadp.py --n_epochs=100 --tol=1.0 --noise_sigma=2.0 --batch_size=200
