# ADADP

Code for the experiments of

A. Koskela and A. Honkela.
"Learning rate adaptation for differentially private stochastic gradient descent." arXiv preprint arXiv:1809.03832 (2018).

paper: https://arxiv.org/abs/1809.03832

The current version runs only with GPU.

Usage, e.g.

python3 main_adadp.py --n_epochs=100 --tol=1.0 --noise_sigma=2.0 --batch_size=200
