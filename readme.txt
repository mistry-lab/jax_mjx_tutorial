***Jax and Mujoco MJX tutorial***

This repo uses pipenv for environment mamanagment and requires it to be installed on your machine.
As a good rule of thumb also it is could to use pyenv for python version managment.
Virtual environments (e.g. pipenv) have an easier time setting python version thorugh pyenv.

NOTE: We rely on cuda enabled jax library. This is not included in the Pipfile. (Simply following the steps below should work)
- run pipenv shell
- Depending on preference, either install 
	- pipenv run pip install --upgrade "jax[cuda12_local]" # for local CUDA toolkit
	- pipenv run pip install -U "jax[cuda12]" # for using provided bins
- run pipenv install for the rest of the deps

The notebook is gives a general overview of the main components of functinal programming that transfer to jax.
The python script trains a simple one dimensinoal network that tries to get the system to go to zero.

The main components that you should pay attention to are:
- Jax transformations: jit, grad, vmap and pmap (pmap not included in the tutorials)
- Jax scan operation for sequential code e.g. simulating dynamics
- The Equinox library convention for setting up neural nets. E.g. each class has type definition predefined, this is important for jax transformations
