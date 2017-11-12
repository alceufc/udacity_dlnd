# Configuring Conda Environment in Windows

To create the enviroment run on the Anaconda Prompt:

```
conda create -n proj3env python=3.5
```

Activate the enviroment:

```
activate proj3env
```

Go to the directory of the project and run the following command:

```
pip install -r requirements.txt
```

The command above installs the requirements (packages) for this project.
In order to be able to select which kernel we want in Jupyter, we have to
install the package `nb_conda`.

```
conda install nb_conda
```
