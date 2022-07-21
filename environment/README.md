# Conda Environment Commands

`environment_Ubuntu.yml` contains the exact packages used on our evaluation machine
running Ubuntu 20.04.4 LTS (GNU/Linux 5.13.0-48-generic x86_64).

`environment.yml` contains the packages without build specifications for other OSes,
but we cannot guarantee that this environment will work flawlessly. The file has been tested with 
macOS Monterey on an Apple MacBook Pro M1 2020.

## GPU Usage

By default, environment.yml installs tensorflow and not tensorflow-gpu.
If your machine has a GPU, uncomment the tensorflow-gpu line in
[environment.yml](environment.yml) instead.

## Create environment.yml file via conda

With your conda environment activated (`conda activate RAoPT`),
run the following command to generate dependency yaml file:

    conda env export --name raopt > environment_Ubuntu.yml

## Create conda environment from YML

    conda env create --name raopt -f environment_Ubuntu.yml

or

    conda env create --name raopt -f environment.yml

## Activate environment

Before execution of a script the environment has to be activated via:

    conda activate raopt