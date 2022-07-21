# Reconstruction Attack on Protected Trajectories (RAoPT)

Artifacts for [ACSAC'22](https://www.acsac.org/2022/) paper 'Reconstruction Attack on Differential Private Trajectory Protection Mechanisms'.

## Table of Contents

* [Abstract](#abstract)
* [Citations](#citations)
* [Licence](#licence)
* [Acknowledgements](#acknowledgements)
* [Requirements](#requirements)
* [Setup](#setup)
* [Tests](#tests)
* [Configuration](#configuration)
* [Execution](#execution)
   * [Download datasets into data/ directory](#download-datasets-into-data-directory)
   * [Pre-Processing](#pre-processing)
   * [Protection Mechanisms](#protection-mechanisms)
   * [Evaluation](#evaluation)
   * [Manual Execution](#manual-execution)
      * [Step 1: Creating Train and Test Sets](#step-1-creating-train-and-test-sets)
      * [Step 2: Training](#step-2-training)
      * [Step 3: Prediction/Evaluation](#step-3-predictionevaluation)
   * [Measure Reconstruction Runtime](#measure-reconstruction-runtime)
   * [Adding a Datasets](#adding-a-dataset)
* [Contact](#contact)
* [References](#references)

## Abstract

This repository contains the source code for the Reconstruction Attack on Protected Trajectories (RAoPT)-model.
Additionally, the protection mechanism SDD, CNoise, PNoise, and GNoise from (Jiang et al., 2013) are included,
as well as the pre-processing scripts for the GeoLife (Zhou et al., 2010) and T-Drive (Yuan et al., 2010) datasets.
The datasets themselves need to be downloaded separately from the respective websites due to their size.
The code has been evaluated on Ubuntu 20.04.4 LTS (GNU/Linux 5.13.0-48-generic x86_64) using one GPU.

> Location trajectories collected by smartphones and other sensor-equipped  devices represent a valuable data source for
> analytics services such as transport optimisation, location-based services, and contact tracing.
> Likewise, trajectories have the potential to reveal sensitive information about individuals, such as religious beliefs,
> social connections, or sexual orientation. 
> Accordingly, trajectory datasets require appropriate protection before publication. 
> Due to their strong theoretical privacy guarantees, differential private publication mechanisms have received much
> attention in the past. 
> However, the large amount of noise that needs to be added to achieve differential privacy yields trajectories that
> differ significantly from the original trajectories.
> These structural differences, e.g., ship trajectories passing over land, or car trajectories not following roads,
> can be exploited to reduce the level of privacy provided by the publication mechanism.
> We propose a deep learning-based Reconstruction Attack on Protected Trajectories (RAoPT), that leverages the mentioned
> differences to partly reconstruct the original trajectory from a differential private release.
> The evaluation shows that our RAoPT model can reduce the Euclidean and Hausdorff distances of released trajectories
> to the original trajectories by over 65% on the T-Drive dataset even under protection with e ≤ 1.
> Trained on the T-Drive dataset, the model can still reduce both distances by over 48% if applied to GeoLife
> trajectories protected with a state-of-the-art protection mechanism and e = 0.1.
> This work aims to highlight shortcomings of current publication mechanisms for trajectories and thus motivates further
> research on privacy-preserving publication schemes.

## Citations

If you use any portion of our work, please cite our publication.

To be added, accepted at [ACSAC'22](https://www.acsac.org/2022/).

## Artifacts Evaluation

This artifact was submitted to the
[ACSAC 2022 Artifacts Evaluation](https://www.acsac.org/2022/submissions/papers/artifacts/)
and was evaluated as ... .

## Licence

MIT License

Copyright © Cyber Security Research Centre Limited 2022.
This work has been supported by the Cyber Security Research Centre (CSCRC) Limited
whose activities are partially funded by the Australian Government’s Cooperative Research Centres Programme.
We are currently tracking the impact CSCRC funded research. If you have used this code/data in your project,
please contact us at contact@cybersecuritycrc.org.au to let us know.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the “Software”), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Acknowledgements

The authors would like to thank the University of New South Wales,
the Commonwealth of Australia,
and the Cybersecurity Cooperative Research Centre Limited, 
whose activities are partially funded by the Australian Government’s Cooperative Research Centres Programme,
for their support.

## Requirements

For details see [environment/environment.yml](environment/environment.yml).

* Python 3.9
* h5py~=2.10.0
* haversine~=2.5.1
* matplotlib~=3.5.1
* numpy==1.19.2
* pandas~=1.4.2
* scikit-learn~=1.0.2
* scipy~=1.7.3
* shapely~=1.7.1
* tensorboard~=2.4.0
* tensorflow~=2.4.1
* tqdm~=4.64.0
* utm~=0.7.0

We tested the code with tensorflow-gpu v2.4.1.
If the code is supposed to run on CPU some modifications might be required.

## Setup

Install Conda environment:

    conda env create --name VENV_NAME -f environment/environment.yml
    conda activate VENV_NAME

For more details on the environment setup, we refer to the file [environment/README.md](environment/README.md).

Alternatively, the pip requirements file `requirements.txt` can be used for setup.
However, we only tested the code with the conda environment!

## Tests

The project contains some tests in the `test/` directory. 
These are by no means exhaustive, but can be used to test whether the setup was successful.
Run `python3 -m unittest` from this directory to run all tests.

## Configuration

All important configurations such as file paths, enabling caching or parallelization can be set in the
configuration file `config/config.ini`.
Evaluation cases are defined in `config/cases.csv`.

## Execution

Before the model can be evaluated, the dataset needs to be pre-processed.
Due to the large size, we could not include the pre-processed datasets into this repository.

**Note for ACSAC'22 AE reviewers:**
To allow you to test the RAoPT model without performing the computationally expensive pre-processing,
we uploaded the pre-processed datasets as CSV files to the location stated under `Additional URL`
in our submission.
In case you decide to use these datasets, please copy them into the directory `processed_csv/`
(or as configured in `config/config.ini` for `CSV_DIR`).
Then, you can skip directly to step [Evaluation](#evaluation).

### Download datasets into `data/` directory

* You can download the T-Drive dataset here: [https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/)
* You can download the GeoLife dataset here: [https://www.microsoft.com/en-us/download/details.aspx?id=52367&from=https%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2Fb16d359d-d164-469e-9fd4-daa38f2b2e13%2F](https://www.microsoft.com/en-us/download/details.aspx?id=52367&from=https%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2Fb16d359d-d164-469e-9fd4-daa38f2b2e13%2F)

Then, unzip both datasets into the `data/` directory, resulting in the following structure
    
    data/
        geolife/
            data/
                000/
                ...
                181/
        tdrive/
            taxi_log_2008_by_id/
                1.txt
                ...
                10357.txt

### Pre-Processing 

For the two aforementioned datasets, preprocessing scripts are provided.
The pre-processed datasets are stored in the directory `processed_csv/` or as defined in the configuration file.
If caching is enabled, the corresponding pickle files are stored in `processed_cache/` or as defined
in the configuration file.
For other datasets the provided functions in `raopt.preprocessing.preprocess`
can be used to quickly develop an adapted pre-processing script.

For T-Drive call:

    python3 -m raopt.preprocessing.tdrive

For GeoLife call:

    python3 -m raopt.preprocessing.geolife

The generated files will be stored as `processed_csv/tdrive/originals.csv` and
`processed_csv/geolife/originals.csv`, respectively.
The location of the CSV directory `processed_csv/` can be modified in `config/config.ini`.

### Protection Mechanisms

For the reconstruction attack, protected trajectories are required. 
Protection mechanisms can be applied by calling:

    python3 -m raopt.eval.apply_mechanism [-m SENSITIVITY] DATASET MECHANISM EPSILON VERSION

Currently, supported **datasets** are `geolife` and `tdrive`.
Supported **mechanisms** are `sdd` and `cnoise`.
If no **sensitivity** is specified explicitly, the value defined a `M` in `config/config.ini` is used.
If this value is not defined either, the sensitivity is set to `OUTLIER_SPEED * INTERVAL`.
Common values for **epsilon** found in literature are from within the interval [0,10].
**Version** can be set to `1` unless multiple protected versions of the same configuration shall be created.

Example:

    # SDD with Epsilon = 1 and Sensitivity = 16500m (default value for T-Drive)
    python3 -m raopt.eval.apply_mechanism -m 16500 tdrive sdd 1.0 1

This call protects the pre-processed T-Drive trajectories with the SDD Mechanism (Epsilon=1.0, Sensitivity=16500)
and stores the results into `processed_csv/tdrive/sdd_M16500_e1.0_1.csv`.
If caching is activated a pickle file with the same basename will be written to `processed_caching/tdrive/`.
The Naming convention of the files is `MECHANISM_MSENSITIVITY_eEPSILON_VERSION.csv` or `.pickle`.

Example 2:

    # CNoise with Epsilon = 1 and default sensitivity defined in config
    python3 -m raopt.eval.apply_mechanism tdrive cnoise 1.0 1

Note: To run an evaluation the protection does not need to run until the end. It can be stopped at any point via
ctrl + c. **Only press the combination once**, and wait until the already created trajectories are saved properly.

### Evaluation

After all the protected trajectories required for a certain evaluation have been generated, the main
evaluation script can be run. To avoid lengthy command line arguments, evaluation cases are defined in
`config/cases.csv`. The evaluation script will only run cases with `Todo = True`. I.e., all other cases in this
file are ignored. It is required to fill all columns within the file when adding a new evaluation case.

The activated cases can be run by:

    python3 -m raopt.eval.main [-g GPU]

A specific case can be run with:

    python3 -m raopt.eval.main -c CASE_ID [-g GPU]

The GPU option allows to only utilize one GPU if multiple GPUs are available.
The results will be stored into `output/caseX/` where X is the case ID defined in `config/cases.csv`.
The output directory can be modified in `config/config.ini`.

Note: To run a case, the required files in the CSV directory need to exist.
I.e., a case using the T-Drive dataset requires the file `processed_csv/tdrive/originals.csv` to exists. 
If particularly the protection mechanism SDDe=1 (and default M=16500) 
is considered, the file `processed_csv/tdrive/sdd_M16500_e1.0_1.csv` needs to exist, too.

### Manual Execution

The model can also be manually trained and used for prediction/evaluation without defining evaluation cases.
The previous steps of pre-processing and protection mechanism need to be completed beforehand.

#### Step 1: Creating Train and Test Sets

    python3 -m raopt.ml.split_dataset [-h] [-s SPLIT] ORIGINAL_FILE PROTECTED_FILE OUTPUT_DIR

The value `SPLIT € [0,1]` defines the share of the trajectories used for the test set.

Example:

    python3 -m raopt.ml.split_dataset -s 0.2 processed_csv/tdrive/originals.csv processed_csv/tdrive/cnoise_M16500_e1.0_1.csv tmp/example/

This will split the provided trajectories into an 80/20 split of train and test set
and write them into the following files:

    tmp/example/train_p.csv  # Protected Trajectories for training, i.e., trainX
    tmp/example/train_o.csv  # Original Trajectories for training, i.e., trainY
    tmp/example/test_p.csv   # Protected Trajectories for prediction, i.e., testX
    tmp/example/test_o.csv   # Original Trajectories for evaluation, i.e., testY
    
#### Step 2: Training

The model can be trained by calling

    python3 -m raopt.ml.train [-h] [-b BATCH] [-e EPOCHS] [-l LEARNING_RATE] [-s EARLY_STOP] ORIGINAL_FILE PROTECTED_FILE PARAMETER_FILE MAX_LENGTH

Example:
    
    python3 -m raopt.ml.train -b 512 -e 200 -l 0.001 -s 20 tmp/example/train_o.csv tmp/example/train_p.csv tmp/example/parameters.hdf5 100

`MAX_LENGTH` is 100 for T-Drive trajectories and 200 for GeoLife.
If multiple datasets are mixed, the larger value has to be chosen.
The `PARAMETER_FILE` is used to store the parameters of the trained model.
Note, the _reference point_ and _scaling factor_ are written to stdout and the log 
during training, and these can be used during prediction/evaluation.

#### Step 3: Prediction/Evaluation

The trained model can be used for prediction or evaluation with:

    python3 -m raopt.ml.predict [-h] [-e ORIGINAL_FILE] [-r LATITUDE LONGITUDE] [-s LATITUDE LONGITUDE] INPUT_FILE OUTPUT_FILE PARAMETER_FILE MAX_LENGTH

The `INPUT_FILE` contains the protected trajectories to reconstruct from, i.e., textX.
Without the `-e` option, the script is used for _prediction_, with the option for _evaluation._
In the evaluation case, the `ORIGINAL_FILE` contains the unprotected trajectories to evaluate against, i.e., trainY.
The `OUTPUT_FILE` is used to write the results:

* In case of _prediction_: The file contains the reconstructed trajectories.
* In case of _evaluation_: The file contains the computed distances.

The parameter file contains the model parameters generated in step 2.
`MAX_LENGTH` is as described for step 2.
With `-r`, a reference point can be provided and with `-s` a scaling factor.

Example Prediction:

    python3 -m raopt.ml.predict tmp/example/test_p.csv tmp/example/reconstructed.csv tmp/example/parameters.hdf5 100

Example Evaluation:

    python3 -m raopt.ml.predict -e tmp/example/test_o.csv tmp/example/test_p.csv tmp/example/reconstructed.csv tmp/example/parameters.hdf5 100

### Measure Reconstruction Runtime

To measure the time that is required to reconstruct on protected trajectory with a trained model, the script
`raopt.eval.execution_time` can be used.
The script takes the following arguments:

    execution_time.py [-h] [-g GPU] [-s SAMPLE] PARAMETER_FILE PROTECTED_FILE OUTPUT_FILE

`GPU` defines the GPU to use for the prediction.
With `SAMPLE`, a number of trajectories can be provided that are used for the evaluation as using all trajectories
might take too long. By default, all trajectories are used for the evaluation.
`PARAMETER_FILE` contains the parameters of the trained model.
`PROTECTED_FILE` is a CSV file containing the protected trajectories which shall be reconstructed.
`OUTPUT_FILE` is used to store the results of this evaluation.

Example:

    python -m raopt.eval.execution_time -s 10000 output/case16/parameters_fold_1.hdf5 processed_csv/geolife/sdd_M16500_e0.1_1.csv tmp/execution_times.csv

### Adding a Dataset

Of course, it is possible to add further datasets not currently included. 
To do so, the following steps need to be complete:

1. In `raopt/utils/config.py`, the name of the new dataset (in capital letters) needs to be added to the list
`DATASETS` at the top of the file.
2. A section for the new dataset needs to created in `config/config.ini`.
Minimal keys are `MIN_LENGTH`, `MAX_LENGTH`, `OUTLIER_SPEED`, and `INTERVAL`.
3. The dataset needs to be converted into a CSV with columns `trajectory_id`, `uid`, `latitude`, and
`longitude`. This CSV has to be stored into `processed_csv/DATASET_NAME_IN_LOWERCASE/originals.csv`.
You might find the methods `raopt.utils.helpers.read_trajectories_from_csv` and
`raopt.utils.helpers.trajectories_to_csv` useful.
4. After these steps you can use `apply_mechanism` and `eval.main` as described above with the new
dataset name.

## Contact

**Author:** Erik Buchholz ([e.buchholz@unsw.edu.au](mailto:e.buchholz@unsw.edu.au))

**Supervision:**
- Prof. Salil Kanhere
- Dr. Surya Nepal

**Involved Researchers:**
- Dr. Sharif Abuadbba
- Dr. Shuo Wang

**Maintainer E-mail:** [e.buchholz@unsw.edu.au](mailto:e.buchholz@unsw.edu.au)

## References

We mainly referred to LSTM-TrajGAN [4] for our implementation.
The other references are used above in the text.

[1] J. Yuan et al., “T-drive,” in Proceedings of the 18th SIGSPATIAL International Conference on Advances in Geographic Information Systems - GIS ’10, New York, New York, USA, 2010, p. 99. doi: 10.1145/1869790.1869807.

[2] X. Zhou et al., “GeoLife: A Collaborative Social Networking Service among User, Location and Trajectory,” Bulletin of the Technical Committee on Data Engineering, vol. 33, no. 2, pp. 1–69, 2010, doi: 10.1.1.165.6100.

[3] K. Jiang, D. Shao, S. Bressan, T. Kister, and K.-L. Tan, “Publishing trajectories with differential privacy guarantees,” in Proceedings of the 25th International Conference on Scientific and Statistical Database Management - SSDBM, New York, New York, USA, 2013, p. 1. doi: 10.1145/2484838.2484846.

[4] J. Rao, S. Gao, Y. Kang, and Q. Huang, “LSTM-TrajGAN: A Deep Learning Approach to Trajectory Privacy Protection,” Leibniz International Proceedings in Informatics, LIPIcs, vol. 177, no. GIScience, pp. 1–16, 2020, doi: 10.4230/LIPIcs.GIScience.2021.I.12.
