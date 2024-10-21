# SpotV2Net

This repository supports the paper titled **"SpotV2Net: Multivariate Intraday Spot Volatility Forecasting via Vol-of-Vol-Informed Graph Attention Networks"**, authored by **Alessio Brini** and **Giacomo Toscano**. The paper is published in the *International Journal of Forecasting*.

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Instructions](#instructions)
3. [Data Availability](#data-availability)
4. [Computational Resources](#computational-resources)

## Repository Structure

The files in this repository are numbered in the order they should be executed. Below is the structure of the repository:

- `config/` - Configuration files.
- `utils/` - Utility functions for the following scripts.
- `1_downsample_TAQ_data.py` - First script to downsample TAQ data.
- `2_organize_prices_as_tables.py` - Organize prices into tables.
- `3_create_matrix_dataset.py` - Create the volatility and co-volatility of volatility matrix to be passed to the Pytorch Geometric dataset constructor.
- `4_standardize_data.py` - Standardize the data for neural network training.
- `5_train_LSTM_optuna.py` - Train LSTM model using `Optuna` (hyperparameter optimization).
- `5_train_SpotV2Net.py` - Train the RGNN model (single run, no hyperparameter optimization).
- `5_train_SpotV2Net_optuna.py` - Train RGNN using `Optuna` (hyperparameter optimization).
- `6_results.ipynb` - Jupyter notebook containing results.

## Instructions

The files are numbered to indicate the sequence in which they should be executed. Specific instructions are also embedded within each file. This section provides general guidance on what each file does and how to run them in order:

1. **First and Second Scripts**:
   - Start by running `1_downsample_TAQ_data.py` followed by `2_organize_prices_as_tables.py` in sequence. These scripts process the tick-by-tick data for the 30 DJIA constituents over a 3-year period.
   - The scripts filter data for NYSE during market hours and resample it to the daily level, storing the output in the folder `rawdata/taq/by_comp/`. This folder will contain 30 files, one for each company, named in the format `{COMPANY_NAME}_20_23`. An empty file is provided to illustrate the expected structure.
   - After organizing the prices into tables, you will need to use the *Fourier-Malliavin Volatility (FMVol) MATLAB library* referenced in the paper *Sanfelici, S., & Toscano, G. (2024)*, available [here](https://it.mathworks.com/matlabcentral/fileexchange/72999-fsda-flexible-statistics-data-analysis-toolbox).
   - The MATLAB package allows you to estimate univariate volatilities, multivariate volatilities, and co-volatilities. The results will be saved into four structured folders:
     - `processed_data/vol/` - Univariate volatilities (30 files, one per company).
     - `processed_data/covol/` - Multivariate co-volatilities (435 files, one for each entry in a 30x30 upper triangular matrix).
     - `processed_data/vol_of_vol/` - Univariate volatility of volatilities.
     - `processed_data/covol_of_vol/` - Multivariate co-volatility of volatilities (similar to the covolatility folder, 435 files).

2. **Third Script**:
   - The next step is to run `3_create_matrix_dataset.py`. This script starts from the structured data in the four folders mentioned above. It aggregates the volatilities, co-volatilities, and volatility of volatility into sequences of matrices. These matrices will be used in later steps to construct the graph dataset, as described in the paper.

3. **Fourth Script**:
   - Run `4_standardize_data.py` to standardize the data for neural network training. This script saves the parameters used for standardization, enabling you to rescale the output after training.

4. **Fifth Set of Scripts**:
   - There are three options for training neural network models in step 5:
     - `5_train_LSTM_optuna.py` uses the `Optuna` Python package for hyperparameter optimization of the LSTM model.
     - `5_train_SpotV2Net.py` trains the `SpotV2Net` model in a single run, without hyperparameter optimization.
     - `5_train_SpotV2Net_optuna.py` performs hyperparameter optimization on `SpotV2Net` using `Optuna`.
   - The configuration for these training processes is controlled by the YAML file in the `config/` folder. If running the `Optuna` version of `SpotV2Net`, the choice of the hyperparameter grid (specified below line 40) becomes important. For a single run, select specific hyperparameters above line 40.
   - The YAML file also specifies which data in H5 format (produced by `3_create_matrix_dataset.py`) to use for training.

5. **Sixth Script**:
   - `6_results.ipynb` is a Jupyter notebook that performs several tasks essential for generating the results presented in the paper:
     - It fits the Multivariate HAR model used in the paper. It also includes the hyperparameter optimization and testing of the `XGBoost` model, since this process does not require `Optuna` but can be done using `scikit-learn`.
     - The notebook provides evaluations for both single-step and multi-step forecasting models.
     - Additionally, it loads the results from the neural network runs, generates the figures from the paper, and produces values for the tables on losses, MCS, and DM tests.
   - The notebook is structured into clear sections to enhance readability, and it contains instructions for modifying parameters to explore different results.

## Data Availability

The data used in this reproducibility check comes from the **Trade and Quote (TAQ)** database via **WRDS**. The TAQ database contains daily intraday transactions data (trades and quotes) for all securities listed on the New York Stock Exchange (NYSE) and American Stock Exchange (AMEX), as well as Nasdaq National Market System (NMS) and SmallCap issues. 

The data must be purchased separately through WRDS, and we recommend running the query on WRDS once per year due to the large size of the data, especially since we are dealing with tick-by-tick data.

## Computational Resources

For training the GNN and LSTM models, we used a server equipped with an **NVIDIA GeForce RTX 2080 Ti** with 12 GB of memory. These models require substantial computational resources, and we recommend using a similar or higher-spec GPU for efficient training.

The **HAR** model and **XGBoost** can be run locally on a standard laptop without the need for extensive hardware requirements.

The required packages and their versions for running the code are listed below:

- `dask`: 2024.4.1
- `h5py`: 3.11.0
- `numpy`: 1.26.4
- `optuna`: 3.6.1
- `pandas`: 2.2.2
- `pandas_market_calendars`: 4.4.0
- `scikit-learn`: 1.2.2
- `torch`: 2.2.2
- `torch_geometric`: 2.3.0
- `tqdm`: 4.66.2
