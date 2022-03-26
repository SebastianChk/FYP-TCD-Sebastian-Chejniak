# Comments
This repository only includes the code used to train and evaluate the models to get the results in the thesis. The (countless) amounts of code used for troubleshooting and experimentation are omitted to keep everything tidy. I also omitted the code to create some of the plots

# Environment setup
Unfortunately gpflow seems to have conflicts with some of the other packages I used. Ensuring I had the correct packages when running the code was a task that caused me many issues. I ended up having to create two new anaconda environments, fyp_mxnet and fyp_gpflow. I attempted to create a requirements.txt file that one could easily use to import the packages in either of these environments, however I unfortunately did not succeed. So below is a general list of the important packages one needs to run the code. I may have missed some packages, but on attempting to run the code you will of course be informed if a necessary package is missing, so you can install it then.

To install gluonTS and mxnet, one needs something called C++ build tools installed.

## fyp_mxnet
gluonts, mxnet, pandas, numpy, scipy, sktime, properscoring

## fyp_gpflow
gpflow, pandas, numpy, scipy, click

# Credit
Some of the code is copied from/largely inspired by code I found online. Some of the .py files are entirely my own creations, but almost all of the code in the .ipynb files are my own. Here is a summary:
- src/data_preparation/
    - All mine
- src/models/
    - croston.py 
        - The implementation of the two croston methods (the functions Croston and Croston_TSB) is from https://towardsdatascience.com/croston-forecast-model-for-intermittent-demand-360287a17f5f
        - The rest of the code (implementing least squares, allowing the code to take in M5 data, etc.) is my own.
    - deepAR.ipynb, croston.ipynb
        - Entirely my own
    - gp_src/
        - GP.ipynb is my own, except for the part where 3 restarts are performed and the model is chosen based on the highest Evidence Lower Bound (ELBO) - My supervisor wrote this
        - timeseries/
            - Includes the implementation of the GP method, provided by my supervisor Alessio Benavoli. I did however modify the source code of this method in models/gaussian_process.py, as I discovered some bugs/errors in the implementation. Moreover, I adjusted the gaussian_process.py code to also provide a monte carlo sample of the predicted PMFs, not just the mean and variance estimates
- src/_metrics/
    - utils/
        - Code to aggregate the M5 data into the 12 levels (as discussed in thesis), for the purposes of finding the WRMSSE and WSPL.
        - evaluation.py and misc.py are from https://github.com/aldente0630/mofc-demand-forecast/tree/main/utils - used for WRMSSE
        - m5_helpers.py - used for WSPL - from https://www.kaggle.com/chrisrichardmiles/m5-helpers
    - reconciliation.py
        - Copies some code from m5_helpers.py, so that I could create reconciled sample forecasts
    - WSPL.ipynb, 305_metrics.ipynb, metrics.py
        - Entirely my own
    

# How to run
Obtain the data described in the next section and put it into data/external

Unless otherwise specified, simply run the following .py files and notebooks in the given order, in the fyp_mxnet environment:

data_preparation/deepar_prepare_data.py
   - Convert the M5 data into a gluonTS ListDataset object which one can input into deepAR in gluonTS
   
data_preparation/gp_prepare_data.ipynb
   - Creates time variable to be used as input to the GP method
   - Creates the sample of time series on which the models will be trained
   - Pickles sales_train_evaluation.csv for convenience
   
models/croston.ipynb
   - Runs both croston methods on the entire dataset (note: this is unnecessary, one only has to run it on the sample. It just happened that running it on the entire data and then subsetting that was most convenient for me. This should only take 1 or 2 hours to run anyway)

models/deepAR.ipynb
   - Trains deepAR and forecasts on the 305 time series subset
   
Switch to the fyp_gpflow environment and run: models/gp_src/GP.ipynb
   - Trains and forecasts the GP model on the 305 time series subset 
   
Switch back to the fyp_mxnet environment

models/_metrics/WSPL.ipynb 
   - See the markdown cell at the top of the notebook for an explanation on how to run it. This notebook was not designed to be run top-to-bottom
   - Extract and reconcile the quantile forecasts for deepAR and GP for the purposes for finding the WSPL. 
   - Extract sample forecasts from the forecasted pmf's of the GP method, for the purposes of finding the CRPS
   
models/_metrics/305_metrics.ipynb
   - Finds all the relevant metrics for all methods, as well as the point forecast metrics of a dummy forecast which forecasts all zeros

# Structure

## data
#### external
Data which I did not modify, but rather downloaded from the internet. Should contain the following files, which can be found in https://www.kaggle.com/c/m5-forecasting-accuracy:
- sales_train_evaluation.csv
- calendar.csv
- sell_prices.csv

and the following from https://www.kaggle.com/c/m5-forecasting-uncertainty:
- sample_submission.csv

#### interim 
Contains intermediate data that I created to help create the final(data/processed/) data

#### processed
Final data which I input into the models

## models
Contains folders corresponding to each trained model and its predictions

## src
Source code
#### data_preparation
Prepares data for input into deepAR and the GP method
#### models
Code which implements/trains models
#### _metrics
Implements metrics and evaluates metrics for predictions found in src/models

