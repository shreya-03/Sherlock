This readme gives a brief outline for executing files to get the results and plots.  
Note : All code is written in Python 2.7

## Stage 1 : Detecting whether the stream is botted or not

For this stage, there is only one file for execution which is StreamClassification. The command for execution is: `python StreamClassification.py`. The code takes features extracted from the synthetic dataset as input (stored as a *csv* file), trains ML classifiers, and tests on the two sample files provided as test input. Both files are correctly classified by all the classifiers except the NN-MLP classifier.

To get the plots in **Figure 2c, 2d** (IMD bins vs. #IMDs in that bin) of the paper, run `python Plots.py` file. Sketches plots for the provided botted and real sample chatlogs.

## Stage 2 : Determining the constituent bot users in a flagged botted stream from stage 1

There are basically 4 files used for this stage - `main.py`, `UserClustering.py`, `Entropy.py` and `UserBins.py` amongst which the `main.py` file is to be executed and the remaining are dependent modules (their filenames indicate what they do).  
The command for executing is `python main.py`. By default, it is set to run on the provided `../Sample Data/sample_chatlog_chatbotted.txt` file. Outputs the plots in **Figure 3** of the paper and gives a list of users who are labelled as bots on the terminal. Performing stage 2 on the other file i.e. `../Sample Data/sample_chatlog_real.txt` will yield sub-par results as stage 2 is engineered on the assumption that a given stream has been flagged as botted by stage 1, and with a high probability, there will exist both types of users - bots and real. `sample_chatlog_real.txt` would not be flagged by stage 1 and hence stage 2 would not be performed on it.

## Baseline : A supervised method which uses Corrected Conditional Entropy (CCE) of IMDs and Message Lengths

Execute `python baseline_model.py`. Runs and reports the accuracy of the baseline model trained on the synthetic dataset (features stored as a *csv* file) and tested on the provided sample botted and real chatlogs. Refer **NOTE** in the `README.md` in the root directory for a note about the training dataset for the baseline approach.
