This readme gives a brief outline for executing files to get the results and plots.  
Note : All code is written in python 2.7

## Stage 1 : Detecting whether the stream is botted or not

For this stage, there is only one file for execution which is StreamClassification. The command for execution is: `python StreamClassification.py`
This will give the results mentioned in the **Table 1** of the paper.

To get the plots in **Figure 2**, run the `Plots.py` file. In that file you can enter the filename of your choice at the end of the file. This code runs on files from the synthetic dataset. At line no. *136* enter the *merged data filename* (the generated synthetic chatlog) and at *137* enter the corresponding *real data filename* (the real stream chatlog which was used for generation of the synthetic chatlog). The functions have been set with the filenames which gave the plots in the paper.

## Stage 2 : Determining the constituent bot users in a flagged botted stream from stage 1

There are basically 4 files used for this stage - `main.py`, `UserClustering.py`, `Entropy.py` and `UserBins.py` amongst which the `main.py` file is to be executed and the remaining are dependent modules (their filenames indicate what they do).  
The command for executing is `python main.py`. By default, it is set to run on the provided `../Sample Data/sample_chatlog_chatbotted.txt` file. Outputs the plots in Figure 3 of the paper and gives a list of users who are labelled as bots on the terminal. Performing stage 2 on the other file i.e. `../Sample Data/sample_chatlog_real.txt` will yield sub-par results as stage 2 is engineered on the assumption that a given stream has been flagged as botted by stage 1, and with a high probability, there will exist both types of users - bots and real. `sample_chatlog_real.txt` would not be flagged by stage 1 and hence stage 2 wouldn't be performed on it.

## Baseline : A supervised method which uses Corrected Conditional Entropy (CCE) of IMDs and Message Lengths

Execute `filename.py` Describe
