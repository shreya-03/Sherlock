This readme gives a brief outline for executing files to get the results and plots.

Note : All codes are written in python 2.7

Stage 1 : Detecting whether the stream is botted or not

For this stage, there is only one file for execution which is StreamClassification. The command for execution is :
python StreamClassification.py
This will give the results mentioned in the table 1 of the paper.

To get the plots in Figure 2, run plots.py file. In that file you can enter the filename of your choice at the end of the file. At line no 136 enter the merged data filename and at 137 enter the corresponding real data filename. Already the filenames are fed into the functions which gave the plots in the paper. If wish to experiment on other files, take them from path ./Data/Real Data/Merged Data/ and ./Data/Real Data.


Stage 2 : Determining the constituent bot users in detected botted stream from stage 1

There are basically 4 files mainly used for this stage namely modified_main, modified_UserClustering, entropy and UserBins amongst which main file is to be executed and remaining others are modules which are imported in the main file. 


modified_main is the main executable file which imports all other files. The command for executing is
python modified_main.py <Merged users file> <Real user file> <Bot file>

Here Merged user file comprises of both real users and bots and is present in Merged_Data repository which is present in Real Data repo of Data folder.
Real users file consists of legitimate real users and is present in Real data folder of Data folder.
Bot file has all bots and path is Data/Bot Data. For better results we considered random1 and chatterscontrolled.

After executing modofied_main file on one such set(Merged data file, corresponding real user data, corresponding bot data) of files, we will obtain plots in figure 3. 
To get the exact plots in figure 3 of the paper, the filenames are already fed to the function at lines 454-455.
 