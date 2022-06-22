# recsys22Challenge-
Code of team DataLab HMU.GR of RecSys Challenge 2022


You have to download first the .mat files that that store the dataset 
and some precomputed propabilities, similarities used to speed up the computations.
You can download them from https://teicrete-my.sharepoint.com/:u:/g/personal/cpanag_hmu_gr/EZMArO3-q9FJt-j_VM3f1IkB10c3hJ9qiOsPWg3jsUnlBA?e=nvhi3U

The proposed system can be executed according to the following two options: 

1. Training and Testing the whole system.
RUN: runRecLSTM_SINGLE_CLASS_Feats.m
The script runs the LSTM train and next it test the proposed system under initial Leaderboard
or Final Leaderboard datasets.

If the the parameter runFINALTEST is one, it selects the Final Leaderboard dataset.  

2. Testing the whole system.
RUN: If you want to test the pretrained LSTM with the Propability model to can use
the function:
runRecSubmitLSTM_SINGLE_CLASSFeats(LSTMname,fWeight,MAXSIZE,runFINALTEST);
with parameters: 
runRecSubmitLSTM_SINGLE_CLASSFeats('LSTM_73_5_100_100_400',8,150,1);
