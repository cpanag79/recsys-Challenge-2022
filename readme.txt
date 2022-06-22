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

**In the folder, there exists some .mat files that stored the dataset 
and some precomputed propabilities, similarities to speed up the computations.

