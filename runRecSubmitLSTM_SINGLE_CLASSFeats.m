function [fn] = runRecSubmitLSTM_SINGLE_CLASSFeats(LSTMname,fWeight,MAXSIZE,runFINALTEST)

load('dataMat.mat');
load('PPP.mat');
load('PR_12_feats.mat');
load('SIM_2_test.mat');

%parameters 
if isempty(LSTMname)
    LSTMname = 'LSTM_9_50_100_400'; %network
end
if isempty(fWeight)
    fWeight = 0.01;
end
if isempty(MAXSIZE)
    MAXSIZE = 100; 
end

fn = strcat('net',LSTMname,'.mat');
load(fn);
LSTMname
fWeight
MAXSIZE

%http://www.recsyschallenge.com/2022/dataset.html
%candidateitems, N x 1 (4990x1 double)
%itemfeatures,  N x 3  (item_id,feature_category_id, feature_value_id) 471.751x3 double
%MTEFS,  N x 3  (session_id, item_id, date) 226.138x3 table
%MTELS, N x 3  (session_id, item_id, date) 229.354x3 table
%MTRP, N x 3  (session_id, item_id, date) 1.000.000x3 table
%MTRS, N x 3  (session_id, item_id, date) 4.743.820x3 table

%create a new training set (about 10% of last sessions)



N = size(MTRP,1);
t = MTRP(:,3);
vec = sort(t);
UseSparse = 1;
sapo = 0.8;
NoCD = setdiff(itemfeatures(:,1),candidateitems);
t1 = vec(round(sapo*N));%parameter to change > 0.85 ????
t2 = max(vec);%parameter to change > 0.85 ????

TRAIN2 = find(t >= t1 & t <= t2);

CI = candidateitems;%candidateitems
%[prop] = getPropsBuy(CI,max(itemfeatures(:,1)),MTRP(TRAIN2,:));
NMAX = max(itemfeatures(:,1));
[propBV] = getPropsBuyView(CI,NMAX,SES_TR,MTRS,MTRP,TRAIN2);
prop = propBV;


[Feats] = getItemFeats(itemfeatures);

CI_B = CI;
CI_V = unique(MTRS(:,2));

[PR,~] = getPropA_B(SES_TR,MTRS,MTRP,TRAIN2,CI_B,CI_V,UseSparse); %propability
[PRsv,~] = getPropA_B_single_view(SES_TR,MTRS,MTRP,TRAIN2,CI_B,CI_V,UseSparse); %propability
[PRap,~] = getPropA_B_all_pairs(SES_TR,MTRS,MTRP,TRAIN2,CI_B,CI_V,UseSparse); %propability
[PRap2,~] = getPropA_B_all_pairs2(SES_TR,MTRS,MTRP,TRAIN2,CI_B,CI_V,UseSparse); %propability
[PRaa,~] = getPropA_A_1(SES_TR,MTRS,MTRP,TRAIN2,CI_B,CI_V,UseSparse); %propability
[PRaa2,~] = getPropA_A_12(SES_TR,MTRS,MTRP,TRAIN2,CI_B,CI_V,UseSparse); %propability
[PRaa_k1,~] = getPropA_A_1_k(SES_TR,MTRS,MTRP,TRAIN2,CI_B,CI_V,UseSparse,1); %propability
[PRaa_k2,~] = getPropA_A_1_k(SES_TR,MTRS,MTRP,TRAIN2,CI_B,CI_V,UseSparse,2); %propability

[PR_feats,~] = getPropA_Feats(SES_TR,MTRS,MTRP,TRAIN2,CI_B,CI_V,UseSparse,Feats);

%weights = [0.1350 0.1298  0.1400 0.1386 0.1116 0.1068];
weights = [0.1350 0.1298  0.1400 0.1386 0.1116 0.1068 0.1251 0.1072 0.5];

%writeResult(MTEFS,Rank_FS,SES_TEFS,'tefs.csv');

if runFINALTEST == 0 %Validation LS  set 
    [Rank_LS] = getRankPr2LSTM_SINGLE_CLASS9Feats(fWeight,MAXSIZE,net,Feats,MTELS,SES_TELS,PR,PRsv,PRap,PRaa,PRaa2,PRap2,prop,NoCD,weights,PR1,PR2,SIM,PRaa_k1,PRaa_k2,PR_feats);
    fn = strcat('tels',LSTMname,'__',num2str(MAXSIZE),'_',num2str(fWeight),'.csv');
    writeResult(MTELS,Rank_LS,SES_TELS,fn);
else  %final results  - FS set
    [Rank_FS] = getRankPr2LSTM_SINGLE_CLASS9Feats(fWeight,MAXSIZE,net,Feats,MTEFS,SES_TEFS,PR,PRsv,PRap,PRaa,PRaa2,PRap2,prop,NoCD,weights,PR1,PR2,SIM,PRaa_k1,PRaa_k2,PR_feats);
    fn = strcat('tefs',LSTMname,'__',num2str(MAXSIZE),'_',num2str(fWeight),'.csv');
    writeResult(MTEFS,Rank_FS,SES_TEFS,fn);
end

















