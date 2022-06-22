%TRAIN LSTM 

close all;
clear all;
load('dataMat.mat');
%load('PPP.mat');
load('PR_12_feats.mat');
%load('SIM.mat');
load('SIM_2.mat');
SIM = SIM+SIM';

%Parameters

RUNTEST = 1;
NUMFEATS = 73;
APO_F = 100;
EOS_F = 400;

%load('PR_I_Fast_338.97.mat');
%load('PR_I_FastAllViews_365.02.mat');
%PR_I = PR_I_T;
%PR_I(isnan(PR_I)) = 0;

%http://www.recsyschallenge.com/2022/dataset.html
%candidateitems, N x 1 (4990x1 double)
%itemfeatures,  N x 3  (item_id,feature_category_id, feature_value_id) 471.751x3 double
%MTEFS,  N x 3  (session_id, item_id, date) 226.138x3 table
%MTELS, N x 3  (session_id, item_id, date) 229.354x3 table
%MTRP, N x 3  (session_id, item_id, date) 1.000.000x3 table
%MTRS, N x 3  (session_id, item_id, date) 4.743.820x3 table

tic;
%create a new training set (about 10% of last sessions)
UseSparse = 1;
t = MTRP(:,3);
vec = sort(t);
t0 = vec(round(0.9*length(vec)));
XV = 0.75;
i10 = zeros(1,100);

[Feats] = getItemFeats(itemfeatures);
PR2 = PR2.^2;

[PP] = normalizeMatrix(PR2);
[PP1] = normalizeMatrix(PR1);

dt = XV;

t1 = vec(round(dt*length(vec)));
t2 = t0;

TRAIN2 = find(t >= t1 & t < t2);

TRAIN = find(t < t0);
TEST = setdiff([1:length(t)],TRAIN);
SES_TEST = MTRP(TEST,1);
SES_TRAIN = MTRP(TRAIN2,1);


CI_B = unique(MTRP(TEST,2));%candidateitems  BUY
CI_V = unique(MTRP(TRAIN2,2));%candidateitems VIEW

NoCD = setdiff(itemfeatures(:,1),CI_B);

NMAX = max(itemfeatures(:,1));
%[prop] = getPropsBuy(CI_B,NMAX);
[propBV] = getPropsBuyView(CI_B,NMAX,SES_TR,MTRS,MTRP,TRAIN2);
prop = propBV;

[PR,~] = getPropA_B(SES_TR,MTRS,MTRP,TRAIN2,CI_B,CI_V,UseSparse); %propability
%    [PR2,~] = getPropA_B2(SES_TR,MTRS,MTRP,TRAIN2,CI_B,CI_V,UseSparse); %propability
[PRsv,~] = getPropA_B_single_view(SES_TR,MTRS,MTRP,TRAIN2,CI_B,CI_V,UseSparse); %propability
%    [PRsv2,~] = getPropA_B_single_view2(SES_TR,MTRS,MTRP,TRAIN2,CI_B,CI_V,UseSparse); %propability
[PRap,~] = getPropA_B_all_pairs(SES_TR,MTRS,MTRP,TRAIN2,CI_B,CI_V,UseSparse); %propability
[PRap2,~] = getPropA_B_all_pairs2(SES_TR,MTRS,MTRP,TRAIN2,CI_B,CI_V,UseSparse); %propability
[PRaa,~] = getPropA_A_1(SES_TR,MTRS,MTRP,TRAIN2,CI_B,CI_V,UseSparse); %propability
[PRaa2,~] = getPropA_A_12(SES_TR,MTRS,MTRP,TRAIN2,CI_B,CI_V,UseSparse); %propability

PRM{1} = PR;
PRM{2} = PRsv;
PRM{3} = PRap;
PRM{4} = PRap2;
PRM{5} = PRaa;
PRM{6} = PRaa2;
PRM{7} = SIM;
PRM{8} = PP;

for i=1:8
    PRM{i} = correctEmbeddings(PRM{i},NoCD);
end
[~,pos] = sort(prop,'descend');
Rank = zeros(length(TEST),100);
%pos0 = pos(1:100);
% weights = [0.1350 0.1298  0.1400 0.1386 0.1 0.1068 ];
weights = [0.1350 0.1298  0.1400 0.1386 0.1116 0.1068 0.5];
%weights = [0.1350 0.1298  0.1400 0.1386 0.1116 0.1068];
%    weights = [1 1  1 1 1 1];

%Pefomance (one to six feats): 0.1350    0.1445    0.1489    0.1528 0.1561 0.1567
%weights = [0.1344 0.1273 0.1388 0.1124];
%PPP = weights(1)*PR+weights(3)*PRap+weights(4)*PRaa+weights(5)*PRaa2+weights(6)*PRaa2;

PPP = weights(1)*PR;
for j=3:length(PRM)-2
    PPP = PPP+weights(j)*PRM{j};
end
SW = sum(weights)-weights(2);
PPP = PPP / SW;
[PPP] = correctEmbeddings(PPP,NoCD);

PRM{2} = correctEmbeddings(PRM{2},NoCD);


SIM = correctEmbeddings(SIM,NoCD);
PPP = correctEmbeddings(PPP,NoCD);
PP = correctEmbeddings(PP,NoCD);


XTrain = [];
YTrain = [];
k = 1;
for i=1:length(TRAIN2)
 
    id = TRAIN2(i);
    b0 = MTRP(id,2);
    apo = SES_TR(id,1);
    eos = SES_TR(id,2);
    views = MTRS(apo:eos,2);
    lastView = views(length(views));
    [SimLW] = getSimWeights(views,lastView,Feats);
    sw = 2;


    vec = 0.026*prop+(SimLW*PPP(:,views)')+sw*(SimLW*PP(:,views)');
    vec = vec+ (weights(2)/SW)*(PRM{2}(:,lastView)')+0.09*(SimLW*SIM(:,views)');
    vec1 = vec;
    propNew = vec;
    propNew(views) = -1;
    propNew(NoCD) = -1;
    propNew(b0) = -1;
    [~,pos] = sort(propNew,'descend');

   % vecC = cell(1,length(views));
   % if NUMFEATS == 10
   %     for j=1:length(views)
   %         vecC{j} = 0.026*prop+(SimLW(j)*PPP(:,views(j))')+sw*(SimLW(j)*PP(:,views(j))');
   %         vecC{j} = vecC{j}+ (weights(2)/SW)*(PRM{2}(:,views(j))')+0.09*(SimLW(j)*SIM(:,views(j))');
   %     end
   % end


    for ex = 1:4
        bbest = pos(1);
        if ex == 1
            b = b0;
       % elseif ex == 2 && b0 ~= bbest
       %     b = bbest;
        else
            while 1
                b = pos(APO_F+randi(EOS_F));
                if b ~= b0
                    break;
                end
            end
        end
        A = zeros(NUMFEATS,length(views));
        for j=1:length(views)

            id = views(j);
            [feat] = getLSTM_FeaturesExtraFeats(NUMFEATS,PRM,SimLW,b,id,j,Feats);
            A(:,j) = feat;
        end
        XTrain{k} = A;

        if ex == 1
            YTrain{k} = sprintf('%d',1);
       % elseif ex == 2 && b0 ~= bbest
       %     YTrain{k} = sprintf('%d',1);
        else
            YTrain{k} = sprintf('%d',0);
        end

        k = k+1;

    end
end
YTrain = categorical(YTrain)';


numClasses = 2;
%LSTM parameters 
numFeatures = size(XTrain{1},1);
numHiddenUnits = 100;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','last')
%    dropoutLayer(0.2)
%    lstmLayer(round(0.75*numHiddenUnits),'OutputMode','sequence')
%    dropoutLayer(0.2)
%    lstmLayer(round(0.5*numHiddenUnits),'OutputMode','last')
%    dropoutLayer(0.2)
    %lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

maxEpochs = 25;
miniBatchSize = 100;


options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ... %  'GradientThreshold',1, ...'Verbose',false, ...
    'Plots','training-progress');



numObservations = numel(XTrain);
for i=1:numObservations
    sequence = XTrain{i};
    sequenceLengths(i) = size(sequence,2);
end

[sequenceLengths,idx] = sort(sequenceLengths);
XTrain = XTrain(idx);
YTrain = YTrain(idx);


if 1
    SetVal = randi(numel(XTrain),round(0.1*numel(XTrain)),1);
    XValidation = XTrain(SetVal);
    YValidation = YTrain(SetVal);
    MM = numel(XTrain);
    XTrain = XTrain(setdiff([1:MM],SetVal));
    YTrain = YTrain(setdiff([1:MM],SetVal));
    
    options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ... %  'GradientThreshold',1, ...'Verbose',false, ...
        'ValidationData',{XValidation,YValidation},...
        'OutputNetwork','best-validation-loss',...
        'Plots','training-progress');
end

net = trainNetwork(XTrain,YTrain,layers,options);

timeVal = toc
ss = [];
ss{1} = net.Layers(1).InputSize;
ss{2} = length(net.Layers);
ss{3} = net.Layers(2).NumHiddenUnits;
ss{4} = APO_F;
ss{5} = EOS_F;
LSTMname = 'LSTM';

for i=1:length(ss)
    ss{i} = num2str(ss{i});
    LSTMname = strcat(LSTMname,'_',ss{i});
end
fn = strcat('net',LSTMname,'.mat')
save(fn, 'net');


if RUNTEST == 1 %TEST LSTM + PROP MODEL
    fWeight = 8;
    MAXSIZE = 150;
    runFINALTEST = 0;
    [fn] = runRecSubmitLSTM_SINGLE_CLASSFeats(LSTMname,fWeight,MAXSIZE,runFINALTEST);
end








