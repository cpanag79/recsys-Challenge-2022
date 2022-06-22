function [Rank] = getRankPr2LSTM_SINGLE_CLASS9Feats(fWeight,MAXSIZE,net,Feats,TEST,SES_TEST,PR,PRsv,PRap,PRaa,PRaa2,PRap2,prop,NoCD,weights,PR1,PR2,SIM,PRaa_k1,PRaa_k2,PR_feats)
N = size(SES_TEST,1);
PR2 = PR2.^2;
[PP] = normalizeMatrix(PR2);

PRM{1} = PR;
PRM{2} = PRsv;
PRM{3} = PRap;
PRM{4} = PRap2;
PRM{5} = PRaa;
PRM{6} = PRaa2;
PRM{7} = PRaa_k1;
PRM{8} = PRaa_k2;

PRM{9} = SIM;
PRM{10} = PP;

for i=1:10
    PRM{i} = correctEmbeddings(PRM{i},NoCD);
end
PPP = weights(1)*PR;
for j=3:length(PRM)-2
    PPP = PPP+weights(j)*PRM{j};
end
SW = sum(weights)-weights(2);
PPP = PPP / SW;
[PPP] = correctEmbeddings(PPP,NoCD);
PP = correctEmbeddings(PP,NoCD);
PRM{2} = correctEmbeddings(PRM{2},NoCD);
SIM = correctEmbeddings(SIM,NoCD);
PR_feats = correctEmbeddings(PR_feats,NoCD);

%[PP1] = normalizeMatrix(PR1);

Rank = zeros(N,100);
%[~,pos] = sort(prop,'descend');
%pos0 = pos(1:100);
%PPP = weights(1)*PR+weights(3)*PRap+weights(4)*PRaa+weights(5)*PRaa2;
k = 1;
%get Test for LSTM
XTest = cell(1,MAXSIZE*N);
for i=1:N

    apo = SES_TEST(i,1);
    eos = SES_TEST(i,2);
    views = TEST(apo:eos,2);
    lastView = views(length(views));

    [SimLW] = getSimWeights(views,lastView,Feats);
    sw = 2;


    vec = 0.026*prop+(SimLW*PPP(:,views)')+sw*(SimLW*PP(:,views)');
    vec = vec+ (weights(2)/SW)*(PRM{2}(:,lastView)')+0.09*(SimLW*SIM(:,views)');

    vecF = 0*vec;
    for j=1:length(views)
        F1 = Feats.F{views(j)};
        F1 = F1(:);
        y = find(F1 == 1);
        vecF = vecF+sum(PR_feats(:,y)');
    end
    vecF = vecF+sum(PR_feats(:,y)');
    vecF = vecF/sum(vecF);
    vec = vec/sum(vec);
    vec = vec+0.12*vecF;

    %[~,pos] = sort(vec,'descend');
    %[propF] = getFeatureBasedProp(views, SimLW,Feats,pos(1:50),NMAX);
    %vec = vec+2*propF;

    propNew = vec;
    propNew(views) = -1;
    propNew(NoCD) = -1;
    [~,pos] = sort(propNew,'descend');

    vecC = cell(1,length(views));
    if net.Layers(1).InputSize == 10
        for j=1:length(views)
            vecC{j} = 0.026*prop+(SimLW(j)*PPP(:,views(j))')+sw*(SimLW(j)*PP(:,views(j))');
            vecC{j} = vecC{j}+ (weights(2)/SW)*(PRM{2}(:,views(j))')+0.09*(SimLW(j)*SIM(:,views(j))');
        end
    end

    for j1=1:MAXSIZE
        A = zeros(net.Layers(1).InputSize,length(views));
        for j=1:length(views)
            id = views(j); 
            [feat] = getLSTM_FeaturesExtraFeats(net.Layers(1).InputSize,PRM,SimLW,pos(j1),id,j,Feats);
            A(:,j) = feat;
        end
        XTest{k} = A;
        k = k+1;
    end
end

[~,POSTERIOR] = classify(net,XTest,'MiniBatchSize',100);

apo1 = 1;
eos1 = MAXSIZE;
for i=1:N
    apo = SES_TEST(i,1);
    eos = SES_TEST(i,2);
    views = TEST(apo:eos,2);
    %views = unique(TEST(apo:eos,2),'stable');
    lastView = views(length(views));
    %sw = 4*sum(weights);
    sw = 2;
    [SimLW] = getSimWeights(views,lastView,Feats);
    %SimLW = ones(1,length(SimLW));
    SimLW = SimLW/sum(SimLW);
    
    vec = 0.026*prop+(SimLW*PPP(:,views)')+sw*(SimLW*PP(:,views)');
    vec = vec+ (weights(2)/SW)*(PRM{2}(:,lastView)')+0.09*(SimLW*SIM(:,views)');
    vec = vec/sum(vec);

    vecF = 0*vec;
    for j=1:length(views)
        F1 = Feats.F{views(j)};
        F1 = F1(:);
        y = find(F1 == 1);
        vecF = vecF+sum(PR_feats(:,y)');
    end
    vecF = vecF+sum(PR_feats(:,y)');
    vecF = vecF/sum(vecF);
    vec = vec/sum(vec);
    vec = vec+0.12*vecF;


    %[~,pos] = sort(vec,'descend');
    %[propF] = getFeatureBasedProp(views, SimLW,Feats,pos(1:50),NMAX);
    %vec = vec+2*propF;
        
    propNew = vec;
    propNew(views) = -1;
    propNew(NoCD) = -1;
    [~,pos] = sort(propNew,'descend');

    propRF = POSTERIOR(apo1:eos1,2);

    propRF = propRF/max(0.00000001,sum(propRF));
    
    
    if fWeight == 0 %product rule
        for j=1:MAXSIZE
            vec(pos(j)) = 1000+vec(pos(j))*propRF(j);
        end
    else
        for j=1:MAXSIZE
            vec(pos(j)) = fWeight*vec(pos(j))+propRF(j);
        end
    end

    propNew = vec;
    propNew(views) = -1;
    propNew(NoCD) = -1;
    [~,pos] = sort(propNew,'descend');
    pos = pos(1:100);

    Rank(i,1:100) = pos;
    apo1 = eos1+1;
    eos1 = apo1+MAXSIZE-1;
end
