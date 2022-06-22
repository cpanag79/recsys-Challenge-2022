function [SimLW] = getSimWeights(views,lastView,Feats)

SimLW = zeros(1,length(views));
i1 = lastView;
F1 = Feats.F{i1};
for i=1:length(views)
    i2 = views(i);
    F2 = Feats.F{i2};
    [IoU,IoM] = getFeatDist(F1,F2,Feats.V,Feats.VL);
    %SimLW(i) = IoM*IoU^2;
    
    if IoU > 0.9
        SimLW(i) = IoM;
    else
        SimLW(i) = IoM/10;
    end
end

SimLW = SimLW/sum(SimLW);
