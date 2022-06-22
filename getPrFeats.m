function [PR1,PR2] = getPrFeats(Feats)

N = length(Feats.F);

PR1 = zeros(N,N);
PR2 = zeros(N,N);

for i=1:N
    F1 = Feats.F{i};
    for j=i+1:N
        F2 = Feats.F{j};
        [IoU,IoM] = getFeatDist(F1,F2,Feats.V,Feats.VL);
        if IoU > 0.6
            PR1(i,j) = IoU;
            PR2(i,j) = IoM;
        end
    end
end

PR1 = sparse(PR1);
PR2 = sparse(PR2);