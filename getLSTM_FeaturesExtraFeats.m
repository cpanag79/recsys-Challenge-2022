function [feat] = getLSTM_FeaturesExtraFeats(N,PRM,SimLW,b,viewID,j0,Feats)

if N == 73
    [V1,V2] = getFeatDistVec(Feats.F{b},Feats.F{viewID});
    feat = V2;
    return;
end

feat = zeros(1,N);
k = 0;
for j=1:length(PRM)
   k = k+1;
   vec = PRM{j}(:,viewID);
   feat(k) = vec(b);
end
k = k+1;
feat(k) = SimLW(j0);

[V1,V2] = getFeatDistVec(Feats.F{b},Feats.F{viewID});
feat(k+1:N) = V1;

%feat(k+1:k+length(V1)) = V1;
%if N > 100
%   feat(length(V1)+1:N) = V2;
%end

%feat(N) = SimLW(j0);


function [feat] = getLSTM_FeaturesExtra2old(N,PRM,SimLW,b,viewID,j0,Feats)

feat = zeros(1,N);
k = 0;
for j=1:length(PRM)
    k = k+1;
    vec = PRM{j}(:,viewID);
    %feat(k) = vec(b)/max(0.0000001,max(vec));
   % k = k+1;
    feat(k) = vec(b);
end
%vec = PP1(:,viewID);
%k = k+1;
%feat(k) = vec(b)/max(0.0000001,max(vec));
%k = k+1;
%feat(k) = vec(b);

%[IoU,IoM] = getFeatDist(Feats.F{b},Feats.F{viewID},[],[]);
%k = k+1;
%feat(k) = IoU;
%k = k+1;
%feat(k) = IoM;

%if N > 19
%    [V1,V2] = getFeatDistVec(Feats.F{b},Feats.F{viewID});
%    feat(k+1:k+length(V1)) = V1;
%end
%if N > 100
%    feat(k+length(V1)+1:N-1) = V2;
%end
if N > length(PRM)
    feat(N) = SimLW(j0);
end






function [V1,V2] = getFeatDistVec(F1,F2)
    

s1 = sum(F1');
s2 = sum(F2');

V1 = min(s1,s2)./max(0.000001,max(s1,s2)); 

L = sum(F1'.*F2');

M = max(sum(F1'),sum(F2'));

V2 = L ./ max(0.00000001,M);
