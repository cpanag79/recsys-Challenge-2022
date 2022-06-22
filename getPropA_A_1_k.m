%Computes the propability Matrix P(A/B) where A is a product that is viewed
%at time t and B is the object that is viewed at time t-1

function [PR,PR_V] = getPropA_A_1_k(SES_TR,MTRS,MTRP,TRAIN2,CI_B,CI_V,UseSparse,kd)

N1 = max(CI_B);
N2 = max(CI_V);
N1 = max(N1,N2);
N2 = max(N1,N2);

PR = zeros(N1,N2);
NumView = zeros(N2,1);
for i=1:length(TRAIN2)
    id = TRAIN2(i);
    b = MTRP(id,2);
    apo = SES_TR(id,1);
    eos = SES_TR(id,2);
    %views = unique(MTRS(apo:eos,2),'stable');
    views = MTRS(apo:eos,2);
    for j=kd+2:length(views)
        PR(views(j),views(j-1-kd)) = PR(views(j),views(j-1-kd))+1;
        NumView(views(j-1-kd)) = NumView(views(j-1-kd))+1;
    end
   %  PR(b,(length(views))) = PR(b,(length(views)))+1;
   %  NumView((length(views))) = NumView((length(views)))+1;
   j = length(views);
   if j > kd
       PR(b,views(j-kd)) = PR(b,views(j-kd))+1;
       NumView(views(j-kd)) = NumView(views(j-kd))+1;
   end
end

for i=1:N2
    if NumView(i) > 0
        PR(:,i) = PR(:,i)/NumView(i);
    end
end
PR_V = NumView/sum(NumView);

%PR = sparse(PR);


if UseSparse == 1
    PR = sparse(PR);
end


