%Computes the propability Matrix P(A/B) where A is a product that is bought
%under the assupmption that B has been viewed 

function [PR,PR_B] = getPropA_B(SES_TR,MTRS,MTRP,TRAIN2,CI_B,CI_V,UseSparse)

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
    for j=1:length(views)
        PR(b,views(j)) = PR(b,views(j))+1;
        NumView(views(j)) = NumView(views(j))+1;
    end
end

for i=1:N2
    if NumView(i) > 0
        PR(:,i) = PR(:,i)/NumView(i);
    end
end
PR_B = NumView/sum(NumView);

%PR = sparse(PR);

if UseSparse == 1
    PR = sparse(PR);
end



