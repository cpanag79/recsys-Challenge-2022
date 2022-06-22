
%Computes the propability to buy a product according to the total sales

function [prop] = getPropsBuyView(candidateitems,N,SES_TR,MTRS,MTRP,TRAIN2)
prop = zeros(1,N);

for i=1:length(TRAIN2)
    id = TRAIN2(i);
    b = MTRP(id,2);
    apo = SES_TR(id,1);
    eos = SES_TR(id,2);
    %views = unique(MTRS(apo:eos,2),'stable');
    views = MTRS(apo:eos,2);
    views(length(views)+1) = b;
    for j1=1:length(views)
        c = views(j1);
        prop(c) = prop(c)+1;
    end
end

vec = zeros(1,N);
vec(candidateitems) = 1;
prop = prop.*vec;

prop = prop / sum(prop);