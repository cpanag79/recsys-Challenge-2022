function [Feats] = getItemFeats(itemfeatures)
IF = itemfeatures;
NUM_ITEMS = max(IF(:,1));
NUM_FEATS = max(IF(:,2));

for i=1:NUM_FEATS
    v = find(IF(:,2) == i);
    g = IF(v,3);
    ValsL(i) = length(unique(g));
    Vals{i} = unique(g);
end

NUM_VALS = max(ValsL);

Feats = cell(NUM_ITEMS,1);
%Feats = zeros(NUM_ITEMS,NUM_FEATS,NUM_VALS);

pid = unique(IF(:,1));
for i=1:length(pid)
    Feats{pid(i)} = zeros(NUM_FEATS,NUM_VALS);
end

for i=1:size(IF,1)
    pid = IF(i,1);
    x = IF(i,2);
    y = IF(i,3);
    z = Vals{x}(1:ValsL(x));
    pos = find(z == y);
    Feats{pid}(x,pos(1)) = 1;
end

F = Feats;
Feats = [];
Feats.F = F;
Feats.V = Vals;
Feats.VL = ValsL;





