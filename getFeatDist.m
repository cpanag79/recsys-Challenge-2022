function [IoU,IoM] = getFeatDist(F1,F2,V,VL)

if numel(F1) ~= numel(F2)
    IoU = 0;
    IoM = 0;
    return;
end
    

s1 = max(F1');
s2 = max(F2');

IoU = sum(min(s1,s2))/sum(max(s1,s2)); 


L = sum(sum(F1.*F2));

M = max(sum(F1(:)),sum(F2(:)));
IoM = L / M;



