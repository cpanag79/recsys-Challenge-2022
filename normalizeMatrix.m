function [PN] = normalizeMatrix(P)
PN = full(P);

for i=1:size(PN,1)
    x = sum(P(:,i));
    if x > 0
        PN(:,i) = P(:,i)/x;
    end
end

PN = sparse(P);
