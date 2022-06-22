function [PN] = correctEmbeddings(P,NoCD)
PN = full(P);

for i=1:size(P,2)
    PN(NoCD,i) = 0;
    s = sum(PN(:,i));
    if s > 0
        PN(:,i) = PN(:,i)/s;
    end
end
PN = sparse(PN);

