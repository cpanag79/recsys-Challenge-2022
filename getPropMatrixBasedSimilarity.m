%It computes the similarity based on given propability matrix

function [SIM] = getPropMatrixBasedSimilarity(PPP,NoCD)

N = size(PPP,1);
PR = PPP;

for i=1:size(PR,1)
    x = sum(PR(:,i));
    if x > 0
        PR(:,i) = PR(:,i)/x;
        PR(i,i) = 1;
        PR(:,i) = PR(:,i)/2;
    end
end

SIM = zeros(N,N);

for i=1:N
    if sum(PR(:,i)) == 0
        continue;
    end
    vec1 = PR(:,i);
    for j=i+1:N
        if sum(PR(:,j)) == 0
            continue;
        end
        vec2 = PR(:,j);
        %vec1(i) = vec2(i);
        %vec2(j) = vec1(j);
        %vec1(i) = 1;
        %vec2(j) = 1;
        %vec2 = vec2/sum(vec2);
        %vec1 = vec1/sum(vec1);
        SIM(i,j) = sum(vec1.*vec2);
        %SIM(j,i) = SIM(i,j);
        %vec1 = PR(:,i);
    end
   if rem(i,500) == 0
       i/N
   end
end
SIM = correctEmbeddings(SIM,NoCD);
SIM = sparse(SIM);



