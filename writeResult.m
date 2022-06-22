function [ok] = writeResult(TEST,RANK,SES_TEST,fname)
ok = 1;
N = size(SES_TEST,1);
M = zeros(100*N,3);

k = 0;
for i=1:N
    apo = SES_TEST(i,1);
    for j=1:100
        k = k+1;
        M(k,1) = TEST(apo,1);
        M(k,2) = RANK(i,j);
        M(k,3) = j;
    end
end

T = array2table(M);
T.Properties.VariableNames(1:3) = {'session_id','item_id','rank'};
writetable(T,fname);
