function [FI]=Fisher_inf_gaus(c,A,B,sig2)

D = size(A,2);
var = zeros(D,1);
FI = 0;

for i=1:D
    var(i) = c'*B(:,:,i)*c+sig2(i);
    FI = FI + (A(:,i)*A(:,i)'/var(i)) + 2*(B(:,:,i)*c)*(c'*B(:,:,i))/var(i)^2;
end