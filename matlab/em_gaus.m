function [Sig2,Hg,Eg,Lg,Lgte] = em_gaus(X,Xte,C,Sig2,alpha,beta)

[K,n] = size(C);
[~,D] = size(X);
[nte,~] = size(Xte);

B = zeros(K,K,D);
A = zeros(K,D);
Hg = zeros(K,K,n);
Eg = zeros(K,n);
Lg = zeros(n,1);
Lgte = zeros(nte,1);

for i=1:D
    Sinv = diag(1./Sig2(:,i));

    % E-step
    B(:,:,i) = (C*Sinv*C'+eye(K))\eye(K);    % posterior covariance of the latent variables
    A(:,i) = B(:,:,i)*C*Sinv*X(:,i);    % posterior mean of the latent variables

    % M-step
    for j=1:n
        mu = C(:,j)'*A(:,i);
        Sig2(j,i) = ( (X(j,i)-mu)^2 + C(:,j)'*B(:,:,i)*C(:,j) + 2/beta) / (2*(alpha+1)+1);
        % For coefficient update
        Hg(:,:,j) = Hg(:,:,j) + (B(:,:,i)+A(:,i)*A(:,i)')/Sig2(j,i);
        Eg(:,j) = Eg(:,j) + X(j,i)*A(:,i)/Sig2(j,i);
        
        Lg(j) = Lg(j) + log(normpdf(X(j,i),mu,sqrt(Sig2(j,i))));
        if j<=nte
            Lgte(j) = Lgte(j) + log(normpdf(Xte(j,i),mu,sqrt(Sig2(j,i))));
        end
    end
       
end