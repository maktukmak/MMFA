function [mse,L,Lte,Lr,Lrte,Ce] = coeff_gaus_mult(C,Dg,Dm,M,max_iter,lam,nte,alpha,beta)

% Data Generation
[K,n] = size(C);
Sig2 = ones(n,Dg);

U = randn(K,Dg);
V = randn(K,Dm-1);

mX = C'*U;
X = normrnd(mX,sqrt(Sig2));
Xte = normrnd(mX(1:nte,:),sqrt(Sig2(1:nte,:)));
LX = sum(sum(log(normpdf(X,mX,sqrt(Sig2)))));
LXte = sum(sum(log(normpdf(Xte,mX(1:nte,:),sqrt(Sig2(1:nte,:))))));
% LX = sum(sum(log(normpdf(X,zeros(n,Dg),sqrt(repmat(diag(C'*C),1,Dg)+Sig2)))));
Eta_real = C'*V;
Z_score = exp(Eta_real);
Z_prob = Z_score./repmat(1+sum(Z_score,2),1,Dm-1);
Eta = Eta_real;
P_eta = Z_prob;
Z_prob = [Z_prob,1-sum(Z_prob,2)];
Z = mnrnd(M,Z_prob);
Zte = mnrnd(M,Z_prob(1:nte,:));
LZ = sum(sum(log(mnpdf(Z,Z_prob))));
LZte = sum(sum(log(mnpdf(Zte,Z_prob(1:nte,:)))));
Lr = (LX/Dg+LZ/Dm)/n;
Lrte = (LXte/Dg+LZte/Dm)/nte;
Ce = rand(K,n);
Ce = Ce./repmat(sqrt(sum(Ce.^2)),K,1);
L = zeros(max_iter,1);
Lte = zeros(max_iter,1);
mse = zeros(max_iter,1);
ch_Eta = zeros(max_iter,1);

% EM iterations
for iter=1:max_iter
    [Sig2,Hg,Eg,Lg,Lgte] = em_gaus(X,Xte,Ce,Sig2,alpha,beta);
    [Fi,Del,Phi,Zt,Eta,P_eta,ch_Eta(iter),Lm,Lmte] = em_mult(Z,Zte,Ce,Eta,P_eta,K,Dm);
    Phis = sum(Phi,2);
    H = M/2 * ( Fi*(Dm-1)^2/Dm + Del*(Dm-1)/Dm + Phi*Phi' - Phis*Phis'/Dm );
%     H = 0;
    for i=1:n
        H = H+Hg(:,:,i);       
        H = H+eye(K)*lam*mean(diag(H));
        Em = Phi*Zt(i,:)';
%         Em = 0;
        e = Em + Eg(:,i);
%         e = Em;
        Ce(:,i) = H\e;
    end
%     Ce = Ce./repmat(sqrt(sum(Ce.^2)),K,1);
%     Ce = rotatefactors(Ce');
%     Ce = Ce';
    
%     Lg = 0;
    L(iter) = sum(Lg/Dg+Lm/Dm)/n;
    Lte(iter) = sum(Lgte/Dg+Lmte/Dm)/nte;
    %mse(iter) = mean(sum(Ce-C).^2);
    mse(iter) = mean(sum((Ce-C).^2));
    
%     if rem(iter,10)==0
%         fprintf('iter=%d, L=%f, L1=%f\n',iter,L(iter),Lte(iter))
%     end
end
