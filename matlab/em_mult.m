function [Fi,Del,Phi,Zt,Eta,P_eta,ch_Eta,Lm,Lmte] = em_mult(Z,Zte,C,Eta,P_eta,K,D)

Mi = diag(sum(Z,2));
[nte,~] = size(Zte);

Cc = C*Mi*C';
I = eye(K);
Fi = (Cc/2+I)\I;
G = Cc/(2*D);
Del = Fi*G*( I + (D-1)*((G+I)\I)*G )*Fi;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Posterior Mean of Latent Factor Scores
Zt = Z(:,1:end-1) - Mi*( P_eta - ( Eta-repmat(sum(Eta,2)/D,1,D-1) )/2 );
Phi = Fi*C*Zt + Del*C*repmat(sum(Zt,2),1,D-1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Eta_old = Eta;
Eta = C'*Phi;

exp_nor = 0;
oflow = max(max(Eta));
if oflow>=700    % to avoid exp(710)=inf
    exp_nor = oflow-700;        
end
Exp_eta = exp(Eta-exp_nor);
P_eta = Exp_eta./repmat(sum(Exp_eta,2)+exp(-exp_nor),1,D-1);

Lm = log(mnpdf(Z,[P_eta,1-sum(P_eta,2)]));
Lmte = log(mnpdf(Zte,[P_eta(1:nte,:),1-sum(P_eta(1:nte,:),2)]));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Eta = Eta./repmat(sqrt(sum(Eta.^2,2)),1,D)*100;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ch_Eta = norm(Eta-Eta_old,'fro')/norm(Eta_old,'fro');
if isnan(ch_Eta)
    keyboard
end