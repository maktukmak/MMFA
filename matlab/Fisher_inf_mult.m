function [FI]=Fisher_inf_mult(K,N,D,M,c)

% K = 5;
% N = 1e3;
% D = 4;
% M = 50;

V = randn(K,D,N);
Z = zeros(N,D+1);
Prob = zeros(N,D+1);
p_z = zeros(N,1);
Vsum = zeros(K,N);
P_z_der = zeros(K,N);

for i=1:N
    eta = c'*V(:,:,i);   
    prob = exp(eta)./(1+sum(exp(eta)));
    Prob(i,:) = [prob,1-sum(prob)];    
end

for i=1:N    
    Z(i,:) = mnrnd(M,Prob(i,:));
    Pz = mnpdf(repmat(Z(i,:),N,1),Prob);
    p_z(i) = mean(Pz);
    for j=1:N
        Vsum(:,j) = V(:,:,j)*(Z(i,1:D)-M*Prob(j,1:D))';
    end
    P_z_der(:,i) = Vsum*Pz/N;
end

FI = P_z_der*diag(p_z.^-2)*P_z_der'/N;
% FI_inv = inv(FI);