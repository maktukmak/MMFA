clear

K = 3;
N = 2e3;
Dg = 5;
Dm = 5;
M = 40;
n = 100;
nte = 10;
max_iter = 2e2;
lam = 1e-6;
alpha = 1e0;
beta = 10^-1;

tr = 1e1;
L_av = zeros(max_iter,tr);
Lte_av = zeros(max_iter,tr);
mse = zeros(max_iter,tr);

A = randn(K,Dg);
B = repmat(eye(K),1,1,Dg);
sig2 = ones(Dg,1);

C = randn(K,n);
C = C./repmat(sqrt(sum(C.^2)),K,1);
    
for i=1:tr

%     C = rotatefactors(C');
%     C = C';
%     c = C(:,1); 

    [mse(:,i),L,Lte,Lr,Lrte,Ce] = coeff_gaus_mult(C,Dg,Dm,M,max_iter,lam,nte,alpha,beta);
    
    % figure, plot(mse)
    % hold on, plot(repmat(mmse,length(mse),1),'r--')
    
%     figure, plot(exp(L(max_iter*.05:end)-Lr))
    % figure, plot(exp(L(30:110)-Lr))
    % hold on, plot(repmat(Lr,length(L(max_iter*.1:end)),1),'r--')
%     figure, plot(exp(Lte(max_iter*.05:end)-Lrte))
    % figure, plot(exp(Lte(30:110)-Lrte))
    % hold on, plot(repmat(Lrte,length(Lte(max_iter*.1:end)),1),'r--')
    
    L_av(:,i) = L-Lr;
    Lte_av(:,i) = Lte-Lrte;    

    trial=i
end

L_avg = mean(L_av,2);
Lte_avg = mean(Lte_av,2);
figure, plot(exp(L_avg))
figure, plot(exp(Lte_avg))

[FI_gaus]=Fisher_inf_gaus(C(:,1),A,B,sig2);
[FI_mult]=Fisher_inf_mult(K,N,Dm,M,C(:,1));
% % FI_mult = 0;
FI_inv = inv(FI_gaus+FI_mult);
mmse = trace(FI_inv);
mmse_g = trace(inv(FI_gaus));
mmse_m = trace(inv(FI_mult));

mse_avg = mean(mse,2);
figure, plot(mse_avg(1:end),'linewidth',2)    
hold on, plot(repmat(mmse,max_iter,1),'r--','linewidth',2)
hold on, plot(repmat(mmse_g,max_iter,1),'k:','linewidth',2)
hold on, plot(repmat(mmse_m,max_iter,1),'m-.','linewidth',2)
% hold on, plot(repmat(.15,max_iter,1),'r--')
leg = legend('MMFA','CRLB','CRLB-gaus','CRLB-mult');
set(leg,'fontsize',14,'interpreter','latex')
xlabel('iterations','fontsize',16,'interpreter','latex')
ylabel('$E[\|\mathbf{c}_i-\hat{\mathbf{c}}_i\|^2]$','fontsize',16,'interpreter','latex')
