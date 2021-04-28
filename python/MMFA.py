import numpy as np
from numpy.linalg import inv
from utils import softmax
from utils import logsumexp
from utils import logdet

class MMFA(object):
    
    def __init__(self, I, Du, Mu, K = 10, MM = 0):
        self.MM = MM
        self.logeval = 0
        self.modelparams = self.model_params(Du, Mu, K, I)
        self.latentparams = self.latent_params(self.modelparams, I)
    
    class model_params:
        def __init__(self, D, M, K, I):
            self.D = D
            self.M = M
            self.I = I
            self.K = K
            self.W = np.random.multivariate_normal(np.zeros(K), np.identity(K), D)
            self.H = np.random.multivariate_normal(np.zeros(K), np.identity(K), np.sum(M))
            self.Sigma_x = np.identity(D)
            self.Prec_x = np.identity(D)       
            self.U_mean_prior = 1*np.random.normal(0, 1, K)
            #self.U_mean_prior = np.zeros(K)
            self.nu0 = -(self.K+1)
            self.alpha = 0.00
            self.alpha_mu = 0
            self.param_a = 3
            self.param_b = 0.5
            self.s0 = 0 * np.eye(self.K)
            self.Xon = 1
            self.Yon = 1
            self.LogLikOn = 1
            self.F_u = []
            for i in range(0,M.shape[0]):
                self.F_u.append(1/2 * (np.identity(M[i]) - (1/(M[i]+1)) * np.ones((M[i],1)) * np.ones((M[i],1)).T))
        
    class latent_params:
        def __init__(self, modelparams, I):
            self.SS_SecMoment = 0
            self.SS_Mean = 0
            self.U_mean = np.tile(modelparams.U_mean_prior[:,None], [1,I])
            self.Psi_u = modelparams.H @ self.U_mean
            self.Prec_u_prior = 1 * np.identity(modelparams.K)
            self.U_SecMoment = np.tile(self.Prec_u_prior + modelparams.U_mean_prior[:,None].T @ modelparams.U_mean_prior[:,None], [I, 1, 1])
            self.LogLik = 0
            self.LogRating = 0
            self.Psd_X = np.zeros((np.sum(modelparams.M) + modelparams.D, I))
            self.Sigma_u = np.zeros((I, modelparams.K, modelparams.K))

            
    def e_step(self, modelparams, latentparams, X, Y, MM):
          
        # Fetch model parameters
        M = modelparams.M
        #I = modelparams.I
        I = X.shape[1]
        D = modelparams.D
        K = modelparams.K
        H = modelparams.H
        W = modelparams.W
        Prec_x = modelparams.Prec_x
        Sigma_x = modelparams.Sigma_x
        U_mean_prior = modelparams.U_mean_prior
        Xon = modelparams.Xon
        Yon = modelparams.Yon
        F_u = modelparams.F_u
        
        # Infer posterior covariance and precision
        Sigma_u = np.zeros((I, K, K))
        InfPrec = latentparams.Prec_u_prior.copy()
        if Xon == 1:
            InfPrec = InfPrec + (W.T @ Prec_x @ W)
        if Yon == 1:
            InfCat = 0
            for i in range(0,M.shape[0]):
                ind = range(np.sum(M[0:i]), np.sum(M[0:i+1]))
                InfCat = InfCat +  H[ind,:].T @ F_u[i] @ H[ind]
            InfPrec = InfPrec + InfCat 
        Sigma_u = inv(InfPrec)
        
        # Infer posterior mean
        if Yon == 1:
            iter_psi = 10 # Iterations of bound convergence for categorical info
        else:
            iter_psi = 1 # No bound iteration in the absence of categorical info
        

        for iterPsi in range(0,iter_psi):
            G_u = np.zeros((np.sum(M), I))
            for i in range(0,M.shape[0]):
                ind = range(np.sum(M[0:i]), np.sum(M[0:i+1]))
                Psi_u_d = softmax(latentparams.Psi_u[ind, :])
                G_u[ind,:] = F_u[i] @ latentparams.Psi_u[ind, :] - Psi_u_d[0:-1, :]

            SS_SecMoment = np.zeros((K, K))
            InfSum = np.zeros((K, I))
            if MM == 0:
                InfSum = InfSum + (latentparams.Prec_u_prior @ U_mean_prior)[None].T
            if Xon == 1:
                InfSum = InfSum + (W.T @ (X / np.diag(Sigma_x)[None].T))
            if Yon == 1:
                InfSum = InfSum + (H.T @ (Y + G_u))
           
            #U_mean = np.einsum('ijk,ik->ij', Sigma_u, InfSum.T).T
            U_mean = (Sigma_u @ InfSum)
            if MM == 0:
                U_SecMoment = Sigma_u + np.einsum('ijk,ikl->ijl', U_mean[None].T, np.reshape(U_mean.T, (I, 1, K)))
            else:
                U_SecMoment = np.einsum('ijk,ikl->ijl', U_mean[None].T, np.reshape(U_mean.T, (I, 1, K)))
            SS_SecMoment = np.sum(U_SecMoment, axis = 0)
        
            Psi_u_old = latentparams.Psi_u.copy()
            Psi_u = H @ U_mean
            latentparams.Psi_u = Psi_u
            conv = np.sum((Psi_u_old - Psi_u)**2) / (Psi_u.shape[0] * Psi_u.shape[1])
            #print(conv)
            if conv < 1e-5:
                #print("Converged")
                break;
        
        
        # Fuse multimodal observations
        Psd_Cov = np.zeros((np.sum(M) + D, np.sum(M) + D))
        Psd_Prec = np.zeros((np.sum(M) + D, np.sum(M) + D))
        Psd_X = np.zeros((np.sum(M) + D, I))
        Psd_X[0:D, :] = X

        Psd_Cov[0:D,0:D] =  Sigma_x
        Psd_Prec[0:D,0:D] = Prec_x
        for i in range(0,M.shape[0]):
            ind = range(np.sum(M[0:i]), np.sum(M[0:i+1]))
            Psi_u_d = softmax(Psi_u[ind, :])
            G_u[ind,:] = F_u[i] @ Psi_u[ind, :] - Psi_u_d[0:-1, :]
            ind_tilde = range(np.sum(M[0:i]) + D, np.sum(M[0:i+1]) + D)
            Y_tilde = inv(F_u[i]) @ (Y[ind, :] + G_u[ind,:]) 
            Psd_X[ind_tilde, :] = Y_tilde
            Psd_Cov[np.ix_((ind_tilde),(ind_tilde))] = inv(F_u[i])
            Psd_Prec[np.ix_((ind_tilde),(ind_tilde))] = F_u[i]
            
        latentparams.SS_Mean = Psd_X @ U_mean.T
        latentparams.Psi_u = Psi_u
        latentparams.SS_SecMoment =  SS_SecMoment
        latentparams.U_SecMoment = U_SecMoment
        latentparams.U_mean = U_mean
        latentparams.Psd_X = Psd_X
        latentparams.Sigma_u = Sigma_u

    
    def m_step(self, modelparams, latentparams, X, Y):
        
            YY = np.sum(X * X, axis = 1)
            D = modelparams.D
            
            U_mean_prior = np.mean(latentparams.U_mean,axis = 1)
            
            #Beta = latentparams.SS_Mean @ (inv(latentparams.SS_SecMoment) + modelparams.alpha * np.eye(modelparams.K))
            W = latentparams.SS_Mean[0:D, :] @ (inv(latentparams.SS_SecMoment) + modelparams.alpha * np.eye(modelparams.K))
            #W = Beta[0:D, :]
            #Sigma_x = np.diag((2*modelparams.param_b + YY  - np.diag(W @ latentparams.SS_Mean[0:D, :].T)) / (modelparams.I + 2*(modelparams.param_a+1)))
            Sigma_x = np.diag((2*modelparams.param_b + YY - 2 * np.diag(W @ latentparams.SS_Mean[0:D, :].T) + np.diag(W @ latentparams.SS_SecMoment @ W.T) + np.sum((W ** 2) * modelparams.alpha , axis = 1)) / (modelparams.I + 2*(modelparams.param_a+1)))
            
            Prec_x = np.diag(1/np.diag(Sigma_x))
            #H = Beta[D:, :]
            H = latentparams.SS_Mean[D:, :] @ (inv(latentparams.SS_SecMoment))
            
            modelparams.U_mean_prior = U_mean_prior
            modelparams.Prec_x = Prec_x
            modelparams.Sigma_x = Sigma_x
            modelparams.W = W
            modelparams.H = H
            
    def loglik(self, modelparams, X, Y):
        

        M = modelparams.M
        #I = modelparams.I
        I = X.shape[1]
        D = modelparams.D
        K = modelparams.K
        
        F_u = modelparams.F_u
        Xon = modelparams.Xon
        Yon = modelparams.Yon
        W = modelparams.W
        H = modelparams.H
        U_mean_prior = modelparams.U_mean_prior
        

        latentparams = self.latent_params(modelparams, I)
        self.e_step(modelparams, latentparams, X, Y, 0)
        
        U_mean = latentparams.U_mean
        Sigma_u = latentparams.Sigma_u
        Psi_u = latentparams.Psi_u
        
        LogInst = 0
        LogMult = 0
        Psd_Prec = np.zeros((np.sum(M) + D, np.sum(M) + D))
        Psd_X = np.zeros((np.sum(M) + D, I))
        Psd_X[0:D, :] = X
        
        Psd_Prec[0:D,0:D] = modelparams.Prec_x
        if Yon == 1:
            for i in range(0,M.shape[0]):
                ind = range(np.sum(M[0:i]), np.sum(M[0:i+1]))
                Psi_u_d = softmax(Psi_u[ind, :])
                G_u = F_u[i] @ Psi_u[ind, :] - Psi_u_d[0:-1, :]
                ind_tilde = range(np.sum(M[0:i]) + D, np.sum(M[0:i+1]) + D)
                Y_tilde = inv(F_u[i]) @ (Y[ind, :] + G_u) 
                Psd_X[ind_tilde, :] = Y_tilde
                Psd_Prec[np.ix_((ind_tilde),(ind_tilde))] = F_u[i]
                LogMult_c = 0.5 * np.sum(Psi_u[ind, :] * (F_u[i] @ Psi_u[ind, :]), axis = 0) - np.sum(Psi_u_d[0:-1, :] * Psi_u[ind, :], axis = 0) + logsumexp(Psi_u[ind, :])
                LogMult = LogMult + 0.5 * np.log(2*np.pi) * M[i] + 0.5 * logdet(inv(F_u[i])) + 0.5 * np.sum(Y_tilde * (F_u[i] @ Y_tilde), axis = 0) - LogMult_c
            
            LogInst = LogMult
            LogMult = np.sum(LogMult)
        else:
            Psd_X = X
            Psd_Prec = modelparams.Prec_x
        
        if Xon == 1 or Yon == 1:
            if Xon == 1 and Yon == 1:
                Psd_Beta = np.append(W,H,axis=0)
            elif Xon == 1:
                Psd_Beta = W.copy()
            elif Yon == 1:
                Psd_Beta = H.copy()
            Psd_Mean =  Psd_Beta @ U_mean
            LogLink =  np.sum(0.5 * (logdet(Psd_Prec) - (np.sum(M) + D) * np.log(2*np.pi)) - 0.5 * np.sum((Psd_X - Psd_Mean) * (Psd_Prec @ (Psd_X - Psd_Mean)), axis = 0))
            #for i in range(0, I):
            LogLink = LogLink - 0.5 * np.trace( Psd_Prec @ Psd_Beta @ Sigma_u @ Psd_Beta.T ) * I
        else:
            LogLink = 0
    
        Entropy = 0
        LogLatent = 0.5 * (logdet(latentparams.Prec_u_prior) - (K) * np.log(2*np.pi)) - 0.5 * np.sum((U_mean_prior[None].T - U_mean) * (latentparams.Prec_u_prior @ (U_mean_prior[None].T - U_mean)), axis = 0)
        LogInst = LogInst + LogLatent
        LogLatent = np.sum(LogLatent)
        #for i in range(0, I):
        LogLatent = LogLatent - 0.5 * np.trace( latentparams.Prec_u_prior @ Sigma_u) * I
        Entropy =Entropy + 0.5 * (np.log(2*np.pi) * K + logdet(Sigma_u)) * I
        
        
        LogPrior = -(modelparams.param_a + 1) * np.sum(np.log(np.diag(modelparams.Sigma_x))) - modelparams.param_b * np.sum(np.diag(modelparams.Prec_x))
        LogPrior = Xon * LogPrior + 0.5 * (modelparams.nu0 + K + 1) * logdet(latentparams.Prec_u_prior) - 0.5 * np.trace(modelparams.s0 @ latentparams.Prec_u_prior)
        
        LogLik = (LogLink + LogMult + LogLatent + Entropy + LogPrior) / I
        
        return LogLik, LogInst
            
    def fit(self, X, Y, X_test, Y_test, Y_test_ext, epochno = 10):
        
        tmp_log = 0
        for epoch in range(0,epochno):
            LogLik, LogInst = self.loglik(self.modelparams, X_test, Y_test)
            print('Validation logLikelihood:', LogLik)
            
            self.e_step(self.modelparams, self.latentparams, X, Y, self.MM)
            
            self.m_step(self.modelparams, self.latentparams, X, Y)
            
            LogLik = 0
            if self.modelparams.LogLikOn == 1:
                LogLik, LogInst = self.loglik(self.modelparams, X, Y)
            tmp_log = LogLik
            print('Epoch:', epoch)
            
            print('LogLikelihood:', LogLik)
            self.latentparams.LogInst = LogInst
            