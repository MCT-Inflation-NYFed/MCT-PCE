function [log_likelihood, prediction, filter] = Kalman_filter(Y, SSM)
% KALMAN_FILTER  Kalman filter for state-space model:
%   y_t     = D_t + H_t * x_t + eps_t,
%   x_(t+1) = C_t + F_t * x_t + G_t * eta_t,
%   x_1     ~ N(mu_1, Sigma_1),
%   eps_t   ~ N(0_Nx1, Sigma_(eps, t)),
%   eta_t   ~ N(0_Kx1, Sigma_(eta, t)).
% 
%   LOG_LIKELIHOOD = KALMAN_FILTER(Y, SSM) computes log likelihood of data
%   Y for state-space model SSM:
%     Y is NxT, columns are y_t, missing data is NaN.
%     SSM is struct with state-space matrices (constant or time varying):
%     - SSM.D is Nx1 or NxT, default is zeros(N, 1).
%     - SSM.H is NxM or NxMxT.
%     - SSM.Sigma_eps is NxN or NxNxT, default is zeros(N).
%     - SSM.C is Mx1 or Mx(T-1), default is zeros(M, 1).
%     - SSM.F is MxM or MxMx(T-1).
%     - SSM.G is MxK or MxKx(T-1).
%     - SSM.Sigma_eta is KxK or KxKx(T-1), default is eye(K).
%     - SSM.mu_1 is Mx1.
%     - SSM.Sigma_1 is MxM.
%
%   [LOG_LIKELIHOOD, PREDICTION] = KALMAN_FILTER(Y, SSM) also computes
%   errors and inverse MSEs of the one-period-ahead predictions 
%   E[y_t|y_(1:(t-1))]:
%     PREDICTION is struct:
%     - PREDICTION.error is NxT.
%     - PREDICTION.invMSE is NxNxT.
%     - PREDICTION.gains is NxMxT, matrices are Kalman gains.
%
%   [LOG_LIKELIHOOD, PREDICTION, FILTER] = KALMAN_FILTER(Y, SSM) also gives
%   filtered states E[x_t|y_(1:(t-1))] and MSEs V(x_t|y_(1:(t-1)):
%     FILTER is struct:
%     - FILTER.mu is MxT, columns are filtered states.
%     - FILTER.Sigma is MxMxT, matrices are MSEs.
%
%   Good numerical performance requires MSE of y_t invertible.
%
%   See Durbin and Koopman (2012): Time Series Analysis by State Space 
%   Methods. Oxford Statistical Sciences Series.
%
%   Version: 2020 Dec 28 - Matlab R2017b

% Determine tasks
need_prediction = (nargout > 1);
need_filter     = (nargout > 2);

% Define function handle and options for inverse algorithm
symmetrize    = @(A) (A + A')/2;
option        = struct();
option.SYM    = true;
option.POSDEF = true;

% Unfold state-space struct
H = SSM.H;
F = SSM.F;
G = SSM.G;

% Recover dimensions
[N, T]    = size(Y);
[M, K, ~] = size(G);

% Set state-space matrices to default if needed
if isfield(SSM, 'D'),         Y         = Y - SSM.D;     end
if isfield(SSM, 'Sigma_eps'), Sigma_eps = SSM.Sigma_eps; else, Sigma_eps = zeros(N);    end
if isfield(SSM, 'C'),         C         = SSM.C;         else, C         = zeros(M, 1); end
if isfield(SSM, 'Sigma_eta'), Sigma_eta = SSM.Sigma_eta; else, Sigma_eta = eye(K);      end

% Compute time-varying indicators
isTV_H         = (size(H, 3) > 1);
isTV_Sigma_eps = (size(Sigma_eps, 3) > 1);
isTV_C         = (size(C, 2) > 1);
isTV_F         = (size(F, 3) > 1);
isTV_G         = (size(G, 3) > 1);
isTV_Sigma_eta = (size(Sigma_eta, 3) > 1);


%% KALMAN FILTER RECURSIONS

% Create storage
log_likelihood = -(T*N/2)*log(2*pi);
if need_prediction
    prediction_error  = NaN(N, T);
    prediction_invMSE = NaN(N, N, T);
    prediction_gain   = NaN(M, N, T);
    if need_filter
        filter_mu    = zeros(M, T);
        filter_Sigma = zeros(M, M, T);
    end
end

% Create auxiliary variables
Ct          = C(:, 1);
Ft          = F(:, :, 1);
Gt          = G(:, :, 1);
Sigma_eta_t = Sigma_eta(:, :, 1);
mu          = SSM.mu_1;
Sigma       = symmetrize(SSM.Sigma_1);
Ht          = H(:, :, 1);
Sigma_eps_t = Sigma_eps(:, :, 1);

% Do first iteration of Kalman filter
nonmiss         = ~isnan(Y(:, 1));
Y_aux           = Y(nonmiss, 1);
Ht_aux          = Ht(nonmiss, :);
Sigma_eps_t_aux = Sigma_eps_t(nonmiss, nonmiss);
e               = Y_aux - Ht_aux * mu;
S               = symmetrize( Ht_aux * Sigma * (Ht_aux') + Sigma_eps_t_aux );
S_inv           = linsolve(S, eye(nnz(nonmiss)), option);
Kt              = Sigma * (Ht_aux') * S_inv;
log_likelihood  = log_likelihood - (1/2)*log(det(S)) - (1/2)*(e')*S_inv*e;

% Store prediction and filter output
if need_prediction
    prediction_error(nonmiss, 1)           = e;
    prediction_invMSE(nonmiss, nonmiss, 1) = S_inv;
    prediction_gain(:, nonmiss, 1)         = Kt;
    if need_filter
        filter_mu(:, 1)       = mu;
        filter_Sigma(:, :, 1) = Sigma;
    end
end

% Perform Kalman filter recursions
for t = 2:T
    % Update parameters of transition equation if time varying
    if isTV_C,         Ct          = C(:, t-1);            end    
    if isTV_F,         Ft          = F(:, :, t-1);         end
    if isTV_G,         Gt          = G(:, :, t-1);         end
    if isTV_Sigma_eta, Sigma_eta_t = Sigma_eta(:, :, t-1); end

    % Compute filtered states
    mu    = Ct + Ft * (mu + Kt*e);
    Sigma = symmetrize( Ft*(Sigma - Kt*Ht_aux*(Sigma'))*(Ft') + Gt*Sigma_eta_t*(Gt') );
    
    % Update parameters of measurement equation if time varying
    if isTV_H,         Ht          = H(:, :, t);           end
    if isTV_Sigma_eps, Sigma_eps_t = Sigma_eps(:, :, t);   end
    
    % Compute predictions and log likelihood
    nonmiss         = ~isnan(Y(:, t));
    Y_aux           = Y(nonmiss, t);
    Ht_aux          = Ht(nonmiss, :);
    Sigma_eps_t_aux = Sigma_eps_t(nonmiss, nonmiss);
    e               = Y_aux - Ht_aux * mu;
    S               = symmetrize( Ht_aux * Sigma * (Ht_aux') + Sigma_eps_t_aux );
    S_inv           = linsolve(S, eye(nnz(nonmiss)), option);
    Kt              = Sigma * (Ht_aux') * S_inv;
    log_likelihood  = log_likelihood - (1/2)*log(det(S)) - (1/2)*(e')*S_inv*e;
    
    % Store prediction and filter output
    if need_prediction
        prediction_error(nonmiss, t)           = e;
        prediction_invMSE(nonmiss, nonmiss, t) = S_inv;
        prediction_gain(:, nonmiss, t)         = Kt;
        if need_filter
            filter_mu(:, t)       = mu;
            filter_Sigma(:, :, t) = Sigma;
        end
    end
end

% Pass output as struct
if need_prediction
    prediction        = struct();
    prediction.error  = prediction_error;
    prediction.invMSE = prediction_invMSE;
    prediction.gain   = prediction_gain;
    if need_filter
        filter       = struct();
        filter.mu    = filter_mu;
        filter.Sigma = filter_Sigma;
    end
end

end