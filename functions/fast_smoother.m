function [disturbances, states, MSEs] = fast_smoother(Y, SSM)
% FAST_SMOOTHER  Fast disturbance and state smoother for state-space model:
%   y_t     = D_t + H_t * x_t + eps_t,
%   x_(t+1) = C_t + F_t * x_t + G_t * eta_t,
%   x_1     ~ N(mu_1, Sigma_1),
%   eps_t   ~ N(0_Nx1, Sigma_(eps, t)),
%   eta_t   ~ N(0_Kx1, Sigma_(eta, t)).
% 
%   DISTRUBANCES = FAST_SMOOTHER(Y, SSM) computes smoothed disturbances 
%   E[eta_t|y_(1:T)] and E[eps_t|y_(1:T)] on data Y for model SSM:
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
%     DISTURBANCES is struct:
%     - DISTURBANCES.m_errors is NxT.
%     - DISTURBANCES.shocks is Kx(T-1).
%
%   [DISTURBANCES, STATES] = FAST_SMOOTHER(Y, SSM) also computes smoothed 
%   states E[x_t|y_(1:T)]:
%     STATES is MxT.
%
%   [DISTURBANCES, STATES, MSES] = fast_smoother(Y, SSM) also computes
%   MSE matrices V(eps_t|y_(1:T)), V(eta_t|y_(1:T)) and V(x_t|y_(1:T)):
%     MSES is struct:
%     - MSES.m_errors is NxNxT.
%     - MSES.shocks is KxKx(T-1).
%     - MSES.states is MxMxT.
%
%   Good numerical performance requires MSE of y_t invertible.
%
%   See Durbin and Koopman (2012): Time Series Analysis by State Space 
%   Methods. Oxford Statistical Sciences Series.
%
%   Version: 2020 Dec 28 - Matlab R2017b

% Determine tasks
need_states = (nargout > 1);
need_MSEs   = (nargout > 2);

% Define function handle
symmetrize    = @(A) (A + A')/2;

% Unfold state-space struct
H = SSM.H;
F = SSM.F;
G = SSM.G;

% Recover dimensions
[N, T]    = size(Y);
[M, K, ~] = size(G);

% Set state-space matrices to default if needed
if isfield(SSM, 'Sigma_eps'), Sigma_eps = SSM.Sigma_eps; else, Sigma_eps = zeros(N); end
if isfield(SSM, 'Sigma_eta'), Sigma_eta = SSM.Sigma_eta; else, Sigma_eta = eye(K);   end

% Compute time-varying indicators
isTV_H         = (size(H, 3) > 1);
isTV_Sigma_eps = (size(Sigma_eps, 3) > 1);
isTV_F         = (size(F, 3) > 1);
isTV_G         = (size(G, 3) > 1);
isTV_Sigma_eta = (size(Sigma_eta, 3) > 1);

% Run Kalman filter and recover prediction outputs
if ~need_MSEs
    [~, prediction] = Kalman_filter(Y, SSM);    
else
    [~, prediction, filter] = Kalman_filter(Y, SSM);
end
e     = prediction.error;
S_inv = prediction.invMSE;
Kt    = prediction.gain;


%% DISTURBANCE SMOOTHER RECURSIONS

% Create storage
m_errors = zeros(N, T);
shocks   = zeros(K, T-1);

% Create auxiliary variables
Ft          = F(:, :, 1);
Gt          = G(:, :, 1);
Sigma_eta_t = Sigma_eta(:, :, 1);
if isTV_H, Ht = H(:, :, T); else, Ht = H; end
if isTV_Sigma_eps, Sigma_eps_t = Sigma_eps(:, :, T); else, Sigma_eps_t = Sigma_eps; end

% Do first iteration of disturbance smoother
nonmiss         = ~isnan(e(:, T));
e_aux           = e(nonmiss, T);
S_inv_aux       = S_inv(nonmiss, nonmiss, T);
Ht_aux          = Ht(nonmiss, :);
Sigma_eps_t_aux = Sigma_eps_t(:, nonmiss);                                 % only ignore missing columns
u_aux           = S_inv_aux * e_aux;
r_aux           = (Ht_aux') * u_aux + zeros(M, 1);
m_errors(:, T)  = Sigma_eps_t_aux * u_aux;

% Perform disturbance smoother recursions
for t = (T-1):(-1):1
    % Update parameters of transition equation if time varying
    if isTV_F,         Ft          = F(:, :, t);         end
    if isTV_G,         Gt          = G(:, :, t);         end
    if isTV_Sigma_eta, Sigma_eta_t = Sigma_eta(:, :, t); end

    % Compute smoothed shocks
    shocks(:, t) = Sigma_eta_t * (Gt') * r_aux;
    
    % Update parameters of measurement equation if time varying
    if isTV_H,         Ht          = H(:, :, t);         end
    if isTV_Sigma_eps, Sigma_eps_t = Sigma_eps(:, :, t); end
    
    % Compute smoother measurement errors and variances
    nonmiss         = ~isnan(e(:, t));
    e_aux           = e(nonmiss, t);
    S_inv_aux       = S_inv(nonmiss, nonmiss, t);
    Kt_aux          = Kt(:, nonmiss, t);
    Ht_aux          = Ht(nonmiss, :);
    Sigma_eps_t_aux = Sigma_eps_t(:, nonmiss);                             % only ignore missing columns
    u_aux           = S_inv_aux * e_aux - (Ft*Kt_aux)' * r_aux;
    r_aux           = (Ht_aux') * u_aux + (Ft') * r_aux;
    m_errors(:, t)  = Sigma_eps_t_aux * u_aux;
end

% Pass output as struct
disturbances          = struct();
disturbances.m_errors = m_errors;
disturbances.shocks   = shocks;


%% STATE SMOOTHER RECURSIONS

if need_states    
    % Set state-space matrices to default if needed
    if isfield(SSM, 'C'), C = SSM.C; else, C = zeros(M, 1); end
    
    % Compute time-varying indicators
    isTV_C = (size(C, 2) > 1);
    
    % Create storage
    states = zeros(M, T);
    
    % Create auxiliary variables
    Ct = C(:, 1);
    
    % Do first iteration of fast state smoother
    x            = SSM.mu_1 + SSM.Sigma_1 * r_aux;
    states(:, 1) = x;
    
    % Perform state smoother recursions
    for t = 2:T
        % Update parameters of transition equation if time varying
        if isTV_C, Ct = C(:, t-1);    end
        if isTV_F, Ft = F(:, :, t-1); end
        if isTV_G, Gt = G(:, :, t-1); end
        
        % Compute smoothed shocks
        x            = Ct + Ft * x + Gt * shocks(:, t-1);
        states(:, t) = x;
    end
end


%% MEAN-SQUARE ERROR RECURSIONS

if need_MSEs
    % Create storage
    mse_m_errors = zeros(N, N, T);
    mse_shocks   = zeros(K, K, T-1);
    mse_states   = zeros(M, M, T);

    % Unwrap struct with MSE of filtered states
    Sigma = filter.Sigma;
    
    % Create auxiliary variables
    Ft          = F(:, :, 1);
    Gt          = G(:, :, 1);
    Sigma_eta_t = Sigma_eta(:, :, 1);
    if isTV_H, Ht = H(:, :, T); else, Ht = H; end
    if isTV_Sigma_eps, Sigma_eps_t = Sigma_eps(:, :, T); else, Sigma_eps_t = Sigma_eps; end
    
    % Do first iteration of disturbance smoother
    nonmiss               = ~isnan(e(:, T));
    S_inv_aux             = S_inv(nonmiss, nonmiss, T);
    Ht_aux                = Ht(nonmiss, :);
    Sigma_eps_t_aux       = Sigma_eps_t(:, nonmiss);                       % only ignore missing columns
    D_aux                 = S_inv_aux;
    N_aux                 = (Ht_aux') * D_aux * Ht_aux + zeros(M, M);
    mse_m_errors(:, :, T) = symmetrize( Sigma_eps_t - Sigma_eps_t_aux * D_aux * (Sigma_eps_t_aux') );
    mse_states(:, :, T)   = symmetrize( Sigma(:, :, T) - Sigma(:, :, T) * N_aux * Sigma(:, :, T) );
    
    % Perform disturbance smoother recursions
    for t = (T-1):(-1):1
        % Update parameters of transition equation if time varying
        if isTV_F,         Ft          = F(:, :, t);         end
        if isTV_G,         Gt          = G(:, :, t);         end
        if isTV_Sigma_eta, Sigma_eta_t = Sigma_eta(:, :, t); end
                
        % Update parameters of measurement equation if time varying
        if isTV_H, Ht = H(:, :, t);                          end
        if isTV_Sigma_eps, Sigma_eps_t = Sigma_eps(:, :, t); end
        
        % Compute smoother measurement errors and variances
        nonmiss               = ~isnan(e(:, t));
        S_inv_aux             = S_inv(nonmiss, nonmiss, t);
        Kt_aux                = Kt(:, nonmiss, t);
        Ht_aux                = Ht(nonmiss, :);
        Sigma_eps_t_aux       = Sigma_eps_t(:, nonmiss);                   % only ignore missing columns
        mse_shocks(:, :, t)   = Sigma_eta_t - Sigma_eta_t * (Gt') * N_aux * Gt * Sigma_eta_t;
        D_aux                 = S_inv_aux + (Ft*Kt_aux)' * N_aux * (Ft*Kt_aux);
        N_aux                 = (Ht_aux') * D_aux * Ht_aux + (Ft') * N_aux * Ft ...
                                - (Kt_aux*Ht_aux)' * (Ft') * N_aux * Ft ...
                                - (Ft') * N_aux * Ft * (Kt_aux*Ht_aux);
        mse_m_errors(:, :, t) = symmetrize( Sigma_eps_t - Sigma_eps_t_aux * D_aux * (Sigma_eps_t_aux') );
        mse_states(:, :, t)   = symmetrize( Sigma(:, :, t) - Sigma(:, :, t) * N_aux * Sigma(:, :, t) );
    end
end

% Pass output as struct
if need_MSEs
    MSEs          = struct();
    MSEs.m_errors = mse_m_errors;
    MSEs.shocks   = mse_shocks;
    MSEs.states   = mse_states;
end

end