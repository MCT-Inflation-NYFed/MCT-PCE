function [Y, states, disturbances] = simulate_SSM(SSM, T)
% SIMULATE_SSM  Simulate state-space model:
%   y_t     = D_t + H_t * x_t + eps_t,
%   x_(t+1) = C_t + F_t * x_t + G_t * eta_t,
%   x_1     ~ N(mu_1, Sigma_1),
%   eps_t   ~ N(0_Nx1, Sigma_(eps, t)),
%   eta_t   ~ N(0_Kx1, Sigma_(eta, t)).
% 
%   Y = SIMULATE_SSM(SSM, T) draws data from model SSM w/sample size T:
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
%     T is integer.
%     Y is NxT, columns are y_t.
%
%   [Y, STATES] = SIMULATE_SSM(SSM, T) also draws states:
%     STATES is MxT, columns are x_t.
%
%   [Y, STATES, DISTURBANCES] = SIMULATE_SSM(SSM, T) also draws 
%   disturbances:
%     DISTURBANCES is struct:
%     - DISTURBANCES.m_errors is NxT, columns are eps_t.
%     - DISTURBANCES.shocks is Kx(T-1), columns are eta_t.
%
%   See Durbin and Koopman (2012): Time Series Analysis by State Space 
%   Methods. Oxford Statistical Sciences Series.
%
%   Version: 2020 Dec 28 - Tincho Almuzara - Matlab R2017b

% Determine tasks
need_states       = (nargout > 1);
need_disturbances = (nargout > 2);

% Unfold state-space struct
H = SSM.H;
F = SSM.F;
G = SSM.G;

% Recover dimensions
N         = size(H, 1);
[M, K, ~] = size(G);

% Set state-space matrices to default if needed
if isfield(SSM, 'D'),         D         = SSM.D;         else, D         = zeros(N, 1); end
if isfield(SSM, 'Sigma_eps'), Sigma_eps = SSM.Sigma_eps; else, Sigma_eps = zeros(N);    end
if isfield(SSM, 'C'),         C         = SSM.C;         else, C         = zeros(M, 1); end
if isfield(SSM, 'Sigma_eta'), Sigma_eta = SSM.Sigma_eta; else, Sigma_eta = eye(K);      end

% Compute time-varying indicators
isTV_D         = (size(D, 2) > 1);
isTV_H         = (size(H, 3) > 1);
isTV_Sigma_eps = (size(Sigma_eps, 3) > 1);
isTV_C         = (size(C, 2) > 1);
isTV_F         = (size(F, 3) > 1);
isTV_G         = (size(G, 3) > 1);
isTV_Sigma_eta = (size(Sigma_eta, 3) > 1);

% Define auxiliary function handle
symmetrize = @(A) (A + A')/2;


%% SIMULATION OF STATE-SPACE MODEL

% Create storage
Y = zeros(N, T);
if need_states
    states = zeros(M, T);
    if need_disturbances
        m_errors = zeros(N, T);
        shocks   = zeros(K, T-1);
    end
end

% Create auxiliary variables
Ct          = C(:, 1);
Ft          = F(:, :, 1);
Gt          = G(:, :, 1);
Sigma_eta_t = Sigma_eta(:, :, 1);
Dt          = D(:, 1);
Ht          = H(:, :, 1);
Sigma_eps_t = Sigma_eps(:, :, 1);

% Simulate initial condition
x_t     = mvnrnd(SSM.mu_1, symmetrize(SSM.Sigma_1))';
eps_t   = mvnrnd(zeros(N, 1), Sigma_eps_t)';
Y(:, 1) = Dt + Ht * x_t + eps_t;
if need_states
    states(:, 1) = x_t;
    if need_disturbances
        m_errors(:, 1) = eps_t;
    end
end

% Simulate rest of the sample
for t = 2:T
    % Update parameters of transition equation if time varying
    if isTV_C,         Ct          = C(:, t-1);            end    
    if isTV_F,         Ft          = F(:, :, t-1);         end
    if isTV_G,         Gt          = G(:, :, t-1);         end
    if isTV_Sigma_eta, Sigma_eta_t = Sigma_eta(:, :, t-1); end

    % Draw shocks and update states
    eta_t = mvnrnd(zeros(K, 1), Sigma_eta_t)';
    x_t   = Ct + Ft * x_t + Gt * eta_t;
    
    % Update parameters of measurement equation if time varying
    if isTV_D,         Dt          = D(:, t);            end
    if isTV_H,         Ht          = H(:, :, t);         end
    if isTV_Sigma_eps, Sigma_eps_t = Sigma_eps(:, :, t); end
    
    % Draw measurement errors and compute data
    eps_t   = mvnrnd(zeros(N, 1), Sigma_eps_t)';
    Y(:, t) = Dt + Ht * x_t + eps_t;

    % Store states and disturbances if needed
    if need_states
        states(:, t) = x_t;
        if need_disturbances
            shocks(:, t-1) = eta_t;
            m_errors(:, t) = eps_t;
        end
    end
end

% Pass output as struct
if need_disturbances
    disturbances          = struct();
    disturbances.m_errors = m_errors;
    disturbances.shocks   = shocks;
end

end