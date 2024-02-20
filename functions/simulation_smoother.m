function [states, disturbances] = simulation_smoother(Y, SSM)
% SIMULATION_SMOOTHER  Simulation smoother for state-space model:
%   y_t     = D_t + H_t * x_t + eps_t,
%   x_(t+1) = C_t + F_t * x_t + G_t * eta_t,
%   x_1     ~ N(mu_1, Sigma_1),
%   eps_t   ~ N(0_Nx1, Sigma_(eps, t)),
%   eta_t   ~ N(0_Kx1, Sigma_(eta, t)).
% 
%   STATES = SIMULATION_SMOOTHER(Y, SSM) draws states from conditional 
%   distribution P[x_(1:T)|y_(1:T)] given data Y for model SSM:
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
%     STATES is MxT.
%
%   [STATES, DISTURBANCES] = SIMULATION_SMOOTHER(Y, SSM) also draws 
%   disturbances from P[eps_(1:T)|y_(1:T)] and P[eta_(1:T)|y_(1:T)]:
%     DISTURBANCES is struct:
%     - disturbances.m_errors is NxT.
%     - disturbances.shocks is Kx(T-1).
%
%   Good numerical performance requires MSE of y_t invertible.
%
%   See Durbin and Koopman (2012): Time Series Analysis by State Space 
%   Methods. Oxford Statistical Sciences Series.
%
%   Version: 2020 Dec 28 - Tincho Almuzara - Matlab R2017b

% Determine tasks
need_disturbances = (nargout > 1);

% Recover dimensions
T = size(Y, 2);

% Create auxiliary state-space model with no constants
SSM_aux      = SSM;
SSM_aux.mu_1 = 0*SSM_aux.mu_1;
if isfield(SSM, 'C'), SSM_aux.C = 0*SSM_aux.C; end
if isfield(SSM, 'D'), SSM_aux.D = 0*SSM_aux.D; end


%% DURBIN-KOOPMAN SIMULATION SMOOTHER

% Simulate state-space model and run smoother
if ~need_disturbances
    [Y_sim, states_sim] = simulate_SSM(SSM_aux, T);
    [~, states_smooth]  = fast_smoother(Y - Y_sim, SSM);
else
    [Y_sim, states_sim, disturb_sim] = simulate_SSM(SSM_aux, T);
    [disturb_smooth, states_smooth]  = fast_smoother(Y - Y_sim, SSM);
end

% Compute draw of states
states = states_sim + states_smooth;

% Compute draw of disturbances if needed
if need_disturbances
    disturbances          = struct();
    disturbances.m_errors = disturb_sim.m_errors + disturb_smooth.m_errors;
    disturbances.shocks   = disturb_sim.shocks + disturb_smooth.shocks;
end    

end