function update = decompose_update(y, share, output_MCT, is_timeag, i_depend, T_back)
%
%

% Recover dimensions
[T, n]    = size(y);
q         = repmat(size(output_MCT.theta, 2), [n, 1]);
theta_r   = ((1:max(q)) <= q);
share     = share(:); if ~(length(share) == n), error('Dimension of share vector is wrong'); end
is_depend = (i_depend > 0);

% Extract median parameters and latent variables
beta_tau      = median(output_MCT.beta_tau, 2);
alpha_tau     = median(output_MCT.alpha_tau, 3);
sigma_dtau_c  = median(output_MCT.sigma_dtau_c, 2);
sigma_dtau_i  = median(output_MCT.sigma_dtau_i, 3);
beta_eps      = median(output_MCT.beta_eps, 2);
alpha_eps     = median(output_MCT.alpha_eps, 3);
sigma_eps_c   = median(output_MCT.sigma_eps_c, 2);
s_eps_c       = median(output_MCT.s_eps_c, 2);
sigmaXs_eps_c = sigma_eps_c.*s_eps_c;
sigma_eps_i   = median(output_MCT.sigma_eps_i, 3);
s_eps_i       = median(output_MCT.s_eps_i, 3);
sigmaXs_eps_i = sigma_eps_i.*s_eps_i;
eye_cell      = cell(n, 1); for i = 1:n, if is_timeag(i), eye_cell{i} = repmat(1/12, [1, 12]); else, eye_cell{i} = 1; end; end
eye_blk       = blkdiag(eye_cell{:});
theta         = median(output_MCT.theta, 3);
theta_cell    = cell(n, 1); for i = 1:n, theta_cell{i} = [1, theta(i, theta_r(i, :))]; end
theta_blk     = blkdiag(theta_cell{:});

% Define indexing for state variables
if any(is_timeag), id_tau_c = 1:12; else, id_tau_c = 1; end
id_tmp   = length(id_tau_c);
id_tau_i = cell(n, 1);
for i = 1:n
    if is_timeag(i), id_tau_i{i} = id_tmp+(1:12); else, id_tau_i{i} = id_tmp+1; end
    id_tmp = id_tmp + length(id_tau_i{i});
end
id_eps_c = id_tmp + 1;
id_tmp   = id_tmp + length(id_eps_c);
id_eps_i = cell(n, 1);
for i = 1:n
    id_eps_i{i} = id_tmp + (1:(1+q(i)));
    id_tmp      = id_tmp + length(id_eps_i{i});
end
n_state = id_tmp;

% Initialize state-space model
SSM = struct();
%%% H
SSM.H = zeros(n, n_state, T+1);
for t = 1:T
    SSM.H(~is_timeag, id_tau_c(1), t+1) = alpha_tau(t, ~is_timeag)';
    SSM.H(is_timeag, id_tau_c, t+1)     = (1/12)*[alpha_tau(t:(-1):max(1, t-11), is_timeag)', repmat(alpha_tau(1, is_timeag)', [1, max(12-t, 0)])];
    SSM.H(:, [id_tau_i{:}], t+1)        = eye_blk;
    SSM.H(:, id_eps_c(1), t+1)          = alpha_eps(t, :)';
    SSM.H(:, [id_eps_i{:}], t+1)        = theta_blk;
    for i = 1:n
        if is_depend(i)
            i_aux                          = i_depend(i);
            SSM.H(i, id_tau_i{i_aux}, t+1) = beta_tau(i)*SSM.H(i_aux, id_tau_i{i_aux}, t+1);
            SSM.H(i, id_eps_i{i_aux}, t+1) = beta_eps(i)*SSM.H(i_aux, id_eps_i{i_aux}, t+1);
        end
    end    
end
%%% Sigma_eps
SSM.Sigma_eps = (1e-4)*eye(n);
%%% F
SSM.F                     = zeros(n_state);
n_tmp                     = length(id_tau_c);
SSM.F(id_tau_c, id_tau_c) = [1, zeros(1, n_tmp-1); eye(n_tmp-1), zeros(n_tmp-1, 1)];
for i = 1:n
    n_tmp                           = length(id_tau_i{i});
    SSM.F(id_tau_i{i}, id_tau_i{i}) = [1, zeros(1, n_tmp-1); eye(n_tmp-1), zeros(n_tmp-1, 1)];
end
n_tmp                     = length(id_eps_c);
SSM.F(id_eps_c, id_eps_c) = [zeros(1, n_tmp); eye(n_tmp-1), zeros(n_tmp-1, 1)];
for i = 1:n
    n_tmp                           = length(id_eps_i{i});
    SSM.F(id_eps_i{i}, id_eps_i{i}) = [zeros(1, n_tmp); eye(n_tmp-1), zeros(n_tmp-1, 1)];
end
%%% G
SSM.G                = zeros(n_state, 2*(n+1));
n_tmp                = length(id_tau_c);
SSM.G(id_tau_c, 1)   = [1; zeros(n_tmp-1, 1)];
for i = 1:n
    n_tmp                   = length(id_tau_i{i});
    SSM.G(id_tau_i{i}, 1+i) = [1; zeros(n_tmp-1, 1)];
end
n_tmp                = length(id_eps_c);
SSM.G(id_eps_c, 2+n) = [1; zeros(n_tmp-1, 1)];
for i = 1:n
    n_tmp                     = length(id_eps_i{i});
    SSM.G(id_eps_i{i}, 2+n+i) = [1; zeros(n_tmp-1, 1)];
end
%%% Sigma_eta
SSM.Sigma_eta = zeros(2*(n+1), 2*(n+1), T);
for t = 1:T
    SSM.Sigma_eta(:, :, t) = diag([sigma_dtau_c(t), sigma_dtau_i(t, :), ...
                                  sigmaXs_eps_c(t), sigmaXs_eps_i(t, :)].^2);    
end
%%% mu_1 and Sigma_1
SSM.mu_1                        = zeros(n_state, 1);
SSM.Sigma_1                     = 1e6*eye(n_state);
SSM.Sigma_1(id_tau_c, id_tau_c) = 0;
SSM.Sigma_1(id_eps_c, id_eps_c) = 0;


%% UPDATE DECOMPOSITION

% Compute trend and forecasts based on previous month data
y_tmp          = [NaN(1, n); y(1:(T-1), :); NaN(1, n)];
[~, state_tmp] = fast_smoother(y_tmp', SSM);
tau_c          = state_tmp(id_tau_c(1), T+1)';
tau_i          = NaN(1, n);
for i = 1:n
    tau_i(i) = state_tmp(id_tau_i{i}(1), T+1)';
end
trend_old      = (tau_c*alpha_tau(T, :) + tau_i)*share;
forecasts_last = SSM.H(:, :, T+1)*state_tmp(:, T+1);

% Compute trend based on current month data
y_tmp          = [NaN(1, n); y];
[~, state_tmp] = fast_smoother(y_tmp', SSM);
tau_c          = state_tmp(id_tau_c(1), T+1)';
tau_i          = NaN(1, n);
for i = 1:n
    tau_i(i) = state_tmp(id_tau_i{i}(1), T+1)';
end
trend_new      = (tau_c*alpha_tau(T, :) + tau_i)*share;
realized_last  = y(T, :)';

% Compute weights
weights_last   = NaN(n, 1);
y_tmp          = [NaN(1, n); zeros(T, n)];
[~, state_tmp] = fast_smoother(y_tmp', SSM);
tau_c          = state_tmp(id_tau_c(1), T+1)';
tau_i          = NaN(1, n);
for i = 1:n
    tau_i(i) = state_tmp(id_tau_i{i}(1), T+1)';
end
trend_0        = alpha_tau(T, :)*tau_c + tau_i;
for i = 1:n
    y_tmp           = [NaN(1, n); zeros(T, n)];
    y_tmp(T+1, i)   = 1;
    [~, state_tmp]  = fast_smoother(y_tmp', SSM);
    tau_c           = state_tmp(id_tau_c(1), T+1)';
    tau_i           = NaN(1, n);
    for j = 1:n
        tau_i(j) = state_tmp(id_tau_i{j}(1), T+1)';
    end
    trend_1         = tau_c*alpha_tau(T, :) + tau_i;
    weights_last(i) = (trend_1 - trend_0)*share;
end

% Store results
update           = struct();
update.trend_old = trend_old;
update.trend_new = trend_new;
update.impacts   = weights_last .* (realized_last - forecasts_last);
update.weights   = weights_last;
update.shares    = share;
update.news      = realized_last - forecasts_last;
update.realized  = realized_last;
update.forecasts = forecasts_last;


%% FILTER WEIGHT CALTULATION

if isempty(T_back)
    filter = [];
else
    filter         = NaN(n, T_back);
    filter(:, 1)   = weights_last;
    y_tmp          = [NaN(1, n); zeros(T, n)];
    [~, state_tmp] = fast_smoother(y_tmp', SSM);
    tau_c          = state_tmp(id_tau_c(1), T+1)';
    tau_i          = NaN(1, n);
    for i = 1:n
        tau_i(i) = state_tmp(id_tau_i{i}(1), T+1)';
    end
    trend_0        = tau_c*alpha_tau(T, :) + tau_i;
    for t = 2:T_back
    for i = 1:n
        y_tmp               = [NaN(1, n); zeros(T, n)];
        y_tmp(T+1-(t-1), i) = 1;
        [~, state_tmp]      = fast_smoother(y_tmp', SSM);
        tau_c               = state_tmp(id_tau_c(1), T+1)';
        tau_i               = NaN(1, n);
        for j = 1:n
            tau_i(j) = state_tmp(id_tau_i{j}(1), T+1)';
        end
        trend_1             = tau_c*alpha_tau(T, :) + tau_i;
        filter(i, t)        = (trend_1 - trend_0)*share;
    end
    end
end

% Store filter weights
update.filter = filter;


%% FORECASTING

% Extend state-space model
T_fore = max(q+12*is_timeag) + 1;
for t = T+(1:T_fore)
    SSM.H(~is_timeag, id_tau_c(1), t+1) = alpha_tau(T, ~is_timeag)';
    SSM.H(is_timeag, id_tau_c, t+1)     = (1/12)*[repmat(alpha_tau(T, is_timeag)', [1, min(12, t-T)]), alpha_tau(T:(-1):(t-11), is_timeag)'];
    SSM.H(:, [id_tau_i{:}], t+1)        = eye_blk;
    SSM.H(:, id_eps_c(1), t+1)          = alpha_eps(T, :)';
    SSM.H(:, [id_eps_i{:}], t+1)        = theta_blk;
    for i = 1:n
        if is_depend(i)
            i_aux                          = i_depend(i);
            SSM.H(i, id_tau_i{i_aux}, t+1) = beta_tau(i)*SSM.H(i_aux, id_tau_i{i_aux}, t+1);
            SSM.H(i, id_eps_i{i_aux}, t+1) = beta_eps(i)*SSM.H(i_aux, id_eps_i{i_aux}, t+1);
        end
    end        
    SSM.Sigma_eta(:, :, t)              = diag([sigma_dtau_c(T), sigma_dtau_i(T, :), ...
                                               sigmaXs_eps_c(T), sigmaXs_eps_i(T, :)].^2);
end

% Compute future point forecasts
y_tmp            = [NaN(1, n); y; NaN(T_fore, n)];
[~, state_tmp]   = fast_smoother(y_tmp', SSM);
forecasts_future = NaN(n, T_fore);
for t = 1:T_fore
    forecasts_future(:, t) = (SSM.H(:, :, T+1+t)*state_tmp(:, T+1+t));
end
aggregate_future = (share')*forecasts_future;

% Store future point forecasts
update.forecasts_future = forecasts_future;
update.aggregate_future = aggregate_future;

end