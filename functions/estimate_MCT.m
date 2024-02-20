function output = estimate_MCT_new(y, prior, settings, initial)
% ESTIMATE_MCT  Estimate Multivariate Core Trend model
%
% Levels of alpha_tau*tau_c and tau_i are not identified.
% Signs of tau_c, alpha_tau, eps_c, alpha_eps are not identified.

% Determine progress report
if isfield(settings, 'show_progress') 
    show_progress = settings.show_progress; 
    del_str       = '';
    avg_time      = NaN;
    tic    
    fig0          = figure();
    set(fig0, 'units', 'normalized', 'position', [0 0 1 1])
else
    show_progress = false; 
end

% Define function handle
symmetrize = @(A) (A + A')/2;

% Recover settings and dimensions
n_draw = settings.n_draw;
n_burn = settings.n_burn;
n_thin = settings.n_thin;
[T, n] = size(y);
q      = settings.n_lags(:);

% Recover indicators for time-aggregated and cross-dependent series
if isfield(settings, 'is_timeag'), is_timeag = settings.is_timeag; else, is_timeag = false(n, 1); end % which series are time-aggregated
t_skip    = max(q+12*is_timeag)+1;
if isfield(settings, 'i_depend'), i_depend = settings.i_depend; else, i_depend = zeros(n, 1); end % indicator of series with which cross-dependence is allowed
is_depend = (i_depend > 0);

% Set beta prior
if isfield(prior, 'prec_beta'), prec_beta = prior.prec_beta; else, prec_beta = 1; end

% Set theta restrictions and prior
if (length(q) == 1), q = repmat(q, [n, 1]); end
theta_r = ((1:max(q)) <= q);
prec_MA = prior.prec_MA;

% Set lambda/gamma priors
nu_lam = prior.nu_lam;
s2_lam = prior.s2_lam;
nu_gam = prior.nu_gam;
s2_gam = prior.s2_gam;

% Set support for s_eps and ps prior
a_ps     = prior.a_ps;
b_ps     = prior.b_ps;
n_s_vals = 40;
s_vals   = [1; linspace(2, 10, n_s_vals-1)'];

% Preallocate output
output              = struct();
output.tau_c        = NaN(T, n_draw);
output.alpha_tau    = NaN(T, n, n_draw);
output.sigma_dtau_c = NaN(T, n_draw);
output.tau_i        = NaN(T, n, n_draw);
output.sigma_dtau_i = NaN(T, n, n_draw);
output.eps_c        = NaN(T, n_draw);
output.alpha_eps    = NaN(T, n, n_draw);
output.sigma_eps_c  = NaN(T, n_draw);
output.s_eps_c      = NaN(T, n_draw);
output.eps_i        = NaN(T, n, n_draw);
output.sigma_eps_i  = NaN(T, n, n_draw);
output.s_eps_i      = NaN(T, n, n_draw);
output.theta        = NaN(n, max(q), n_draw);
output.lam_tau      = NaN(n, n_draw);
output.beta_tau     = NaN(n, n_draw);
output.gam_dtau_c   = NaN(1, n_draw);
output.gam_dtau_i   = NaN(n, n_draw);
output.lam_eps      = NaN(n, n_draw);
output.beta_eps     = NaN(n, n_draw);
output.gam_eps_c    = NaN(1, n_draw);
output.ps_c         = NaN(1, n_draw);
output.gam_eps_i    = NaN(n, n_draw);
output.ps_i         = NaN(n, n_draw);
output.y_draw       = NaN(T, n, n_draw);

% Initialize latent variables and parameters
if (nargin < 4)
    y_scale      = std(y, 'omitnan');
    alpha_tau    = (y_scale/16).*ones(T, n);
    sigma_dtau_c = ones(T, 1);
    sigma_dtau_i = (y_scale/16).*ones(T, n);
    alpha_eps    = (y_scale/16).*ones(T, n);
    sigma_eps_c  = ones(T, 1);
    s_eps_c      = ones(T, 1);
    sigma_eps_i  = (y_scale/16).*ones(T, n);
    s_eps_i      = ones(T, n);
    theta        = zeros(n, max(q));
    lam_tau      = sqrt(s2_lam)*ones(n, 1);
    beta_tau     = zeros(n, 1);
    gam_dtau_c   = sqrt(s2_gam);
    gam_dtau_i   = sqrt(s2_gam)*ones(n, 1);
    lam_eps      = sqrt(s2_lam)*ones(n, 1);
    beta_eps     = zeros(n, 1);
    gam_eps_c    = sqrt(s2_gam);
    gam_eps_i    = sqrt(s2_gam)*ones(n, 1);
    ps           = a_ps/(a_ps+b_ps);
    s_probs      = [ps, (1-ps)/(n_s_vals-1)*ones(1, n_s_vals-1)];
    s_probs_c    = s_probs;
    s_probs_i    = repmat(s_probs, n, 1);    
else
    alpha_tau     = initial.alpha_tau;
    sigma_dtau_c  = initial.sigma_dtau_c;
    sigma_dtau_i  = initial.sigma_dtau_i;
    alpha_eps     = initial.alpha_eps;
    sigma_eps_c   = initial.sigma_eps_c;
    s_eps_c       = initial.s_eps_c;
    sigma_eps_i   = initial.sigma_eps_i;
    s_eps_i       = initial.s_eps_i;
    theta         = initial.theta;
    lam_tau       = initial.lam_tau;
    if isfield(initial, 'beta_tau'), beta_tau = initial.beta_tau; else, beta_tau = zeros(n, 1); end
    gam_dtau_c    = initial.gam_dtau_c;
    gam_dtau_i    = initial.gam_dtau_i;
    lam_eps       = initial.lam_eps;
    if isfield(initial, 'beta_eps'), beta_eps = initial.beta_eps; else, beta_eps = zeros(n, 1); end
    gam_eps_c     = initial.gam_eps_c;
    gam_eps_i     = initial.gam_eps_i;
    s_probs_c     = [initial.ps_c, (1-initial.ps_c)/(n_s_vals-1)*ones(1, n_s_vals-1)];
    s_probs_i     = [initial.ps_i, (1-initial.ps_i)/(n_s_vals-1)*ones(1, n_s_vals-1)];
end

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

% Define auxiliary variables
eye_cell      = cell(n, 1); for i = 1:n, if is_timeag(i), eye_cell{i} = repmat(1/12, [1, 12]); else, eye_cell{i} = 1; end; end
eye_blk       = blkdiag(eye_cell{:});
theta_cell    = cell(n, 1); for i = 1:n, theta_cell{i} = [1, theta(i, theta_r(i, :))]; end
theta_blk     = blkdiag(theta_cell{:});
sigmaXs_eps_c = sigma_eps_c.*s_eps_c;
sigmaXs_eps_i = sigma_eps_i.*s_eps_i;

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
SSM.Sigma_eps = (1e-6)*eye(n);
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
SSM.mu_1(id_tau_c)              = 0;
SSM.Sigma_1                     = zeros(n_state);
for i = 1:n
    SSM.Sigma_1(id_tau_i{i}, id_tau_i{i}) = eye(length(id_tau_i{i})) + 1e1*ones(length(id_tau_i{i}));
    SSM.Sigma_1(id_eps_i{i}, id_eps_i{i}) = eye(length(id_eps_i{i}));
end
SSM.Sigma_1(id_tau_c, id_tau_c) = 0;
SSM.Sigma_1(id_eps_c, id_eps_c) = 0;

% Define indexing for TVCs
id_TVC = cell(n, 1);
id_tmp = 0;
for i = 1:n
    if is_timeag(i), id_TVC{i} = id_tmp + (1:12); else, id_TVC{i} = id_tmp + 1; end
    id_tmp = id_tmp + length(id_TVC{i});
end
n_TVC  = id_tmp;

% Initialize state-space model for TVCs
SSM_TVC = struct();
%%% H
SSM_TVC.H = zeros(n, n_TVC+n+size(theta_blk, 2), T);
%%% Sigma_eps
SSM_TVC.Sigma_eps = (1e-6)*eye(n);
%%% F
SSM_TVC.F = blkdiag(zeros(n_TVC+n), SSM.F([id_eps_i{:}], [id_eps_i{:}]));
for i = 1:n
    n_tmp                           = length(id_TVC{i});
    SSM_TVC.F(id_TVC{i}, id_TVC{i}) = [1, zeros(1, n_tmp-1); eye(n_tmp-1), zeros(n_tmp-1, 1)];
end
SSM_TVC.F(n_TVC+(1:n), n_TVC+(1:n)) = eye(n);
%%% G
SSM_TVC.G = blkdiag(zeros(n_TVC+n, 2*n), SSM.G([id_eps_i{:}], 2+n+(1:n)));
for i = 1:n
    n_tmp                   = length(id_TVC{i});
    SSM_TVC.G(id_TVC{i}, i) = [1; zeros(n_tmp-1, 1)];
end
SSM_TVC.G(n_TVC+(1:n), n+(1:n)) = eye(n);
%%% Sigma_eta
SSM_TVC.Sigma_eta = zeros(3*n, 3*n, T-1);
%%% mu_1 and Sigma_1
SSM_TVC.mu_1    = zeros(n_TVC+n+size(theta_blk, 2), 1);
Sigma_aux       = zeros(n_TVC); for i = 1:n, Sigma_aux(id_TVC{i}, id_TVC{i}) = 1; end
SSM_TVC.Sigma_1 = eye(n_TVC+n+size(theta_blk, 2)) ...
     + 1e1*blkdiag(Sigma_aux, ones(n), zeros(size(theta_blk, 2)));

% Initialize state variables
state_smooth = simulation_smoother([NaN(1, n); y]', SSM);
tau_c        = state_smooth(id_tau_c(1), 2:(T+1))';
tau_i        = NaN(T, n);
for i = 1:n
    tau_i(:, i) = state_smooth(id_tau_i{i}(1), 2:(T+1))';
end
eps_c        = state_smooth(id_eps_c(1), 2:(T+1))';
eps_i        = NaN(T, n);
for i = 1:n
    eps_i(:, i) = state_smooth(id_eps_i{i}(1), 2:(T+1))';
end

for i_draw = (-n_burn):n_draw
    
    if show_progress
        % Report progress        
        message = sprintf('MCT model - Draw %d/%d - Time left: %d mins %02d secs\n', i_draw, n_draw, floor(avg_time/60), round(mod(avg_time, 60)));
        fprintf([del_str, message])
        del_str  = repmat('\b', 1, length(message));
        avg_time = toc*(n_draw-i_draw)/(i_draw+1+n_burn);

        % Plot progress
        if (mod(i_draw, 10) == 0)
            y_total  = mean(y, 2, 'omitnan');
            t_common = mean(alpha_tau, 2) .* tau_c;
            t_total  = t_common + mean(tau_i, 2);
            data_all = [y_total, t_common, t_total];
            ax0 = subplot(2, 3, 1); plot0 = plot(ax0, data_all);
            xlim(ax0, [1, T]), xlabel('t')
            ylim(ax0, [min(data_all(:))-0.5, max(data_all(:))+0.5])
            ylabel(ax0, 'Common (yellow) and total trend (blue)', 'Interpreter', 'latex')
            set(plot0, {'linewidth'}, {1; 1.5; 1.5})
            set(plot0, {'color'}, {[0, 0, 0]; [252, 204,  76]/255; [108, 172, 228]/255})
            ax0 = subplot(2, 3, 2); plot0 = plot(ax0, [sigma_dtau_i, sigma_eps_i]);
            xlim(ax0, [1, T]), xlabel('t')
            ylabel(ax0, '$\sigma_{\tau, i}$ (red) and $\sigma_{\varepsilon, i}$ (blue)', 'Interpreter', 'latex')
            set(plot0, 'linewidth', 1)
            set(plot0, {'color'}, [repmat({[2/3, 0, 0]}, [n, 1]); repmat({[0, 0, 2/3]}, [n, 1])])
            ax0 = subplot(2, 3, 3); plot0 = plot(ax0, [alpha_tau, alpha_eps]);
            xlim(ax0, [1, T]), xlabel('t')
            ylabel(ax0, '$\alpha_{\tau, i}$ (red) and $\alpha_{\varepsilon, i}$ (blue)', 'Interpreter', 'latex')
            set(plot0, 'linewidth', 1)
            set(plot0, {'color'}, [repmat({[2/3, 0, 0]}, [n, 1]); repmat({[0, 0, 2/3]}, [n, 1])])
            ax0 = subplot(2, 3, 4); plot0 = plot([sigma_dtau_c, sigma_eps_c]);
            xlim(ax0, [1, T]), xlabel('t')
            ylabel(ax0, '$\sigma_{\tau, c}$ (red) and $\sigma_{\varepsilon, c}$ (blue)', 'Interpreter', 'latex')
            set(plot0, 'linewidth', 1.5)
            set(plot0, {'color'}, {[2/3, 0, 0]; [0, 0, 2/3]})
            ax0 = subplot(2, 3, 5); plot0 = plot(ax0, tau_c);
            xlim(ax0, [1, T]), xlabel('t')
            ylabel(ax0, '$\tau_{c}$', 'Interpreter', 'latex')
            set(plot0, 'linewidth', 1.5)
            set(plot0, 'color', [2/3, 0, 0])
            ax0 = subplot(2, 3, 6); plot0 = plot(ax0, eps_c);
            xlim(ax0, [1, T]), xlabel('t')
            ylabel(ax0, '$\varepsilon_{c}$', 'Interpreter', 'latex')
            set(plot0, 'linewidth', 1.5)
            set(plot0, 'color', [0, 0, 2/3])            
            drawnow
        end        
    end

    for i_thin = 1:n_thin

        % Define auxiliary variables
        theta_cell    = cell(n, 1); for i = 1:n, theta_cell{i} = [1, theta(i, theta_r(i, :))]; end
        theta_blk     = blkdiag(theta_cell{:});
        sigmaXs_eps_c = sigma_eps_c.*s_eps_c;
        sigmaXs_eps_i = sigma_eps_i.*s_eps_i;

        % Draw loadings
        H_aux = SSM.H(:, :, T+1);
        H_aux(:, [id_eps_i{:}]) = theta_blk;
        for i = 1:n
            if (i_depend(i) > 0)
                i_aux                     = i_depend(i);
                H_aux(i, id_tau_i{i_aux}) = beta_tau(i)*H_aux(i_aux, id_tau_i{i_aux});
                H_aux(i, id_eps_i{i_aux}) = beta_eps(i)*H_aux(i_aux, id_eps_i{i_aux});
            end
        end
        y_TVC = (y') - H_aux(:, [id_tau_i{:}])*state_smooth([id_tau_i{:}], 2:(T+1));
        for t = 1:T
            for i = 1:n
                if is_timeag(i)
                    SSM_TVC.H(i, id_TVC{i}, t) = (1/12)*[tau_c(t:(-1):max(1, t-11))', repmat(tau_c(1), [1, max(12-t, 0)])];
                else
                    SSM_TVC.H(i, id_TVC{i}, t) = tau_c(t);
                end
            end
            SSM_TVC.H(:, n_TVC+(1:n), t) = eps_c(t)*eye(n);
            SSM_TVC.H(:, n_TVC+n+(1:size(theta_blk, 2)), t) = H_aux(:, [id_eps_i{:}]);
        end
        for t = 2:T
            SSM_TVC.Sigma_eta(:, :, t-1) = diag([lam_tau', lam_eps', sigmaXs_eps_i(t, :)].^2);
        end
        alpha_smooth = simulation_smoother(y_TVC, SSM_TVC);
        for i = 1:n
            alpha_tau(:, i) = alpha_smooth(id_TVC{i}(1), :)';
            alpha_eps(:, i) = alpha_smooth(n_TVC+i, :)';
        end
        
        % Draw lambda
        lam_tau = update_gam(diff(alpha_tau((t_skip+1):T, :), 1, 1), nu_lam, s2_lam);
        lam_eps = update_gam(diff(alpha_eps((t_skip+1):T, :), 1, 1), nu_lam, s2_lam);
        
        % Update state-space representation
        for t = 1:T
            SSM.H(~is_timeag, id_tau_c(1), t+1) = alpha_tau(t, ~is_timeag)';
            SSM.H(is_timeag, id_tau_c, t+1)     = (1/12)*[alpha_tau(t:(-1):max(1, t-11), is_timeag)', repmat(alpha_tau(1, is_timeag)', [1, max(12-t, 0)])];
            SSM.H(:, id_eps_c(1), t+1)          = alpha_eps(t, :)';
            SSM.H(:, [id_eps_i{:}], t+1)        = theta_blk;
            SSM.Sigma_eta(:, :, t)              = diag([sigma_dtau_c(t), sigma_dtau_i(t, :), ...
                                                       sigmaXs_eps_c(t), sigmaXs_eps_i(t, :)].^2);
            for i = 1:n
                if (i_depend(i) > 0)
                    i_aux                          = i_depend(i);
                    SSM.H(i, id_tau_i{i_aux}, t+1) = beta_tau(i)*SSM.H(i_aux, id_tau_i{i_aux}, t+1);
                    SSM.H(i, id_eps_i{i_aux}, t+1) = beta_eps(i)*SSM.H(i_aux, id_eps_i{i_aux}, t+1);
                end
            end
        end
        SSM.H(:, :, 1) = SSM.H(:, :, 2);
                
        % Draw states
        state_smooth = simulation_smoother([NaN(1, n); y]', SSM);
        tau_c        = state_smooth(id_tau_c(1), 2:(T+1))';
        dtau_c       = (state_smooth(id_tau_c(1), 2:(T+1)) - state_smooth(id_tau_c(1), 1:T))';
        tau_i        = NaN(T, n);
        dtau_i       = NaN(T, n);
        for i = 1:n
            tau_i(:, i)  = state_smooth(id_tau_i{i}(1), 2:(T+1))';
            dtau_i(:, i) = (state_smooth(id_tau_i{i}(1), 2:(T+1)) - state_smooth(id_tau_i{i}(1), 1:T))';
        end
        eps_c        = state_smooth(id_eps_c(1), 2:(T+1))';
        eps_i        = NaN(T, n);
        for i = 1:n
            eps_i(:, i) = state_smooth(id_eps_i{i}(1), 2:(T+1))';            
        end
        ups_i        = (theta_blk*state_smooth([id_eps_i{:}], 2:(T+1)))';
        
        % Draw dependence loadings
        for i = 1:n
            if is_depend(i)
                i_aux       = i_depend(i);
                y_beta      = (beta_tau(i) .* tau_i((t_skip+1):T, i_aux) + beta_eps(i) .* ups_i((t_skip+1):T, i_aux) + eps_i((t_skip+1):T, i)) ./ sigmaXs_eps_i((t_skip+1):T, i);
                X_beta      = [tau_i((t_skip+1):T, i_aux), ups_i((t_skip+1):T, i_aux)] ./ sigmaXs_eps_i((t_skip+1):T, i);
                P_beta      = (X_beta')*(X_beta);
                Pinv_beta   = pinv(prec_beta*eye(2) + P_beta);
                m_beta      = Pinv_beta*(X_beta')*y_beta;
                b_beta      = mvnrnd(m_beta, symmetrize(Pinv_beta));
                beta_tau(i) = b_beta(1);
                beta_eps(i) = b_beta(2);
            end
        end

        % Draw theta
        for i = 1:n
            if (q(i) > 0)
                y_theta = ups_i((t_skip+1):T, i) ./ sigmaXs_eps_i((t_skip+1):T, i);
                x_theta = NaN(T-t_skip, q(i));
                for lag = 1:q(i), x_theta(:, lag) = eps_i((t_skip-lag+1):(T-lag), i) ./ sigmaXs_eps_i((t_skip+1):T, i); end
                theta(i, 1:q(i)) = update_theta(y_theta, x_theta, prec_MA);
            end
        end
        
        % Draw volatilities
        sigma_dtau_c = update_vol([NaN(t_skip, 1); dtau_c((t_skip+1):T)], sigma_dtau_c, gam_dtau_c, 0);
        for i = 1:n
            sigma_dtau_i(:, i) = update_vol([NaN(t_skip, 1); dtau_i((t_skip+1):T, i)], sigma_dtau_i(:, i), gam_dtau_i(i));
        end
        sigma_eps_c  = update_vol([NaN(t_skip, 1); eps_c((t_skip+1):T)./s_eps_c((t_skip+1):T)], sigma_eps_c, gam_eps_c, 0);
        for i = 1:n
            sigma_eps_i(:, i)  = update_vol([NaN(t_skip, 1); eps_i((t_skip+1):T, i)./s_eps_i((t_skip+1):T, i)], sigma_eps_i(:, i), gam_eps_i(i));
        end
                   
        % Draw gamma
        gam_dtau_c = update_gam(2*diff(log(sigma_dtau_c((t_skip+1):T)), 1, 1), nu_gam, s2_gam);
        gam_dtau_i = update_gam(2*diff(log(sigma_dtau_i((t_skip+1):T, :)), 1, 1), nu_gam, s2_gam);
        gam_eps_c  = update_gam(2*diff(log(sigma_eps_c((t_skip+1):T)), 1, 1), nu_gam, s2_gam);
        gam_eps_i  = update_gam(2*diff(log(sigma_eps_i((t_skip+1):T, :)), 1, 1), nu_gam, s2_gam);

        % Draw scales (outliers)
        s_eps_c = update_scl([NaN(t_skip, 1); eps_c((t_skip+1):T)./sigma_eps_c((t_skip+1):T)], s_vals, s_probs_c);
        for i = 1:n
            s_eps_i(:, i) = update_scl([NaN(t_skip, 1); eps_i((t_skip+1):T, i)./sigma_eps_i((t_skip+1):T, i)], s_vals, s_probs_i(i, :));                
        end
        
        % Draw ps 
        ps_c      = update_ps(s_eps_c((t_skip+1):T), a_ps, b_ps);
        ps_i      = update_ps(s_eps_i((t_skip+1):T, :), a_ps, b_ps);
        s_probs_c = [ps_c, (1-ps_c)/(n_s_vals-1)*ones(1, n_s_vals-1)];
        s_probs_i = [ps_i, (1-ps_i)/(n_s_vals-1)*ones(1, n_s_vals-1)];
                
    end
    
    % Compute data draws
    y_draw = NaN(n, T);
    for t = 1:T
        y_draw(:, t) = SSM.H(:, :, t+1)*state_smooth(:, t+1);
    end

    % Save draws
    if (i_draw > 0)
        output.tau_c(:, i_draw)           = tau_c;
        output.alpha_tau(:, :, i_draw)    = alpha_tau;
        output.sigma_dtau_c(:, i_draw)    = sigma_dtau_c;
        output.tau_i(:, :, i_draw)        = tau_i;
        output.sigma_dtau_i(:, :, i_draw) = sigma_dtau_i;
        output.eps_c(:, i_draw)           = eps_c;
        output.alpha_eps(:, :, i_draw)    = alpha_eps;
        output.sigma_eps_c(:, i_draw)     = sigma_eps_c;
        output.s_eps_c(:, i_draw)         = s_eps_c;
        output.eps_i(:, :, i_draw)        = eps_i;
        output.sigma_eps_i(:, :, i_draw)  = sigma_eps_i;
        output.s_eps_i(:, :, i_draw)      = s_eps_i;
        output.theta(:, :, i_draw)        = theta;
        output.lam_tau(:, i_draw)         = lam_tau;
        output.beta_tau(:, i_draw)        = beta_tau;
        output.gam_dtau_c(i_draw)         = gam_dtau_c;
        output.gam_dtau_i(:, i_draw)      = gam_dtau_i;
        output.lam_eps(:, i_draw)         = lam_eps;
        output.beta_eps(:, i_draw)        = beta_eps;
        output.gam_eps_c(i_draw)          = gam_eps_c;
        output.ps_c(i_draw)               = ps_c;
        output.gam_eps_i(:, i_draw)       = gam_eps_i;
        output.ps_i(:, i_draw)            = ps_i;
        output.y_draw(:, :, i_draw)       = y_draw';
    end
    
end

end
