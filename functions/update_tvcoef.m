function alpha = update_tvcoef(y, x, sigma, lambda, var_init)
 % UPDATE_ALPHA  Update coefficients in the TVP regression model:
%   y_t          = alpha_t*x_t + sigma_t .* eps_t,
%   vec(alpha_t) = vec(alpha_(t-1)) + lambda .* ups_t,
%   alpha_0        ~ N(0, var_init),
%   (eps_t, ups_t) ~ N(0, I).
% 
%   ALPHA = UPDATE_ALPHA(Y, X, SIGMA, LAMBDA, VAR_INIT) returns updated
%   coefficients ALPHA based on regressands Y, regressors X, std of errors
%   SIGMA, std of coef's LAMBDA and initial condition variance VAR_INIT:
%     ALPHA is Tx(N*K), row t is vec(alpha_t).
%     Y is TxN.
%     X is TxK.
%     SIGMA is TxN.
%     LAMBDA is Nx1.
%     VAR_INIT is (N*K)x(N*K), defaults to 1e-1*eye(N*K)+1e2*kron(eye(K), ones(N)).
%
%   Version: 2021 Dec 01 - Matlab R2020a

% Recover dimensions
[T, N] = size(y);
K      = size(x, 2);
n_coef = N*K;

% Define symmetrize function handle
symmetrize = @(A) (A + A')/2;

% Pre-allocate coefficients
alpha = NaN(T, n_coef);

% Define state-space objects
Sigma_eta = diag(lambda.^2);
if (nargin < 5), var_init = 1e-1*eye(n_coef) + 1e2*kron(eye(K), ones(N)); end

% Pre-allocate filtering output
X1_KF = zeros(n_coef, T);
P1_KF = zeros(n_coef, n_coef, T);
X2_KF = zeros(n_coef, T);
P2_KF = zeros(n_coef, n_coef, T);
X1    = zeros(n_coef, 1);
P1    = var_init;

% Run Kalman filter
for t = 1:T
    
    % Create time-varying state-space objects
    y_t       = y(t, :)';
    H         = kron(x(t, :), eye(N));
    Sigma_eps = diag((sigma(t, :).^2));
        
    % Do prediction step
    miss = isnan(y_t);
    X2   = X1;
    P2   = P1 + Sigma_eta;
           
    % Compute Kalman gain
    
    % Do updating step
    if all(miss)
        X1 = X2;
        P1 = P2;        
    else
        e  = y_t(~miss) - H(~miss, :)*X2;
        [K_denom, n_cond] = linsolve(H(~miss, :)*P2*(H(~miss, :)') + Sigma_eps(~miss, ~miss), eye(length(y_t(~miss))));
        if (n_cond < 1e-12), K_denom = pinv(H(~miss, :)*P2*(H(~miss, :)') + Sigma_eps(~miss, ~miss)); end
        K  = P2*(H(~miss, :)') * K_denom;
        X1 = X2 + K*e;
        P1 = (eye(n_coef) - K*H(~miss, :))*P2;
    end
    
    % Store filter results
    X1_KF(:, t)    = X1;
    P1_KF(:, :, t) = P1;
    X2_KF(:, t)    = X2;
    P2_KF(:, :, t) = P2;
    
end

% Draw state via Carter-Kohn algorithm
X3 = X1;
P3 = P1;
X  = mvnrnd(X3, symmetrize(P3))';
alpha(T, :) = X';

for t = (T-1):(-1):1
    
    % Recover means and variances
    X1    = X1_KF(:, t);
    P1    = P1_KF(:, :, t);
    X2    = X2_KF(:, t+1);
    P2    = P2_KF(:, :, t+1);
    
    % Draw state
    [P3_denom, n_cond] = linsolve(P2, P1);
    if (n_cond < 1e-12), P3_denom = pinv(P2)*P1; end
    X3 = X1 + P3_denom'*(X - X2);
    P3 = P1 - P3_denom'*P1;
    X  = mvnrnd(X3, symmetrize(P3))';
    alpha(t, :) = X';
end

end