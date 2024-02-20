function theta = update_theta(y, x, prec_prior)
% UPDATE_THETA  Update theta in the moving-average model:
%   x_t   = (1 + theta_1*L + ... + theta_q*L^q) * sigma_t * eps_t,
%   eps_t ~ N(0, 1).
% 
%   THETA0 = UPDATE_THETA(THETA, X, SIGMA, VAR_PRIOR) returns updated
%   moving-average coefficients THETA based on observables X, scales SIGMA
%   and prior precision PREC_PRIOR:
%     THETA is Qx1.
%     X is Tx1.
%     SIGMA is Tx1.
%     PREC_PRIOR is a positive scalar.
%
%   Version: 2021 Dec 01 - Matlab R2020a

% Compute dimensions and set function handle
q_MA       = size(x, 2);
symmetrize = @(A) (A + A')/2;

% Construct prior precision matrix
P = zeros(q_MA);
for i_MA = 1:q_MA, P(i_MA, i_MA) = prec_prior*(i_MA^2); end 

% Draw coefficients from posterior
Pinv  = pinv(P + (x')*x);
m     = Pinv*((x')*y);
theta = mvnrnd(m, symmetrize(Pinv));

% Impose invertibility
MA_roots                    = roots([flip(theta) 1]);
MA_roots(abs(MA_roots) > 1) = 1./MA_roots(abs(MA_roots) > 1);
theta                       = poly(MA_roots);
theta                       = [theta(2:end) zeros(1, q_MA-length(theta)+1)]; % because roots removes leading zeros

end