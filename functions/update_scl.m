function s = update_scl(x, vals, probs)
% UPDATE_SCL  Update scales in the model:
%   x_t   = s_t * eps_t,
%   s_t   ~ Discrete(vals, probs),
%   eps_t ~ N(0, 1).
% 
%   S = UPDATE_SCL(X, VALS, PROBS) returns updated scales S based on
%   observables X, values VALS and probabilities PROBS:
%     X is Tx1.
%     VALS is N_Sx1.
%     PROBS is N_Sx1.
%
%   Version: 2021 Dec 01 - Matlab R2020a

% Transform inputs to column vector
x     = x(:);
vals  = vals(:);
probs = probs(:);
T     = size(x, 1);
n_s   = length(probs);

% Define arrays for likelihood computations
x_rep     = repmat(x, 1, n_s);
vals_rep  = repmat(vals', T, 1);
probs_rep = repmat(probs', T, 1);

% Apply Bayes rule
likelihood       = (exp(-(1/2)*(x_rep./vals_rep).^2))./vals_rep;
pxlikelihood     = likelihood .* probs_rep;
xmlikelihood     = sum(pxlikelihood, 2);
xmlikelihood_rep = repmat(xmlikelihood, 1, n_s);
posteriors       = pxlikelihood./xmlikelihood_rep;

% Set posterior to prior if data is missing
posteriors(isnan(posteriors)) = probs_rep(isnan(posteriors));

% Draw scales from posterior
weights = mnrnd(1, posteriors);
s       = weights*vals;

end