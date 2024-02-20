function ps = update_ps(x, a_prior, b_prior)
% UPDATE_PS  Update probs in the model:
%   x_t ~ Bernoulli(ps),
%   with ps ~ Beta(a_prior, b_prior).
% 
%   PS = UPDATE_PS(X, ALPHA_PRIOR, BETA_PRIOR) returns updated probs PS
%   based on observables X and prior parameters A_PRIOR and B_PRIOR:
%     PS is Nx1.
%     X is TxN.
%     A_PRIOR is Nx1.
%     B_PRIOR is Nx1.
%
%   Version: 2021 Dec 01 - Matlab R2020a

% Transform inputs to row vector
a_prior = a_prior(:);
b_prior  = b_prior(:);

% Draws from beta distribution
alpha_post = a_prior + sum(x == 1, 1)';
beta_post  = b_prior + sum(~(x == 1), 1)';
ps         = betarnd(alpha_post, beta_post);

end