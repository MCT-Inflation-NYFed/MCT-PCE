function gam = update_gam(x, nu_prior, s2_prior)
% UPDATE_GAM  Update gamma in the model:
%   x_t   = gamma * eps_t,
%   eps_t ~ N(0, 1).
% 
%   GAM = UPDATE_GAM(X, NU_PRIOR, S2_PRIOR) returns updated std GAM based
%   observables X, prior degrees of freedom NU_PRIOR and prior location
%   S2_PRIOR:
%     GAM is Nx1.
%     X is TxN.
%     NU_PRIOR is Nx1.
%     S2_PRIOR is Nx1.
%
%   Version: 2021 Dec 01 - Matlab R2020a

% Transform inputs to row vector
nu_prior = nu_prior(:);
s2_prior = s2_prior(:);

% Draws from inverse gamma distribution
T       = size(x, 1);
nu_post = nu_prior + T;
s2_post = (nu_prior./nu_post).*s2_prior + (1./nu_post).*(sum(x.^2, 1)');
gam     = 1./sqrt(gamrnd(nu_post/2, 2./(nu_post.*s2_post)));

end
