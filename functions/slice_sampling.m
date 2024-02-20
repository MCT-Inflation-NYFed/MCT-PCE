function [x1, f1, rejection] = slice_sampling(x0, f0, log_density, settings)
% SLICE_SAMPLING  Perform slice-sampling updates.
%
%   X1 = SLICE_SAMPLING(X0, F0, LOG_DENSITY, SETTINGS) returns updated
%   value X1 based on initial value X0, initial unnormalized log-density
%   F0, log-density function LOG_DENSITY and settings SETTINGS:
%     X0 and F0 are scalar.
%     LOG_DENSITY is function handle, F0 = LOG_DENSITY(X0).
%     SETTING is struct:
%     - SETTINGS.WIDTH is positive scalar.
%     - SETTINGS.N_MOVES is positive integer.
%
%   [X1, F1] = SLICE_SAMPLING(X0, F0, LOG_DENSITY, SETTINGS) also returns
%   updated value of unnormalized log-density F1 = LOG_DENSITY(X1).
%
%   [X1, F1, REJECTION] = SLICE_SAMPLING(X0, F0, LOG_DENSITY, SETTINGS)
%   also records rejections on left step, right step and shrinking step.
%
%   Version: 2019 Sep 28 - Matlab R2017b

% Preliminaries: inputs
width   = settings.width;
n_moves = settings.n_moves;

% Initialize rejection
rejection = zeros(3, 1);

% Draw from vertical slice
y0 = f0 - gamrnd(1, 1);

% Draw from horizontal slice
% Initial interval around x0
x_left  = x0 - rand()*width;
x_right = x_left + width;

% Lateral moves
n_moves_left  = floor(rand()*n_moves);
n_moves_right = n_moves - 1 - n_moves_left;

% Stepping out to the left
while (n_moves_left > 0) && (y0 < log_density(x_left))
    rejection(1) = rejection(1) + 1;
    n_moves_left = n_moves_left - 1;
    x_left       = x_left - width;
end

% Stepping out to the right
while (n_moves_right > 0) && (y0 < log_density(x_right))
    rejection(2)  = rejection(2) + 1;
    n_moves_right = n_moves_right - 1;
    x_right       = x_right + width;
end

% Shrinking
x1 = x_left + rand(1, 1)*(x_right - x_left);
f1 = log_density(x1);

while (f1 < y0)
    rejection(3) = rejection(3) + 1;
    if (x1 < x0)
        x_left  = x1;
    else
        x_right = x1;
    end
    x1 = x_left + rand()*(x_right - x_left);
    f1 = log_density(x1);
end

end