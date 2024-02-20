%%% AUTOMATED ESTIMATION OF INFLATION MODELS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate Univariate and Multivariate Core Trend models on the most recent
% vintage of PCE data. Summaries and a report are produced.
%
% Version: 2022 Mar 03 - Matlab R2020a
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear memory
clear
close all
clc

% Determine tasks
estimation = true;
date_used  = '202310';

% Set directories
rng(2022)
addpath('functions');
data_path = [pwd filesep 'data' filesep];
res_path  = [pwd filesep 'results' filesep];
fig_path  = [pwd filesep 'figures' filesep];
tab_path  = [pwd filesep 'tables' filesep];
repo_path = [pwd filesep 'reports' filesep];


%% DATA

% Extract data
pce_m59           = load([data_path 'pce_m59_' date_used '.mat']);
date_str          = pce_m59.date_str;
dates             = pce_m59.dates;
sample            = (dates >= datetime(1960, 1, 1));
dates             = dates(sample);
labels_short      = pce_m59.labels_short;
infla_agg         = pce_m59.infla_agg(sample, :);
infla_agg_xfe     = pce_m59.infla_agg_xfe(sample, :);
infla_disagg      = pce_m59.infla_disagg(sample, :);
infla_12m_agg     = pce_m59.infla_12m_agg(sample, :);
infla_12m_agg_xfe = pce_m59.infla_12m_agg_xfe(sample, :);
infla_12m_disagg  = pce_m59.infla_12m_disagg(sample, :);
share             = pce_m59.share(sample, :);
share_xfe         = pce_m59.share_xfe(sample, :);
is_xfe            = (mean(share_xfe, 1) > 0.001);
fprintf('- PCE data for %s/%s\n\n', date_str(5:6), date_str(1:4))

% Create directory for figures and update path
if ~exist([fig_path 'current'], 'dir')
    mkdir([fig_path 'current'])
end
if ~exist([fig_path date_str], 'dir')
    mkdir([fig_path date_str])
end
fig_current_path = [fig_path 'current' filesep];
fig_archive_path = [fig_path date_str filesep];

% Set dimensions and tail probabilities for interval estimates
T      = length(dates);
n      = size(infla_disagg, 2);
signif = 1/6;

% Set indexes for aggregation
agg_list         = {1:8, 10:17, 9};
n_agg            = length(agg_list);
agg_names        = {'Goods', 'Services ex. housing', 'Housing'};
infla_aggreg     = NaN(length(pce_m59.dates), n_agg);
infla_12m_aggreg = NaN(length(pce_m59.dates), n_agg);
infla_m_disagg   = 100*((1+pce_m59.infla_disagg/100).^(1/12)-1);
for i_agg = 1:n_agg
    infla_aggreg(:, i_agg) = sum(pce_m59.share_xfe(:, agg_list{i_agg}) .* infla_m_disagg(:, agg_list{i_agg}), 2)./sum(pce_m59.share_xfe(:, agg_list{i_agg}), 2);
    for t = 12:size(infla_aggreg, 1)
        infla_12m_aggreg(t, i_agg) = 100*( prod((1+infla_aggreg((t-11):t, i_agg)/100), 1) - 1);
    end
end
infla_aggreg     = 100*((1+infla_aggreg(sample, :)/100).^12 - 1);
infla_12m_aggreg = infla_12m_aggreg(sample, :);


%% ESTIMATION

% Set estimation settings
settings               = struct();
settings.show_progress = true;
settings.n_draw        = 3000;
settings.n_burn        = 3000;
settings.n_thin        = 2;

% Set theta/lambda/gamma/ps priors
nper          = 12;
ps_mean       = 1-1/(4*nper);
ps_prior_obs  = 10*nper;
prior         = struct();
prior.prec_MA = 0.1;
prior.nu_lam  = 12;
prior.s2_lam  = 0.25^2/60/nper;
prior.nu_gam  = 60;
prior.s2_gam  = 1/60/nper;
prior.a_ps    = ps_mean*ps_prior_obs;
prior.b_ps    = (1-ps_mean)*ps_prior_obs;

% Set number of MA lags, time-aggregated sectors and dependent sectors
n_lags    = repmat(3, [n, 1]);
is_timeag = false(n, 1); %is_timeag(9) = true;
i_depend  = zeros(n, 1);


if (estimation == true)
    
    % Perform estimation of multivariate model
    fprintf('Estimating MCT model\n')
    prior.nu_gam       = 60;
    settings.n_lags    = n_lags;
    settings.is_timeag = is_timeag;
    settings.i_depend  = i_depend;
    output_MCT         = estimate_MCT(infla_disagg, prior, settings);

    % Normalize common components
    output_MCT.alpha_tau    = output_MCT.alpha_tau .* permute(repmat(output_MCT.sigma_dtau_c, 1, 1, n), [1 3 2]);
    output_MCT.tau_c        = output_MCT.tau_c ./ output_MCT.sigma_dtau_c;
    output_MCT.sigma_dtau_c = output_MCT.sigma_dtau_c ./ output_MCT.sigma_dtau_c;
    output_MCT.alpha_eps    = output_MCT.alpha_eps .* permute(repmat(output_MCT.sigma_eps_c, 1, 1, n), [1 3 2]);
    output_MCT.eps_c        = output_MCT.eps_c ./ output_MCT.sigma_eps_c;
    output_MCT.sigma_eps_c  = output_MCT.sigma_eps_c ./ output_MCT.sigma_eps_c;

    % Recover number of draws
    n_draw = settings.n_draw;

    % Compute multivariate core trend
    trend_sector_c_draws = squeeze(output_MCT.alpha_tau.*permute(repmat(output_MCT.tau_c, 1, 1, n), [1 3 2]));
    trend_sector_i_draws = output_MCT.tau_i;
    trend_sector_draws   = trend_sector_c_draws + trend_sector_i_draws;
    MCT_c_draws          = squeeze(sum(repmat(share_xfe, 1, 1, n_draw).*trend_sector_c_draws, 2));
    MCT_i_draws          = squeeze(sum(repmat(share_xfe, 1, 1, n_draw).*trend_sector_i_draws, 2));
    MCT_draws            = MCT_c_draws + MCT_i_draws;
    trend_sector_c       = median(trend_sector_c_draws, 3);
    trend_sector_i       = median(trend_sector_i_draws, 3);
    trend_sector         = median(trend_sector_draws, 3);
    MCT_c                = quantile(MCT_c_draws, [signif, 0.5, 1-signif], 2);
    MCT_i                = quantile(MCT_i_draws, [signif, 0.5, 1-signif], 2);
    MCT                  = quantile(MCT_draws,   [signif, 0.5, 1-signif], 2);

    % Compute decomposition of sectoral contributions
    MCT_sector_part   = median(repmat(share_xfe, 1, 1, n_draw).*trend_sector_draws, 3);
    MCT_agg_part      = NaN(T, n_agg);
    MCT_agg_share     = NaN(T, n_agg);
    for i_agg = 1:n_agg
        MCT_agg_part(:, i_agg)  = sum(MCT_sector_part(:, agg_list{i_agg}), 2);
        MCT_agg_share(:, i_agg) = sum(share_xfe(:, agg_list{i_agg}), 2);
    end
    trend_sector_part = mean(repmat(share, 1, 1, n_draw).*trend_sector_draws, 3);

    % Compute decomposition of sectoral contributions to common/idiosyncratic
    MCT_sector_c_part = median(repmat(share_xfe, 1, 1, n_draw).*trend_sector_c_draws, 3);
    MCT_agg_c_part    = NaN(T, n_agg);
    for i_agg = 1:n_agg
        MCT_agg_c_part(:, i_agg) = sum(MCT_sector_c_part(:, agg_list{i_agg}), 2);
    end
    MCT_sector_i_part = median(repmat(share_xfe, 1, 1, n_draw).*trend_sector_i_draws, 3);
    MCT_agg_i_part    = NaN(T, n_agg);
    for i_agg = 1:n_agg
        MCT_agg_i_part(:, i_agg) = sum(MCT_sector_i_part(:, agg_list{i_agg}), 2);
    end

    % Compute multivariate core volatility
    MCV_c_draws = sqrt( squeeze( sum(repmat(share_xfe, 1, 1, n_draw) .* output_MCT.alpha_eps, 2).^2 ) ...
        .* (output_MCT.sigma_eps_c.^2) );
    theta_draws = permute(repmat(ones(n, n_draw) + squeeze(sum(output_MCT.theta.^2, 2)), [1, 1, T]), [3, 1, 2]);
    MCV_i_draws = sqrt( squeeze( sum(theta_draws .* (repmat(share_xfe, 1, 1, n_draw) .* output_MCT.sigma_eps_i).^2, 2) ) );
    MCV_draws   = sqrt(MCV_c_draws.^2 + MCV_i_draws.^2);
    MCV_c       = quantile(MCV_c_draws, [signif, 0.5, 1-signif], 2);
    MCV_i       = quantile(MCV_i_draws, [signif, 0.5, 1-signif], 2);
    MCV         = quantile(MCV_draws,   [signif, 0.5, 1-signif], 2);

    % Compute outlier probabilities
    outlier_ind = output_MCT.s_eps_i;
    outlier_ind(outlier_ind == 1) = 0;
    outlier_ind(outlier_ind > 1)  = 1;
    outlier_ind = mean(outlier_ind, 3);

    % Decompose updates
    update    = cell(n_agg + 1, 1);
    share_tmp = share_xfe(end, :);
    update{1} = decompose_update(infla_disagg, share_tmp, output_MCT, is_timeag, i_depend, 12);
    for i_agg = 1:n_agg
        share_tmp                  = zeros(1, n);
        share_tmp(agg_list{i_agg}) = share_xfe(end, agg_list{i_agg}) ./ sum(share_xfe(end, agg_list{i_agg}), 2);
        update{i_agg+1}            = decompose_update(infla_disagg, share_tmp, output_MCT, is_timeag, i_depend, 12);
    end

    % Save results
    output                        = struct();
    output.param_MCT              = struct();
    output.param_MCT.theta        = quantile(output_MCT.theta, [signif, 0.5, 1-signif], 3);
    output.param_MCT.beta_tau     = quantile(output_MCT.beta_tau, [signif, 0.5, 1-signif], 2);
    output.param_MCT.lam_tau      = quantile(output_MCT.lam_tau, [signif, 0.5, 1-signif], 2);
    output.param_MCT.gam_dtau_c   = quantile(output_MCT.gam_dtau_c, [signif, 0.5, 1-signif], 2);
    output.param_MCT.gam_dtau_i   = quantile(output_MCT.gam_dtau_i, [signif, 0.5, 1-signif], 2);
    output.param_MCT.beta_eps     = quantile(output_MCT.beta_eps, [signif, 0.5, 1-signif], 2);
    output.param_MCT.lam_eps      = quantile(output_MCT.lam_eps, [signif, 0.5, 1-signif], 2);
    output.param_MCT.gam_eps_c    = quantile(output_MCT.gam_eps_c, [signif, 0.5, 1-signif], 2);
    output.param_MCT.gam_eps_i    = quantile(output_MCT.gam_eps_i, [signif, 0.5, 1-signif], 2);
    output.param_MCT.alpha_tau    = quantile(output_MCT.alpha_tau, [signif, 0.5, 1-signif], 3);
    output.param_MCT.alpha_eps    = quantile(output_MCT.alpha_eps, [signif, 0.5, 1-signif], 3);
    output.param_MCT.sigma_dtau_c = quantile(output_MCT.sigma_dtau_c, [signif, 0.5, 1-signif], 2);
    output.param_MCT.sigma_dtau_i = quantile(output_MCT.sigma_dtau_i, [signif, 0.5, 1-signif], 3);
    output.param_MCT.sigma_eps_c  = quantile(output_MCT.sigma_eps_c, [signif, 0.5, 1-signif], 2);
    output.param_MCT.sigma_eps_i  = quantile(output_MCT.sigma_eps_i, [signif, 0.5, 1-signif], 3);
    output.trend_sector_c         = trend_sector_c;
    output.trend_sector_i         = trend_sector_i;
    output.trend_sector           = trend_sector;
    output.MCT_c                  = MCT_c;
    output.MCT_i                  = MCT_i;
    output.MCT                    = MCT;
    output.MCT_sector_part        = MCT_sector_part;
    output.MCT_agg_part           = MCT_agg_part;
    output.MCT_agg_share          = MCT_agg_share;
    output.trend_sector_part      = trend_sector_part;
    output.MCT_sector_c_part      = MCT_sector_c_part;
    output.MCT_agg_c_part         = MCT_agg_c_part;
    output.MCT_sector_i_part      = MCT_sector_i_part;
    output.MCT_agg_i_part         = MCT_agg_i_part;
    output.MCV_c                  = MCV_c;
    output.MCV_i                  = MCV_i;
    output.MCV                    = MCV;
    output.outlier_ind            = outlier_ind;
    output.update                 = update;
    save([res_path 'results_current.mat'], '-struct', 'output')
    save([res_path sprintf('results_%s.mat', date_str)], '-struct', 'output')
end



%% FIGURES

% Load or re-load estimation results
load([res_path 'results_' date_used '.mat'])

% Define subsample for recent data
subsample        = (year(dates) >= 2017); 
subsubsample     = (year(dates) >= 2017); 
trend_c_norm     = mean(MCT_c(year(dates) <= 2019 & year(dates) >= 2017, 2));
trend_i_norm     = mean(MCT_i(year(dates) <= 2019 & year(dates) >= 2017, 2));
trend_agg_c_norm = mean(MCT_agg_c_part(year(dates) <= 2019 & year(dates) >= 2017, :), 1);
trend_agg_i_norm = mean(MCT_agg_i_part(year(dates) <= 2019 & year(dates) >= 2017, :), 1);

% Define colors and figure format
blue    = [  1,  87, 155]/255;
red     = [4/5, 1/5, 1/5];
green   = [2/5, 4/5, 2/5];
lgblue  = [108, 172, 228]/255;
yellow  = [252, 204,  76]/255;
orange  = [255, 165,   0]/255;
pink    = [254,   1, 154]/255;
violet  = [143,   0, 255]/255;
black   = [  0,   0,   0];
white   = [  1,   1,   1];
dec_color = {lgblue; yellow};
agg_color = {red; blue; green};
agg_extended_color = {pink; lgblue; yellow; red; blue; green};
sec_color = {0.5*black+0.5*red;    yellow; red;    0.5*white+0.5*red;    orange; white; ...
             0.5*black+0.5*lgblue; green;  lgblue; 0.5*white+0.5*lgblue; violet; 0.7*green+0.3*lgblue; 0.7*white+0.2*black; black};
fig_fmt = 'png';

% Define auxiliary names
agg_extended_names = cell(n_agg, 2);
for i_agg = 1:n_agg
    agg_extended_names(i_agg, :) = {['Sector-specific ' lower(agg_names{i_agg})], ...
                                    ['Common ' lower(agg_names{i_agg})]};
end
agg_extended_names = reshape(agg_extended_names, [1, 2*n_agg]);


% Plot multivariate trend over full sample %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig0  = figure();
ax0   = axes();
fill(ax0, [dates', fliplr(dates')], [MCT(:, 1)', fliplr(MCT(:, 3)')], ...
    blue, 'facealpha', 0.2, 'linestyle', ':', 'linewidth', 0.1, 'edgecolor', blue);
hold('on')
plot0 = plot(ax0, dates, [infla_12m_agg, infla_12m_agg_xfe, MCT(:, 2)]); 
hold('off')

% Add extras
xlim(ax0, [dates(1), dates(end)])
ylim(ax0, [-5, 15])
xticks(ax0, datetime(year(dates(1)):5:year(dates(end)), 1, 1))

% Tune ax handle
set(ax0, 'box', 'on')
grid(ax0, 'on')

% Tune plot
legend(plot0, {'Headline PCE inflation (YoY)', 'Core PCE Inflation (YoY)', 'Multivariate Core Trend'}, 'location', 'best')
set(plot0, {'linewidth'}, {2; 2; 3})
set(plot0, {'color'}, {black; black; blue})
set(plot0, {'linestyle'}, {':'; '-'; '-'})

% Tune and save figure
set(fig0, 'units', 'inches', 'position', [0 0 10 6])
print(fig0, [fig_current_path 'MCT_full'], ['-d' fig_fmt])
print(fig0, [fig_archive_path 'MCT_full'], ['-d' fig_fmt])

% Tune and save figure in recent sample
xlim(ax0, [dates(find(subsample, 1, 'first')), dates(end)])
ylim(ax0, [0, 7])
xticks(ax0, datetime(unique(year(dates(subsample))), 1, 1))
legend(plot0, {'Headline PCE inflation (YoY)', 'Core PCE Inflation (YoY)', 'Multivariate Core Trend'}, ...
    'location', 'best')
set(fig0, 'units', 'inches', 'position', [0 0 10 6])
print(fig0, [fig_current_path 'MCT_recent'], ['-d' fig_fmt])
print(fig0, [fig_archive_path 'MCT_recent'], ['-d' fig_fmt])



% Plot sector-specific/common decomposition over full sample %%%%%%%%%%%%%%
fig0  = figure();
ax0   = axes();
plot0 = area(ax0, dates, [MCT_i(:, 2)-trend_i_norm, MCT_c(:, 2)-trend_c_norm]);

% Add extras
xlim(ax0, [dates(1), dates(end)])
xticks(ax0, datetime(year(dates(1)):5:year(dates(end)), 1, 1))

% Tune ax handle
set(ax0, 'box', 'on')
grid(ax0, 'on')

% Tune plot
legend(plot0, {'Sector-specific', 'Common'}, 'location', 'best')
set(plot0, {'facecolor'}, dec_color)
set(plot0, {'edgecolor'}, dec_color)
set(plot0, 'facealpha', 0.7)

% Tune and save figure
set(fig0, 'units', 'inches', 'position', [0 0 10 6])
print(fig0, [fig_current_path 'dec_specific_common_full'], ['-d' fig_fmt])
print(fig0, [fig_archive_path 'dec_specific_common_full'], ['-d' fig_fmt])

% Tune and save figure in recent sample
xlim(ax0, [dates(find(subsubsample, 1, 'first')), dates(end)])
xticks(ax0, datetime(unique(year(dates(subsubsample))), 1, 1))
legend(plot0, {'Sector-specific', 'Common'}, 'location', 'best')
set(fig0, 'units', 'inches', 'position', [0 0 10 6])
print(fig0, [fig_current_path 'dec_specific_common_recent'], ['-d' fig_fmt])
print(fig0, [fig_archive_path 'dec_specific_common_recent'], ['-d' fig_fmt])

% Plot sector-specific/common decomposition (non-stacked) %%%%%%%%%%%%%%%%%
fig0  = figure();
ax0   = axes();
fill(ax0, [dates(subsubsample)', fliplr(dates(subsubsample)')], [MCT_i(subsubsample, 1)'-trend_i_norm, fliplr(MCT_i(subsubsample, 3)'-trend_i_norm)], ...
    dec_color{1}, 'facealpha', 0.2, 'linestyle', ':', 'linewidth', 0.1, 'edgecolor', dec_color{1});
hold('on')
fill(ax0, [dates(subsubsample)', fliplr(dates(subsubsample)')], [MCT_c(subsubsample, 1)'-trend_c_norm, fliplr(MCT_c(subsubsample, 3)'-trend_c_norm)], ...
    dec_color{2}, 'facealpha', 0.2, 'linestyle', ':', 'linewidth', 0.1, 'edgecolor', dec_color{2});
hold('on')
plot0 = plot(ax0, dates(subsubsample), [MCT(subsubsample, 2)-trend_i_norm-trend_c_norm, ...
    MCT_i(subsubsample, 2)-trend_i_norm, MCT_c(subsubsample, 2)-trend_c_norm]);
hold('off')

% Add extras
xlim(ax0, [dates(find(subsubsample, 1, 'first')), dates(end)])
xticks(ax0, datetime(unique(year(dates(subsubsample))), 1, 1))

% Tune ax handle
set(ax0, 'box', 'on')
grid(ax0, 'on')

% Tune plot
legend(plot0, {'Multivariate Core Trend', 'Sector-specific', 'Common'}, 'location', 'best')
set(plot0, {'linewidth'}, {1.5; 2; 2})
set(plot0, {'linestyle'}, {'-'; '-'; '-'})
set(plot0, {'color'}, [black; dec_color])

% Tune and save figure
set(fig0, 'units', 'inches', 'position', [0 0 10 6])
print(fig0, [fig_current_path 'dec_specific_common_recent_line'], ['-d' fig_fmt])
print(fig0, [fig_archive_path 'dec_specific_common_recent_line'], ['-d' fig_fmt])


% Plot aggregates decomposition over full sample %%%%%%%%%%%%%%%%%%%%%%%%%%
fig0  = figure();
ax0   = axes();
plot0 = area(ax0, dates, MCT_agg_part-(trend_agg_i_norm+trend_agg_c_norm));

% Add extras
xlim(ax0, [dates(1), dates(end)])
xticks(ax0, datetime(year(dates(1)):5:year(dates(end)), 1, 1))

% Tune ax handle
set(ax0, 'box', 'on')
grid(ax0, 'on')

% Tune plot
legend(plot0, agg_names, 'location', 'best')
set(plot0, {'facecolor'}, agg_color)
set(plot0, {'edgecolor'}, agg_color)
set(plot0, 'facealpha', 0.7)

% Tune and save figure
set(fig0, 'units', 'inches', 'position', [0 0 10 6])
print(fig0, [fig_current_path 'dec_aggregates_full'], ['-d' fig_fmt])
print(fig0, [fig_archive_path 'dec_aggregates_full'], ['-d' fig_fmt])

% Tune and save figure in recent sample
xlim(ax0, [dates(find(subsubsample, 1, 'first')), dates(end)])
xticks(ax0, datetime(unique(year(dates(subsubsample))), 1, 1))
legend(plot0, agg_names, 'location', 'best')
set(fig0, 'units', 'inches', 'position', [0 0 10 6])
print(fig0, [fig_current_path 'dec_aggregates_recent'], ['-d' fig_fmt])
print(fig0, [fig_archive_path 'dec_aggregates_recent'], ['-d' fig_fmt])

% Plot aggregates decomposition (non-stacked) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig0  = figure();
ax0   = axes();
plot0 = plot(ax0, dates(subsubsample), [MCT(subsubsample, 2)-trend_i_norm-trend_c_norm, MCT_agg_part(subsubsample, :)-(trend_agg_i_norm+trend_agg_c_norm)]);

% Add extras
xticks(ax0, datetime(unique(year(dates(subsubsample))), 1, 1))

% Tune ax handle
set(ax0, 'box', 'on')
grid(ax0, 'on')

% Tune plot
legend(plot0, ['Multivariate Core Trend', agg_names], 'location', 'best')
set(plot0, {'linewidth'}, [{1.5}; repmat({2}, [n_agg, 1])])
set(plot0, {'linestyle'}, [{'-'}; repmat({'-'}, [n_agg, 1])])
set(plot0, {'color'}, [black; agg_color])

% Tune and save figure
set(fig0, 'units', 'inches', 'position', [0 0 10 6])
print(fig0, [fig_current_path 'dec_aggregates_recent_line'], ['-d' fig_fmt])
print(fig0, [fig_archive_path 'dec_aggregates_recent_line'], ['-d' fig_fmt])


% Plot sector-specific/common*aggregates decomposition over full sample %%%
fig0  = figure();
ax0   = axes();
plot0 = area(ax0, dates, [MCT_agg_i_part-trend_agg_i_norm, MCT_agg_c_part-trend_agg_c_norm]);

% Add extras
xlim(ax0, [dates(1), dates(end)])
xticks(ax0, datetime(year(dates(1)):5:year(dates(end)), 1, 1))

% Tune ax handle
set(ax0, 'box', 'on')
grid(ax0, 'on')

% Tune plot
legend(plot0, agg_extended_names, 'location', 'best')
set(plot0, {'facecolor'}, agg_extended_color)
set(plot0, {'edgecolor'}, agg_extended_color)
set(plot0, 'facealpha', 0.7)

% Tune and save figure
set(fig0, 'units', 'inches', 'position', [0 0 10 6])
print(fig0, [fig_current_path 'dec_specific_common_aggregates_full'], ['-d' fig_fmt])
print(fig0, [fig_archive_path 'dec_specific_common_aggregates_full'], ['-d' fig_fmt])

% Tune and save figure in recent sample
xlim(ax0, [dates(find(subsubsample, 1, 'first')), dates(end)])
xticks(ax0, datetime(unique(year(dates(subsubsample))), 1, 1))
legend(plot0, agg_extended_names, 'location', 'best')
set(fig0, 'units', 'inches', 'position', [0 0 10 6])
print(fig0, [fig_current_path 'dec_specific_common_aggregates_recent'], ['-d' fig_fmt])
print(fig0, [fig_archive_path 'dec_specific_common_aggregates_recent'], ['-d' fig_fmt])

% Plot common/idiosyncratic+aggregates decomposition (non-stacked) %%%%%%%%
fig0  = figure();
ax0   = axes();
plot0 = plot(ax0, dates(subsubsample), [MCT(subsubsample, 2)-trend_i_norm-trend_c_norm, ...
    MCT_agg_i_part(subsubsample, :)-trend_agg_i_norm, MCT_agg_c_part(subsubsample, :)-trend_agg_c_norm]);

% Add extras
xticks(ax0, datetime(unique(year(dates(subsubsample))), 1, 1))

% Tune ax handle
set(ax0, 'box', 'on')
grid(ax0, 'on')

% Tune plot
legend(plot0, ['Multivariate Core Trend', agg_extended_names], 'location', 'best')
set(plot0, {'linewidth'}, [{1.5}; repmat({2}, [2*n_agg, 1])])
set(plot0, {'linestyle'}, [{'-'}; repmat({'-'}, [2*n_agg, 1])])
set(plot0, {'color'}, [black; agg_extended_color])

% Tune and save figure
set(fig0, 'units', 'inches', 'position', [0 0 10 6])
print(fig0, [fig_current_path 'dec_specific_common_aggregates_recent_line'], ['-d' fig_fmt])
print(fig0, [fig_archive_path 'dec_specific_common_aggregates_recent_line'], ['-d' fig_fmt])


% Plot fine sectoral decomposition over full sample %%%%%%%%%%%%%%%%%%%%%%%
fig0  = figure();
ax0   = axes();
plot0 = area(ax0, dates, MCT_sector_part(:, is_xfe));

% Add extras
xlim(ax0, [dates(1), dates(end)])
xticks(ax0, datetime(year(dates(1)):5:year(dates(end)), 1, 1))

% Tune ax handle
set(ax0, 'box', 'on')
grid(ax0, 'on')

% Tune plot
legend(plot0, labels_short(is_xfe), 'location','best')
set(plot0, {'facecolor'}, sec_color)
set(plot0, {'edgecolor'}, sec_color)
set(plot0, 'facealpha', 0.7)

% Tune and save figure
set(fig0, 'units', 'inches', 'position', [0 0 10 6])
print(fig0, [fig_current_path 'dec_sectoral_full'], ['-d' fig_fmt])
print(fig0, [fig_archive_path 'dec_sectoral_full'], ['-d' fig_fmt])

% Tune and save figure in recent sample
xlim(ax0, [dates(find(subsubsample, 1, 'first')), dates(end)])
xticks(ax0, datetime(unique(year(dates(subsubsample))), 1, 1))
legend(plot0, labels_short(is_xfe), 'location','best')
set(fig0, 'units', 'inches', 'position', [0 0 10 6])
print(fig0, [fig_current_path 'dec_sectoral_recent'], ['-d' fig_fmt])
print(fig0, [fig_archive_path 'dec_sectoral_recent'], ['-d' fig_fmt])


% Plot sectoral details %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:n
    
    % Create figure
    fig0 = figure();
    ax0  = axes();
    yyaxis(ax0, 'right')
    area(ax0, dates, outlier_ind(:, i), 'facecolor', blue, 'facealpha', 0.25, 'edgecolor', white)
    ylabel(ax0, 'Outlier Probability')
    ylim(ax0, [0 1])
    yyaxis(ax0, 'left')
    hold on
    plot(ax0, dates, infla_disagg(:, i), 'color', black, 'linestyle', '--', 'linewidth', 1);
    plot(ax0, dates, trend_sector(:, i), 'color', blue, 'linestyle', '-', 'linewidth', 2)
    plot(ax0, dates, trend_sector_i(:, i), 'color', red, 'linestyle', '-', 'linewidth', 2)
    ylabel(ax0, 'Inflation (%)')
    hold off

    % Add extras
    xlim(ax0, [dates(1), dates(end)])
    xticks(ax0, datetime(year(dates(1)):5:year(dates(end)), 1, 1))
    title(ax0, labels_short{i}, 'fontsize', 16)
    
    % Tune ax handle
    set(ax0, 'box', 'on')
    grid(ax0, 'on')
    ax0.YAxis(1).Color = black;
    ax0.YAxis(2).Color = black;

    % Tune plot
    legend('Sectoral inflation (Annualized MoM)', 'Total trend', 'Sector-specific trend', 'Outlier probability (right axis)')
    legend('location', 'best')

    % Tune and save figure
    set(fig0, 'units', 'inches', 'position', [0 0 10 6])
    print(fig0, [fig_current_path sprintf('sector_%d_full', i)], ['-d' fig_fmt])
    print(fig0, [fig_archive_path sprintf('sector_%d_full', i)], ['-d' fig_fmt])

    % Save figure in recent sample
    xlim(ax0, [dates(find(subsample, 1, 'first')), dates(end)])
    xticks(ax0, datetime(unique(year(dates(subsample))), 1, 1))
    legend('Sectoral inflation (Annualized MoM)', 'Total trend', 'Sector-specific trend', 'Outlier probability (right axis)')
    legend('location', 'best')
    set(fig0, 'units', 'inches', 'position', [0 0 10 6])
    print(fig0, [fig_current_path sprintf('sector_%d_recent', i)], ['-d' fig_fmt])
    print(fig0, [fig_archive_path sprintf('sector_%d_recent', i)], ['-d' fig_fmt])
    
    
    % Create figure
    fig0 = figure();
    
    % Plot alpha_tau
    sign_tau = sign( squeeze(param_MCT.alpha_tau(end, i, 2)) );
    ax0      = subplot(2, 2, 1);
    plot0    = plot(ax0, dates, sign_tau*squeeze(param_MCT.alpha_tau(:, i, :)).*param_MCT.sigma_dtau_c, ...
        'color', black, 'linewidth', 2);
    xlim(ax0, [dates(1), dates(end)])
    ylabel(ax0, '$\alpha_{\tau, i}\sigma_{\Delta\tau, i}$', 'interpreter', 'latex')
    set(ax0, 'box', 'on')
    grid(ax0, 'on')
    set(plot0, {'linestyle'}, {':'; '-'; ':'})

    % Plot alpha_eps
    sign_eps = sign( squeeze(param_MCT.alpha_eps(end, i, 2)) );
    ax0      = subplot(2, 2, 2);
    plot0    = plot(ax0, dates, sign_eps*squeeze(param_MCT.alpha_eps(:, i, :)).*param_MCT.sigma_eps_c, ...
        'color', black, 'linewidth', 2);
    xlim(ax0, [dates(1), dates(end)])
    ylabel(ax0, '$\alpha_{\varepsilon, i}\sigma_{\varepsilon, i}$', 'interpreter', 'latex')
    set(ax0, 'box', 'on')
    grid(ax0, 'on')
    set(plot0, {'linestyle'}, {':'; '-'; ':'})

    % Plot sigma_dtau_i
    ax0   = subplot(2, 2, 3);
    plot0 = plot(ax0, dates, squeeze(param_MCT.sigma_dtau_i(:, i, :)), ...
        'color', black, 'linewidth', 2);
    xlim(ax0, [dates(1), dates(end)])
    ylabel(ax0, '$\sigma_{\Delta\tau, i}$', 'interpreter', 'latex')
    set(ax0, 'box', 'on')
    grid(ax0, 'on')
    set(plot0, {'linestyle'}, {':'; '-'; ':'})

    % Plot sigma_eps_i
    ax0   = subplot(2, 2, 4);
    plot0 = plot(ax0, dates, squeeze(param_MCT.sigma_eps_i(:, i, :)), ...
        'color', black, 'linewidth', 2);
    xlim(ax0, [dates(1), dates(end)])
    ylabel(ax0, '$\sigma_{\varepsilon, i}$', 'interpreter', 'latex')
    set(ax0, 'box', 'on')
    grid(ax0, 'on')
    set(plot0, {'linestyle'}, {':'; '-'; ':'})
    
    % Tune and save figure
    sgtitle(fig0, labels_short{i})
    set(fig0, 'units', 'inches', 'position', [0 0 10 6])
    print(fig0, [fig_current_path sprintf('sector_tvp_%d', i)], ['-d' fig_fmt])
    print(fig0, [fig_archive_path sprintf('sector_tvp_%d', i)], ['-d' fig_fmt])
    
end


%% TABLES

% Tabulate chart data for public data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
chart_fname_public      = [tab_path 'mct-charts-public_' date_str '.xlsx'];          

if exist(chart_fname_public, 'file'), delete(chart_fname_public); end
copyfile([tab_path 'charts_heading_public.xlsx'], chart_fname_public)

% Create table with chart data
results_chart      = [infla_12m_agg, infla_12m_agg_xfe, MCT,MCT(:, 2)-trend_i_norm-trend_c_norm, NaN(T, 3*n_agg)];
j                  = 1;
for i = 7:(6+n_agg)
    results_chart(:, i) = MCT_agg_part(:, j) - (trend_agg_i_norm(j)+trend_agg_c_norm(j)); 
    j                   = j + 1;    
end                                 
j                  = 1;
for i = (7+n_agg):2:size(results_chart, 2)
    results_chart(:, i)   = MCT_agg_c_part(:, j) - trend_agg_c_norm(j);
    results_chart(:, i+1) = MCT_agg_i_part(:, j) - trend_agg_i_norm(j);
    j                     = j + 1;    
end                                 
results_chart_cell = num2cell(round(results_chart, 2));
results_chart_cell = [mat2cell(datestr(dates), ones(length(dates), 1)), results_chart_cell];
tab_chart          = cell2table(results_chart_cell);

% Write to excel
writetable(tab_chart, chart_fname_public, 'sheet', 'Charts', 'range', 'A7', 'writevariablenames', false);
