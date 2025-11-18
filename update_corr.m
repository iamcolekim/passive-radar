%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PASSIVE RADAR PROCESSING USING BUILT-IN dsp.LMSFilter (NLMS MODE)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

%% ----------------------------
% File paths
% ----------------------------
fid_x = fopen('/MATLAB Drive/9f_880M_01_11_car_2.cs8', 'r');    % Reference
fid_y = fopen('/MATLAB Drive/9f_880M_01_11_car_2.cs8', 'r');    % Surveillance

if fid_x < 0 || fid_y < 0
    error('Could not open one or both input files.');
end

%% ----------------------------
% Parameters
% ----------------------------
len = 2^15;       % samples per block
Fs  = 20e6;       % sampling frequency (for plotting only)

L   = 2^15;        % NLMS filter length
mu  = 0.3;        % NLMS step size (0.1–0.5 typical)

%% ----------------------------
% Create the NLMS clutter canceller
% ----------------------------
nlms = dsp.LMSFilter( ...
    'Length', L, ...
    'StepSize', mu, ...
    'Method', 'Normalized LMS');

disp('Starting continuous passive radar processing...');

block_idx = 1;

%% ----------------------------
% Main processing loop
% ----------------------------
while true

    %% --- Read raw bytes ---
    raw_x = fread(fid_x, 2*len, 'int8');
    raw_y = fread(fid_y, 2*len, 'int8');

    if numel(raw_x) < 2*len || numel(raw_y) < 2*len
        disp('End of file reached.');
        break;
    end

    %% --- Convert to complex ---
    x = double(raw_x(1:2:end)) + 1i * double(raw_x(2:2:end));
    y = double(raw_y(1:2:end)) + 1i * double(raw_y(2:2:end));

    %% --- Remove DC ---
    x = x - mean(x);
    y = y - mean(y);

    %% ----------------------------------------------------
    % NLMS clutter cancellation using built-in dsp.LMSFilter:
    %
    % [y_hat, err] = nlms(x , y);
    %
    % where:
    %   y_hat = NLMS estimate of clutter/DPI
    %   err   = y - y_hat = CLEAN TARGET CHANNEL
    %
    % -----------------------------------------------------
    [y_hat, y_clean] = nlms(x, y);

    %% --- Cross-correlate reference with clean surveillance ---
    [R, lags] = xcorr(x, y_clean);
    R_norm = abs(R) ./ max(abs(R));

    %% --- Plot results ---
    figure(1); clf;
    plot(lags/Fs*1e6, 20*log10(R_norm + eps));
    title(sprintf('Cross-Correlation After NLMS (Block %d)', block_idx));
    xlabel('Lag (µs)');
    ylabel('Normalized Correlation (dB)');
    ylim([-60 10]);
    xlim([-100 100]);
    grid on;
    drawnow;

    block_idx = block_idx + 1;
end

%% Cleanup
fclose(fid_x);
fclose(fid_y);

disp('Processing complete.');

