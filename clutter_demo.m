%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PASSIVE RADAR PROCESSING USING dsp.LMSFilter (NLMS MODE)
% - Simulated target injected AFTER NLMS has converged
% - Only first 60 blocks processed
% - Results recorded, then played back
% - Combined AVI video (3x1 layout)
% - Progress bar in command window
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

%% ----------------------------
% File paths
% ----------------------------
fid_x = fopen('/MATLAB Drive/9f_880M_01_11_car_2.cs8', 'r');    % Reference
fid_y = fopen('/MATLAB Drive/9f_880M_01_11_car_2.cs8', 'r');    % Surveillance

if fid_x < 0 || fid_y < 0
    error('Could not open input files.');
end

%% ----------------------------
% Parameters
% ----------------------------
len = 2^15;        
Fs  = 20e6;        

L   = 2^12;        
mu  = 0.3;         

adapt_blocks = 50; 
max_record   = 60; 

delay_samp = 400;   
doppler_hz = 30;    
atten      = 0.01;  

nlms = dsp.LMSFilter( ...
    'Length',   L, ...
    'StepSize', mu, ...
    'Method',   'Normalized LMS');

%% Storage
R_before_db_all     = cell(max_record,1);
R_after_db_all      = cell(max_record,1);
R_before_norm_all   = cell(max_record,1);
R_after_norm_all    = cell(max_record,1);
lags_us = [];

disp('Starting...');

block_idx = 1;

%% ========================================================================
% MAIN PROCESSING LOOP (EXIT AFTER 60 BLOCKS)
%% ========================================================================
while true

    %% Progress bar
    if block_idx <= max_record
        pct = (block_idx / max_record) * 100;
        bar_len = 30;
        filled  = round((pct/100) * bar_len);
        empty   = bar_len - filled;

        fprintf('[%s%s] %5.1f%% (%d/%d blocks)\r', ...
            repmat('#',1,filled), repmat('-',1,empty), ...
            pct, block_idx, max_record);
    end

    %% Read block
    raw_x = fread(fid_x, 2*len, 'int8');
    raw_y = fread(fid_y, 2*len, 'int8');

    if numel(raw_x) < 2*len || numel(raw_y) < 2*len
        fprintf('\nEOF reached early.\n');
        break;
    end

    x = double(raw_x(1:2:end)) + 1i * double(raw_x(2:2:end));
    y = double(raw_y(1:2:end)) + 1i * double(raw_y(2:2:end));

    x = x - mean(x);
    y = y - mean(y);

    %% Inject target AFTER NLMS converges
    if block_idx > adapt_blocks
        t = (0:length(y)-1).' / Fs;
        y_delayed = [zeros(delay_samp,1); y(1:end-delay_samp)];
        y_doppler = y_delayed .* exp(1i*2*pi*doppler_hz*t);
        y = y + atten * y_doppler;
    end

    %% NLMS clutter cancellation
    [~, y_clean] = nlms(x, y);

    %% Correlations
    [R_before, lags] = xcorr(x, y);
    R_after          = xcorr(x, y_clean);

    R_before_db = 20*log10(abs(R_before) + 1e-12);
    R_after_db  = 20*log10(abs(R_after)  + 1e-12);

    R_before_norm_db = 20*log10(abs(R_before / max(abs(R_before))) + 1e-12);
    R_after_norm_db  = 20*log10(abs(R_after  / max(abs(R_after)))  + 1e-12);

    %% Store for playback
    if block_idx <= max_record

        if isempty(lags_us)
            lags_us = lags/Fs * 1e6;
        end

        R_before_db_all{block_idx}   = R_before_db;
        R_after_db_all{block_idx}    = R_after_db;

        R_before_norm_all{block_idx} = R_before_norm_db;
        R_after_norm_all{block_idx}  = R_after_norm_db;
    end

    %% EXIT after exactly 60 blocks
    if block_idx == max_record
        fprintf('\nRecording complete. Exiting processing loop...\n');
        break;
    end

    block_idx = block_idx + 1;
end

fclose(fid_x);
fclose(fid_y);

fprintf('Starting playback and video generation...\n');

%% ========================================================================
% SET UP COMBINED VIDEO (AVI — works in MATLAB Online)
%% ========================================================================
video_filename = 'PassiveRadar_Corr_Combined.avi';
v = VideoWriter(video_filename, 'Motion JPEG AVI');
v.FrameRate = 20;
open(v);

fig = figure(1);
set(fig, 'Position', [100 100 900 900]);

%% ========================================================================
% PLAYBACK + VIDEO CAPTURE
%% ========================================================================
for k = 1:max_record

    figure(fig); clf;

    %% ---------- 1) Non-normalized BEFORE/AFTER ----------
    subplot(3,1,1);
    plot(lags_us, R_before_db_all{k}, 'b', 'LineWidth', 1.2); hold on;
    plot(lags_us, R_after_db_all{k},  'r', 'LineWidth', 1.2);
    title(sprintf('Non-normalized Cross-Correlation (Block %d)', k));
    xlabel('Lag (\mus)');
    ylabel('Correlation (dB)');
    legend('Before NLMS','After NLMS');
    grid on;
    ylim([-200 200]); xlim([-100 100]);

    %% ---------- 2) Normalized BEFORE ----------
    subplot(3,1,2);
    plot(lags_us, R_before_norm_all{k}, 'b', 'LineWidth', 1.2);
    title(sprintf('Normalized BEFORE NLMS (Block %d)', k));
    xlabel('Lag (\mus)');
    ylabel('Norm Corr (dB)');
    grid on;
    ylim([-80 5]); xlim([-100 100]);

    %% ---------- 3) Normalized AFTER ----------
    subplot(3,1,3);
    plot(lags_us, R_after_norm_all{k}, 'r', 'LineWidth', 1.2);
    title(sprintf('Normalized AFTER NLMS (Block %d)', k));
    xlabel('Lag (\mus)');
    ylabel('Norm Corr (dB)');
    grid on;
    ylim([-80 5]); xlim([-100 100]);

    drawnow;

    % Add frame to video
    frame = getframe(fig);
    writeVideo(v, frame);
end

close(v);
fprintf('Finished! Video saved to: %s\n', video_filename);

