%Antenna Results MDE - F25-05  (LP step, smoothing & interpolation + BANDPASS @ 2.4 GHz)
%11/08/2025

%% 0) Housekeeping
clear; clc; close all;

%% 1) File paths & knobs
scanCsv = 'nanovnaf_pattern_4_6_outside_w_box.csv';   % NanoVNA sweep CSV
anFile  = 'vivald_E.txt';                            % anechoic baseline
freq_c  = 5.0e9;                                      % comparison frequency (Hz)
angStep = 3;                                          % output angle grid (deg)

% Smoothing options for the ANGULAR pattern (post-normalization)
SMOOTH.use     = true;
SMOOTH.method  = 'sgolay';    % 'sgolay' | 'movmean' | 'none'
SMOOTH.win     = 7;           % odd window length (5–11 typical)
SMOOTH.poly    = 2;           % sgolay order (<= win-1)
SMOOTH.movN    = 5;           % moving-average window if method='movmean'

% LP transform display (no gating)
LP.window      = 'hann';      % 'hann'|'kaiser'|'none'
LP.show_angle  = 0;           % which angle to display impulse+step (nearest)

% NEW: Band-pass transform display (no gating, centered at 2.4 GHz)
BP.enable      = true;        % turn on/off BP display section
BP.window      = 'hann';      % 'hann'|'kaiser'|'none'
BP.show_angle  = 0;           % angle to show BP impulse (nearest)
% (Uses your measured span around freq_c automatically)

%% 2) Load & prepare anechoic baseline (normalized)
[A_deg, A_amp_dB] = load_anechoic_at_freq(anFile, freq_c*1e-9);
A_amp_norm_dB     = A_amp_dB - max(A_amp_dB);

%% 3) Load NanoVNA scan and extract RAW pattern @ freq_c (normalized)
T = readtable(scanCsv);
need = {'angle_deg','freq_Hz','S21_re','S21_im','S21_dB'};
assert(all(ismember(need, T.Properties.VariableNames)), ...
    'CSV must contain: %s', strjoin(need,', '));

angles_scan = unique(T.angle_deg(:).','stable');
scan_raw_dB = zeros(size(angles_scan));
Per = struct('ang',[],'f',[],'H',[]);
Per(numel(angles_scan)).ang = [];

for k = 1:numel(angles_scan)
    sub = T(T.angle_deg==angles_scan(k),:);
    [f,ix] = sort(sub.freq_Hz);
    H = sub.S21_re(ix) + 1j*sub.S21_im(ix);
    Per(k).ang = angles_scan(k);
    Per(k).f   = f(:);
    Per(k).H   = H(:);
    [~,iC] = min(abs(f - freq_c));
    scan_raw_dB(k) = 20*log10(abs(H(iC))+1e-15);
end
scan_raw_norm_dB = scan_raw_dB - max(scan_raw_dB);

%% 4) Interpolate both patterns to a common grid and smooth (no gating)
angGrid = 0:angStep:355;

ane_on_grid  = resample_circular(A_deg,            A_amp_norm_dB,   angGrid);
scan_on_grid = resample_circular(angles_scan,      scan_raw_norm_dB, angGrid);

if SMOOTH.use
    scan_on_grid = smooth_angular(scan_on_grid, SMOOTH);
    % If you prefer baseline untouched, comment the next line:
    ane_on_grid  = smooth_angular(ane_on_grid,  SMOOTH);
end

%% 5) Plot individual datasets (normalized)
plot_dataset_normalized('Anechoic (baseline)', freq_c, angGrid, ane_on_grid);
plot_dataset_normalized('Scan (RAW, interp+smooth)', freq_c, angGrid, scan_on_grid);

%% 6) Compare (normalized)
compare_two_patterns('Scan vs Anechoic (No time gating)', ...
    freq_c, angGrid, scan_on_grid, ane_on_grid);

% Quick console stats
err = scan_on_grid - ane_on_grid;
fprintf('\n--- Comparison @ %.3f GHz (No time gating) ---\n', freq_c*1e-9);
fprintf('MAE  = %.2f dB\n', mean(abs(err),'omitnan'));
fprintf('RMSE = %.2f dB\n', sqrt(mean(err.^2,'omitnan')));

%% 7) Show VNA-like Low-Pass impulse & step (one angle, for reference only)
repAngLP = pick_nearest_angle(angles_scan, LP.show_angle);
subLP    = T(T.angle_deg==repAngLP,:);
[fLP,ix] = sort(subLP.freq_Hz);
HLP      = subLP.S21_re(ix) + 1j*subLP.S21_im(ix);
[t_lp, h_lp, s_lp] = vna_like_lowpass_impulse(fLP, HLP, LP.window);

figure('Name',sprintf('LP Impulse & Step @ %d° (no gating)', repAngLP),'Color','w');
subplot(2,1,1); plot(t_lp*1e9, abs(h_lp),'LineWidth',1.2); grid on;
xlabel('Time (ns)'); ylabel('|Impulse| (linear)'); title('Low-Pass Impulse (0 \rightarrow F_{nyq})');
subplot(2,1,2); plot(t_lp*1e9, abs(s_lp),'LineWidth',1.2); grid on;
xlabel('Time (ns)'); ylabel('|Step| (linear)');   title('Low-Pass Step Response');

%% 8) NEW — Band-pass transform at 2.4 GHz (display only, no gating)
if BP.enable
    repAngBP = pick_nearest_angle(angles_scan, BP.show_angle);
    subBP    = T(T.angle_deg==repAngBP,:);
    [fBP,ix] = sort(subBP.freq_Hz);
    HBP      = subBP.S21_re(ix) + 1j*subBP.S21_im(ix);

    [t_bp, h_bp, env_bp, meta] = vna_like_bandpass_impulse(fBP, HBP, freq_c, BP.window);

    figure('Name',sprintf('Band-pass Impulse @ %.1f GHz, angle %d°', freq_c/1e9, repAngBP),'Color','w');
    subplot(2,1,1);
    plot(t_bp*1e9, real(h_bp),'LineWidth',1.1); hold on;
    plot(t_bp*1e9, imag(h_bp),'LineWidth',1.1); grid on;
    xlabel('Time (ns)'); ylabel('h_{BP}(t)'); 
    title(sprintf('Band-pass Impulse (fs = %.2f GHz, span = %.2f GHz)', meta.fs/1e9, meta.span/1e9));
    legend('Real','Imag','Location','best');

    subplot(2,1,2);
    plot(t_bp*1e9, env_bp,'LineWidth',1.4); grid on;
    xlabel('Time (ns)'); ylabel('|h_{BP}(t)|');
    title('Band-pass Envelope (no time gating)');
end

%% ============================ Helpers ============================

function yq = resample_circular(x_deg, y_dB, xq_deg)
    % Wrap, collapse duplicate angles, pad ±360, interpolate (linear)
    wrap = @(x) mod(x,360);
    [xu, yu] = collapse_duplicates(wrap(x_deg(:)), y_dB(:));
    [xp, yp] = periodic_pad(xu, yu);
    yq = interp1(xp, yp, mod(xq_deg,360), 'linear');
end

function [xu, yu] = collapse_duplicates(x, y)
    tol = 1e-9;
    xr = round(x/tol)*tol;
    [~,~,ic] = unique(xr,'stable');
    xu = accumarray(ic, x, [], @mean);
    yu = accumarray(ic, y, [], @mean);
    [xu,s] = sort(xu); yu = yu(s);
end

function [xp, yp] = periodic_pad(x, y)
    xp = [x-360; x; x+360];
    yp = [y;     y; y    ];
    [xp,s] = sort(xp); yp = yp(s);
end

function y = smooth_angular(y, S)
    y = y(:).';
    switch lower(S.method)
        case 'sgolay'
            if exist('sgolayfilt','file')==2
                win = max(3, S.win + mod(S.win+1,2)); % enforce odd
                poly = min(S.poly, win-1);
                y = sgolayfilt(y, poly, win);
            else
                y = movmean(y, max(3,S.win));
            end
        case 'movmean'
            y = movmean(y, max(3,S.movN));
        otherwise
            % none
    end
end

function plot_dataset_normalized(labelStr, freq_c, angGrid, patt_dB)
    lin = 10.^(patt_dB/20);
    figure('Color','w','Name',[labelStr ' (Polar)']);
    if exist('polarpattern','file')==2
        polarpattern(angGrid, patt_dB,'LineWidth',1.4);
        title(sprintf('%s @ %.3f GHz (Normalized dB)', labelStr, freq_c*1e-9));
    else
        polarplot(deg2rad(angGrid), lin,'LineWidth',1.6); rlim([0 1]); grid on;
        title(sprintf('%s @ %.3f GHz (Normalized linear)', labelStr, freq_c*1e-9));
    end
    figure('Color','w','Name',[labelStr ' (dB vs Angle)']);
    plot(angGrid, patt_dB,'LineWidth',1.6); grid on; ylim([-40 0]);
    xlabel('Angle (deg)'); ylabel('Normalized Gain (dB)');
    title(sprintf('%s @ %.3f GHz', labelStr, freq_c*1e-9));
end

function compare_two_patterns(titleStr, freq_c, angGrid, patt1_dB, patt2_dB)
    lin1 = 10.^(patt1_dB/20); lin2 = 10.^(patt2_dB/20);
    figure('Color','w','Name',[titleStr ' (Polar)']);
    if exist('polarpattern','file')==2
        polarpattern(angGrid, patt1_dB,'b-','LineWidth',1.4); hold on;
        polarpattern(angGrid, patt2_dB,'r-','LineWidth',1.4);
        legend('Scan','Anechoic','Location','southoutside');
        title(sprintf('%s @ %.3f GHz (Normalized dB)', titleStr, freq_c*1e-9));
    else
        polarplot(deg2rad(angGrid), lin1,'LineWidth',1.6); hold on; grid on;
        polarplot(deg2rad(angGrid), lin2,'LineWidth',1.6);
        rlim([0 1]); legend('Scan','Anechoic','Location','southoutside');
        title(sprintf('%s @ %.3f GHz (Normalized linear)', titleStr, freq_c*1e-9));
    end
    figure('Color','w','Name',[titleStr ' (dB overlay)']);
    plot(angGrid, patt1_dB,'LineWidth',1.6); hold on; grid on;
    plot(angGrid, patt2_dB,'LineWidth',1.6);
    xlabel('Angle (deg)'); ylabel('Normalized Gain (dB)'); ylim([-40 0]);
    legend('Scan','Anechoic','Location','southoutside');
    title(sprintf('%s (dB) @ %.3f GHz', titleStr, freq_c*1e-9));
    figure('Color','w','Name',[titleStr ' Error (dB)']);
    plot(angGrid, patt1_dB - patt2_dB, 'LineWidth',1.4); grid on; ylim([-20 20]);
    xlabel('Angle (deg)'); ylabel('Scan - Anechoic (dB)');
    title(sprintf('Error vs Angle @ %.3f GHz', freq_c*1e-9));
end

function [A_deg, A_amp_dB] = load_anechoic_at_freq(anFile, freqGHz)
    fid = fopen(anFile,'r'); assert(fid>0, 'Cannot open %s', anFile);
    first = fgetl(fid);
    pPos  = strfind(first,'Phase');  assert(~isempty(pPos), 'Header must contain "Phase".');
    ampSeg = strtrim(first(1:pPos-1));
    aPos  = strfind(ampSeg,'Amp');   assert(~isempty(aPos),  'Header must contain "Amp".');
    nums  = regexp(ampSeg(aPos+3:end),'([\d]+\.?[\d]*)','match');
    freqs = str2double(nums);        % GHz list
    Nf = numel(freqs);
    C = textscan(fid, repmat('%f',1,3+Nf+Nf), 'Delimiter',{' ','\t',','}, ...
        'MultipleDelimsAsOne',true);
    fclose(fid);
    A_deg = C{2}(:).';
    ampMat = zeros(numel(A_deg), Nf);
    for i = 1:Nf, ampMat(:,i) = C{3+i}; end
    [~,iF] = min(abs(freqs - freqGHz));
    A_amp_dB = ampMat(:,iF).';
end

function [t_s, h_imp, s_step] = vna_like_lowpass_impulse(freq_Hz, H_complex, windowName)
    % Build a 0..Fnyq spectrum from band-limited S-parameters and IFFT it
    [f,ix] = sort(freq_Hz(:)); H = H_complex(:); H = H(ix);
    df  = median(diff(f));
    fun = (f(1):df:f(end)).';
    if numel(fun)~=numel(f) || max(abs(f - fun)) > max(1,1e-9*df)
        Hr = interp1(f, real(H), fun, 'linear','extrap');
        Hi = interp1(f, imag(H), fun, 'linear','extrap');
        f  = fun; H = Hr + 1j*Hi;
    end
    Fnyq = f(end);
    Npos = floor(Fnyq/df) + 1;
    fpos = (0:Npos-1).' * df;
    Hpos = zeros(Npos,1);
    i0 = round(f(1)/df) + 1;
    i1 = min(i0 + numel(f) - 1, Npos);
    Hpos(i0:i1) = H(1:(i1-i0+1));
    switch lower(windowName)
        case 'hann',   w = hann(Npos);
        case 'kaiser', w = kaiser(Npos,6);
        otherwise,     w = ones(Npos,1);
    end
    Hpos = Hpos .* w;
    Hfull = [Hpos; conj(Hpos(end-1:-1:2))];
    h_imp = ifft(Hfull, 'symmetric');
    fs = 2*Fnyq; dt = 1/fs;
    t_s = (0:numel(h_imp)-1).' * dt;
    s_step = cumsum(h_imp) * dt;
end

function [t_s, h_bp, env_bp, meta] = vna_like_bandpass_impulse(freq_Hz, H_complex, f_center_Hz, windowName)
    % VNA-like BAND-PASS transform (centered at f_center_Hz)
    % Steps:
    %   1) Sort & reinterpolate S(f) to uniform df
    %   2) Form baseband spectrum Hbb(f_off) with f_off = f - f_center
    %   3) Zero-pad outside measured offsets, apply window, ifftshift -> IFFT
    %   4) fs = span (B); t resolution = 1/B. h_bp is COMPLEX. env_bp = |h_bp|.
    [f,ix] = sort(freq_Hz(:)); H = H_complex(:); H = H(ix);
    df  = median(diff(f));
    fun = (f(1):df:f(end)).';
    if numel(fun)~=numel(f) || max(abs(f - fun)) > max(1,1e-9*df)
        Hr = interp1(f, real(H), fun, 'linear','extrap');
        Hi = interp1(f, imag(H), fun, 'linear','extrap');
        f  = fun; H = Hr + 1j*Hi;
    end

    % Offset frequency axis around center
    f_off = f - f_center_Hz;
    Bspan = f(end) - f(1);     % total measured span (Hz)
    fs    = Bspan;             % baseband sampling rate for IFFT
    dt    = 1/fs;

    % Build uniform offset grid covering [-B/2, +B/2]
    % Ensure df_off consistent with df
    df_off = df;
    fmin_off = -Bspan/2;
    fmax_off = +Bspan/2;
    fgrid = (fmin_off:df_off:fmax_off).';   % zero-centered grid
    % Place measured H onto this grid
    Hbb = zeros(size(fgrid));
    % Find indices where measured offsets fall within the grid
    Hbb = interp1(f_off, H, fgrid, 'linear', 0);

    % Window
    switch lower(windowName)
        case 'hann',   w = hann(numel(Hbb));
        case 'kaiser', w = kaiser(numel(Hbb), 6);
        otherwise,     w = ones(numel(Hbb),1);
    end
    Hbb_win = Hbb .* w;

    % Zero frequency is at the center -> use ifftshift before ifft
    h_bp = ifft(ifftshift(Hbb_win), 'symmetric');  % complex band-pass impulse
    t_s  = (0:numel(h_bp)-1).' * dt;
    env_bp = abs(h_bp);

    % meta
    meta.fs   = fs;
    meta.span = Bspan;
    meta.df   = df_off;
end

function r = rms(x), r = sqrt(mean(x.^2,'omitnan')); end
function a = pick_nearest_angle(angList, target), [~,i]=min(abs(angList-target)); a=angList(i); end
