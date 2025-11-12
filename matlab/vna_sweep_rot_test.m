%Antenna Results MDE - F25-05 
%11/08/2025

%% 0) Housekeeping
clear; clc; close all;

%% 1) File paths & main knobs
scanCsv = 'nanovnaf_pattern_3_6.csv';   % NanoVNA sweep CSV
anFile  = 'vivald_E.txt';                            % anechoic baseline
freq_c  = 2.4e9;                                      % comparison frequency (Hz)
angStep = 3;                                          % resample grid (deg)
gate_ns = 4;                                         % time gate end (ns) for LP transform
td_window = 'hann';                                   % LP transform window: 'hann'|'kaiser'|'none'

%% 2) Load & prepare measured data (anechoic baseline)
% Robust parser for "Amp <freqs> Phase <freqs>" header
[A_deg, A_amp_dB] = load_anechoic_at_freq(anFile, freq_c*1e-9);
% Normalize to 0 dB peak (apples-to-apples)
anechoic_norm_dB = A_amp_dB - max(A_amp_dB);
% Angle grid for plotting
angGrid = 0:angStep:355;


% Individual plots: Anechoic
plot_dataset_normalized('Anechoic (baseline)', freq_c, angGrid, ...
    resample_circular(A_deg, anechoic_norm_dB, angGrid));

%% 3) Load & prepare NanoVNA scan (raw)
T = readtable(scanCsv);
need = {'angle_deg','freq_Hz','S21_re','S21_im','S21_dB'};
assert(all(ismember(need, T.Properties.VariableNames)), ...
    'CSV must contain: %s', strjoin(need,', '));

angles_scan = unique(T.angle_deg(:).','stable');

% Build per-angle complex S21(f)
S_per = struct('ang',[],'f',[],'H',[]);
S_per(numel(angles_scan)).ang = [];
for k = 1:numel(angles_scan)
    sub = T(T.angle_deg==angles_scan(k),:);
    [f,ix] = sort(sub.freq_Hz);
    H = sub.S21_re(ix) + 1j*sub.S21_im(ix);
    S_per(k).ang = angles_scan(k);
    S_per(k).f   = f(:);
    S_per(k).H   = H(:);
end

% Extract RAW pattern at freq_c
scan_raw_dB = zeros(size(angles_scan));
for k = 1:numel(angles_scan)
    f = S_per(k).f; H = S_per(k).H;
    [~,i] = min(abs(f - freq_c));
    scan_raw_dB(k) = 20*log10(abs(H(i)) + 1e-15);
end
scan_raw_norm_dB = scan_raw_dB - max(scan_raw_dB);

% Individual plots: Scan (raw)
plot_dataset_normalized('Scan (RAW)', freq_c, angGrid, ...
    resample_circular(angles_scan, scan_raw_norm_dB, angGrid));

%% 4) Compare RAW scan to Anechoic (normalized)
scan_raw_on_grid = resample_circular(angles_scan, scan_raw_norm_dB, angGrid);
ane_on_grid      = resample_circular(A_deg, anechoic_norm_dB, angGrid);

compare_two_patterns('RAW Scan vs Anechoic', freq_c, angGrid, ...
    scan_raw_on_grid, ane_on_grid);

%% 5) VNA-like Low-Pass transform + time-gate + rebuild scan at freq_c
% Gate 0..gate_ns (rect, with Hann edges applied in impulse domain)
scan_gated_dB = zeros(size(angles_scan));
repAng = pick_nearest_angle(angles_scan, 0);  % show impulse/step at ~0 deg

for k = 1:numel(angles_scan)
    f_meas = S_per(k).f;  H_meas = S_per(k).H;

    % --- LP impulse (0..Fnyq with zeros at 0..fmin and >fmax) ---
    [t_s, h_imp, s_step, fs, fpos, Hpos_win] = vna_like_lowpass_impulse(f_meas, H_meas, td_window);

    % --- Simple rectangular time gate 0..gate_ns ---
    g = zeros(size(t_s));
    g(t_s >= 0 & t_s <= gate_ns*1e-9) = 1.0;

    % Apply gate and FFT back to positive-frequency spectrum
    h_gated = h_imp .* g;
    Hpos_g  = time_to_posfreq(h_gated);             % 0..Fnyq bins
    % Re-sample gated spectrum onto measured f grid (within 0..Fnyq)
    Hrec = interp1(fpos, Hpos_g, f_meas, 'linear', 0);

    % Read level at freq_c
    [~,iC] = min(abs(f_meas - freq_c));
    scan_gated_dB(k) = 20*log10(abs(Hrec(iC)) + 1e-15);

    % One demo angle: show impulse/step + gate
    if angles_scan(k) == repAng
        figure('Name',sprintf('LP Impulse & Step @ %d°', repAng),'Color','w');
        subplot(2,1,1);
        plot(t_s*1e9, abs(h_imp), 'LineWidth',1.2); grid on; hold on;
        yyaxis right; plot(t_s*1e9, g, '--','LineWidth',1.2); ylabel('Gate');
        yyaxis left;  ylabel('|Impulse| (linear)');
        xlabel('Time (ns)');
        title(sprintf('Impulse & Time Gate 0..%.1f ns', gate_ns));

        subplot(2,1,2);
        plot(t_s*1e9, abs(s_step), 'LineWidth',1.2); grid on;
        xlabel('Time (ns)'); ylabel('|Step| (linear)');
        title('Low-Pass Step Response');
    end
end

scan_gated_norm_dB = scan_gated_dB - max(scan_gated_dB);

% Individual plots: Scan (GATED)
plot_dataset_normalized(sprintf('Scan (GATED 0..%.1f ns)', gate_ns), freq_c, angGrid, ...
    resample_circular(angles_scan, scan_gated_norm_dB, angGrid));

%% 6) Compare GATED scan to Anechoic (normalized)
scan_gated_on_grid = resample_circular(angles_scan, scan_gated_norm_dB, angGrid);

compare_two_patterns(sprintf('GATED (0..%.1f ns) Scan vs Anechoic', gate_ns), ...
    freq_c, angGrid, scan_gated_on_grid, ane_on_grid);

% Quick console stats
raw_err   = scan_raw_on_grid   - ane_on_grid;
gated_err = scan_gated_on_grid - ane_on_grid;
fprintf('\n--- Comparison @ %.3f GHz ---\n', freq_c*1e-9);
fprintf('RAW   : MAE = %.2f dB, RMSE = %.2f dB\n', mean(abs(raw_err),'omitnan'), rms(raw_err));
fprintf('GATED : MAE = %.2f dB, RMSE = %.2f dB\n', mean(abs(gated_err),'omitnan'), rms(gated_err));

%% ============================ Helpers ============================

function [A_deg, A_amp_dB] = load_anechoic_at_freq(anFile, freqGHz)
    % Parse "Amp <freqs> Phase <freqs>" header, pick closest freq column
    fid = fopen(anFile,'r'); assert(fid>0, 'Cannot open %s', anFile);
    first = fgetl(fid);
    pPos  = strfind(first,'Phase');  assert(~isempty(pPos), 'Header must contain "Phase".');
    ampSeg = strtrim(first(1:pPos-1));
    aPos  = strfind(ampSeg,'Amp');   assert(~isempty(aPos),  'Header must contain "Amp".');
    nums  = regexp(ampSeg(aPos+3:end),'([\d]+\.?[\d]*)','match');
    freqs = str2double(nums);        % GHz list
    assert(all(isfinite(freqs)),'Failed to parse Amp freqs from header');

    Nf = numel(freqs);
    C = textscan(fid, repmat('%f',1,3+Nf+Nf), 'Delimiter',{' ','\t',','}, ...
        'MultipleDelimsAsOne',true);
    fclose(fid);

    A_deg = C{2}(:).';
    ampMat = zeros(numel(A_deg), Nf);
    for i=1:Nf, ampMat(:,i) = C{3+i}; end

    [~,iF] = min(abs(freqs - freqGHz));
    A_amp_dB = ampMat(:,iF).';
end

function yq = resample_circular(x_deg, y_dB, xq_deg)
    % Wrap, collapse duplicates, pad ±360, then interpolate (linear)
    wrap = @(x) mod(x,360);
    [xu, yu] = collapse_duplicates(wrap(x_deg(:)), y_dB(:));
    [xp, yp] = periodic_pad(xu, yu);
    yq = interp1(xp, yp, mod(xq_deg,360),'linear');
end

function [xu, yu] = collapse_duplicates(x, y)
    % Merge duplicate/near-duplicate angle samples by averaging
    tol = 1e-9;
    xr = round(x/tol)*tol;
    [ur,~,ic] = unique(xr,'stable');
    xu = accumarray(ic, x, [], @mean);
    yu = accumarray(ic, y, [], @mean);
    [xu,s] = sort(xu); yu = yu(s);
end

function [xp, yp] = periodic_pad(x, y)
    % Provide support around 0..360 for interpolation
    xp = [x-360; x; x+360];
    yp = [y;     y; y    ];
    [xp,s] = sort(xp); yp = yp(s);
end

function plot_dataset_normalized(labelStr, freq_c, angGrid, patt_dB)
    lin = 10.^(patt_dB/20);
    % Polar
    figure('Color','w','Name',[labelStr ' (Polar)']);
    if exist('polarpattern','file')==2
        polarpattern(angGrid, patt_dB,'LineWidth',1.4); % polarpattern uses dB radius
        title(sprintf('%s @ %.3f GHz (Normalized dB)', labelStr, freq_c*1e-9));
    else
        polarplot(deg2rad(angGrid), lin,'LineWidth',1.6); rlim([0 1]); grid on;
        title(sprintf('%s @ %.3f GHz (Normalized linear)', labelStr, freq_c*1e-9));
    end
    % dB vs angle
    figure('Color','w','Name',[labelStr ' (dB vs Angle)']);
    plot(angGrid, patt_dB,'LineWidth',1.6); grid on; ylim([-40 0]);
    xlabel('Angle (deg)'); ylabel('Normalized Gain (dB)');
    title(sprintf('%s @ %.3f GHz', labelStr, freq_c*1e-9));
end

function compare_two_patterns(titleStr, freq_c, angGrid, patt1_dB, patt2_dB)
    lin1 = 10.^(patt1_dB/20); lin2 = 10.^(patt2_dB/20);
    % Polar overlay
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
    % dB overlay
    figure('Color','w','Name',[titleStr ' (dB)']);
    plot(angGrid, patt1_dB,'LineWidth',1.6); hold on; grid on;
    plot(angGrid, patt2_dB,'LineWidth',1.6);
    xlabel('Angle (deg)'); ylabel('Normalized Gain (dB)'); ylim([-40 0]);
    legend('Scan','Anechoic','Location','southoutside');
    title(sprintf('%s (dB) @ %.3f GHz', titleStr, freq_c*1e-9));
    % Error
    figure('Color','w','Name',[titleStr ' Error (dB)']);
    plot(angGrid, patt1_dB - patt2_dB,'LineWidth',1.4); grid on; ylim([-20 20]);
    xlabel('Angle (deg)'); ylabel('Scan - Anechoic (dB)');
    title(sprintf('Error vs Angle @ %.3f GHz', freq_c*1e-9));
end

function [t_s, h_imp, s_step, fs, fpos, Hpos_win] = vna_like_lowpass_impulse(freq_Hz, H_complex, windowName)
    % Sort + (re)interpolate to uniform df
    [f,ix] = sort(freq_Hz(:)); H = H_complex(:); H = H(ix);
    df = median(diff(f));
    f_uni = (f(1):df:f(end)).';
    if numel(f_uni)~=numel(f) || max(abs(f - f_uni))>max(1,1e-9*df)
        Hr = interp1(f, real(H), f_uni, 'linear','extrap');
        Hi = interp1(f, imag(H), f_uni, 'linear','extrap');
        f = f_uni; H = Hr + 1j*Hi;
    end

    % Build 0..Fnyq positive spectrum (choose Fnyq=fmax)
    Fnyq = f(end);
    Npos = floor(Fnyq/df) + 1;
    fpos = (0:Npos-1).' * df;
    Hpos = zeros(Npos,1);
    i0 = round(f(1)/df) + 1;
    i1 = min(i0 + numel(f) - 1, Npos);
    Hpos(i0:i1) = H(1:(i1-i0+1));

    % Window to control sidelobes
    switch lower(windowName)
        case 'hann',   w = hann(Npos);
        case 'kaiser', w = kaiser(Npos,6);
        otherwise,     w = ones(Npos,1);
    end
    Hpos_win = Hpos .* w;

    % Hermitian symmetry -> real impulse
    Hfull = [Hpos_win; conj(Hpos_win(end-1:-1:2))];
    h_imp = ifft(Hfull, 'symmetric');

    % Time axis and step
    fs   = 2*Fnyq;                 % "sampling rate" in Hz
    dt   = 1/fs;
    t_s  = (0:numel(h_imp)-1).' * dt;
    s_step = cumsum(h_imp) * dt;
end

function Hpos = time_to_posfreq(h)
    % Return positive-frequency spectrum for a real time sequence h
    N = numel(h);
    H = fft(h);
    if mod(N,2)==0
        Npos = N/2 + 1;
    else
        Npos = (N+1)/2;
    end
    Hpos = H(1:Npos);
end

function r = rms(x), r = sqrt(mean(x.^2,'omitnan')); end

function a = pick_nearest_angle(angList, target)
    [~,i] = min(abs(angList - target)); a = angList(i);
end
