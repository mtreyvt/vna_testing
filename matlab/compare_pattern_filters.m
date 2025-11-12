function runTimeGatingAndCompare()
    % Settings
    patternFile  = 'nanovnaf_pattern_4_6_nf.csv';
    anechoicFile = 'vivald_E.txt';
    freq_c  = 5.6e9;    % comparison frequency (Hz)
    angStep = 3;        % output angle grid (degrees)

    % Options for smoothing and flipping
    doSmooth  = true;
    smoothWin = 7;
    smoothOrd = 2;
    doFlip    = true;

    % ==== Time‑gating search ====
    data   = readtable(patternFile);
    angles = unique(data.angle_deg);
    freqs  = unique(data.freq_Hz);
    N      = numel(freqs);
    [ampFreqs, AAngles, ampData] = readAnechoic(anechoicFile);
    gateWidth = round(0.1 * N);
    alpha     = 0.5;
    offsets   = 1:10:(N - gateWidth + 1);

    % Precompute IFFT for each angle
    tfCache = cell(numel(angles),1);
    for k = 1:numel(angles)
        rows = data.angle_deg == angles(k);
        S21  = data.S21_re(rows) + 1i*data.S21_im(rows);
        tfCache{k} = ifft(S21);
    end

    % Gating search
    minRMSE = Inf;
    bestStart = offsets(1);
    for s = offsets
        gate = zeros(N,1);
        gate(s:s+gateWidth-1) = tukeywin(gateWidth,alpha);
        err = 0;
        for k = 1:numel(angles)
            hg   = tfCache{k} .* gate;
            mag  = 20*log10(abs(fft(hg)));
            magInterp = interp1(freqs/1e9, mag, ampFreqs, 'linear');
            refAmp    = getRefAmp(angles(k), AAngles, ampData);
            err = err + mean((magInterp - refAmp).^2);
        end
        rmse = err / numel(angles);
        if rmse < minRMSE
            minRMSE = rmse;
            bestStart = s;
        end
    end

    % ==== Compute gated scan at freq_c ====
    [~, idxF] = min(abs(freqs - freq_c));
    bestGate = zeros(N,1);
    bestGate(bestStart:bestStart+gateWidth-1) = tukeywin(gateWidth,alpha);
    gatedScan = zeros(numel(angles),1);
    for k = 1:numel(angles)
        hg    = tfCache{k} .* bestGate;
        S21g  = fft(hg);
        gatedScan(k) = 20*log10(abs(S21g(idxF)));
    end
    gatedScan = gatedScan - max(gatedScan);  % normalize

    % ==== Extract anechoic amplitude at freq_c ====
    % Find which column in the anechoic data corresponds to 5.6 GHz
    [~, idxRef] = min(abs(ampFreqs - freq_c*1e-9));
    anechAmp = zeros(numel(AAngles),1);
    for i = 1:numel(AAngles)
        anechAmp(i) = ampData{i}(idxRef);
    end
    anechAmp = anechAmp - max(anechAmp);   % normalize
    AAngles  = AAngles(:);                 % ensure column vector
    anechAmp = anechAmp(:);                % ensure column vector

    % ==== Force column shape for scan data ====
    angles    = angles(:);
    gatedScan = gatedScan(:);

    % ==== Interpolate both patterns onto a uniform grid ====
    angGrid    = (0:angStep:357)';  % column vector
    scanInterp = interp1(mod(angles,360), gatedScan, mod(angGrid,360), 'linear','extrap');
    refInterp  = interp1(mod(AAngles,360),  anechAmp,  mod(angGrid,360), 'linear','extrap');

    % ==== Optional flip and smooth ====
    if doFlip
        scanInterp = flipud(scanInterp);
    end
    if doSmooth && exist('sgolayfilt','file') == 2
        win  = max(3, smoothWin + mod(smoothWin+1,2));
        poly = min(smoothOrd, win-1);
        scanInterp = sgolayfilt(scanInterp, poly, win);
        refInterp  = sgolayfilt(refInterp,  poly, win);
    end

    % ==== Plot results ====
    figure;
    polarplot(deg2rad(angGrid), 10.^(scanInterp/20), 'b-', 'LineWidth',1.5); hold on;
    polarplot(deg2rad(angGrid), 10.^(refInterp/20),  'r--','LineWidth',1.5);
    title(sprintf('Normalized S_{21} at %.1f GHz – %s', freq_c/1e9, patternFile),'Interpreter','none');
    legend('Gated scan','Anechoic reference','Location','southoutside');
end

% ===== Helpers =====
function [freqs, angles, ampData] = readAnechoic(file)
    fid    = fopen(file, 'r');
    header = strsplit(strtrim(fgetl(fid)));
    ampIdx   = find(strcmp(header,'Amplitude'));
    phaseIdx = find(strcmp(header,'Phase'));
    freqs    = str2double(header(ampIdx+1:phaseIdx-1));  % frequency list (GHz)
    C = textscan(fid, repmat('%f',1,3+2*numel(freqs)), 'CollectOutput', true);
    fclose(fid);
    angles = C{1}(:,2);  % column vector
    ampData = cell(numel(angles),1);
    ampMat  = C{1}(:,4:(3+numel(freqs))); % amplitude rows
    for i = 1:numel(angles)
        ampData{i} = ampMat(i,:);  % store row vector of amplitude values
    end
end

function refAmp = getRefAmp(angle, AAngles, ampData)
    [~, idx] = min(abs(AAngles - angle));
    refAmp = ampData{idx};
end
