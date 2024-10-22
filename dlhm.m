% dlhm -  Digital Lensless Holography Simulation Model
% This function simulates digital lensless holography and returns the
% hologram.

% Input:
%   sample   - Input complex field (wavefront) to be simulated.            
%   L        - Distance from the point source to the hologram plane.
%   z        - Distance from the point source to the sample plane.
%   W_c      - Width of the sensor.
%   dx_in    - Resampling resolution parameter for input.
%   dx_out   - Pixel size at the sensor.
%   lambda   - Wavelength of the light.
%   x0, y0   - Offsets to adjust the center of the cropped sample.
%   NA_s     - Numerical Aperture of the source.

% Output:
%   holo     - Simulated hologram.

% Author: mariajlopera, mloper23@eafit.edu.co -
% maria.josef.lopera.acosta@vub.be
% Date: 06/12/2023

function holo = dlhm(sample, dx_in, L, z, W_c, dx_out, lambda, x0, y0, NA_s)

    % Uncomment for default values
    % % Default initialization in case optional values are not provided
    % if (~exist('opt', 'var'))
    %     x0 = 0;
    %     y0 = 0;
    %     NA_s = 0;
    % end

    % Determine the size of the input sample
    [N, M] = size(sample);
    
    % Calculate the magnification factor
    Mag = L / z;
    % Calculate the width of the sample plane 
    W_s = W_c / Mag;

    % Re-sample and magnify the sample if input pixel size specified
    if dx_in ~= 0
        % Calculate the resolution difference
        res_d = dx_in * Mag / dx_out;
        % Resize sample to new resolution
        sample = imresize(sample, 1 / res_d);
        [N_s, M_s] = size(sample);  
        % Calculate the width of the resampled sample
        W_sample = N_s * dx_out / Mag;

        % Crop the sample according to magnification if it's larger than
        % the field of view
        if W_sample > W_s
            sample = sample(N_s/2 - N/2 : N_s/2 + N/2 - 1, M_s/2 - M/2 : M_s/2 + M/2 - 1);
        else
            % Resize back to original dimensions if needed
            sample = imresize(sample, [N, M]);
        end

        % Further crop sample to align with magnification
        sample = sample(N/2-N/(Mag*2)+x0 : N/2+N/(Mag*2)-1+x0, M/2-M/(Mag*2)+y0 : M/2+M/(2*Mag)-1+y0);   
        [N_s, ~] = size(sample);
        % Final resample to match sensor characteristics
        rs = N_s / (W_c / dx_out);
        sample = imresize(sample, 1 / rs);
    else
        % Direct resample to sensor characteristics (if no input resampling)
        rs = N / (W_c / dx_out);
        sample = imresize(sample, 1 / rs);
    end

    % Update sample dimensions
    [Q, P] = size(sample);

    % Calculate wave number based on wavelength
    k = 2 * pi / lambda;
    
    % Establish spatial coordinates on sensor plane
    x = linspace(-W_c / 2, W_c / 2, P);
    y = linspace(-W_c / 2, W_c / 2, Q);
    [u, v] = meshgrid(x, y);
    
    % Calculate radial distances for source wavefront
    r = sqrt(u .^ 2 + v .^ 2 + L .^ 2);

    % Determine the illumination pattern based on Numerical Aperture
    if NA_s ~= 0
        % Use Gaussian distribution for restricted illumination
        PS = exp(-2 * r / (L * tan(asin(NA_s))));
    else
        % Default full illumination
        PS = exp(-r);
    end
    % Normalize the point spread function
    PS = PS - min(min(PS));
    PS = PS / max(max(PS));
    
    % Calculate spatial frequency coordinates at sample's plane    
    dfx = Mag/(dx_out*P);
    dfy = Mag/(dx_out*Q);
    [fx, fy] = meshgrid(-P/2*dfx : dfx : P/2*dfx-dfx, -Q/2*dfy : dfy : Q/2*dfy-dfy);
    
    % Compute the propagation kernel for the Angular Spectrum Method (ASM)
    E = exp((-1i * (L - z) * sqrt(k^2 - 4 * pi^2 * (fx.^2 + fy.^2))));
    
    % Compute hologram using inverse Fourier transform
    Uz = ifts(fts(sample) .* E);
    holo = abs(Uz);

    % Calculate maximum distortion due to distance differences
    Max_mag = sqrt(W_c^2 / 2 + L^2) / z;
    Max_D = abs(Max_mag - Mag);


    k = [P 0 P/2; 0 Q Q/2; 0 0 1];
    radialDistortion = [-Max_D / (P/2) 0]; 
    cameraParams = cameraParameters("K", k, "RadialDistortion", radialDistortion);
    holo = undistortImage(holo, cameraParams);

    % Normalize and post-process the hologram
    holo = holo - min(min(abs(holo)));
    holo = holo / max(max(holo));
    holo = holo .* PS;
    holo = holo - min(min(abs(holo)));
    holo = holo / max(max(holo));
    holo = holo * 2^8;
    holo = round(holo, 0);
end
