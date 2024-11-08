% Script for reconstructing digital lensless holograms using ASM with
% spherical wavefront source
% @mariajlopera
% For questions contact mloper23@eafit.edu.co;
% maria.josef.lopera.acosta@vub.be

% Load and process hologram
hologram = im2double(imread('./data/Simulated_hologram.png')); % Load hologram image
[P, Q] = size(hologram); % Get size of hologram

% Parameters
lambda = 532e-9; % Wavelength (m)
dx = 1.85e-6; % Pixel size in x-direction (m)
dy = dx; % Pixel size in y-direction (m)
Wcx = P * dx; % Width of sensor area in x
Wcy = Q * dy; % Width of sensor area in y
L = 11e-3; % Distance to screen (m)
z = 4.95e-3; % Distance to sample (m)
k = 2 * pi / lambda; % Wavenumber

% Calculate sampling rate at sample plane
sampling_sample_plane = lambda * sqrt((Wcx/2)^2 + (Wcy/2)^2 + z^2) / Wcx;
oversampling_factor = dx / sampling_sample_plane; % Determine oversampling factor

% Uncomment the line below if using non-correct sampling (for fast approaches)
% oversampling_factor = 1;

% Resize hologram based on oversampling factor
resized_hologram = imresize(hologram, oversampling_factor); % Resize hologram
[N, M] = size(resized_hologram); % Get new size after resizing

% Generate reference wave
reference = point_src(N, M, L, 0, 0, lambda, (dx * P) / N); % Reference wave using point source

% Perform angular spectrum propagation
U = angular_spectrum(reference .* resized_hologram, Wcx, Wcy, k, (L - z)); % Sample to sensor
U0 = angular_spectrum(reference .* ones(N, M), Wcx, Wcy, k, (L - z)); % Reference propagation

% Reconstructed image
Rec = U .* conj(U0); % Calculate the interference pattern
figure(1), 
subplot(1,2,1), imagesc(abs(Rec)), colormap gray, axis off, title('Amplitude')
subplot(1,2,2), imagesc(angle(Rec)), colormap gray, axis off, title('Phase') % Display phase of reconstructed field

%% Functions

% Point source function to generate reference wave
function P = point_src(N, M, z, x0, y0, lambda, dx)
    % point_src Generates a point source illumination centered at (x0, y0)
    % and observed in a plane at a distance z.
    % 
    % Parameters:
    %   N, M - Number of points in the y and x dimensions
    %   z    - Distance to screen
    %   x0, y0 - Center coordinates of point source
    %   lambda - Wavelength
    %   dx     - Sampling pitch in x (and y if dx = dy)
    %
    % Output:
    %   P - Complex field of point source illumination

    dy = dx; % Set y-pitch same as x-pitch
    [m, n] = meshgrid(1-M/2 : M/2, 1-N/2 : N/2); % Mesh grid for coordinates

    k = 2 * pi / lambda; % Wavenumber
    r = sqrt(z^2 + (m * dx - x0).^2 + (n * dy - y0).^2); % Radial distance from source

    P = exp(1i * k * r) ./ r; % Complex field with spherical phase
end

% Angular spectrum propagation function
function B = angular_spectrum(A, Wx, Wy, k, z)
    % angular_spectrum Propagates field A using angular spectrum method.
    % 
    % Parameters:
    %   A  - Input field
    %   Wx, Wy - Physical width in x and y
    %   k  - Wavenumber
    %   z  - Propagation distance
    %
    % Output:
    %   B - Output field after propagation

    [P, Q] = size(A); % Get size of input field
    dfx = 1 / Wx; % Frequency sampling interval in x
    dfy = 1 / Wy; % Frequency sampling interval in y
    
    % Generate frequency grid
    [fx, fy] = meshgrid(-P/2 * dfx : dfx : (P/2 - 1) * dfx, -Q/2 * dfy : dfy : (Q/2 - 1) * dfy);

    % Complex exponential term for propagation
    E = exp(-1i * z * sqrt(k^2 - 4 * pi^2 * (fx.^2 + fy.^2)));

    % Perform Fourier transform, apply propagation, and inverse Fourier transform
    B = ifts(fts(A) .* E);
end
