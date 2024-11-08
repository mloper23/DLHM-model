% Main script for simulating realistic digital lensless holograms
% @mariajlopera
% For questions contact mloper23@eafit.edu.co;
% maria.josef.lopera.acosta@vub.be

% Load and preprocess the input image
% Convert the image to grayscale and normalize its intensity
intensityImage = im2double(rgb2gray(imread('./data/BenchmarkTarget.png')));
intensityImage = imresize(intensityImage, 0.3);
% Define the wavelength of the light used in the simulation
lambda = 532e-9;  % Wavelength in meters (532 nm)

% Maximum height difference in the sample
h_max = 350e-9;  % in meters

% Generate the complex wavefront of the sample
% The wavefront is based on the phase modulation by the sample
% The refractive index contrast here is 1.51 (RI of sample) - 1 (RI of medium)
sample = exp(-1i * 2 * pi * (1.51 - 1) * h_max * intensityImage / lambda);
% Display the phase of the sample to verify
figure(1);
imagesc(angle(sample))
colormap gray;
axis image;
title('Phase of the Sample');


%% Simulation parameters setup

% Set the system parameters for the holography simulation
L = 8e-3;     % Distance from the source to the hologram plane in meters
z = 2e-3;     % Distance from the source to the sample plane in meters
W_c = 5.55e-3; % Width of the sensor in meters
dx = 1.85e-6; % Pixel size on the sensor in meters

% Repeated definition of lambda (already defined earlier)

% Call the realistic_dlhm function to simulate digital lensless holograms
it = 20;
zs = linspace(7e-3,7e-3,it);

for i = 1:it
    hologram = dlhm(sample, 1.85e-6, L, zs(i), W_c, dx, lambda, 1, 1, 0);
    
    % Display the simulated hologram
    figure(1);
    imagesc(hologram);
    colormap gray;
    axis image;
    title('Simulated Hologram');
end

