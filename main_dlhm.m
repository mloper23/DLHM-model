% Main script for simulating realistic digital lensless holograms

% Load and preprocess the input image
% Convert the image to grayscale and normalize its intensity
intensityImage = im2double(rgb2gray(imread('./data/BenchmarkTarget.png')));

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
L = 4e-3;     % Distance from the source to the hologram plane in meters
z = 1e-3;     % Distance from the source to the sample plane in meters
W_c = 5.55e-3; % Width of the sensor in meters
dx = 1.85e-6; % Pixel size on the sensor in meters

% Repeated definition of lambda (already defined earlier)

% Call the realistic_dlhm function to simulate digital lensless holograms
% Use numerical aperture of 0.05 for the source
hologram = realistic_dlhm(sample, 1.85e-6, L, z, W_c, dx, lambda, 1, 1, 0.1);

% Display the simulated hologram
figure(1);
imagesc(hologram);
colormap gray;
axis image;
title('Simulated Hologram');

% End of main script
