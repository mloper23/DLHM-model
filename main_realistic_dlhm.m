% Main script for simulating realistic digital lensless hologrms

% Load a sample (convert to grayscale and normalize)
sample = rgb2gray(im2double(imread('./Experimental holograms/USAF_mask.png')));

% Simulation parameters
L = 8e-3;          % Distance from the source to the hologram plane
z = 4e-3;          % Distance from the source to the sample's plane
W_c = 4.71e-3;     % Width of the sensor 
lambda = 532e-9;   % Wavelength of the light

% Call the realistic_dlhm function to simulate digital lensless holograms
[holo, ref] = realistic_dlhm(sample, L, z, W_c, lambda);

% Display the simulated hologram and reference wave
figure(1)
subplot(1,2,1), imagesc(holo), colormap gray, axis image
title('Simulated Hologram')
subplot(1,2,2), imagesc(ref), colormap gray, axis image
title('Reference Wave')

% End of main script
