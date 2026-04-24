# Script to simulate Configuration 1 of the paper, working.

# @mariajloepra, 24/04/2026
import numpy as np

# Libraries
from dlhm import *
import plotly.express as px

# Load a sample (convert to grayscale and normalize)
intensityImage = np.array(cv.imread('data/Complete_Benchmark.png', cv.IMREAD_GRAYSCALE)).astype(float) / 255

# Parameters and complex-valued sample creation
lambda_ = 532e-9  # Wavelength
h_max = 350e-9
sample = np.exp(-1j * 2 * np.pi * (1.51 - 1) * h_max * intensityImage / lambda_)

P, Q = sample.shape
# Just more parameters
N = 3000
M = 4000
dx_out = 1.85e-6
L = 31.45e-3  # Distance from the source to the hologram plane
z = 8.2e-3  # Distance from the source to the sample's plane

Mag = L / z
W_cy = N * dx_out # Width of the sensor
W_cx = M * dx_out
lambda_ = 532e-9  # Wavelength
k = 2 * np.pi / lambda_
dx_in = 0.39e-6 # Physical size of the image

# Pixel pitch at the sample plane
dx_sample = dx_out / Mag

# As we don't want to over-compute stuff, we are limited by the sampling criteria
sampling_factor =  dx_in / dx_sample
resized_sample = (resize(sample, Q*sampling_factor, P*sampling_factor))  # Resize hologram
P, Q = resized_sample.shape

# Here I create the spherical wavefront source
X = np.linspace(-W_cx / 2, W_cx / 2, Q)
Y = np.linspace(-W_cy / 2, W_cy / 2, P)
x, y = np.meshgrid(X, Y)
r = np.sqrt(x ** 2 + y ** 2 + L ** 2)
spherical = np.exp(1j * k * r) / r

# Spherical wavefront source at the sensor's plane
spherical_sample_plane = angular_spectrum(spherical, W_cx, W_cy, k, (L - z))
# Spherical wavefront source at the sample's plane
sample_plane = spherical_sample_plane * resized_sample
# I am deleting this variable to reduce memory
del spherical_sample_plane
# I compute the hologram with the paper's approach
hologram_phys_img = dlhm(resized_sample, dx_sample, L, z, W_cx, W_cy, dx_out, lambda_, x0=-1000, y0=0, NA_s=0)
# Again, freeing memory
del resized_sample
# I continue with the ASM with spherical wavefront
sensor_plane = angular_spectrum(sample_plane, W_cx, W_cy, k, -(L - z))
# Final discretization
sensor_plane = resize(sensor_plane, M, N)
hologram_spherical_asm = np.abs(sensor_plane) ** 2
#
del sensor_plane
# Reconstruction. We will discuss this later, I know you'll have many questions
rec_spherical_asm = np.angle(angular_spectrum(hologram_spherical_asm, W_cx, W_cy, k, Mag*(L - z)))
rec_phys_img = np.angle(angular_spectrum(hologram_phys_img, W_cx, W_cy, k, Mag*(L - z)))

# Just plotting
fig = px.imshow(hologram_spherical_asm, color_continuous_scale='gray')
fig.write_html('hologram_sph_asm.html')
fig = px.imshow(rec_spherical_asm, color_continuous_scale='gray')
fig.write_html('rec_sph_asm.html')
fig = px.imshow(rec_phys_img, color_continuous_scale='gray')
fig.write_html('rec_phys_img.html')
fig = px.imshow(hologram_phys_img, color_continuous_scale='gray')
fig.write_html('hologram_phys_img.html')
