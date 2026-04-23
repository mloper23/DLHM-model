from dlhm import *
import plotly.express as px

import numpy as np
import plotly.express as px
from scipy.ndimage import zoom

# Load and process hologram - updated
hologram = np.array(cv.imread('./data/Simulated_hologram.png', cv.IMREAD_GRAYSCALE)).astype(float) / 255  # Load hologram image
P, Q = hologram.shape  # Get size of hologram

# Parameters
lambda_ = 532e-9  # Wavelength (m)
dx = 1.85e-6  # Pixel size in x-direction (m)
dy = dx  # Pixel size in y-direction (m)
Wcx = P * dx  # Width of sensor area in x
Wcy = Q * dy  # Width of sensor area in y
L = 11e-3  # Distance to screen (m)
z = 4.95e-3  # Distance to sample (m)
k = 2 * np.pi / lambda_  # Wavenumber

# Calculate sampling rate at sample plane
sampling_sample_plane = lambda_ * np.sqrt((Wcx/2)**2 + (Wcy/2)**2 + z**2) / Wcx
oversampling_factor = dx / sampling_sample_plane  # Determine oversampling factor
# oversampling_factor = 1
# Resize hologram based on oversampling factor
resized_hologram = resize(hologram, P*oversampling_factor, Q*oversampling_factor)  # Resize hologram
N, M = resized_hologram.shape  # Get new size after resizing

# Generate reference wave
reference = point_src(N, M, L, 0, 0, lambda_, (dx * P) / N)  # Reference wave using point source

# Perform angular spectrum propagation
U = angular_spectrum(reference * resized_hologram, Wcx, Wcy, k, (L - z))  # Sample to sensor
U0 = angular_spectrum(reference * np.ones((N, M)), Wcx, Wcy, k, (L - z))  # Reference propagation

# Reconstructed image
Rec = U * np.conj(U0)  # Calculate the interference pattern

# Plot amplitude and phase using Plotly Express
amp_fig = px.imshow(np.abs(Rec), color_continuous_scale='gray', title="Amplitude")
amp_fig.update_layout(coloraxis_showscale=False)
amp_fig.show()

phase_fig = px.imshow(np.angle(Rec), color_continuous_scale='gray', title="Phase")
phase_fig.update_layout(coloraxis_showscale=False)
phase_fig.show()