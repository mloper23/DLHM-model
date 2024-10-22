# Main script for simulating realistic digital lensless holography
from dlhm import *
import plotly.express as px
# Load a sample (convert to grayscale and normalize)
intensityImage = np.array(cv.imread('data/BenchmarkTarget.png', cv.IMREAD_GRAYSCALE)).astype(float) / 255
lambda_ = 532e-9  # Wavelength of the light


h_max = 350e-9

sample = np.exp(-1j * 2 * np.pi * (1.51 - 1) * h_max * intensityImage / lambda_)
# sample = 1-intensityImage

# Simulation parameters
L = 8e-3  # Distance from the source to the hologram plane
z = 2e-3  # Distance from the source to the sample's plane
W_c = 5.55e-3  # Width of the sensor
lambda_ = 532e-9  # Wavelength
dx_in = 1.85e-6

# Call the dlhm function to simulate digital lensless holograms
holo = dlhm(sample, dx_in, L, z, W_c, dx_in, lambda_, x0=0, y0=0, NA_s=0.1)

# Display the simulated hologram
fig = px.imshow(holo, color_continuous_scale='gray')
fig.write_html('test.html')
