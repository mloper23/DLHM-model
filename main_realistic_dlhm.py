# Main script for simulating realistic digital lensless holography

from realistic_dlhm import *
# Load a sample (convert to grayscale and normalize)
sample =Image.open('Experimental holograms/USAF_mask.png').convert('L')
sample = np.asarray(sample) / 255

# Simulation parameters
L = 8e-3  # Distance from the source to the hologram plane
z = 2e-3  # Distance from the source to the sample's plane
W_c = 4.71e-3  # Width of the sensor
lambda_ = 532e-9  # Wavelength of the light

# Call the realistic_dlhm function to simulate digital lensless holograms
holo, ref = realistic_dlhm(sample, L, z, W_c, lambda_)

# Display the simulated hologram and reference wave
fig, axs = plt.subplots(1, 2)
axs[0].imshow(holo, cmap='gray')
axs[0].axis('image')
axs[0].set_title('Simulated Hologram')

axs[1].imshow(ref, cmap='gray')
axs[1].axis('image')
axs[1].set_title('Reference Wave')

plt.show()

# End of main script
