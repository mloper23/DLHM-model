import numpy as np
import cv2 as cv


def dlhm(sample, dx_in, L, z, W_c, dx_out, wavelength, x0=0, y0=0, NA_s=0):
    """
    Digital Lensless Holography Model (DLHM) Simulation.

    Args:
    sample: np.ndarray
        Input complex field (wavefront) to be simulated.
    dx_in: float
        Resampling resolution parameter for input.
    L: float
        Distance from the point source to the hologram plane.
    z: float
        Distance from the point source to the sample plane.
    W_c: float
        Width of the sensor.
    dx_out: float
        Pixel size at the sensor.
    wavelength: float
        Wavelength of the light.
    x0, y0: float, optional
        Offsets to adjust the center of the cropped sample.
    NA_s: float, optional
        Numerical Aperture of the source.

    Returns:
    holo: np.ndarray
        Simulated hologram as intensity.
    """

    # Determine the size of the input sample
    N, M = sample.shape

    # Magnification factor
    Mag = L / z
    W_s = W_c / Mag

    # Re-sample if desired
    if dx_in != 0:
        res_d = dx_in * Mag / dx_out
        sample = resize(sample, int(M / res_d), int(N / res_d))
        N_s, M_s = sample.shape
        W_sample = N_s * dx_out / Mag

        if W_sample > W_s:
            sample = sample[N_s // 2 - N // 2:N_s // 2 + N // 2, M_s // 2 - M // 2:M_s // 2 + M // 2]
        else:
            sample = resize(sample, M, N)

        # Crop the sample based on the magnification and offsets
        start_x = int(N / 2 - N / (Mag * 2) + x0)
        end_x = int(N / 2 + N / (Mag * 2) + x0)
        start_y = int(M / 2 - M / (Mag * 2) + y0)
        end_y = int(M / 2 + M / (Mag * 2) + y0)
        sample = sample[start_x:end_x, start_y:end_y]

        # Resample to match sensor characteristics
        # rs = sample.shape[0] / (W_c / dx_out)
        sample = resize(sample, M, N)

    # Ensure the updated sample size
    N, M = sample.shape

    # Wave number
    k = 2 * np.pi / wavelength

    # Spatial coordinates in the camera's plane
    x = np.linspace(-W_c / 2, W_c / 2, M)
    y = np.linspace(-W_c / 2, W_c / 2, N)
    u, v = np.meshgrid(x, y)

    # Radial distances for source wavefront
    r = np.sqrt(u ** 2 + v ** 2 + L ** 2)

    # Determine illumination pattern and normalization
    if NA_s != 0:
        PS = np.exp(-2 * r / (L * np.tan(np.arcsin(NA_s))))
    else:
        PS = np.exp(-r)
    PS = PS - np.min(PS)
    PS = PS / np.max(PS)

    # Spatial frequency coordinates at the sample's plane
    dfx = Mag / (dx_out * M)
    dfy = Mag / (dx_out * N)
    fx, fy = np.meshgrid(np.arange(-M / 2 * dfx, M / 2 * dfx, dfx),
                         np.arange(-N / 2 * dfy, N / 2 * dfy, dfy))

    # Propagation kernel for the Angular Spectrum Method (ASM)
    E = np.exp(-1j * (L - z) * np.sqrt(k ** 2 - 4 * np.pi ** 2 * (fx ** 2 + fy ** 2)))

    # Compute hologram using inverse Fourier transform
    Uz = ifts(fts(sample) * E)
    holo = np.abs(Uz)

    # Distortion of the coordinates
    # Calculate maximum distortion due to distance differences
    Max_D = np.abs((L + np.abs(np.sqrt(W_c**2 / 2 + L**2) - L)) / z - L / z)

    # Apply distortion to the hologram
    camMat = np.array([[N, 0, N / 2], [0, M, M / 2], [0, 0, 1]])
    distCoeffs = np.array([-Max_D / (N/2), 0, 0, 0, 0])  # Radial distortion parameters
    holo = cv.undistort(holo.astype(np.float32), camMat, distCoeffs)

    # Normalize and post-process the hologram
    holo = holo - np.min(holo)
    holo = holo / np.max(holo)
    holo = holo * PS
    holo = holo * 2 ** 8
    holo = np.round(holo).astype(np.uint8)

    return holo


def ifts(A):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(A)))


def fts(A):
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(A)))

# Customized function for complex resize
def resize(B, dx, dy):
    am = np.abs(B)
    ph = np.angle(B)
    am_r = cv.resize(am, (int(dx), int(dy)), interpolation=cv.INTER_LINEAR)
    ph_r = cv.resize(ph, (int(dx), int(dy)), interpolation=cv.INTER_LINEAR)
    B_r = am_r * np.exp(1j * ph_r)
    return B_r
