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

    # The discretization on the camera is set as the same input's discretization
    P = M
    Q = N

    # Magnification factor
    Mag = L / z
    W_s = W_c / Mag  # the size of the window (field of view) that will be propagated

    # Re-sample if desired
    if dx_in != 0:
        res_d = dx_in * Mag / dx_out  # ratio of resizing considering the input sample's pixel and the desired
        # magnification
        sample = resize(sample, int(M / res_d), int(N / res_d))
        N_s, M_s = sample.shape
        W_sample = N_s * dx_out / Mag  # the size of the provided field of view is computed

        if W_sample > W_s:  # if the field of view provided is larger that the field of view that the system captures,
            # then only the illuminated portion is taken. Else, it is assumed that the provided field
            # of view is the illuminated portion.
            sample = sample[N_s // 2 - N // 2:N_s // 2 + N // 2, M_s // 2 - M // 2:M_s // 2 + M // 2]
        else:
            sample = resize(sample, P, Q)

        # Crop the sample based on the magnification and offsets
        start_x = int(N / 2 - N / (Mag * 2) + x0)
        end_x = int(N / 2 + N / (Mag * 2) + x0)
        start_y = int(M / 2 - M / (Mag * 2) + y0)
        end_y = int(M / 2 + M / (Mag * 2) + y0)
        sample = sample[start_x:end_x, start_y:end_y]

        # Resample to match sensor characteristics
        sample = resize(sample, P, Q)

    # Wave number
    k = 2 * np.pi / wavelength

    # Spatial coordinates in the camera's plane
    x = np.linspace(-W_c / 2, W_c / 2, P)
    y = np.linspace(-W_c / 2, W_c / 2, Q)
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
    Mag_max = np.sqrt(W_c ** 2 / 2 + L ** 2) / z
    Dist_max = np.abs(Mag_max - Mag)
    # Apply distortion to the hologram
    camMat = np.array([[P, 0, P / 2], [0, Q, Q / 2], [0, 0, 1]])
    distCoeffs = np.array([-Dist_max / (2 * Mag), 0, 0, 0, 0])  # Radial distortion parameters
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
    r = np.real(B)
    img = np.imag(B)
    r_r = cv.resize(r, (int(dx), int(dy)), interpolation=cv.INTER_LINEAR)
    img_r = cv.resize(img, (int(dx), int(dy)), interpolation=cv.INTER_LINEAR)
    B_r = r_r + 1j * img_r
    return B_r


def point_src(N, M, z, x0, y0, lambda_, dx):
    """
    Generates a point source illumination centered at (x0, y0)
    and observed in a plane at distance z.
    """
    dy = dx  # Set y-pitch same as x-pitch
    m, n = np.meshgrid(np.arange(-M / 2, M / 2), np.arange(-N / 2, N / 2))  # Coordinate mesh grid
    k = 2 * np.pi / lambda_  # Wavenumber
    r = np.sqrt(z ** 2 + (m * dx - x0) ** 2 + (n * dy - y0) ** 2)  # Radial distance from source
    return np.exp(1j * k * r) / r  # Complex field with spherical phase


def angular_spectrum(A, Wx, Wy, k, z):
    """
    Propagates field A using angular spectrum method.
    """
    P, Q = A.shape  # Get size of input field
    dfx = 1 / Wx  # Frequency sampling interval in x
    dfy = 1 / Wy  # Frequency sampling interval in y

    # Generate frequency grid using linspace
    fx = np.linspace(-P/2 * dfx, (P/2 - 1) * dfx, P)
    fy = np.linspace(-Q/2 * dfy, (Q/2 - 1) * dfy, Q)
    fx, fy = np.meshgrid(fx, fy)


    # Complex exponential term for propagation
    E = np.exp(-1j * z * np.sqrt(k ** 2 - (2 * np.pi * fx) ** 2 - (2 * np.pi * fy) ** 2))

    # Perform Fourier transform, apply propagation, and inverse Fourier transform
    return ifts(fts(A) * E)
