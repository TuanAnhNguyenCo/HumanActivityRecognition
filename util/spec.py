
import numpy as np


def _create_spectrogram(signal: np.ndarray) -> np.ndarray:
    """Creates spectrogram for signal, and returns it.

    The resulting spectrogram represents time on the x axis, frequency
    on the y axis, and the color shows amplitude.
    """
    n = len(signal)  # 128
    sigma = 3
    time_list = np.arange(n)
    spectrogram = np.zeros((n, n))

    for (i, time) in enumerate(time_list):
        # We isolate the original signal at a particular time by multiplying
        # it with a Gaussian filter centered at that time.
        g = _get_gaussian_filter(time, time_list, sigma)
        ug = signal * g
        # Then we calculate the FFT. Some FFT values may be complex, so we take
        # the absolute value to guarantee that they're all real.
        # The FFT is the same size as the original signal.
        ugt = np.abs(np.fft.fftshift(np.fft.fft(ug)))
        # The result becomes a column in the spectrogram.
        spectrogram[:, i] = ugt

    return spectrogram


def _get_gaussian_filter(b: float, b_list: np.ndarray,
                         sigma: float) -> np.ndarray:
    """Returns the values of a Gaussian filter centered at time value b, for all
    time values in b_list, with standard deviation sigma.
    """
    a = 1 / (2 * sigma**2)
    return np.exp(-a * (b_list - b)**2)
