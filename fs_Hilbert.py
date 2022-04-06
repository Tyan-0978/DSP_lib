# ------------------------------------------------------------
# Discrete Hilbert transform by frequency sampling method
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import cmath

def freq_sampling_Hilbert(k):
  # discrete Hilbert transform filter by frequency sampling method
  # return the impulse response
  # transition band is set on 1, k, k+1, 2k points
  N = 2 * k + 1 # length

  # sampled frequency response
  trans_err = 0.1   # error on transition points, larger for smaller ripple
  H_sampled = np.hstack((np.array([0]), np.full(k, complex(0,-1)), np.full(k, complex(0,1))))
  H_sampled[1] = H_sampled[k] = complex(0, -1 + trans_err)
  H_sampled[k+1] = H_sampled[2*k] = complex(0, 1 - trans_err)

  # inverse discrete Fourier transform
  n = np.arange(N)
  m = np.arange(N).reshape(N, 1)
  r_matrix = H_sampled.reshape(N, 1) * np.exp(complex(0, 1) * 2 * np.pi * m * n / N)
  r = np.sum(r_matrix, axis=0) / N

  # return impulse response
  h = np.hstack((r[k+1:], r[:k+1]))
  return h

if __name__ == "__main__":
  # parameters
  k = 8
  N = 2 * k + 1
  fs = 1e4

  n = np.arange(N)
  F = np.linspace(0, 1, int(fs), endpoint=False)

  # get impulse response h[n] and calculate DTFT (centered at n=0)
  h = freq_sampling_Hilbert(k)
  m = np.arange(N).reshape(N, 1) - k # centered at n = 0
  H_matrix = h.real.reshape(N, 1) * np.exp(complex(0, -1) * 2 * np.pi * F * m)
  H = np.sum(H_matrix, axis=0)

  # plot results
  plt.figure()
  plt.stem(n, h.real, use_line_collection=True)
  plt.title('impulse response')
  plt.xlabel('n')
  plt.ylabel('h[n]')
  plt.figure()
  plt.plot(F, H.imag)
  plt.title('imaginary part of frequency response')
  plt.xlabel('F')
  plt.ylabel('imaginary part of H(F)')
  plt.show()

# ------------------------------
# end of the program
# ------------------------------
