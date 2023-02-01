# ------------------------------------------------------------
# number theoretical transform matrix generator
# ------------------------------------------------------------

import numpy as np

# function for finding inverse (mod N)
def get_inv(x, N):
  for n in range(1, N):
    if (x * n) % N == 1:
      return n
  return 0

def NTTm(N, M):
  '''
  N: number of points
  M: modulus
  return: forward and inverse transform matrices
  '''

  # find the smallest number a such that:
  #   a**N (mod M) = 1
  #   a**n (mod M) != 1 for n = 0, 1, 2, ... N-1
  a = 1
  found_a = False
  while not found_a:
    a += 1
    if (a >= M):
      print('error: a cannot be found')
      return

    power_a = a
    for i in range(2, N+1):
      power_a = (power_a * a) % M
      if power_a == 1:
        if i == N: 
          found_a = True
        else:
          break

  # find inverse of a, N ---------------------------
  a_inv = get_inv(a, M)
  N_inv = get_inv(N, M)

  # construct NTT matrix ---------------------------
  ntt_mat = np.ones((N, N))
  ntt_mat_inv = np.ones((N, N))

  for i in range(1, N): # first row
    ntt_mat[1, i] = (ntt_mat[1, i-1] * a) % M
    ntt_mat_inv[1, i] = (ntt_mat_inv[1, i-1] * a_inv) % M
  for j in range(2, N): # other part
    ntt_mat[j, 1:N] = (ntt_mat[j-1, 1:N] * ntt_mat[1, 1:N]) % M
    ntt_mat_inv[j, 1:N] = (ntt_mat_inv[j-1, 1:N] * ntt_mat_inv[1, 1:N]) % M

  ntt_mat_inv = (ntt_mat_inv * N_inv) % M

  return ntt_mat, ntt_mat_inv

# ------------------------------------------------------------
# test
# ------------------------------------------------------------

if __name__ == '__main__':
  N = 10
  M = 8191
  A, B = NTTm(N, M)
  #print(A)
  #print(B)
  C = np.matmul(A, B) % M
  print(C) # should be an unit matrix

# ------------------------------------------------------------
# end
# ------------------------------------------------------------
