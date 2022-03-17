# ------------------------------------------------------------
# FIR mini-max filter design
# currently only low-pass or high-pass filters available
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

def mini_max_filter(length, passbd, transbd, weight, analog_intv):
  # length: length of the final impulse response
  # passbd: list of the boundaries of passband, e.g. [0, 0.2]
  # transbd: list of the boundaries of transition band, e.g. [0.2, 0.3]
  # weight: dictionary of passband/stopband weight, e.g. {"pass": 1, "stop": 0.5}
  # analog_intv: analog frequency interval

  k = (length - 1) // 2
  analog_freq = int(1 / analog_intv)
  F = np.linspace(0, 0.5, analog_freq + 1)
  edge_freq = sum(transbd) / 2 # transition band center
  is_lowpass = (passbd[1] == transbd[0])

  # set weight function and desired transfer function
  w = np.zeros(analog_freq + 1)
  Hd = np.ones(analog_freq + 1)
  for i in range(analog_freq + 1):
    if passbd[0] <= F[i] <= passbd[1]:
      w[i] = weight["pass"]
    elif transbd[0] < F[i] < transbd[1]:
      if is_lowpass:
        if F[i] >= edge_freq:
          Hd[i] = 0
      else: # highpass
        if F[i] <= edge_freq:
          Hd[i] = 0
    else: # stopband
      w[i] = weight["stop"]
      Hd[i] = 0

  #plt.plot(F, w)
  #plt.plot(F, Hd)

  # initialize extreme points randomly
  extreme_points = []
  random_points = np.random.choice(analog_freq, k + 2)
  random_points[0] = sum(passbd) * analog_freq // 2 # at least one in passband
  for p in random_points:
    if transbd[0] < F[p] < transbd[1]:
      alt_p = p
      while transbd[0] < F[alt_p] < transbd[1]:
        alt_p = np.random.choice(analog_freq, 1)[0]
      extreme_points.append(F[alt_p])
    else:
      extreme_points.append(F[p])
  extreme_points.sort()
  extreme_points = np.array(extreme_points)

  #print(extreme_points)

  # main loop
  max_err = 200
  prev_max_err = 0
  loop_count = 0
  while not 0 <= prev_max_err - max_err <= analog_intv:
    loop_count += 1

    # square matrix
    sqr = np.ones((k + 2) ** 2).reshape(k+2, k+2)
    for i in range(1, k+1):
      sqr[:,i] = np.cos(2 * np.pi * i * extreme_points)
    for j in range(k + 2):
      sqr[j, k+1] = ((-1) ** j) / w[np.where(F == extreme_points[j])]

    # desired transfer function at extreme points
    Hd_extreme = np.ones(k + 2)
    for i in range(k + 2):
      Hd_extreme[i] = Hd[np.where(F == extreme_points[i])]
    Hd_extreme = Hd_extreme.reshape(k+2, 1) # reshape to vector

    # calculate s[n]
    s = np.matmul(np.linalg.inv(sqr), Hd_extreme)
    s = s.flatten()

    #print(sqr)
    #print(np.linalg.inv(sqr))
    #print(s)

    # calculate frequency response R(F) and error
    R = np.ones(analog_freq + 1) * s[0]
    for i in range(1, k+1):
      R = R + (s[i] * np.cos(2 * np.pi * i * F))
    error = (R - Hd) * w
    abs_error = np.absolute(error)

    # plt.figure()
    # plt.plot(F, Hd, label='Hd')
    # plt.plot(F, R, label='R')
    # plt.plot(F, error, label='error')
    # plt.legend()

    # find new extreme points and maximum error
    prev_max_err = max_err
    max_err = np.amax(abs_error)
    new_ext = []
    for i in range(analog_freq + 1):
      if i == 0:
        is_local_max = error[i] > 0 and error[i] > error[i+1]
        is_local_min = error[i] < 0 and error[i] < error[i+1]
      elif i == analog_freq:
        is_local_max = error[i] > error[i-1] and error[i] > 0
        is_local_min = error[i] < error[i-1] and error[i] < 0
      else:
        is_local_max = error[i] > error[i-1] and error[i] > error[i+1]
        is_local_min = error[i] < error[i-1] and error[i] < error[i+1]
      if is_local_max or is_local_min:
        new_ext.append(F[i])

    # if there are more than k+2 extreme points
    while len(new_ext) > (k + 2):
      min_err = 200
      min_err_freq = 1
      # check 4 frequency boundaries
      if 0 in new_ext and abs_error[0] < min_err:
        min_err = abs_error[0]
        min_err_freq = 0
      if transbd[0] in new_ext and abs_error[np.where(F == transbd[0])] < min_err:
        min_err = abs_error[np.where(F == transbd[0])]
        min_err_freq = transbd[0]
      if transbd[1] in new_ext and abs_error[np.where(F == transbd[1])] < min_err:
        min_err = abs_error[np.where(F == transbd[1])]
        min_err_freq = transbd[1]
      if 0.5 in new_ext and abs_error[analog_freq] < min_err:
        min_err = abs_error[analog_freq]
        min_err_freq = 0.5
      new_ext.remove(min_err_freq)

    extreme_points = np.array(new_ext)

    print(f"Maximum error in loop {loop_count}: {max_err}")
  # end main loop

  # plot 
  plt.figure()
  plt.plot(F, Hd, label='desired')
  plt.plot(F, R, label='R')
  #plt.plot(F, error, label='error')
  plt.legend()
  plt.show()

  # return impulse response
  impulse_res = np.concatenate((np.flip(s[1:k+1]) * 0.5, s[0:1], s[1:k+1] * 0.5))
  return impulse_res

if __name__ == "__main__":
  imp_res = mini_max_filter(
    length=17,
    passbd=[0, 1200 / 6000],
    transbd=[1200 / 6000, 1500 / 6000],
    weight={"pass": 1, "stop": 0.6},
    analog_intv=1e-4
  )
  print(imp_res)

# ------------------------------
# end
# ------------------------------
