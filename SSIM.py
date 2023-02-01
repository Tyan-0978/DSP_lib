# ------------------------------------------------------------
# Structural Similarity measurement
# ------------------------------------------------------------

import random
import numpy as np
import cv2

def SSIM(A, B, c1, c2):
  '''
  returns the structural similarity of A, B
  c1, c2 are adjustable constants
  '''
  L = 255   # 255 for images

  # check if A, B have the same dimension
  if not np.shape(A) == np.shape(B):
    print('Error: A, B should have the same size for computing SSIM.')
    return

  # compute means of A, B
  u_A, u_B = np.mean(A), np.mean(B)

  # compute variances of A, B
  var_A = np.mean((A - u_A) ** 2)
  var_B = np.mean((B - u_B) ** 2)

  # compute covariance of A, B
  covar = np.mean((A - u_A) * (B - u_B))

  # compute SSIM
  num1 = 2 * u_A * u_B + (c1 * L) ** 2
  num2 = 2 * covar + (c2 * L) ** 2
  den1 = u_A ** 2 + u_B ** 2 + (c1 * L) ** 2
  den2 = var_A + var_B + (c2 * L) ** 2
  SSIM_result = num1 * num2 / (den1 * den2)
  return SSIM_result

if __name__ == '__main__':
  # for testing, prepare any image, place it in the same directory 
  # as the SSIM.py file, and change the path below
  img_path = './cat.jpg'

  # reference image
  img1 = cv2.imread(img_path)

  # same as reference image but brighter
  # this image has large error but high structural similarity
  alpha = 0.5 # brightness parameter
  img_temp = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
  img_temp[:,:,0] = 255 * (img_temp[:,:,0] / 255) ** alpha
  img2 = cv2.cvtColor(img_temp, cv2.COLOR_YCrCb2BGR)

  # ramdom image
  img_ran = np.random.rand(img1.size).reshape(img1.shape) * 255

  # noisy image
  # this image may have smaller error but looks like a random image
  noise_amp = 50 # noise amplitude
  img_noise = (img1 - (img_ran * noise_amp / 255)) % 255

  print(f'Average error for changing brightness: {np.mean(np.abs(img1 - img2))}')
  print(f'Average error for adding noises: {np.mean(np.abs(img1 - img_noise))}')
  print(f'Average error for a random image: {np.mean(np.abs(img1 - img_ran))}')
  print(f'SSIM for changing brightness: {SSIM(img1, img2, 1/16, 1/16)}')
  print(f'SSIM for adding noises: {SSIM(img1, img_noise, 1/16, 1/16)}')
  print(f'SSIM for a random image: {SSIM(img1, img_ran, 1/16, 1/16)}')

  cv2.imshow('original image', img1)
  cv2.imshow('image with different brightness', img2)
  cv2.imshow('image with random noise', img_noise)
  cv2.imshow('random image', img_ran)
  cv2.waitKey()

  #cv2.imwrite('brighter.jpg', img2)
  #cv2.imwrite('noise.jpg', img_noise)
  #cv2.imwrite('random.jpg', img_ran)

# ------------------------------
# end
# ------------------------------
