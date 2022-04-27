# ------------------------------------------------------------
# Music File Generating Function
# Generate .wav file by simplified music notation (numbers)
# ------------------------------------------------------------

import scipy.io.wavfile as scipy_wav
import numpy as np
import matplotlib.pyplot as plt

def numbers_to_music(score, beat, name='music', bpm=180, f_base=524,
                     chord=[], chord_beat=[], tc=0.5, fs=22050):
  # function for generating music files by score and beat numbers
  # score should contain non-negative numbers only
  # options: 
  # - bpm (beat per minute), default is 180
  # - base note frequency, default is (roughly) the C that is
  #   8 degree higher than the middle C at 440Hz standard
  # - time constant, used for amplitude decay
  # - sampling frequency
  
  # check whether the score and beat have the same length
  if len(score) != len(beat):
    print('Error: score and beat should have the same length.')
    return

  # parameters
  ppb = int(60 / bpm * fs) # points per beat
  note_num = len(score)    # number of notes
  note_convert = {         # used for note to frequency conversion
    1: 0, 1.5: 1,
    2: 2, 2.5: 3,
    3: 4,
    4: 5, 4.5: 6,
    5: 7, 5.5: 8,
    6: 9, 6.5: 10,
    0: 11
  }

  # generate music signal (melody)
  music_signal = np.empty(0)
  for i in range(note_num):
    if score[i] == 0: # silent note
      music_signal = np.append(music_signal, np.zeros(beat[i] * ppb))
      continue

    # compute frequency exponent
    e = (note_convert[score[i] % 7] / 12) + (score[i] // 7)
    if score[i] // 7 == 0: # fix exponent if the note is B
      e = e - 1

    f = f_base * (2 ** e)             # note frequency
    t = np.arange(int(beat[i] * ppb)) / fs # time vector

    # amplitude, decades exponentially with time constant
    A_init = 1
    A = A_init * np.exp(-t / tc)

    # append note to the music signal
    note_signal = A * np.cos(2 * np.pi * f * t)
    music_signal = np.append(music_signal, note_signal)

  # generate chord if any
  # frequency of chord notes are lower 8
  if len(chord) != 0:
    if len(chord) != len(chord_beat):
      print('Error: chord and chord_beat have different length,')
      print('so chord are not generated.')
    elif sum(chord_beat) != sum(beat):
      print('Error: beat and chord_beat have different sum,')
      print('so chord are not generated.')
    else:
      chord_num = len(chord)
      chord_signal = np.empty(0)
      for i in range(chord_num):
        chord_size = len(chord[i])              # number of notes in the chords
        chord_length = int(chord_beat[i] * ppb) # length of the signal (vector)

        if chord_size == 0: # silent note
          chord_signal = np.append(chord_signal, np.zeros(chord_length))
          continue

        # time vector
        t = np.arange(chord_length) / fs

        # amplitude
        A_init = 0.2   # initial amplitude
        A = A_init * np.exp(-t / tc)

        single_chord_signal = np.zeros(chord_length)
        for j in range(chord_size):
          # compute frequency exponent
          e = (note_convert[chord[i][j] % 7] / 12) + (chord[i][j] // 7) - 1
          if chord[i][j] // 7 == 0: # fix exponent if the note is B
            e = e - 1
          f = f_base * (2 ** e)
          note_signal = A * np.cos(2 * np.pi * f * t)
          single_chord_signal = single_chord_signal + note_signal
        
        chord_signal = np.append(chord_signal, single_chord_signal)

      # add chord into music signal
      music_signal = music_signal + chord_signal

  # write wave file
  # wave module generates noisy wave file, so I use scipy instead
  #wav_file = wave.open(f'{name}.wav', 'wb')
  #wav_file.setnchannels(2)
  #wav_file.setsampwidth(2)
  #wav_file.setframerate(fs)
  #wav_file.writeframes(music_signal.tobytes())
  #wav_file.close()
  scipy_wav.write(f'{name}.wav', fs, music_signal)
  print(f'{name}.wav file is generated successfully.')

  return

if __name__ == '__main__':
  score_tk = [1,1,5,5,6,6,5]
  beat_tk = [1,1,1,1,1,1,2]
  chord_tk = [[1,3,5], [1,4,6]]
  chord_beat_tk = [4,4]
  numbers_to_music(score_tk, beat_tk, name='twinkle', bpm=150,
                   chord=chord_tk, chord_beat=chord_beat_tk)

  # Mariage d'amour
  score_md = np.array([
    9, 5, 6.5, 9, 8,   9, 5, 6.5, 9, 8,   9, 5, 6.5, 9.5, 9,   9.5, 5, 6.5, 9.5, 9,
    9.5, 9.5, 9, 9.5, 10,   11, 11, 12, 11, 12,   9,
    -7
  ]) + 7
  beat_md = np.array([
    2, 1,1,1,1, 2, 1,1,1,1, 2, 1,1,1,1, 2, 1,1,1,1,
    2, 1,1,1,1, 2, 1,1,1,1, 6,
    6
  ])
  chord_md = np.array([
    [5],[9],[13.5],[9],[13.5],[9], [5],[9],[13.5], [8],[12],[16.5],
    [12],[16.5],[12], [4],[8],[13], [6.5],[13.5],[6,13],
    [5,12]
  ], dtype=object)
  chord_beat_md = np.zeros(len(chord_md)) + 2
  chord_beat_md[-1] = 6
  numbers_to_music(score_md, beat_md, name='Mariage_damour', f_base=524,
                   chord=chord_md, chord_beat=chord_beat_md, 
                   bpm=280, tc=0.2, fs=44100)

# ------------------------------
# end of the program
# ------------------------------
