from audiomentations import *

def get_transformations():
  """
  Define audio augmentation transformations
  """
  musan_dir = "./musan"
  augment = Compose([
      Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.1),
      PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
      TimeStretch(min_rate=0.9, max_rate=1.1, p=0.4, leave_length_unchanged=False),
      OneOf([
      AddBackgroundNoise(
                      sounds_path=musan_dir, min_snr_in_db=1.0, max_snr_in_db=5.0, noise_transform=PolarityInversion(), p=1.
                  ),
      AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.),],  p=1.0)

      ])

  return augment

def augment_dataset(batch):
  sample = batch['audio']

  augmentation = get_transformations()
    # apply augmentation
  augmented_waveform = augmentation(sample["array"], sample_rate=sample["sampling_rate"])
  batch['audio']["array"] = augmented_waveform
  return batch




