import librosa
import soundfile as sf
import numpy as np


class AudioUtil:
    # ----------------------------
    # Loads an audio file from disk at a specified sample rate and returns its
    # waveform array and sample rate.
    # ----------------------------
    @staticmethod
    def open(audio_path, sample_rate=22050, mono=True):
        y, sr = librosa.load(audio_path, sr=sample_rate, mono=mono)
        return (y, sr)

    # ----------------------------
    # Saves the given audio array and sample rate to a file with the specified
    # name and extension.
    # ----------------------------
    @staticmethod
    def write(y, output_path, audio_name, extension, sample_rate=22050):
        if extension == "npy":
            np.save(f"{output_path}/{audio_name}.{extension}", y)
            return

        sf.write(
            f"{output_path}/{audio_name}.{extension}",
            y.T,
            sample_rate,
        )

    # ----------------------------
    # Divides the audio into fixed-length overlapping segments, padding the
    # final segment if it’s shorter than the window.
    # ----------------------------
    @staticmethod
    def split(audio, window_ms, overlap_ms):
        y, sr = audio
        chunks = []

        y_length = y.shape[-1]
        window_length = int(sr * window_ms / 1000)
        overlap_length = int(sr * overlap_ms / 1000)

        if y_length < window_length:
            padded_y, sr = AudioUtil.pad_trunc(audio, window_ms)
            chunks.append((padded_y, sr))
            return chunks

        hop_start = 0

        while hop_start <= y_length:
            if y.ndim == 1:
                chunk = y[hop_start : hop_start + window_length]
            else:
                chunk = y[:, hop_start : hop_start + window_length]

            if len(chunk) < window_length:
                padded_chunk, sr = AudioUtil.pad_trunc((chunk, sr), window_ms)
                chunks.append((padded_chunk, sr))
                break

            chunks.append((chunk, sr))

            hop_start += window_length - overlap_length

        return chunks

    # ----------------------------
    # Adjusts the audio length to a target duration by padding with silence or
    # truncating as needed.
    # ----------------------------
    @staticmethod
    def pad_trunc(audio, duration_ms, axis=-1):
        y, sr = audio
        size = int(sr * duration_ms / 1000)
        padded_y = librosa.util.fix_length(y, size=size, axis=axis)
        return (padded_y, sr)

    # ----------------------------
    # Shifts the audio in time by a given number of milliseconds, filling the
    # gap with zeros at the start or end.
    # ----------------------------
    @staticmethod
    def time_shift_zero_pad(audio, duration_ms):
        y, sr = audio

        y_length = y.shape[-1]
        shift_length = int(sr * abs(duration_ms) / 1000)

        if y.ndim == 1:
            zero_pad = np.zeros(abs(shift_length))

            if duration_ms < 0:
                padded_y = np.concatenate((y[shift_length:], zero_pad))
            else:
                padded_y = np.concatenate((zero_pad, y[: y_length - shift_length]))

            return (padded_y, sr)

        zero_pad = np.zeros((y.shape[0], abs(shift_length)))

        if duration_ms < 0:
            padded_y = np.concatenate((y[:, shift_length:], zero_pad), axis=1)
        else:
            padded_y = np.concatenate(
                (zero_pad, y[:, : y_length - shift_length]), axis=1
            )

        return (padded_y, sr)

    # ----------------------------
    # Generate agit  Spectrogram.
    # ----------------------------
    @staticmethod
    def melspectrogram(audio):
        y, sr = audio
        spec = librosa.feature.melspectrogram(y=y, sr=sr)
        spec = librosa.power_to_db(S=spec, ref=np.max)
        return spec

    # ----------------------------
    # Time mask
    # ----------------------------
    @staticmethod
    def time_mask(spec, T):
        aug_spec = spec.copy()
        _, n_frames = aug_spec.shape
        t = np.random.randint(0, T)
        t0 = np.random.randint(0, n_frames - t)
        aug_spec[:, t0 : t0 + t] = 0
        return aug_spec

    # ----------------------------
    # Frequency mask
    # ----------------------------
    @staticmethod
    def freq_mask(spec, F):
        aug_spec = spec.copy()
        n_mels, _ = aug_spec.shape
        f = np.random.randint(0, F)
        f0 = np.random.randint(0, n_mels - f)
        aug_spec[f0 : f0 + f, :] = 0
        return aug_spec
