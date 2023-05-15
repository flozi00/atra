# copied from https://github.com/snakers4/silero-vad

import numpy as np
import torch

from atra.utils import timeit

languages = ["ru", "en", "de", "es"]


def silero_vad(onnx=True):
    """Silero Voice Activity Detector
    Returns a model with a set of utils
    Please see https://github.com/snakers4/silero-vad for usage examples
    """
    model = OnnxWrapper("silero_vad.onnx")

    return model, get_speech_timestamps


class OnnxWrapper:
    def __init__(self, path):
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = onnxruntime.InferenceSession(
            path, providers=["CPUExecutionProvider"], sess_options=sess_options
        )
        self.session.intra_op_num_threads = 1
        self.session.inter_op_num_threads = 1

        self.reset_states()

    def reset_states(self):
        self._h = np.zeros((2, 1, 64)).astype("float32")
        self._c = np.zeros((2, 1, 64)).astype("float32")

    def __call__(self, x, sr: int):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.dim()}")

        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[::step]
            sr = 16000

        if x.shape[0] > 1:
            raise ValueError("Onnx model does not support batching")

        if sr not in [16000]:
            raise ValueError(f"Supported sample rates: {[16000]}")

        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        ort_inputs = {"input": x.numpy(), "h0": self._h, "c0": self._c}
        ort_outs = self.session.run(None, ort_inputs)
        out, self._h, self._c = ort_outs

        out = torch.tensor(out).squeeze(2)[:, 1]  # make output type match JIT analog

        return out


@timeit
def get_speech_probs(
    audio: torch.Tensor,
    model,
    sampling_rate: int = 16000,
    window_size_samples: int = 1536,
):
    """
    This method is used for splitting long audios into speech chunks using silero VAD

    Parameters
    ----------
    audio: torch.Tensor, one dimensional
        One dimensional float torch.Tensor, other types are casted to torch if possible

    model: preloaded .jit silero VAD model

    sampling_rate: int (default - 16000)
        Currently silero VAD models support 8000 and 16000 sample rates

    window_size_samples: int (default - 1536 samples)
        Audio chunks of window_size_samples size are fed to the silero VAD model.
        WARNING! Silero VAD models were trained using 512, 1024, 1536 samples
        for 16000 sample rate and 256, 512, 768 samples for 8000 sample rate.
        Values other than these may affect model perfomance!!

    Returns
    ----------
    speeches: list of dicts
        list containing ends and beginnings of speech chunks
        (samples or seconds based on return_seconds)
    """

    if not torch.is_tensor(audio):
        try:
            audio = torch.Tensor(audio)
        except Exception:
            raise TypeError("Audio cannot be casted to tensor. Cast it manually")

    if len(audio.shape) > 1:
        for i in range(len(audio.shape)):  # trying to squeeze empty dimensions
            audio = audio.squeeze(0)
        if len(audio.shape) > 1:
            raise ValueError(
                """More than one dimension in audio. 
                Are you trying to process audio with 2 channels?"""
            )

    if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
        step = sampling_rate // 16000
        sampling_rate = 16000
        audio = audio[::step]
    else:
        step = 1

    model.reset_states()

    audio_length_samples = len(audio)

    speech_probs = []
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample : current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(
                chunk, (0, int(window_size_samples - len(chunk)))
            )
        speech_prob = model(chunk, sampling_rate).item()
        speech_probs.append(speech_prob)

    return speech_probs


def get_speech_timestamps(
    audio: torch.Tensor,
    speech_probs,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    window_size_samples: int = 1536,
    speech_pad_ms: int = 30,
    return_seconds: bool = False,
):
    """
    This method is used for splitting long audios into speech chunks using silero VAD

    Parameters
    ----------
    audio: torch.Tensor, one dimensional
        One dimensional float torch.Tensor, other types are casted to torch if possible

    threshold: float (default - 0.5)
        Speech threshold. Silero VAD outputs speech probabilities for each audio chunk,
        probabilities ABOVE this value are considered as SPEECH.
        It is better to tune this parameter for each dataset separately,
        but "lazy" 0.5 is pretty good for most datasets.

    sampling_rate: int (default - 16000)
        Currently silero VAD models support 8000 and 16000 sample rates

    min_speech_duration_ms: int (default - 250 milliseconds)
        Final speech chunks shorter min_speech_duration_ms are thrown out

    min_silence_duration_ms: int (default - 100 milliseconds)
        In the end of each speech chunk
        wait for min_silence_duration_ms before separating it

    window_size_samples: int (default - 1536 samples)
        Audio chunks of window_size_samples size are fed to the silero VAD model.
        WARNING! Silero VAD models were trained using 512, 1024, 1536 samples
        for 16000 sample rate and 256, 512, 768 samples for 8000 sample rate.
        Values other than these may affect model perfomance!!

    speech_pad_ms: int (default - 30 milliseconds)
        Final speech chunks are padded by speech_pad_ms each side

    return_seconds: bool (default - False)
        whether return timestamps in seconds (default - samples)

    Returns
    ----------
    speeches: list of dicts
        list containing ends and beginnings of speech chunks
        (samples or seconds based on return_seconds)
    """

    if not torch.is_tensor(audio):
        try:
            audio = torch.Tensor(audio)
        except Exception:
            raise TypeError("Audio cannot be casted to tensor. Cast it manually")

    if len(audio.shape) > 1:
        for i in range(len(audio.shape)):  # trying to squeeze empty dimensions
            audio = audio.squeeze(0)
        if len(audio.shape) > 1:
            raise ValueError(
                """More than one dimension in audio. 
                Are you trying to process audio with 2 channels?"""
            )

    if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
        step = sampling_rate // 16000
        sampling_rate = 16000
        audio = audio[::step]
    else:
        step = 1

    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000

    audio_length_samples = len(audio)

    triggered = False
    speeches = []
    current_speech = {}
    neg_threshold = threshold - 0.15
    temp_end = 0

    # iterate over the speech probabilities
    for i, speech_prob in enumerate(speech_probs):
        # if we've found speech and haven't already found
        # the end of a speech segment
        if (speech_prob >= threshold) and temp_end:
            # set temp_end to 0 to indicate we've found
            # the end of the speech segment
            temp_end = 0

        # if we've found speech and haven't already found
        # the start of a speech segment
        if (speech_prob >= threshold) and not triggered:
            # set triggered to True to indicate we've found
            # the start of the speech segment
            triggered = True
            # set the start of the speech segment to the current sample
            current_speech["start"] = window_size_samples * i
            # continue to the next iteration of the loop
            continue

        # if we haven't found speech, but we have found the
        # start of a speech segment
        if (speech_prob < neg_threshold) and triggered:
            # if we haven't already found the end of the
            # speech segment, set it to the current sample
            if not temp_end:
                temp_end = window_size_samples * i
            # if the current sample is not within
            # min_silence_samples of the last sample we found
            if (window_size_samples * i) - temp_end < min_silence_samples:
                # continue to the next iteration of the loop
                continue
            # if the current sample is within
            # min_silence_samples of the last sample we found
            else:
                # set the end of the speech segment to the last sample we found
                current_speech["end"] = temp_end
                # if the length of the speech segment is greater than min_speech_samples
                if (
                    current_speech["end"] - current_speech["start"]
                ) > min_speech_samples:
                    # append the speech segment to the list of speeches
                    speeches.append(current_speech)
                # set temp_end to 0 to indicate we've found
                # the end of the speech segment
                temp_end = 0
                # reset current_speech to an empty dict
                current_speech = {}
                # set triggered to False to indicate we haven't
                # found the start of the speech segment
                triggered = False
                # continue to the next iteration of the loop
                continue

    if (
        current_speech
        and (audio_length_samples - current_speech["start"]) > min_speech_samples
    ):
        # When the current speech is longer than the minimum speech length,
        # set the end of the current speech to the end of the audio.
        current_speech["end"] = audio_length_samples
        # Add the current speech to the list of speeches.
        speeches.append(current_speech)

    # Iterate over each speech segment and adjust the start and end indices
    for i, speech in enumerate(speeches):
        # If first speech segment, adjust the start index
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        # If not the last speech segment, adjust the start and end indices
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            # If the silence duration is too short, adjust the end index of the
            # current speech segment and the start index of the next speech segment
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - silence_duration // 2)
                )
            # If the silence duration is long enough, adjust the end index of the
            # current speech segment and the start index of the next speech segment
            else:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + speech_pad_samples)
                )
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - speech_pad_samples)
                )
        # If the current speech segment is the last speech segment, adjust the end index
        else:
            speech["end"] = int(
                min(audio_length_samples, speech["end"] + speech_pad_samples)
            )

    if return_seconds:
        for speech_dict in speeches:
            speech_dict["start"] = round(speech_dict["start"] / sampling_rate, 1)
            speech_dict["end"] = round(speech_dict["end"] / sampling_rate, 1)
    elif step > 1:
        for speech_dict in speeches:
            speech_dict["start"] *= step
            speech_dict["end"] *= step

    return speeches
