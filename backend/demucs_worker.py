import argparse
import wave
from pathlib import Path

import numpy as np
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model


def read_wav(path: Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        samplerate = wf.getframerate()
        frame_count = wf.getnframes()
        if sample_width != 2:
            raise RuntimeError(f"仅支持 16-bit PCM WAV 输入: {path}")
        raw = wf.readframes(frame_count)

    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if data.size == 0:
        raise RuntimeError(f"WAV 为空: {path}")
    data = data.reshape(-1, channels).T.copy()
    return torch.from_numpy(data), samplerate


def write_wav(path: Path, wav_tensor: torch.Tensor, samplerate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = wav_tensor.detach().cpu().numpy()
    wav = np.clip(wav, -1.0, 1.0)
    pcm = (wav.T * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(wav.shape[0])
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Demucs vocals separation on a prepared WAV file.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="htdemucs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shifts", type=int, default=1)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--segment", type=int, default=None)
    args = parser.parse_args()

    wav, samplerate = read_wav(args.input)
    model = get_model(args.model)
    model.cpu()
    model.eval()

    if samplerate != model.samplerate:
        raise RuntimeError(
            f"输入采样率 {samplerate} 与模型采样率 {model.samplerate} 不一致。"
        )

    ref = wav.mean(0)
    wav = wav - ref.mean()
    std = ref.std()
    if float(std) > 1e-8:
        wav = wav / std
    else:
        std = torch.tensor(1.0)

    sources = apply_model(
        model,
        wav[None],
        device=args.device,
        shifts=args.shifts,
        split=True,
        overlap=args.overlap,
        progress=False,
        num_workers=args.jobs,
        segment=args.segment,
    )[0]
    sources = sources * std
    sources = sources + ref.mean()

    if "vocals" not in model.sources:
        raise RuntimeError(f"模型不支持 vocals stem: {args.model}")
    vocals = sources[model.sources.index("vocals")]
    write_wav(args.output, vocals, model.samplerate)


if __name__ == "__main__":
    main()
