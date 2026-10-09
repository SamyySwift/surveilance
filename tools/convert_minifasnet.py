"""
Convert Silent-Face-Anti-Spoofing's MiniFASNet weights (PyTorch) to ONNX for
liveness.py, verify the conversion, and record where everything came from.

The server doesn't need PyTorch; only this one-off conversion does. Run it
in a throwaway environment:

    python -m venv /tmp/convert-env
    /tmp/convert-env/bin/pip install torch onnx onnxruntime numpy "opencv-python-headless<5" python-dotenv

(OpenCV 5 dropped the Caffe loader the original repo's face detector needs.)
    git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing /tmp/sfas
    /tmp/convert-env/bin/python -I tools/convert_minifasnet.py /tmp/sfas models/liveness

What it does:
  - loads each checkpoint with torch.load(weights_only=True), so no pickled
    code from the download is executed;
  - exports it to ONNX;
  - checks ONNX Runtime matches PyTorch on random inputs, and that the full
    ONNX path (crop + preprocessing as in liveness.py) reproduces the
    original repo's verdicts on its own sample images;
  - writes MANIFEST.json (source commit, SHA-256 of every input and output,
    license) and copies the Apache-2.0 license next to the models.
"""

import datetime
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import onnxruntime as ort
import torch

# (checkpoint file, model class name, crop scale, output file)
MODELS = [
    ("2.7_80x80_MiniFASNetV2.pth", "MiniFASNetV2", 2.7, "minifasnet_v2_scale2.7_80x80.onnx"),
    ("4_0_0_80x80_MiniFASNetV1SE.pth", "MiniFASNetV1SE", 4.0, "minifasnet_v1se_scale4.0_80x80.onnx"),
]
INPUT_SIZE = (80, 80)
LIVE_CLASS_INDEX = 1  # Silent-Face's label 1 = real face
# The repo's sample images and the verdict its own test.py gives for them.
SAMPLES = {"image_T1.jpg": True, "image_F1.jpg": False, "image_F2.jpg": False}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model_module(repo):
    # Load only the network definition file (it imports nothing but torch),
    # rather than putting the downloaded repo on sys.path.
    path = os.path.join(repo, "src", "model_lib", "MiniFASNet.py")
    spec = importlib.util.spec_from_file_location("minifasnet_def", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(module, class_name, checkpoint):
    kernel = ((INPUT_SIZE[1] + 15) // 16, (INPUT_SIZE[0] + 15) // 16)
    model = getattr(module, class_name)(conv6_kernel=kernel)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval()


def export(model, out_path):
    dummy = torch.zeros(1, 3, INPUT_SIZE[1], INPUT_SIZE[0])
    kwargs = dict(input_names=["input"], output_names=["logits"], opset_version=13,
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(model, dummy, out_path, dynamo=False, **kwargs)
    except TypeError:  # older torch without the dynamo argument
        torch.onnx.export(model, dummy, out_path, **kwargs)


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def check_random_inputs(model, session):
    x = np.random.default_rng(0).uniform(0, 255, size=(8, 3, INPUT_SIZE[1], INPUT_SIZE[0])).astype(np.float32)
    with torch.no_grad():
        ref = softmax(model(torch.from_numpy(x)).numpy())
    got = softmax(session.run(None, {"input": x})[0])
    diff = float(np.abs(ref - got).max())
    if diff > 1e-4:
        raise SystemExit(f"ONNX output differs from PyTorch by {diff}")
    return diff


def repo_bbox(repo, image):
    """The original repo's face box for an image (its RetinaFace Caffe model), as x1, y1, x2, y2."""
    import math

    det_dir = os.path.join(repo, "resources", "detection_model")
    net = cv2.dnn.readNetFromCaffe(os.path.join(det_dir, "deploy.prototxt"),
                                   os.path.join(det_dir, "Widerface-RetinaFace.caffemodel"))
    height, width = image.shape[:2]
    img = image
    if width * height >= 192 * 192:
        ar = width / height
        img = cv2.resize(image, (int(192 * math.sqrt(ar)), int(192 / math.sqrt(ar))))
    net.setInput(cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123)), "data")
    out = net.forward("detection_out").squeeze()
    i = int(np.argmax(out[:, 2]))
    left, top, right, bottom = out[i, 3] * width, out[i, 4] * height, out[i, 5] * width, out[i, 6] * height
    return [int(left), int(top), int(right), int(bottom)]


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    repo, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    module = load_model_module(repo)
    commit = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()

    entries, sessions = [], []
    for ckpt_name, class_name, scale, out_name in MODELS:
        ckpt = os.path.join(repo, "resources", "anti_spoof_models", ckpt_name)
        out_path = os.path.join(out_dir, out_name)
        model = build(module, class_name, ckpt)
        export(model, out_path)
        session = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        diff = check_random_inputs(model, session)
        print(f"{out_name}: max |ONNX - PyTorch| = {diff:.2e}")
        sessions.append((session, scale))
        entries.append({
            "file": out_name,
            "source_file": f"resources/anti_spoof_models/{ckpt_name}",
            "source_sha256": sha256(ckpt),
            "onnx_sha256": sha256(out_path),
            "architecture": class_name,
            "scale": scale,
            "input_size": list(INPUT_SIZE),
            "input_format": "BGR, values 0-255, not normalised, NCHW",
            "live_class_index": LIVE_CLASS_INDEX,
        })

    # End-to-end: the liveness.py crop + preprocessing on the repo's samples.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from liveness import _context_crop  # noqa: E402

    for name, expect_real in SAMPLES.items():
        image = cv2.imread(os.path.join(repo, "images", "sample", name))
        bbox = repo_bbox(repo, image)
        probs = []
        for session, scale in sessions:
            crop = _context_crop(image, bbox, scale, *INPUT_SIZE)
            x = crop.astype(np.float32).transpose(2, 0, 1)[None]
            probs.append(softmax(session.run(None, {"input": x})[0])[0])
        mean = np.mean(probs, axis=0)
        is_real = int(np.argmax(mean)) == LIVE_CLASS_INDEX
        print(f"{name}: live probability {mean[LIVE_CLASS_INDEX]:.3f} -> {'real' if is_real else 'fake'}")
        if is_real != expect_real:
            raise SystemExit(f"{name}: expected {'real' if expect_real else 'fake'}; conversion is wrong")

    shutil.copy(os.path.join(repo, "LICENSE"), os.path.join(out_dir, "LICENSE-Silent-Face-Anti-Spoofing"))
    manifest = {
        "description": "MiniFASNet passive liveness models from Silent-Face-Anti-Spoofing, converted to ONNX",
        "source": "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing",
        "source_commit": commit,
        "license": "Apache-2.0 (see LICENSE-Silent-Face-Anti-Spoofing)",
        "converted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "converted_with": {"torch": torch.__version__, "onnxruntime": ort.__version__},
        "models": entries,
    }
    with open(os.path.join(out_dir, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(entries)} models and MANIFEST.json to {out_dir}")


if __name__ == "__main__":
    main()
