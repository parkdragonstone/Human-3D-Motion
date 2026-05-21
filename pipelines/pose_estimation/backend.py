import logging


logger = logging.getLogger(__name__)


def setup_backend_device(backend="auto", device="auto"):
    """
    Set up backend & device.

    If device and backend are not specified, they are determined automatically:
      1) CUDA + onnxruntime
      2) ROCm + onnxruntime
      3) MPS/CoreML + onnxruntime
      4) CPU + openvino (fallback)
    """
    if device != "auto" and backend != "auto":
        device = device.lower()
        backend = backend.lower()

    if device == "auto" or backend == "auto":
        if (device == "auto" and backend != "auto") or (device != "auto" and backend == "auto"):
            logger.warning("If you set device or backend to 'auto', set the other to 'auto' too. Auto-detecting both.")

        try:
            import onnxruntime as ort
            import torch
            if torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers():
                device = "cuda"
                backend = "onnxruntime"
                logger.info("\nValid CUDA installation found: using ONNXRuntime backend with GPU.")
            elif torch.cuda.is_available() and "ROCMExecutionProvider" in ort.get_available_providers():
                device = "rocm"
                backend = "onnxruntime"
                logger.info("\nValid ROCM installation found: using ONNXRuntime backend with GPU.")
            else:
                raise RuntimeError("No CUDA/ROCM provider")
        except Exception:
            try:
                import onnxruntime as ort
                if "MPSExecutionProvider" in ort.get_available_providers() or "CoreMLExecutionProvider" in ort.get_available_providers():
                    device = "mps"
                    backend = "onnxruntime"
                    logger.info("\nValid MPS installation found: using ONNXRuntime backend with GPU.")
                else:
                    raise RuntimeError("No MPS/CoreML provider")
            except Exception:
                device = "cpu"
                backend = "openvino"
                logger.info("\nNo valid CUDA installation found: using OpenVINO backend with CPU.")
    return backend, device
