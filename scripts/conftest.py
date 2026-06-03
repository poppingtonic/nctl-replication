"""Pytest configuration for the NCTL benchmark scripts.

Registers the ``cuda`` marker so CUDA-only tests can be skipped cleanly on
machines without a usable GPU (require_cuda_kernel handles the actual skip).
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "cuda: marks tests that require a working CUDA device and the "
        "pre-compiled fmn_mixture_cuda kernel (skipped automatically on "
        "CPU-only machines via require_cuda_kernel).",
    )
