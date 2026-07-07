"""Optico — Multi-Frame Super-Resolution Engine.

Pure optical data, authentic reconstruction. No generative AI.

Modules
-------
alignment : Frame registration and dither quality estimation
masking : Dynamic foreground motion detection
preflight : Scale bounding based on alignment quality
drizzle : Variable-Pixel Linear Reconstruction stacking
deconvolution : Adaptive Wiener deconvolution
pipeline : End-to-end processing pipeline
constants : Configuration and named constants
"""

__version__ = "0.2.0"
__author__ = "chienhaoc"

from .constants import OpticoConfig
from .pipeline import run_pipeline

__all__ = ["OpticoConfig", "run_pipeline", "__version__"]
