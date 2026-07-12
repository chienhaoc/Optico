# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-07-12
### Added
- Focal-Length Driven dedicated PSF base scaling (17mm=0.35, 45mm=0.57, 50mm=0.63) to prevent over-sharpening on wide-angle small faces.
- LR Data-Side Pre-emphasis (Phase 8.0) using high-pass residual filter ($\alpha=0.55$) to restore JPEG high frequencies.
- MD5-based Drizzle Stacking Cache Registry to skip alignment and stacking for matched burst signatures.
- Support and default upgrade to the sharper `lanczos4` Drizzle accumulation kernel.
- `--psf-base` CLI parameter mapping to `OpticoConfig.psf_base`.

### Removed
- OpenCV YuNet face detection protection and non-linear Gamma Cap inside deconvolution to maintain linear gradient deconvolution stability.

## [1.0.0] - 2026-07-06
### Added
- Initial release of Optico.
- Multi-Frame Super Resolution (MFSR) backend core utilizing ECC alignment and True Drizzle stacking.
- Advanced Fourier Wiener Deconvolution for PSF correction.
- Vue 3 frontend with interactive steps, split-view comparison, and optical data dashboard.
- PyWebView desktop wrapper for native application feel.
- i18n support architecture.
