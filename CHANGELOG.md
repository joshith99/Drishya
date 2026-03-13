# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
- [2026-03-13] Cleaned repository structure (moved original scripts to `archive/`)
- [2026-03-13] Added `testing_videos/` directory to store high-quality and degraded video samples
- [2026-03-13] Added `nrvqa_roadmap.md` detailing future plans including AI and Temporal integration
- [2026-03-13] Added `generate_test_videos.py` to create synthetic noise, blur, and compression datasets
- [2026-03-13] Added `brisque.py` and `compare_brisque.py` for executing perception-based NR-VQA

## [1.0.0] - 2026-02-09

### Added
- Initial project setup with three core tools
- [2026-02-09] Created `sample_frames.py` - Frame extraction utility with configurable sampling rates
- [2026-02-09] Created `nr_features.py` - No-Reference quality metrics calculator
  - Laplacian Variance for blurriness detection
  - Blockiness detection (8×8 and 16×16 blocks) for compression artifact analysis
  - Wavelet-based noise estimation using MAD method
  - CSV export functionality for analysis results
- [2026-02-09] Created `compare_videos.py` - Side-by-side video quality comparison tool
  - Multi-panel visualization (6 subplots)
  - Temporal quality graphs for all metrics
  - Statistical summary with improvement percentages
  - Normalized quality radar chart
- [2026-02-09] Added `requirements.txt` with all necessary dependencies
  - opencv-python >= 4.8.0 for video processing
  - numpy >= 1.24.0 for numerical operations
  - PyWavelets >= 1.4.1 for wavelet-based noise estimation
  - matplotlib >= 3.7.0 for visualization
- [2026-02-09] Created CHANGELOG.md for version tracking
- [2026-02-09] Created comprehensive README.md with:
  - Detailed explanation of each quality metric
  - Complete usage examples for all three tools
  - Practical interpretation guidelines for metric values
  - Educational context and technical references
- [2026-02-09] Added `.gitignore` file to exclude virtual environments, cache files, and temporary outputs from version control

### Changed
- [2026-02-09] **Major README rewrite** - Aligned documentation with actual implementation
  - Removed references to unimplemented deep learning features
  - Focused on classical computer vision methods currently implemented
  - Added practical usage examples and interpretation guidelines
  - Included mathematical formulas for transparency
  - Added limitations and future enhancement sections
  - Restructured for clarity and educational value

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

---

## How to Use This Changelog

### Categories
- **Added**: New features or functionality
- **Changed**: Changes to existing functionality
- **Deprecated**: Features that will be removed in future versions
- **Removed**: Features that have been removed
- **Fixed**: Bug fixes
- **Security**: Security-related changes

### Version Format
Use semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Incompatible API changes
- MINOR: Backwards-compatible functionality additions
- PATCH: Backwards-compatible bug fixes

### Example Entry Format
```markdown
## [1.0.0] - 2026-02-09

### Added
- Feature X that does Y
- New model architecture Z

### Fixed
- Bug in data loading causing crash
- Memory leak in inference pipeline
```
