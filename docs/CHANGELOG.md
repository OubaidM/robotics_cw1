# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive error analysis documentation
- Dataset card with detailed statistics
- Macro-F1 scoring to model evaluation
- Per-class performance metrics

## [1.0.0] - 2025-10-28

### Added
- **Phase 3 Training**: Micro fine-tuning specifically for mug class detection
- **Phase 2 Training**: Model fine-tuning with optimized weights
- **Model Evaluation**: Comprehensive test set evaluation with mAP50, mAP50-95, precision, recall, and F1 scores
- **Documentation**: Complete project documentation including dataset card and error analysis
- **Git Configuration**: Proper .gitignore file to exclude training artifacts and large files

### Changed
- **Project Structure**: Reorganized repository structure with proper documentation
- **Model Management**: Removed hotdesk_training directory from version control while preserving local training artifacts
- **README**: Updated with installation instructions, usage examples, and troubleshooting guide

### Fixed
- **Environment Setup**: Added CPU fallback options for broader compatibility
- **Data Configuration**: Corrected data.yaml paths and dataset references

## [0.1.0] - 2025-10-28

### Added
- **Initial Model Training**: Phase 1 training with heavy data augmentation
- **Jupyter Notebook**: Complete training and evaluation pipeline
- **Environment Verification**: ROCm/GPU compatibility checks
- **Model Initialization**: YOLO11m model setup and configuration
- **Data Configuration**: data.yaml for dataset management

### Changed
- **README**: Revised with project overview, environment setup, and model performance metrics

## [Initial Commit] - 2025-10-28

### Added
- Project foundation and repository structure
- Initial commit with basic project setup