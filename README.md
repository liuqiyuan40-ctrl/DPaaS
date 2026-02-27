# DPaaS - Data Prefiltering as a Service

A flexible client-server framework for building and executing data prefiltering pipelines with configurable filters and modality transformations.

## Overview

DPaaS enables distributed data prefiltering by separating data validation and filtering logic between client and server. It supports modular pipeline construction through configurable filters that can process various data modalities (filepaths, numpy arrays, etc.).

## Features

- **Client-Server Architecture**: Filter data remotely with automatic serialization/deserialization
- **Configurable Pipelines**: Define filtering stages via JSON configuration
- **Modality System**: Seamlessly transform data between formats (files, numpy arrays, etc.)
- **Filter Registry**: Extensible filter system with built-in validators and samplers
- **Batch Processing**: Efficient handling of multiple files with progress tracking
- **Compression Support**: Automatic gzip compression for network transfers

## Installation

```bash
pip install -r requirements.txt  # If available
# Or install dependencies manually:
pip install flask requests numpy opencv-python
```

## Quick Start

```bash
# Download test dataset
cd test/scannetpp
bash download.sh
cd ..

# Terminal 1 - Start the server
python3 demo.server.py

# Terminal 2 - Run the client (in a new terminal)
cd test
python3 demo.client.py
```

The demo will process MP4 videos through a multi-stage filtering pipeline and save results to `test/demo_output/`:
- `local_pipeline.txt` - Local pipeline configuration details
- `report.txt` - Filtering results and reports for each file

## Architecture

### Core Components

- **Pipeline**: Sequential execution of filters with modality propagation
- **Filter**: Individual filtering unit with input/output modality
- **Modality**: Data representation format (filepath, numpy array, etc.)
- **Client**: Handles local pipeline + server communication
- **Server**: Manages remote pipelines and task routing

### Data Flow

1. Client performs local validation/prefiltering
2. Data is serialized and sent to server
3. Server applies remote filtering stages
4. Results are returned to client with detailed reports

## Built-in Filters

- **MP4MetaChecker**: Validate video metadata (fps, duration, resolution). We enforce strict spatial and temporal checks (e.g., resolution >= 720P, framerate > 20 fps).
- **FastCutDetector**: Detect and filter out videos with excessive rapid cuts or scene changes. It analyzes cut density over a sliding window (e.g., max 10 cuts per 30s) and overall cuts per minute (CPM).
- **MP4Sampler**: Sample frames from videos at specified intervals and resize them. For example, downsampling to 1 fps and resizing the maximum side to 720P before sending to the server to save bandwidth.
- **VLMViewpointDetector**: Remote filter that uses Vision-Language Models (e.g., Gemini, GPT-4V) to detect first-person vs third-person viewpoints. It analyzes sampled frames and returns a confidence score.
- **RandomFilter**: Probabilistic filter for testing
- *(Extend by creating custom filters with `@dpaas_filter` decorator)*

## Packaging for Suppliers

You can package the client into a standalone executable for suppliers who do not have a Python environment installed.

```bash
# Install PyInstaller
pip install pyinstaller

# Build the executable using the provided spec file
pyinstaller DPaaS_Client.spec
```

This will generate a `dist/DPaaS_Client.exe` file. 

**Distribution to Suppliers:**
1. Copy `dist/DPaaS_Client.exe` to a new folder.
2. Create an empty `data/` folder next to the `.exe`.
3. Send this folder to the supplier. They just need to place their `.mp4` videos in the `data/` folder and double-click the `.exe`. Results will be generated in a `demo_output/` folder.

## Project Structure

```
DPaaS/
├── dpaas/              # Core library
│   ├── client.py       # Client implementation
│   ├── server.py       # Flask server
│   ├── pipeline.py     # Pipeline orchestration
│   ├── filter.py       # Filter base class & registry
│   ├── modality.py     # Data modality definitions
│   └── utils.py        # Helper functions
└── test/               # Examples and tests
    ├── demo.server.py  # Server demo
    ├── demo.client.py  # Client demo
    ├── demo.local.json # Local pipeline config
    └── demo.remote.json # Remote pipeline config
```
