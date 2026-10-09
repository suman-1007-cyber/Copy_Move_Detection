# Copy-Move Image Forgery Detection — Traditional DIP

A Windows-friendly classroom demonstration using Python, OpenCV, NumPy, Streamlit, and DCT-based block matching. It does not use AI, machine learning, deep learning, or CNNs.

## Requirements
- Windows 10/11
- Python 3.10–3.12 recommended
- Internet for the initial package installation only

## Setup (PowerShell)
Open PowerShell in this folder and run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If `py -3` is not recognized, install Python from https://www.python.org/downloads/windows/ and check **Add python.exe to PATH** during installation. Reopen PowerShell after installation.

The app opens in your browser, normally at http://localhost:8501.

## Demo sequence
1. Click **Generate forged demo image**.
2. Compare the input image with the marked result.
3. Explain the DCT, block matching, filtering, and red overlay.
4. Optionally upload another image.
5. Mention that highlights are suspected matches, not definitive proof.

## Algorithm summary
1. Convert to grayscale.
2. Extract overlapping blocks.
3. Compute DCT coefficients for each block.
4. Build normalized, quantized descriptors.
5. Find repeated descriptors at spatially distant locations.
6. Prefer matches sharing a consistent translation vector.
7. Map matches into a mask and overlay them in red.

## Limitations
This is a demonstration prototype, not a forensic-grade tool. Repeated natural textures can produce false positives; rotation, scaling, compression, or heavy edits may reduce detection quality. Always explain these limitations honestly.
