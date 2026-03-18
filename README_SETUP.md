# Swasthi Health Locker - Setup Instructions

## Running the App

### Option 1: Using the run script (Recommended)
```bash
cd /home/pratyush/Desktop/hackathron
./run_app.sh
```

### Option 2: Manual activation
```bash
# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swasthi

# Navigate to project directory
cd /home/pratyush/Desktop/hackathron

# Run Streamlit app
streamlit run streamlit_app.py
```

## API Key Setup

The Groq API key is already configured in `.streamlit/secrets.toml`.

**Important**: After modifying secrets.toml, you MUST restart Streamlit for changes to take effect.

## Dependencies

All dependencies are installed in the `swasthi` conda environment. To install/update:

```bash
conda activate swasthi
pip install -r requirements.txt
```

## OCR Setup (Required for Image Processing)

This app uses Tesseract OCR for processing scanned documents and images. You must install Tesseract separately:

### Windows:
```powershell
winget install --id UB-Mannheim.TesseractOCR -e
```

**Note**: The app is configured to automatically find Tesseract if installed in the default location (`C:\Program Files\Tesseract-OCR\tesseract.exe`).

### Linux:
```bash
sudo apt-get install tesseract-ocr
```

### macOS:
```bash
brew install tesseract
```

## Features

- ✅ Free AI-powered disease information (using Groq)
- ✅ Enhanced Summarized Reports with detailed disease info
- ✅ Overall Summary tracking lifetime disease history
- ✅ PDF and Image processing
- ✅ OCR support for scanned documents

