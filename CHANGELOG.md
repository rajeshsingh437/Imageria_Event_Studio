# IMAGERIA EVENT STUDIO — PROJECT SESSION LOG & CHANGELOG
*This file preserves the exact project status and acts as the kickoff prompt for all future sessions.*

---

## 📌 Current Project Status
- **Active Phase**: Step 1 Verification Testing (`step1_ingestion_test_gemini.py`)
- **Python Runtime**: **Python 3.11.9 (Verified in `.venv`)** — *Confirmed optimal sweet spot for AI/MediaPipe/OpenCV/InsightFace; do NOT upgrade.*
- **System Tools**: `ffmpeg` verified available on PATH.
- **Active Test Dataset**: `F:\HTS` (~3,027 images).
- **Current Active Test**: `step1_ingestion_test_gemini.py` configured for a 20-image comparative test.

---

## 🛡️ Non-Negotiable Collaboration Rules
1. **Explain First, Code Second**: Explain the real-world logic in plain, non-technical English before touching any code.
2. **User Go-Ahead Required**: No code generation or modification without explicit user confirmation.
3. **File Naming & Testing Contract**:
   - Work in progress: `*_test_gemini.py`
   - Production locked: `*_final_gemini.py`
   - We only edit the test file. Once verified on real data with 100% satisfaction, graduate to final.
4. **Accuracy Over Shortcuts**: Zero tolerance for missed portraits or misidentified individuals.
5. **Original & Edited Preservation**: Keep original `_ORIGINAL.[ext]` alongside dynamically enhanced `-ed.jpg` for instant QA and manual retouching access.

---

## 📋 Comprehensive Roadmap & Modules

| Step | Module Description | Target Output | Test File (`_test_gemini.py`) | Final File (`_final_gemini.py`) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Step 1** | RAW Ingestion, Corruption Quarantine & Dynamic Enhancement | Clean, color-balanced master images (`-ed.jpg`) + originals | `step1_ingestion_test_gemini.py` | `step1_ingestion_final_gemini.py` | ✅ **LOCKED (PRODUCTION)** |
| **Step 2** | Precision Face Clustering & Event-Profile Sorting | Event-specific cohort photo folders | `step2_cohort_sorting_test_gemini.py` | `step2_cohort_sorting_final_gemini.py` | ✅ **LOCKED (PRODUCTION)** |
| **Step 3** | Dual-Anchor Golden Ratio Anatomy Cropping | High-res 2160x3000 print portraits | `step3_anatomy_crop_test_gemini.py` | `step3_anatomy_crop_final_gemini.py` | ⏳ **Ready for Logic Review** |
| **Step 4A**| Print Memoir & Album Engine | 300 DPI 4-page print-ready A4 PDFs | `step4a_memoir_pdf_test_gemini.py` | `step4a_memoir_pdf_final_gemini.py` | ⏳ Planned |
| **Step 4B**| Rich HTML Interactive Digital Memoir | Standalone web gallery & showcase (zero-server) | `step4b_rich_html_test_gemini.py` | `step4b_rich_html_final_gemini.py` | ⏳ Planned |
| **Step 4C**| Social Media Video Reel Generator | 9:16 vertical (Insta/FB) + 1:1 square (LinkedIn) | `step4c_social_reels_test_gemini.py` | `step4c_social_reels_final_gemini.py` | ⏳ Planned |

---

## 📝 Session History

### [2026-09-21] Session 1: Setup, Baseline Audit, Step 1 Test Creation
- **Environment Verification**: Confirmed Python 3.11.9 and FFmpeg.
- **Master Files Created**: `PROJECT_ARCHITECTURE.md` and `CHANGELOG.md`.
- **Step 3 Anatomy Cropper Created**: [`step3_anatomy_crop_test_gemini.py`](file:///d:/Project-Python/Imageria_Event_Studio/step3_anatomy_crop_test_gemini.py) featuring:
  - **Eye-Line Horizon Leveling**: Detects camera tilt angle ($\theta$) and rotates canvas to level horizontal eye-lines.
  - **20% Headroom Buffer**: Protects hairstyles and caps from being clipped.
  - **Skeletal Shoulder Grid**: Positions shoulders on the golden 40% vertical line.
  - **Hand & Microphone Protection**: Extends bottom boundary to avoid amputating microphones or diploma folders.
  - **High-Res Resampling**: Resamples with Lanczos interpolation to **2160 x 3000 px @ 300 DPI**.
  - Fast test mode: Tests on the first 3 student/performer folders.

---

## 🚀 Prompt for Next Action
Run `step3_anatomy_crop_test_gemini.py` to generate 2160x3000 print portraits for the first 3 performer folders and inspect the framing!
