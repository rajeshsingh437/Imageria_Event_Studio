# IMAGERIA EVENT STUDIO — MASTER SYSTEM ARCHITECTURE
**Engineered for Autonomous Bulk School & University Event Processing**
*Document Version: 2.0 (Full Photo, Video & Multi-Channel Content Edition)*

---

## 1. Project Mission & High-Volume Context
Imageria Event Studio processes bulk photography and videography for educational institutions:
- **Event Scales**: 2,000 to 3,000+ images per event.
- **Event Types**: 
  - Solo Portraits (Graduation, school yearbooks).
  - School Events (Annual Day, Sports Day, Cultural Festivals).
  - University Convocations (Stage degree awards, VIP/faculty presentations, group celebrations).
- **Deliverables**:
  1. Precision Student/Faculty Folders (Zero false matches, zero missed portraits).
  2. Print-ready high-resolution albums and 4-page memoir PDFs.
  3. Interactive, standalone **Rich HTML Digital Memoirs / Web Showcases**.
  4. Dynamic **Short-Form Social Media Videos** (Instagram Reels, Facebook, LinkedIn).
- **Core Principle**: **100% Free-Tier & Open Source**. No paid subscriptions; maximum production quality.

---

## 2. Environment & Python Runtime Confirmation
- **Python Version**: **Python 3.11.9 (Locked in `.venv`)**
  > [!IMPORTANT]
  > **Python 3.11.9 is the optimal sweet spot** for computer vision, AI, and video automation. Higher versions (Python 3.12+ / 3.14) frequently break pre-compiled binary wheels for InsightFace, MediaPipe, and ONNX. Python 3.11.9 delivers top CPython execution speed (~20% faster than 3.10) with complete library stability. **Do not upgrade Python.**

---

## 3. Recommended 100% Free & Open-Source Toolchain

| Domain | Tool / Library | Purpose & Strength |
| :--- | :--- | :--- |
| **RAW Decoding** | `rawpy` (LibRaw) | 16-bit linear sensor demosaicing; camera-native CR2/CR3/NEF/ARW support. |
| **Computer Vision** | `opencv-python`, `numpy` | Blazing-fast dynamic exposure, color constancy, and image processing. |
| **AI Face Detection** | `InsightFace` (`buffalo_l`) | RetinaFace detection + ArcFace 512-d feature extraction. |
| **Anatomy & Posture** | `MediaPipe` (FaceMesh + Pose) | 468 facial landmarks + 33 skeletal landmarks for golden-ratio cropping. |
| **Video Automation** | `FFmpeg` (System Native) | Industry standard for hardware-accelerated video assembly, audio sync, and 9:16 vertical scaling. *(Already installed on your machine!)* |
| **Video Composition** | `MoviePy` / `OpenCV VideoWriter` | Dynamic pan-zoom (Ken Burns), photo-to-reel montages, text overlays. |
| **Web & Rich HTML** | Pure Semantic HTML5 + Vanilla CSS3 + Canvas | Lightweight, beautiful digital flipbooks/galleries; standalone and self-contained (zero hosting cost). |
| **Print / PDF Engine**| `Pillow`, `pypdf`, `reportlab` | 300 DPI A4 print compilation and memoir book publishing. |

---

## 4. Multi-Stage Production Pipeline

```
[ Step 1: RAW Ingestion & Dynamic Enhancement ]
  │  • Corrupted file quarantine (Zero crashes)
  │  • Dynamic per-frame Exposure & Tone Mapping
  │  • Dynamic Dual-Zone White Balance (Stage LEDs & Skin)
  ▼
[ Step 2: Precision Cohort Sorting & Fan-Out ]
  │  • Dominant-Face Solo Portrait Enrollment
  │  • ArgMax Best-Match + Ambiguity Margin (No lookalike errors)
  │  • Anchor-Locked Face DNA (Zero centroid drift)
  │  • Multi-Person Fan-Out (Stage handshakes, group shots)
  ▼
[ Step 3: Dual-Anchor Anatomy-Based Cropping ]
  │  • MediaPipe FaceMesh + Pose skeletal tracking
  │  • Headroom, shoulder grid, waistline certificate inclusion
  │  • Horizon eye-line leveling (theta = 0 deg)
  ▼
[ Step 4: Multi-Channel Content Generation & Publishing ]
  ├── 4A: Print Memoir Engine (A4 300 DPI PDFs)
  ├── 4B: Rich HTML Digital Memoir (Interactive student showcase gallery)
  └── 4C: Social Media Reels Engine:
        • Instagram Reels & Facebook (9:16 vertical 1080x1920, beat-synced montages)
        • LinkedIn Showcase (1:1 square 1080x1080, professional branding & quotes)
```

---

## 5. Specifications for Rich Content & Social Video

### 5.1. Rich HTML Digital Showcase
- **Zero Server Dependency**: Single standalone interactive HTML bundle.
- **Features**:
  - Instant client-side search by student name or cohort folder.
  - Interactive photo card grid with smooth transitions and full-screen lightbox.
  - Optimized modern WebP compressed previews for instant loading.

### 5.2. Automated Short-Form Social Videos
- **Instagram Reels & Facebook Stories (9:16 Vertical — 1080x1920)**:
  - Automated Ken Burns motion (smooth slow zoom-in on student face and award).
  - Beat-synced photo transitions (15–30 seconds duration).
  - Lower-third title overlay with University/School branding and Student Name.
- **LinkedIn Highlights (1:1 Square or 16:9 Widescreen)**:
  - Professional pacing, convocation dais shots, dignitary handshake highlight, and event statistics overlay.

---

## 6. Governance & Non-Technical Development Rules
1. **Explain First, Code Second**: Every module and function is explained in plain studio terms before coding.
2. **User Go-Ahead Required**: No code generation without explicit user confirmation.
3. **File Naming Standard**:
   - Testing files: `*_test_gemini.py`
   - Production locked: `*_final_gemini.py`
4. **Permanent Memory**: `CHANGELOG.md` tracks all progress and acts as the kickoff prompt for future sessions.
