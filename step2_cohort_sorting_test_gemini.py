"""
IMAGERIA EVENT STUDIO — STEP 2: PRECISION FACE CLUSTERING & EVENT SORTING (TESTING)
File: step2_cohort_sorting_test_gemini.py
Role: Enrolls students with Anchor-Locked Face DNA, routes multi-student
      performances with ArgMax + Ambiguity Gap matching, and prevents
      all lookalike/drift errors.

Pair-Programming Rules:
- Original master photo saved as: [filename].[ext]
- Dynamically Enhanced photo saved as: [filename]-ed.jpg
- Deduplication: Keeps strictly 1 original + 1 -ed.jpg per shot.
"""

import os
import sys
import shutil
import json
import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

# ================= USER CONFIGURATION =================
DEFAULT_INPUT_DIR = r"F:\HTS"
DEFAULT_OUTPUT_DIR = r"F:\HTS\Step2_Sorted_Event_Cohorts"
EVENT_PROFILE = "ANNUAL_DAY"    # "ANNUAL_DAY", "CONVOCATION", "SPORTS_DAY"
MATCH_THRESHOLD = 0.58          # Minimum cosine similarity for positive identification
AMBIGUITY_GAP = 0.08            # Minimum gap required over second-closest candidate
CLUSTERING_THRESHOLD = 0.58     # Threshold for grouping initial anchor portraits
DEFAULT_LIMIT = 50              # Fast test limit (None = full run)
# ======================================================


def init_face_engine(model_root: str = r"D:\Imageria") -> FaceAnalysis:
    """Initializes buffalo_l RetinaFace and ArcFace models."""
    print("[*] Initializing InsightFace Engine (buffalo_l: RetinaFace + ArcFace)...")
    app = FaceAnalysis(
        name='buffalo_l',
        root=model_root,
        allowed_modules=['detection', 'recognition']
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def find_companion_files(fpath: str) -> list[str]:
    """
    Finds matching companion files, keeping strictly:
    1. The unedited master image (e.g. EQ9A0612.png)
    2. The dynamically enhanced version (e.g. EQ9A0612-ed.jpg)
    Avoids redundant intermediate '_ORIGINAL' duplicates.
    """
    parent = os.path.dirname(fpath)
    fname = os.path.basename(fpath)
    stem = Path(fname).stem
    
    # Isolate base photo ID (e.g. 'EQ9A0612')
    base_id = stem.replace("_ORIGINAL", "").replace("-ed", "")
    
    possible_folders = [parent]
    step1_dir = os.path.join(os.path.dirname(parent), "Step1_Ingestion_Preview")
    if os.path.exists(step1_dir):
        possible_folders.append(step1_dir)
    step1_enh = os.path.join(os.path.dirname(parent), "Step1_Ingestion_Enhanced")
    if os.path.exists(step1_enh):
        possible_folders.append(step1_enh)

    orig_master = None
    ed_version = None

    for p_dir in possible_folders:
        if not os.path.exists(p_dir):
            continue
        for f in os.listdir(p_dir):
            if f.startswith(base_id):
                f_full = os.path.join(p_dir, f)
                if "-ed" in f:
                    ed_version = f_full
                elif "_ORIGINAL" in f and orig_master is None:
                    orig_master = f_full
                elif not f.endswith("-ed.jpg") and "_ORIGINAL" not in f:
                    # Camera master raw/png takes highest priority
                    orig_master = f_full

    results = []
    if orig_master:
        results.append(orig_master)
    elif fpath not in results:
        results.append(fpath)

    if ed_version and ed_version not in results:
        results.append(ed_version)

    return results


def run_step2_cohort_sorting(input_dir: str, output_dir: str = DEFAULT_OUTPUT_DIR, limit: int | None = DEFAULT_LIMIT):
    stu_base = os.path.join(output_dir, "01_Performers_and_Students")
    fac_base = os.path.join(output_dir, "02_Teachers_Faculty_and_VIPs")
    stage_base = os.path.join(output_dir, "03_Stage_Performances_and_Group_Acts")

    for d in [stu_base, fac_base, stage_base]:
        os.makedirs(d, exist_ok=True)

    print("=" * 80)
    print(f" IMAGERIA EVENT STUDIO — STEP 2: PRECISION SORTING [{EVENT_PROFILE}]")
    print(f" Source Folder       : {input_dir}")
    print(f" Output Destination  : {output_dir}")
    print(f" Match Calibration   : Threshold={MATCH_THRESHOLD} | Ambiguity Gap={AMBIGUITY_GAP}")
    print(f" Test Scope          : {'First ' + str(limit) + ' images' if limit else 'Full Event Batch (All Images)'}")
    print("=" * 80)

    app = init_face_engine()

    # Discover candidate files (skipping preview/step folders to avoid duplicate scans)
    all_files = []
    for root, _, files in os.walk(input_dir):
        if "Step" in root or "Stage" in root or "_Corrupted" in root:
            continue
        for f in sorted(files):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                all_files.append(os.path.join(root, f))

    total_discovered = len(all_files)
    print(f"\n[*] Discovered {total_discovered} primary event images.")

    if limit and limit > 0:
        scan_files = all_files[:limit]
        print(f"[*] Quick-Test Mode: Processing first {len(scan_files)} images.\n")
    else:
        scan_files = all_files
        print(f"[*] Full Batch Mode: Processing all {len(scan_files)} images.\n")

    # =========================================================================
    # PHASE 1: SINGLE-PASS FEATURE EXTRACTION & IN-MEMORY CACHING
    # =========================================================================
    print("[*] Phase 1: High-Speed Feature Extraction & Anchor Discovery...")
    
    frame_cache = []
    solo_candidates = []

    for idx, fpath in enumerate(scan_files):
        fname = os.path.basename(fpath)
        im = cv2.imread(fpath)
        if im is None:
            continue

        h_orig, w_orig = im.shape[:2]
        target_w = 1280
        scale = target_w / float(w_orig) if w_orig > target_w else 1.0
        im_det = cv2.resize(im, (target_w, int(h_orig * scale))) if scale != 1.0 else im

        faces = app.get(im_det)
        valid_faces = []

        if faces:
            for f in faces:
                bb = (f.bbox / scale).astype(int)
                bw = max(0, bb[2] - bb[0])
                bh = max(0, bb[3] - bb[1])
                area = (bw * bh) / float(h_orig * w_orig)
                
                if bw >= 32 and bh >= 32 and area >= 0.003:
                    norm = np.linalg.norm(f.embedding)
                    emb = f.embedding / (norm + 1e-6) if norm > 0 else f.embedding
                    valid_faces.append({
                        "bbox": bb,
                        "emb": emb,
                        "area": area,
                        "det_score": float(f.det_score) if hasattr(f, 'det_score') else 1.0
                    })

        record = {
            "fpath": fpath,
            "fname": fname,
            "file_idx": idx,
            "valid_faces": valid_faces,
            "num_faces": len(valid_faces)
        }
        frame_cache.append(record)

        valid_faces.sort(key=lambda x: x["area"], reverse=True)
        is_solo = False
        if len(valid_faces) == 1 and valid_faces[0]["area"] >= 0.007:
            is_solo = True
        elif len(valid_faces) >= 2 and valid_faces[0]["area"] >= 0.020:
            if valid_faces[0]["area"] >= (valid_faces[1]["area"] * 3.5):
                is_solo = True

        if is_solo:
            solo_candidates.append({
                "fpath": fpath,
                "fname": fname,
                "file_idx": idx,
                "emb": valid_faces[0]["emb"],
                "area": valid_faces[0]["area"]
            })

        percent = ((idx + 1) / len(scan_files)) * 100.0
        sys.stdout.write(f"\r -> Scanning [{idx+1:03d}/{len(scan_files)}] ({percent:5.1f}%): Anchor Candidates Found: {len(solo_candidates)}")
        sys.stdout.flush()

    print(f"\n[+] Extracted {len(frame_cache)} frames into RAM. Found {len(solo_candidates)} anchor portrait candidates.")

    # =========================================================================
    # CLUSTERING: ANCHOR-LOCKED STUDENT ENROLLMENT
    # =========================================================================
    print("\n[*] Enrolling Students with Anchor-Locked Face DNA (Zero-Drift)...")
    
    student_entities = []

    for cand in solo_candidates:
        best_sim = -1.0
        best_ent = None

        for ent in student_entities:
            sim = float(np.dot(cand["emb"], ent["anchor_emb"]))
            if sim > best_sim:
                best_sim = sim
                best_ent = ent

        if best_ent and best_sim >= CLUSTERING_THRESHOLD:
            best_ent["items"].append(cand)
            if cand["area"] > best_ent["best_area"]:
                best_ent["anchor_emb"] = cand["emb"]
                best_ent["best_area"] = cand["area"]
        else:
            new_id = f"Performer_{len(student_entities)+1:03d}"
            student_entities.append({
                "name": new_id,
                "anchor_emb": cand["emb"],
                "best_area": cand["area"],
                "items": [cand],
                "first_idx": cand["file_idx"],
                "dir": os.path.join(stu_base, new_id)
            })

    print(f"[+] Successfully enrolled {len(student_entities)} unique student/performer cohorts.")

    for s in student_entities:
        os.makedirs(s["dir"], exist_ok=True)

    if student_entities:
        anchor_matrix = np.vstack([s["anchor_emb"] for s in student_entities])
    else:
        anchor_matrix = np.empty((0, 512))

    # =========================================================================
    # PHASE 2: INSTANT MEMORY-ROUTING (ARGMAX + AMBIGUITY GAP)
    # =========================================================================
    print(f"\n[*] Phase 2: Memory-Routing Multi-Kid Acts & Group Shots...")
    
    stats_solo_routed = 0
    stats_group_copies = 0
    stats_ceremony_shots = 0

    for idx, rec in enumerate(frame_cache):
        fpath = rec["fpath"]
        fname = rec["fname"]
        faces = rec["valid_faces"]
        num_faces = len(faces)

        companions = find_companion_files(fpath)
        matched_students = {}

        if len(student_entities) > 0 and num_faces > 0:
            for face in faces:
                emb = face["emb"]
                sims = np.dot(anchor_matrix, emb)
                
                top_indices = np.argsort(sims)[::-1]
                best_idx = top_indices[0]
                best_sim = float(sims[best_idx])
                second_sim = float(sims[top_indices[1]]) if len(top_indices) > 1 else -1.0

                if best_sim >= MATCH_THRESHOLD:
                    is_clear_winner = (second_sim < 0) or ((best_sim - second_sim) >= AMBIGUITY_GAP)
                    if is_clear_winner:
                        cand_student = student_entities[best_idx]
                        matched_students[cand_student["name"]] = cand_student

        num_matched = len(matched_students)

        # ROUTING LOGIC:
        if num_matched == 1 and (num_faces <= 3):
            # Solo or Featured Act: Routed directly to the performer's personal folder
            student = list(matched_students.values())[0]
            for comp in companions:
                dest = os.path.join(student["dir"], os.path.basename(comp))
                if not os.path.exists(dest):
                    shutil.copy2(comp, dest)
            stats_solo_routed += 1

        elif num_matched >= 2:
            # Multi-Kid Group Act (Dance team, choir, skit)
            for comp in companions:
                dest = os.path.join(stage_base, f"GROUP_{os.path.basename(comp)}")
                if not os.path.exists(dest):
                    shutil.copy2(comp, dest)
            stats_ceremony_shots += 1

            for student in matched_students.values():
                for comp in companions:
                    dest = os.path.join(student["dir"], f"GROUP_{os.path.basename(comp)}")
                    if not os.path.exists(dest):
                        shutil.copy2(comp, dest)
                stats_group_copies += 1

        elif num_faces >= 4 or rec["file_idx"] < 12:
            dest_dir = fac_base if rec["file_idx"] < 12 else stage_base
            for comp in companions:
                dest = os.path.join(dest_dir, os.path.basename(comp))
                if not os.path.exists(dest):
                    shutil.copy2(comp, dest)
            stats_ceremony_shots += 1

        percent = ((idx + 1) / len(frame_cache)) * 100.0
        sys.stdout.write(f"\r -> Routing [{idx+1:03d}/{len(frame_cache)}] ({percent:5.1f}%): Featured Solos: {stats_solo_routed} | Group Fan-Outs: {stats_group_copies}")
        sys.stdout.flush()

    manifest = {
        "event_profile": EVENT_PROFILE,
        "total_scanned": len(frame_cache),
        "enrolled_performers": len(student_entities),
        "featured_solos_routed": stats_solo_routed,
        "group_fan_out_copies": stats_group_copies,
        "stage_ensemble_shots": stats_ceremony_shots,
        "cohorts": [
            {
                "name": s["name"],
                "items_count": len(os.listdir(s["dir"])) if os.path.exists(s["dir"]) else 0,
                "folder": s["dir"]
            }
            for s in student_entities
        ]
    }

    manifest_path = os.path.join(output_dir, "sorting_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    print("\n\n" + "=" * 80)
    print(" STEP 2: PRECISION CLUSTERING & COHORT SORTING COMPLETE")
    print(f" Enrolled Performers      : {len(student_entities)}")
    print(f" Featured Solo Shots      : {stats_solo_routed}")
    print(f" Group Fan-Out Copies     : {stats_group_copies}")
    print(f" Stage & Ensemble Shots   : {stats_ceremony_shots}")
    print(f" Output Destination       : {output_dir}")
    print(f" Manifest Report          : {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    target_dir = DEFAULT_INPUT_DIR
    limit = DEFAULT_LIMIT

    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        target_dir = sys.argv[1]
    if "--all" in sys.argv:
        limit = None
    elif "--limit" in sys.argv:
        try:
            lim_idx = sys.argv.index("--limit") + 1
            limit = int(sys.argv[lim_idx])
        except (IndexError, ValueError):
            limit = DEFAULT_LIMIT

    run_step2_cohort_sorting(target_dir, limit=limit)
