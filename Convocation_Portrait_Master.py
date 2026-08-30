"""
+---------------------------------------------------------------------------------+
|                               HOW TO RUN THE PIPELINE                           |
|                                                                                 |
| 1. Input Photos: Place your uncompressed event folder (e.g., Cminds) inside     |
|    the project directory:                                                       |
|    D:\Imageria\[Your_Event_Folder]\                                             |
|                                                                                 |
| 2. Run the Script: Open PowerShell, navigate to D:\Imageria, and run:           |
|    python D:\Imageria\Convocation_Portrait_Master.py "D:\Imageria\[Your_Event_Folder]" |
|                                                                                 |
| 3. Output Location: The pipeline automatically generates isolated subfolders    |
|    directly inside your event folder:                                           |
|    D:\Imageria\[Your_Event_Folder]\Stage2_Universal_Cohorts\                    |
|    ├── 01_Students\                (Folders Student_001 to Student_XXX)         |
|    ├── 02_Faculty_and_VIPs\        (Folders Faculty_01 to Faculty_XX)           |
|    └── 03_Stage_Dais_and_Ceremony\ (Wide Groups, Dais, and Podium shots)        |
+---------------------------------------------------------------------------------+
"""
import os
import sys
import shutil
import cv2
import numpy as np
from insightface.app import FaceAnalysis

SIMILARITY_MATCH = 0.50

def normalize_exposure_buffer(im_bgr: np.ndarray) -> np.ndarray:
    """
    Checks frame luminance. If underexposed or shadowy,
    applies adaptive gamma + CLAHE to the inference buffer.
    """
    lab = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    mean_l = np.mean(l_channel)

    if mean_l < 95.0:
        gamma = max(1.2, min(2.2, 100.0 / (mean_l + 1e-6)))
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        l_boosted = cv2.LUT(l_channel, table)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_boosted = clahe.apply(l_boosted)

        boosted_lab = cv2.merge((l_boosted, a_channel, b_channel))
        return cv2.cvtColor(boosted_lab, cv2.COLOR_LAB2BGR)

    return im_bgr

def run_convocation_master(master_dir: str):
    out_dir = os.path.join(master_dir, "Stage2_Universal_Cohorts")
    stu_base = os.path.join(out_dir, "01_Students")
    fac_base = os.path.join(out_dir, "02_Faculty_and_VIPs")
    stage_base = os.path.join(out_dir, "03_Stage_Dais_and_Ceremony")

    for d in [stu_base, fac_base, stage_base]:
        os.makedirs(d, exist_ok=True)

    print("=" * 75)
    print(" CONVOCATION PORTRAIT MASTER (PRODUCTION UNIVERSAL PIPELINE)")
    print(f" Master Input : {master_dir}")
    print(f" Output Cohort: {out_dir}")
    print("=" * 75)

    app = FaceAnalysis(
        name='buffalo_l', 
        root=r'D:\Imageria',
        allowed_modules=['detection', 'recognition']
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))

    all_files = []
    for root, _, files in os.walk(master_dir):
        if "Stage" in root or "Output" in root: continue
        for f in sorted(files):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append(os.path.join(root, f))

    total_files = len(all_files)
    print(f"[*] Indexed {total_files} event images. Phase 1: Solo Portrait Enrollment...\n")

    solo_candidates = []
    multi_person_files = []

    for idx, fpath in enumerate(all_files):
        fname = os.path.basename(fpath)
        im = cv2.imread(fpath)
        if im is None: continue

        h_orig, w_orig = im.shape[:2]
        target_w = 1280
        scale = target_w / float(w_orig) if w_orig > target_w else 1.0
        im_det = cv2.resize(im, (target_w, int(h_orig * scale))) if scale != 1.0 else im

        im_det_norm = normalize_exposure_buffer(im_det)
        faces = app.get(im_det_norm)

        valid_faces = []
        if faces:
            for f in faces:
                bb = (f.bbox / scale).astype(int)
                bw = bb[2] - bb[0]
                bh = bb[3] - bb[1]
                if bw >= 35 and bh >= 35:
                    norm = np.linalg.norm(f.embedding)
                    emb = f.embedding / (norm + 1e-6) if norm > 0 else f.embedding
                    area_ratio = (bw * bh) / float(h_orig * w_orig)
                    valid_faces.append({"bbox": bb, "emb": emb, "area": area_ratio})

        if len(valid_faces) == 1 and valid_faces[0]["area"] >= 0.005:
            solo_candidates.append({
                "fpath": fpath,
                "fname": fname,
                "file_idx": idx,
                "emb": valid_faces[0]["emb"]
            })
        else:
            multi_person_files.append(fpath)

        sys.stdout.write(f"\r -> [{idx+1:03d}/{total_files}] Scanning: {fname} | Solo Portraits Found: {len(solo_candidates)}")
        sys.stdout.flush()

    print(f"\n\n[+] Found {len(solo_candidates)} solo portrait frames.")

    entities = []
    for cand in solo_candidates:
        matched = None
        best_sim = -1.0
        for ent in entities:
            sim = float(np.dot(cand["emb"], ent["centroid"]))
            if sim >= 0.54 and sim > best_sim:
                best_sim = sim
                matched = ent

        if matched:
            matched["items"].append(cand)
            embs = [it["emb"] for it in matched["items"]]
            avg_vec = np.mean(embs, axis=0)
            matched["centroid"] = avg_vec / (np.linalg.norm(avg_vec) + 1e-6)
        else:
            entities.append({
                "centroid": cand["emb"],
                "items": [cand],
                "first_idx": cand["file_idx"]
            })

    student_entities = []
    faculty_entities = []

    for ent in entities:
        # Universal metric: Faculty appear in early ceremony frames with few burst repeats
        if len(ent['items']) < 2 and ent['first_idx'] < (total_files * 0.40):
            faculty_entities.append(ent)
        else:
            student_entities.append(ent)

    student_entities.sort(key=lambda e: e["first_idx"])
    faculty_entities.sort(key=lambda e: e["first_idx"])

    print(f"\n[ENROLLMENT VERIFICATION]")
    print(f" -> Enrolled Student Cohorts : {len(student_entities)}")
    print(f" -> Enrolled Faculty / VIPs  : {len(faculty_entities)}\n")

    for idx, s in enumerate(student_entities):
        s_name = f"Student_{idx+1:03d}"
        s_dir = os.path.join(stu_base, s_name)
        os.makedirs(s_dir, exist_ok=True)
        s["name"] = s_name
        s["dir"] = s_dir
        for it in s["items"]:
            dest = os.path.join(s_dir, f"SOLO_{it['fname']}")
            if not os.path.exists(dest):
                shutil.copy2(it["fpath"], dest)

    for idx, f in enumerate(faculty_entities):
        f_name = f"Faculty_{idx+1:02d}"
        f_dir = os.path.join(fac_base, f_name)
        os.makedirs(f_dir, exist_ok=True)
        f["name"] = f_name
        f["dir"] = f_dir
        for it in f["items"]:
            dest = os.path.join(f_dir, f"SOLO_{it['fname']}")
            if not os.path.exists(dest):
                shutil.copy2(it["fpath"], dest)

    print("[*] Phase 2: Fan-out matching with Adaptive Shadow Lift...\n")
    handshakes = 0
    groups_fanned = 0
    ceremony_shots = 0

    for idx, fpath in enumerate(multi_person_files):
        fname = os.path.basename(fpath)
        im = cv2.imread(fpath)
        if im is None: continue

        h_orig, w_orig = im.shape[:2]
        target_w = 1280
        scale = target_w / float(w_orig) if w_orig > target_w else 1.0
        im_det = cv2.resize(im, (target_w, int(h_orig * scale))) if scale != 1.0 else im

        im_det_norm = normalize_exposure_buffer(im_det)
        faces = app.get(im_det_norm)
        if not faces: continue

        frame_students = {}
        frame_faculty = {}

        for face in faces:
            norm = np.linalg.norm(face.embedding)
            emb = face.embedding / (norm + 1e-6) if norm > 0 else face.embedding

            for s in student_entities:
                if float(np.dot(emb, s["centroid"])) >= SIMILARITY_MATCH:
                    frame_students[s["name"]] = s
                    break

            for f in faculty_entities:
                if float(np.dot(emb, f["centroid"])) >= SIMILARITY_MATCH:
                    frame_faculty[f["name"]] = f
                    break

        num_faces = len(faces)
        num_students_found = len(frame_students)
        num_faculty_found = len(frame_faculty)

        if num_students_found == 1 and (num_faculty_found >= 1 or num_faces <= 3):
            s = list(frame_students.values())[0]
            dest = os.path.join(s["dir"], f"STAGE_AWARD_{fname}")
            if not os.path.exists(dest):
                shutil.copy2(fpath, dest)
                handshakes += 1

        elif num_students_found >= 2:
            shutil.copy2(fpath, os.path.join(stage_base, f"GROUP_{fname}"))
            ceremony_shots += 1
            for s in frame_students.values():
                dest = os.path.join(s["dir"], f"GROUP_{fname}")
                if not os.path.exists(dest):
                    shutil.copy2(fpath, dest)
                    groups_fanned += 1

        elif num_faces >= 5 or num_faculty_found >= 1:
            shutil.copy2(fpath, os.path.join(stage_base, f"CEREMONY_{fname}"))
            ceremony_shots += 1

        percent = ((idx + 1) / len(multi_person_files)) * 100.0
        sys.stdout.write(f"\r -> Routing [{idx+1:03d}/{len(multi_person_files)}] ({percent:5.1f}%): Handshakes: {handshakes} | Group Copies: {groups_fanned}")
        sys.stdout.flush()

    print("\n\n" + "=" * 75)
    print(" COHORT SORTING & FAN-OUT COMPLETE")
    print(f" 1. Student Cohorts Enrolled     : {len(student_entities)}")
    print(f" 2. Faculty Cohorts Enrolled     : {len(faculty_entities)}")
    print(f" 3. Stage Awards Routed          : {handshakes}")
    print(f" 4. Multi-Student Fan-Out Copies : {groups_fanned}")
    print(f" 5. General Stage & Dais Shots   : {ceremony_shots}")
    print(f" Master Output Location          : {out_dir}")
    print("=" * 75)

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else r"D:\Imageria\Cminds"
    run_convocation_master(src)
