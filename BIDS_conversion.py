import os
import pydicom
import subprocess
from collections import defaultdict
from tabulate import tabulate
import tqdm
import numpy as np
import signal
from contextlib import contextmanager
import shutil

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out reading DICOM")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def safe_dcmread(path, timeout=10):
    try:
        with time_limit(timeout):
            return pydicom.dcmread(path, stop_before_pixels=True)
    except Exception:
        return None

def map_to_bids_modality(series_desc, study_desc, sequence_name, dcm):
    series_desc_lower = series_desc.lower()
    seq_lower = sequence_name.lower()
    
    # Map series/study descriptions to BIDS modality
    if ("t1" in series_desc_lower or "bravo" in series_desc_lower or "spgr" in series_desc_lower):
        if "post" in series_desc_lower:
            return "anat", "T1post"
        else:
            return "anat", "T1pre"
    elif "t2" in series_desc_lower and "flair" not in series_desc_lower or 't2' in study_desc and 'flair' not in study_desc:
        return "anat", "T2"
    elif "flair" in series_desc_lower or 'flair' in study_desc:
        return "anat", "FLAIR"
    elif "dwi" in series_desc_lower or "diff" in seq_lower:
        return "dwi", "dwi"
    elif "perf" in series_desc_lower or "epfid" in seq_lower:
        return "perf", "perf"
    elif "fmri" in series_desc_lower or "bold" in seq_lower:
        return "func", "bold"
    else:
        return "misc", "unknown"

def group_dicoms_by_series(dicom_root):
    series_dict = defaultdict(list)
    for root, _, files in os.walk(dicom_root):
        for file in tqdm.tqdm(files):
            dicom_path = os.path.join(root, file)
            if file.startswith('.'):
                continue
            try:
                # dcm = pydicom.dcmread(dicom_path, stop_before_pixels=True)
                dcm = safe_dcmread(dicom_path)
                
                if dcm is None or not hasattr(dcm, "ImageOrientationPatient"):
                    continue
                
                desc = getattr(dcm, "SeriesDescription", "").lower()
                
                if getattr(dcm, "Modality", "") != "MR":
                    continue
                
                # Skip irrelevant series (screenshots, scouts, localizers)
                if any(k in desc for k in ["screenshot", "mpr", "scout", "localizer", "survey", "pilot", "loc"]):
                    continue
                
                # Skip derived or processed images
                image_type = [str(x).upper() for x in getattr(dcm, "ImageType", [])]
                if any(tag in image_type for tag in ["DERIVED", "SECONDARY", "PROCESSED"]):
                    continue
                
                key = (
                    getattr(dcm, "StudyInstanceUID", "Unknown"),
                    getattr(dcm, "SeriesNumber", "0"),
                    getattr(dcm, "SeriesDescription", "").strip().lower(),
                    # tuple(round(float(x), 3) for x in getattr(dcm, "ImageOrientationPatient", [0,0,0,0,0,0])),
                    getattr(dcm, "SliceThickness", None),
                )
                series_dict[key].append(dicom_path)
                
                # series_uid = getattr(dcm, "SeriesInstanceUID", None)
                # if series_uid:
                #     series_dict[series_uid].append(dicom_path)
            except:
                continue
    return series_dict

def sort_and_check_slices(dicom_files):
    """
    Sort DICOM slices based on ImageOrientationPatient and ImagePositionPatient.
    Also check for missing slices using slice spacing consistency.
    """
    # Read the first DICOM file to get orientation
    dcm0 = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)
    orientation = np.array(dcm0.ImageOrientationPatient, dtype=float)
    row_cos = orientation[:3]
    col_cos = orientation[3:]
    slice_cos = np.cross(row_cos, col_cos) # Slice direction vector

    # Compute slice position along slice direction
    slice_positions = []
    for file in dicom_files:
        dcm = pydicom.dcmread(file, stop_before_pixels=True)
        ipp = np.array(dcm.ImagePositionPatient, dtype=float)
        loc = np.dot(ipp, slice_cos)
        slice_positions.append((loc, file))
        
    # Sort slices by their position
    slice_positions.sort(key=lambda x: x[0])
    sorted_files = [f for _, f in slice_positions]
    
    # Estimate slice spacing
    spacings = np.diff([pos for pos, _ in slice_positions])
    if len(spacings) == 0:
        return sorted_files, False
    median_spacing = np.median(spacings) 
    
    # Detect missing slices if spacing exceeds ±20% of the median spacing
    missing = np.any(spacings > median_spacing * 1.2) if len(spacings) > 0 else False
    return sorted_files, missing

# ----------------- Visualize Z positions -----------------
def visualize_z_positions(dicom_files, small_thresh=0.001, large_thresh=0.5):
    """
    Print Z coordinates and flag duplicates or deviations:
    ↓ = duplicate
    ↘ = small deviation
    ↗ = large deviation
    """
    z_list = []
    info_list = []
    for f in dicom_files:
        try:
            dcm = pydicom.dcmread(f, stop_before_pixels=True)
            z = float(dcm.ImagePositionPatient[2])
            inst = int(getattr(dcm, "InstanceNumber", 0))
            z_list.append(z)
            info_list.append((f, inst, z))
        except:
            continue

    if not z_list:
        print("No valid slices to visualize")
        return

    print("Instance | Z(mm)   | Flag")
    prev_z = None
    for f, inst, z in info_list:
        flag = ""
        if prev_z is not None:
            diff = abs(z - prev_z)
            if diff < small_thresh:
                flag = "↓ duplicate"
            elif diff < large_thresh:
                flag = "↘ small Δ"
            else:
                flag = "↗ large Δ"
        print(f"{inst:6} | {z:7.3f} | {flag}")
        prev_z = z
        
def remove_duplicate_z(dicom_files):
    seen_z = set()
    filtered_files = []
    for f in dicom_files:
        try:
            dcm = pydicom.dcmread(f, stop_before_pixels=True)
            z = round(float(dcm.ImagePositionPatient[2]), 3)
            if z not in seen_z:
                filtered_files.append(f)
                seen_z.add(z)
        except Exception:
            continue
    return filtered_files

def diagnose_series(dicom_files):
    slices = []
    for f in dicom_files:
        dcm = pydicom.dcmread(f, stop_before_pixels=True)
        inst = int(getattr(dcm, "InstanceNumber", 0))
        z = float(dcm.ImagePositionPatient[2])
        orientation = tuple(round(x, 6) for x in dcm.ImageOrientationPatient)
        slices.append((f, inst, z, orientation))

    slices.sort(key=lambda x: x[2])
    duplicates_z = []
    non_monotonic_inst = []
    orientation_set = set()
    spacings = []

    prev_inst = None
    prev_z = None
    for f, inst, z, ori in slices:
        if prev_z is not None:
            if abs(z - prev_z) < 1e-3:
                duplicates_z.append(f)
            spacings.append(z - prev_z)
        if prev_inst is not None and inst < prev_inst:
            non_monotonic_inst.append((f, inst, prev_inst))
        prev_inst = inst
        prev_z = z
        orientation_set.add(ori)

    median_spacing = np.median(spacings) if spacings else 0
    spacing_issue = any(abs(s - median_spacing) > 0.01 for s in spacings)
    orientation_issue = len(orientation_set) > 1

    issues = []
    if duplicates_z:
        issues.append(f"Duplicate Z positions ({len(duplicates_z)} slices) – ignored")
    if non_monotonic_inst:
        issues.append(f"Non-monotonic InstanceNumbers ({len(non_monotonic_inst)} slices)")
    if spacing_issue:
        issues.append(f"Slice spacing inconsistent (median={median_spacing:.4f})")
    if orientation_issue:
        issues.append(f"Orientation mismatch ({len(orientation_set)} orientations)")
    if not issues:
        issues.append("No obvious issue detected")

    return "; ".join(issues)

# ----------------- Convert to BIDS -----------------
def convert_to_bids(series_dict, bids_root):
    """
    Convert grouped DICOM series into NIfTI files following the BIDS structure.
    """
    report = []
    success_list = []
    timeout_list = []
    failed_list = []

    os.makedirs(bids_root, exist_ok=True)
    temp_root = os.path.join(bids_root, "_tmp_series")
    os.makedirs(temp_root, exist_ok=True)

    for idx, (series_uid, dicom_files) in tqdm.tqdm(enumerate(series_dict.items(), 1)):
        # Sort slices and check for missing slices
        sorted_files, missing = sort_and_check_slices(dicom_files)
        if len(sorted_files) == 0:
            print(f"[EMPTY SERIES] {series_uid} has no valid DICOM files, skipping.")
            continue
        # if missing:
        #     continue

        # Remove duplicate Z slices
        sorted_files = remove_duplicate_z(sorted_files)

        # dcm = pydicom.dcmread(sorted_files[0], stop_before_pixels=True)
        dcm = safe_dcmread(sorted_files[0])
        patient_id = getattr(dcm, "PatientID", "Unknown")
        study_date = getattr(dcm, "StudyDate", "Unknown")
        series_number = getattr(dcm, "SeriesNumber", "0")
        series_desc = getattr(dcm, "SeriesDescription", "Unknown").replace(" ", "_")
        study_desc = getattr(dcm, "StudyDescription", "Unknown").replace(" ", "_")
        sequence_name = getattr(dcm, "SequenceName", "Unknown").replace(" ", "_")
        thick = getattr(dcm, "SliceThickness", None)

        # Map to BIDS folder and suffix
        bids_folder, bids_suffix = map_to_bids_modality(series_desc, study_desc, sequence_name, dcm)
        if bids_suffix == 'unknown':
            continue

        # Use StudyDate as session ID
        session_id = study_date if study_date != "Unknown" else "ses-1"
        subject_id = f"sub-{patient_id}"
        session_dir = os.path.join(bids_root, subject_id, f"ses-{session_id}", bids_folder)
        os.makedirs(session_dir, exist_ok=True)

        # Define NIfTI filename
        nifti_name = f"run-{series_number}_thick_{thick}_{bids_suffix}"
        
        series_uid_tmp = getattr(dcm, "SeriesInstanceUID", None)
        series_temp_dir = os.path.join(temp_root, series_uid_tmp)
        os.makedirs(series_temp_dir, exist_ok=True)
        
        for f in sorted_files:
            try:
                shutil.copy(f, series_temp_dir)
            except Exception as e:
                print(f"[COPY ERROR] Cannot copy {f}: {e}")


        cmd = [
            "dcm2niix",
            "-z", "y",
            "-m", "y",
            "-f", nifti_name,
            "-o", session_dir,
            series_temp_dir
        ]

        try:
            # subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            status = "Success"
            success_list.append(series_uid_tmp)
            print(f"[SUCCESS] dcm2niix on: {series_desc}")
        # except subprocess.CalledProcessError as e:
        #     status = f"Failed: {e.stderr.strip()}"
        #     print(f"[ERROR] Series {series_desc} → {status}")

        except subprocess.TimeoutExpired:
            status = "Timeout"
            timeout_list.append(series_uid_tmp)
            print(f"[TIMEOUT] dcm2niix stalled on: {series_desc}")
            shutil.rmtree(session_dir, ignore_errors=True)

        except subprocess.CalledProcessError as e:
            diagnostic_msg = diagnose_series(sorted_files)
            status = f"Failed: {e.stderr.strip()} | Diagnosis: {diagnostic_msg}"
            failed_list.append(f"{series_uid_tmp}: {diagnostic_msg}")
            print(f"[FAILED] dcm2niix on: {series_desc} → {diagnostic_msg}")
            # shutil.rmtree(session_dir, ignore_errors=True)
            
            for f in sorted_files:
                try:
                    shutil.copy(f, session_dir)
                except Exception as e:
                    print(f"[COPY ERROR] Cannot copy {f}: {e}")

        # Save report entry
        report.append([
            patient_id,
            study_date,
            series_number,
            series_desc,
            sequence_name,
            len(sorted_files),
            session_dir,
            status
        ])

    # ----------------- Write conversion report -----------------
    headers = ["PatientID", "StudyDate", "Series#", "SeriesDescription",
               "SequenceName", "DICOM Files", "BIDS Path", "Status"]
    table_str = tabulate(report, headers=headers, tablefmt="grid")
    print(table_str)
    report_path = os.path.join(bids_root, "bids_conversion_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(table_str)
    print(f"\nBIDS Conversion completed: {report_path}")


    # Optionally save lists to files
    with open(os.path.join(bids_root, "timeout_list.txt"), "w") as f:
        f.write("\n".join(timeout_list))
    with open(os.path.join(bids_root, "success_list.txt"), "w") as f:
        f.write("\n".join(success_list))
    with open(os.path.join(bids_root, "failed_list.txt"), "w") as f:
        f.write("\n".join(failed_list))

# ----------------- Main -----------------
if __name__ == "__main__":
    dicom_root = # insert path
    output_root = # insert path
    series_dict = group_dicoms_by_series(dicom_root)
    convert_to_bids(series_dict, output_root)
    
    for key, files in series_dict.items():
        print(f"\nSeries: {key[2]} (SeriesNumber {key[1]})")
        sorted_files, missing = sort_and_check_slices(files)
        visualize_z_positions(sorted_files)
