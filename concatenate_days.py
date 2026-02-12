#!/usr/bin/env python3
"""
Concatenate multiple days of wavetracker output into a single dataset.

This script merges the raw wavetracker outputs (fund_v, idx_v, sign_v, times, ident_v)
from multiple recording days into a single continuous dataset for cleanup analysis.

Usage:
    python concatenate_days.py day1_folder day2_folder day3_folder --output merged_output
    python concatenate_days.py intermediate/2024-01-15 intermediate/2024-01-16 -o intermediate/merged
"""

import argparse
import os
from pathlib import Path

import numpy as np


def validate_day_folder(folder):
    """
    Check if a folder contains all required wavetracker output files.
    
    Parameters
    ----------
    folder : str or Path
        Path to the day's intermediate folder
        
    Returns
    -------
    bool
        True if all required files exist
    list
        List of missing files (empty if all present)
    """
    required_files = ["fund_v.npy", "idx_v.npy", "sign_v.npy", "times.npy", "ident_v.npy"]
    missing = []
    
    for fname in required_files:
        if not os.path.exists(os.path.join(folder, fname)):
            missing.append(fname)
    
    return len(missing) == 0, missing


def load_day_data(folder, verbose=True):
    """
    Load wavetracker output arrays from a single day folder.
    
    Parameters
    ----------
    folder : str or Path
        Path to the day's intermediate folder
    verbose : bool
        Print loading information
        
    Returns
    -------
    dict
        Dictionary with keys: fund_v, idx_v, sign_v, times, ident_v (if exists)
    """
    if verbose:
        print(f"Loading data from: {folder}")
    
    data = {}
    data["fund_v"] = np.load(os.path.join(folder, "fund_v.npy"))
    data["idx_v"] = np.load(os.path.join(folder, "idx_v.npy"))
    data["sign_v"] = np.load(os.path.join(folder, "sign_v.npy"))
    data["times"] = np.load(os.path.join(folder, "times.npy"))
    
    # Load ident_v (required - contains preliminary tracking from wavetracker)
    ident_path = os.path.join(folder, "ident_v.npy")
    if not os.path.exists(ident_path):
        raise FileNotFoundError(
            f"Missing ident_v.npy in {folder}. "
            "This file should be created by wavetracker's tracking stage. "
            "Make sure wavetracker completed successfully before concatenating."
        )
    data["ident_v"] = np.load(ident_path)
    
    if verbose:
        print(f"  Detections: {len(data['fund_v'])}")
        print(f"  Duration: {data['times'][-1] - data['times'][0]:.1f} seconds ({(data['times'][-1] - data['times'][0])/3600:.2f} hours)")
        print(f"  Time indices: {len(data['times'])}")
        print(f"  Channels: {data['sign_v'].shape[1] if len(data['sign_v']) > 0 else 0}")
    
    return data


def concatenate_wavetracker_outputs(day_folders, output_folder, gap_duration=0, verbose=True):
    """
    Merge multiple days of wavetracker output into single dataset.
    
    Parameters
    ----------
    day_folders : list of str/Path
        List of paths to day folders (e.g., ['intermediate/2024-01-15/', 'intermediate/2024-01-16/', ...])
    output_folder : str or Path
        Where to save concatenated results
    gap_duration : float, optional
        Time gap to insert between days in seconds (default: 0 = continuous)
    verbose : bool
        Print progress information
        
    Returns
    -------
    dict
        Statistics about the concatenation
    """
    # Validate all folders first
    print("\n" + "="*60)
    print("VALIDATING INPUT FOLDERS")
    print("="*60)
    
    for folder in day_folders:
        is_valid, missing = validate_day_folder(folder)
        if not is_valid:
            raise FileNotFoundError(
                f"Folder {folder} is missing required files: {missing}"
            )
        print(f"✓ {folder}")
    
    # Sort folders to ensure chronological order (assumes date-based naming)
    day_folders = sorted(day_folders)
    
    print("\n" + "="*60)
    print(f"LOADING {len(day_folders)} DAYS")
    print("="*60)
    
    # Initialize concatenation lists
    all_fund_v = []
    all_sign_v = []
    all_idx_v = []
    all_times = []
    all_ident_v = []
    
    # Track offsets for proper concatenation
    cumulative_time_offset = 0
    cumulative_idx_offset = 0
    max_ident_so_far = -1  # Track max ID to avoid collisions between days
    
    day_stats = []
    
    for i, day_folder in enumerate(day_folders):
        print(f"\n--- Day {i+1}/{len(day_folders)} ---")
        
        # Load day's data
        data = load_day_data(day_folder, verbose=verbose)
        
        # Store original stats
        day_info = {
            "folder": day_folder,
            "n_detections": len(data["fund_v"]),
            "duration_hours": (data["times"][-1] - data["times"][0]) / 3600,
            "time_offset_hours": cumulative_time_offset / 3600,
            "idx_offset": cumulative_idx_offset
        }
        day_stats.append(day_info)
        
        # Adjust times: shift to be continuous from previous day
        adjusted_times = data["times"] - data["times"][0] + cumulative_time_offset
        
        # Adjust idx_v: shift indices to account for previous days' time points
        adjusted_idx_v = data["idx_v"] + cumulative_idx_offset
        
        # Adjust ident_v: shift IDs to avoid collisions between days
        # Keep NaN as NaN, but increment valid IDs
        adjusted_ident_v = data["ident_v"].copy()
        valid_mask = ~np.isnan(adjusted_ident_v)
        if np.any(valid_mask):
            adjusted_ident_v[valid_mask] = adjusted_ident_v[valid_mask] + max_ident_so_far + 1
            max_ident_so_far = np.nanmax(adjusted_ident_v)
        
        # Append to concatenation lists
        all_fund_v.append(data["fund_v"])
        all_sign_v.append(data["sign_v"])
        all_idx_v.append(adjusted_idx_v)
        all_times.append(adjusted_times)
        all_ident_v.append(adjusted_ident_v)
        
        # Update offsets for next day
        day_duration = data["times"][-1] - data["times"][0]
        time_step = data["times"][1] - data["times"][0] if len(data["times"]) > 1 else 0
        
        cumulative_idx_offset += len(data["times"])
        cumulative_time_offset += day_duration + time_step + gap_duration
        
        if verbose:
            print(f"  Adjusted time range: {adjusted_times[0]:.1f} - {adjusted_times[-1]:.1f} s")
            print(f"  Adjusted idx range: {adjusted_idx_v.min()} - {adjusted_idx_v.max()}")
    
    # Concatenate all arrays
    print("\n" + "="*60)
    print("CONCATENATING DATA")
    print("="*60)
    
    merged_fund_v = np.concatenate(all_fund_v)
    merged_idx_v = np.concatenate(all_idx_v)
    merged_sign_v = np.concatenate(all_sign_v)
    merged_times = np.concatenate(all_times)
    merged_ident_v = np.concatenate(all_ident_v)
    
    # Verify indices are within bounds
    if merged_idx_v.max() >= len(merged_times):
        print(f"WARNING: Maximum idx_v ({merged_idx_v.max()}) exceeds times length ({len(merged_times)})")
        print("This may cause issues in cleanup. Check time alignment.")
    
    # Save concatenated data
    print(f"\nSaving to: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    
    np.save(os.path.join(output_folder, "fund_v.npy"), merged_fund_v)
    np.save(os.path.join(output_folder, "idx_v.npy"), merged_idx_v)
    np.save(os.path.join(output_folder, "sign_v.npy"), merged_sign_v)
    np.save(os.path.join(output_folder, "times.npy"), merged_times)
    np.save(os.path.join(output_folder, "ident_v.npy"), merged_ident_v)
    
    # Calculate statistics about merged tracks
    n_tracked = np.sum(~np.isnan(merged_ident_v))
    n_untracked = np.sum(np.isnan(merged_ident_v))
    unique_ids = len(np.unique(merged_ident_v[~np.isnan(merged_ident_v)])) if n_tracked > 0 else 0
    
    # Save metadata about concatenation
    metadata = {
        "n_days": len(day_folders),
        "day_folders": [str(f) for f in day_folders],
        "total_detections": len(merged_fund_v),
        "total_duration_hours": merged_times[-1] / 3600,
        "total_timepoints": len(merged_times),
        "gap_duration_seconds": gap_duration,
        "n_tracked_detections": int(n_tracked),
        "n_untracked_detections": int(n_untracked),
        "n_preliminary_tracks": int(unique_ids),
        "day_stats": day_stats
    }
    
    import json
    with open(os.path.join(output_folder, "concatenation_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("CONCATENATION SUMMARY")
    print("="*60)
    print(f"Days merged: {len(day_folders)}")
    print(f"Total detections: {len(merged_fund_v):,}")
    print(f"Total duration: {merged_times[-1]/3600:.2f} hours ({merged_times[-1]/3600/24:.2f} days)")
    print(f"Total timepoints: {len(merged_times):,}")
    print(f"Channels: {merged_sign_v.shape[1] if len(merged_sign_v) > 0 else 0}")
    print(f"Frequency range: {merged_fund_v.min():.1f} - {merged_fund_v.max():.1f} Hz")
    print(f"\nPreliminary tracking (from wavetracker):")
    print(f"  Tracked detections: {n_tracked:,} ({n_tracked/len(merged_fund_v)*100:.1f}%)")
    print(f"  Untracked detections: {n_untracked:,} ({n_untracked/len(merged_fund_v)*100:.1f}%)")
    print(f"  Preliminary track count: {unique_ids}")
    print(f"\nOutput folder: {output_folder}")
    print("\n⚠️  Note: Track IDs from different days have been offset to avoid collisions.")
    print("The cleanup script will refine and merge these preliminary tracks.")
    print("\nReady for cleanup!")
    
    return metadata


def main():
    """Command-line interface for day concatenation."""
    parser = argparse.ArgumentParser(
        description="Concatenate multiple days of wavetracker output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Concatenate 3 days
  python concatenate_days.py day1/ day2/ day3/ -o merged/
  
  # With verbose output
  python concatenate_days.py intermediate/2024-01-* -o intermediate/merged -v
  
  # Add 1-hour gap between days (for visualization)
  python concatenate_days.py day1/ day2/ --output merged/ --gap 3600
        """
    )
    
    parser.add_argument(
        "day_folders",
        nargs="+",
        type=Path,
        help="Paths to day folders containing wavetracker output (fund_v.npy, etc.)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output folder for concatenated data"
    )
    
    parser.add_argument(
        "--gap",
        type=float,
        default=0,
        help="Time gap between days in seconds (default: 0 = continuous)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed information"
    )
    
    args = parser.parse_args()
    
    # Convert Path objects to strings for compatibility
    day_folders = [str(f) for f in args.day_folders]
    output_folder = str(args.output)
    
    # Run concatenation
    try:
        metadata = concatenate_wavetracker_outputs(
            day_folders,
            output_folder,
            gap_duration=args.gap,
            verbose=args.verbose
        )
        
        print("\n✓ Concatenation successful!")
        print(f"\nNext step: Run cleanup on merged data:")
        print(f"  python clean_up.py {output_folder} -n <number_of_fish>")
        
    except Exception as e:
        print(f"\n✗ Error during concatenation: {e}")
        raise


if __name__ == "__main__":
    main()
