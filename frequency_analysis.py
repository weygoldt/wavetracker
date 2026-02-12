#!/usr/bin/env python3
"""
Extract and analyze fish EOD frequencies from wavetracker output.

This unified script:
1. Extracts top N frequencies at specific daily timepoints
2. Optionally connects frequencies across time to assign fish IDs
3. Generates temporal trace plots and statistics

Accounts for temperature-dependent frequency changes (Q10 ~ 1.5) by connecting
closest frequencies between consecutive timepoints.

Usage:
    # Extract only
    python frequency_analysis.py G:\data\wavetracker_bottom\2025-*\ --no-analyze
    
    # Extract and analyze (default)
    python frequency_analysis.py G:\data\wavetracker_bottom\2025-*\
"""

import argparse
import configparser
import glob
import re
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config(config_path=None, data_folder=None):
    """
    Load extraction and analysis parameters from config file.
    
    Search hierarchy:
    1. Explicitly provided config_path
    2. frequency_config.cfg in parent directory of data_folder
    3. frequency_config.cfg in data_folder
    4. frequency_config_default.cfg in script directory
    5. Hardcoded defaults
    
    Parameters
    ----------
    config_path : Path or None
        Explicit path to config file
    data_folder : Path or None
        Data folder to search for config
    
    Returns
    -------
    dict
        Configuration parameters for extraction and analysis
    """
    config = configparser.ConfigParser()
    config_loaded = False
    config_source = "hardcoded defaults"
    
    # Search hierarchy
    search_paths = []
    
    if config_path is not None:
        search_paths.append(Path(config_path))
    
    if data_folder is not None:
        data_path = Path(data_folder)
        parent_dir = data_path.parent
        search_paths.append(parent_dir / "frequency_config.cfg")
        search_paths.append(data_path / "frequency_config.cfg")
    
    script_dir = Path(__file__).parent
    search_paths.append(script_dir / "frequency_config_default.cfg")
    
    # Try to load config
    for path in search_paths:
        if path.exists():
            try:
                config.read(path)
                config_loaded = True
                config_source = str(path)
                break
            except Exception as e:
                print(f"Warning: Could not read config from {path}: {e}")
    
    # Parse parameters with defaults
    params = {}
    
    # Extraction parameters
    if config_loaded and 'extraction' in config:
        section = config['extraction']
        
        params['n_fish'] = section.getint('n_fish', fallback=3)
        
        window_str = section.get('window_duration', fallback='600')
        if window_str.endswith('min'):
            params['window_duration'] = float(window_str.rstrip('min')) * 60
        else:
            params['window_duration'] = float(window_str)
        
        params['method'] = section.get('method', fallback='occurrence')
        params['freq_tolerance'] = section.getfloat('freq_tolerance', fallback=2.0)
        
        # Bandpass filter for fundamental frequencies
        params['min_freq'] = section.getfloat('min_freq', fallback=None)
        params['max_freq'] = section.getfloat('max_freq', fallback=None)
        
        timepoints_str = section.get('timepoints', fallback='07:30,13:30,19:30,01:30')
        params['timepoints'] = []
        for tp in timepoints_str.split(','):
            tp = tp.strip()
            if ':' in tp:
                hour, minute = map(int, tp.split(':'))
                label = f"{hour:02d}:{minute:02d}"
                params['timepoints'].append((hour, minute, label))
    else:
        params['n_fish'] = 3
        params['window_duration'] = 600
        params['method'] = 'occurrence'
        params['freq_tolerance'] = 2.0
        params['min_freq'] = None
        params['max_freq'] = None
        params['timepoints'] = [
            (7, 30, "morning"),
            (13, 30, "noon"),
            (19, 30, "evening"),
            (1, 30, "night")
        ]
    
    # Analysis parameters
    if config_loaded and 'analysis' in config:
        section = config['analysis']
        params['run_analysis'] = section.getboolean('run_analysis', fallback=True)
        params['trace_tolerance'] = section.getfloat('trace_tolerance', fallback=10.0)
    else:
        params['run_analysis'] = True
        params['trace_tolerance'] = 10.0
    
    print(f"Configuration loaded from: {config_source}")
    
    return params


def get_parent_directory(folders):
    """Get the common parent directory from a list of folder paths."""
    if not folders:
        return Path('.')
    
    first_folder = Path(folders[0])
    return first_folder.parent


# ============================================================================
# Extraction Functions
# ============================================================================

def parse_recording_start(folder_name):
    """
    Parse recording start time from folder name.
    
    Expected format: YYYY-MM-DD_HH-MM
    Example: 2025-07-31_00-03 -> 2025-07-31 00:03:00
    """
    pattern = r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})'
    match = re.search(pattern, folder_name)
    
    if not match:
        raise ValueError(f"Could not parse datetime from folder name: {folder_name}")
    
    year, month, day, hour, minute = map(int, match.groups())
    return datetime(year, month, day, hour, minute)


def calculate_target_offset(recording_start, target_hour, target_minute):
    """
    Calculate time offset in seconds from recording start to target time.
    
    Handles day boundaries correctly (e.g., target at 01:30 after recording 
    started at 23:00 previous day).
    """
    target_time = recording_start.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    
    # If target time is before recording start, it's the next day
    if target_time < recording_start:
        target_time += timedelta(days=1)
    
    offset = (target_time - recording_start).total_seconds()
    
    # Check if offset is reasonable (within 48 hours)
    if offset > 48 * 3600:
        return None
    
    return offset


def extract_window_frequencies(fund_v, idx_v, sign_v, times, center_time, window_duration=600):
    """Extract frequencies from a time window around a target time."""
    half_window = window_duration / 2
    
    if center_time < times[0] or center_time > times[-1]:
        return None
    
    window_start = max(center_time - half_window, times[0])
    window_end = min(center_time + half_window, times[-1])
    
    window_mask = (times[idx_v] >= window_start) & (times[idx_v] <= window_end)
    
    if not np.any(window_mask):
        return None
    
    windowed_fund = fund_v[window_mask]
    windowed_sign = sign_v[window_mask]
    
    actual_start = times[idx_v[window_mask]][0]
    actual_end = times[idx_v[window_mask]][-1]
    actual_duration = actual_end - actual_start
    
    return {
        'frequencies': windowed_fund,
        'powers': windowed_sign,
        'n_detections': len(windowed_fund),
        'actual_duration': actual_duration,
        'window_start': actual_start,
        'window_end': actual_end
    }


def get_top_n_frequencies(frequencies, powers, n_fish, method='occurrence', freq_tolerance=2.0):
    """
    Get top N most prominent frequencies in a window.
    Merges signals within freq_tolerance and includes next ranking frequency.
    """
    if len(frequencies) == 0:
        return []
    
    # Bin frequencies
    bin_width = 2.5  # Hz
    bins = np.arange(np.min(frequencies) - bin_width, np.max(frequencies) + 2*bin_width, bin_width)
    bin_indices = np.digitize(frequencies, bins)
    
    # Calculate statistics per bin
    bin_stats = []
    for bin_idx in np.unique(bin_indices):
        mask = bin_indices == bin_idx
        
        bin_freqs = frequencies[mask]
        bin_powers = powers[mask]
        
        median_freq = np.median(bin_freqs)
        mean_freq = np.mean(bin_freqs)
        std_freq = np.std(bin_freqs)
        n_detections = len(bin_freqs)
        
        max_powers = np.max(bin_powers, axis=1)
        mean_power = np.mean(max_powers)
        
        if method == 'occurrence':
            score = n_detections
        elif method == 'power':
            score = mean_power * n_detections
        else:
            score = n_detections
        
        bin_stats.append({
            'median_freq': median_freq,
            'mean_freq': mean_freq,
            'std_freq': std_freq,
            'n_detections': n_detections,
            'mean_power_db': mean_power,
            'score': score
        })
    
    # Sort by score
    bin_stats.sort(key=lambda x: x['score'], reverse=True)
    
    # Select top N, merging duplicates within freq_tolerance
    selected = []
    candidate_idx = 0
    
    while len(selected) < n_fish and candidate_idx < len(bin_stats):
        candidate = bin_stats[candidate_idx]
        candidate_freq = candidate['median_freq']
        
        is_duplicate = False
        for existing in selected:
            if abs(candidate_freq - existing['median_freq']) < freq_tolerance:
                is_duplicate = True
                break
        
        if not is_duplicate:
            selected.append(candidate)
        
        candidate_idx += 1
    
    return selected


def process_day(folder, n_fish, timepoints, window_duration=600, method='occurrence', freq_tolerance=2.0, min_freq=None, max_freq=None):
    """Process one day of wavetracker output.
    
    Parameters
    ----------
    min_freq : float or None
        Minimum frequency (Hz) for bandpass filter. If None, no lower bound.
    max_freq : float or None
        Maximum frequency (Hz) for bandpass filter. If None, no upper bound.
    """
    print(f"\nProcessing: {folder}")
    
    try:
        recording_start = parse_recording_start(folder.name)
    except ValueError as e:
        print(f"  ⚠️  {e}")
        return []
    
    print(f"  Recording start: {recording_start}")
    
    # Check for recordings subfolder
    data_folder = folder / "recordings"
    if not data_folder.exists():
        data_folder = folder
    
    # Load wavetracker output
    try:
        fund_v = np.load(data_folder / "fund_v.npy")
        idx_v = np.load(data_folder / "idx_v.npy")
        sign_v = np.load(data_folder / "sign_v.npy")
        times = np.load(data_folder / "times.npy")
    except FileNotFoundError as e:
        print(f"  ⚠️  Missing file: {e}")
        print(f"     Looked in: {data_folder}")
        return []
    
    print(f"  Loaded {len(fund_v)} detections, {len(times)} time points ({times[-1]/3600:.1f} hours)")
    
    # Apply bandpass filter for fundamental frequencies
    freq_mask = np.ones(len(fund_v), dtype=bool)
    if min_freq is not None:
        freq_mask &= (fund_v >= min_freq)
    if max_freq is not None:
        freq_mask &= (fund_v <= max_freq)
    
    if not np.all(freq_mask):
        n_filtered = np.sum(~freq_mask)
        fund_v = fund_v[freq_mask]
        idx_v = idx_v[freq_mask]
        sign_v = sign_v[freq_mask]
        print(f"  Bandpass filter ({min_freq}-{max_freq} Hz): kept {len(fund_v)} detections, removed {n_filtered}")
    
    results = []
    
    for hour, minute, label in timepoints:
        offset = calculate_target_offset(recording_start, hour, minute)
        
        if offset is None:
            print(f"  ⊗ {label} ({hour:02d}:{minute:02d}): Not in recording")
            continue
        
        if offset < 0 or offset > times[-1]:
            print(f"  ⊗ {label} ({hour:02d}:{minute:02d}): Outside recording bounds")
            continue
        
        window_data = extract_window_frequencies(
            fund_v, idx_v, sign_v, times, offset, window_duration
        )
        
        if window_data is None:
            print(f"  ⊗ {label} ({hour:02d}:{minute:02d}): No data in window")
            continue
        
        top_freqs = get_top_n_frequencies(
            window_data['frequencies'],
            window_data['powers'],
            n_fish,
            method,
            freq_tolerance
        )
        
        print(f"  ✓ {label} ({hour:02d}:{minute:02d}): {len(top_freqs)} signals, {window_data['n_detections']} detections")
        
        for rank, freq_data in enumerate(top_freqs, start=1):
            results.append({
                'date': recording_start.date(),
                'recording_start': recording_start,
                'timepoint_label': label,
                'timepoint_hour': hour,
                'timepoint_minute': minute,
                'rank': rank,
                'median_freq_hz': freq_data['median_freq'],
                'mean_freq_hz': freq_data['mean_freq'],
                'std_freq_hz': freq_data['std_freq'],
                'n_detections': freq_data['n_detections'],
                'mean_power_db': freq_data['mean_power_db'],
                'window_duration_s': window_data['actual_duration']
            })
    
    return results


# ============================================================================
# Analysis Functions
# ============================================================================

def connect_frequencies_temporal(df, freq_tolerance=10.0):
    """
    Connect frequencies across consecutive timepoints based on proximity.
    Assigns consistent fish IDs by connecting closest frequencies.
    """
    df = df.copy()
    
    # Create datetime column
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + 
                                    df['timepoint_hour'].astype(str).str.zfill(2) + ':' +
                                    df['timepoint_minute'].astype(str).str.zfill(2))
    
    df = df.sort_values('datetime').reset_index(drop=True)
    df['fish_id'] = -1
    next_fish_id = 0
    
    timepoints = df['datetime'].unique()
    
    # First timepoint: assign new IDs
    first_tp_mask = df['datetime'] == timepoints[0]
    first_tp_indices = df[first_tp_mask].index
    for idx in first_tp_indices:
        df.loc[idx, 'fish_id'] = next_fish_id
        next_fish_id += 1
    
    # Connect subsequent timepoints
    for i in range(1, len(timepoints)):
        prev_tp = timepoints[i-1]
        curr_tp = timepoints[i]
        
        prev_data = df[df['datetime'] == prev_tp].copy()
        curr_data = df[df['datetime'] == curr_tp].copy()
        
        curr_assigned = set()
        
        for prev_idx, prev_row in prev_data.iterrows():
            prev_freq = prev_row['median_freq_hz']
            prev_fish_id = prev_row['fish_id']
            
            best_match_idx = None
            best_diff = float('inf')
            
            for curr_idx, curr_row in curr_data.iterrows():
                if curr_idx in curr_assigned:
                    continue
                
                curr_freq = curr_row['median_freq_hz']
                diff = abs(curr_freq - prev_freq)
                
                if diff < best_diff and diff < freq_tolerance:
                    best_match_idx = curr_idx
                    best_diff = diff
            
            if best_match_idx is not None:
                df.loc[best_match_idx, 'fish_id'] = prev_fish_id
                curr_assigned.add(best_match_idx)
        
        # Assign new IDs to unmatched frequencies
        for curr_idx, curr_row in curr_data.iterrows():
            if curr_idx not in curr_assigned:
                df.loc[curr_idx, 'fish_id'] = next_fish_id
                next_fish_id += 1
    
    return df


def plot_temporal_traces(df, output_path, trace_tolerance):
    """Plot all frequency traces over time with connections."""
    df = df.sort_values(['fish_id', 'datetime'])
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    n_fish = df['fish_id'].nunique()
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_fish, 20)))
    if n_fish > 20:
        colors = plt.cm.viridis(np.linspace(0, 1, n_fish))
    
    for fish_idx, fish_id in enumerate(sorted(df['fish_id'].unique())):
        fish_data = df[df['fish_id'] == fish_id].sort_values('datetime')
        
        if len(fish_data) < 2:
            ax.scatter(fish_data['datetime'], fish_data['median_freq_hz'],
                      color=colors[fish_idx % len(colors)], s=50, alpha=0.7, zorder=3)
        else:
            ax.plot(fish_data['datetime'], fish_data['median_freq_hz'],
                   marker='o', linewidth=1.5, markersize=5,
                   color=colors[fish_idx % len(colors)], alpha=0.7,
                   label=f'Fish {fish_id}' if fish_idx < 10 else None)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('EOD Frequency (Hz)', fontsize=12)
    ax.set_title(f'Temporal Frequency Traces (tolerance: {trace_tolerance} Hz)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    if n_fish <= 10:
        ax.legend(loc='best', fontsize=8)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract and analyze fish EOD frequencies from wavetracker output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract and analyze (default)
  python frequency_analysis.py G:\\data\\wavetracker_bottom\\2025-*\\
  
  # Extract only, no analysis
  python frequency_analysis.py G:\\data\\wavetracker_bottom\\2025-*\\ --no-analyze
  
  # With custom config
  python frequency_analysis.py G:\\data\\wavetracker_bottom\\2025-*\\ -c my_config.cfg
  
  # Override trace tolerance
  python frequency_analysis.py G:\\data\\wavetracker_bottom\\2025-*\\ --trace-tol 15
        """
    )
    
    parser.add_argument(
        "folders",
        nargs="+",
        type=Path,
        help="Wavetracker output folder(s) with wildcards"
    )
    
    parser.add_argument(
        "-c", "--config",
        type=Path,
        help="Path to config file"
    )
    
    parser.add_argument(
        "--analyze",
        dest='run_analysis',
        action='store_true',
        default=None,
        help="Run temporal trace analysis (default: from config or True)"
    )
    
    parser.add_argument(
        "--no-analyze",
        dest='run_analysis',
        action='store_false',
        help="Skip analysis, only extract frequencies"
    )
    
    parser.add_argument(
        "-n", "--n-fish",
        type=int,
        help="Number of frequencies to extract (overrides config)"
    )
    
    parser.add_argument(
        "-t", "--timepoints",
        nargs="+",
        help="Timepoints as HH:MM (overrides config)"
    )
    
    parser.add_argument(
        "--window",
        type=float,
        help="Window duration in seconds (overrides config)"
    )
    
    parser.add_argument(
        "--method",
        choices=['occurrence', 'power'],
        help="Ranking method (overrides config)"
    )
    
    parser.add_argument(
        "--freq-tolerance",
        type=float,
        help="Frequency tolerance for extraction (Hz, overrides config)"
    )
    
    parser.add_argument(
        "--trace-tol",
        type=float,
        help="Frequency tolerance for connecting traces (Hz, overrides config)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file or directory (default: parent directory of data)"
    )
    
    args = parser.parse_args()
    
    # Expand glob patterns
    expanded_folders = []
    for folder_pattern in args.folders:
        folder_str = str(folder_pattern)
        if '*' in folder_str or '?' in folder_str:
            matches = glob.glob(folder_str)
            if matches:
                expanded_folders.extend([Path(m) for m in matches])
            else:
                print(f"⚠️  No matches found for pattern: {folder_pattern}")
        else:
            expanded_folders.append(Path(folder_str))
    
    expanded_folders = [f for f in expanded_folders if f.is_dir()]
    
    if not expanded_folders:
        print("❌ Error: No valid directories found!")
        return
    
    args.folders = expanded_folders
    parent_dir = get_parent_directory(args.folders)
    
    # Load config
    config = load_config(config_path=args.config, data_folder=args.folders[0])
    
    # Override with CLI arguments
    if args.n_fish is not None:
        config['n_fish'] = args.n_fish
    if args.window is not None:
        config['window_duration'] = args.window
    if args.method is not None:
        config['method'] = args.method
    if args.freq_tolerance is not None:
        config['freq_tolerance'] = args.freq_tolerance
    if args.trace_tol is not None:
        config['trace_tolerance'] = args.trace_tol
    if args.run_analysis is not None:
        config['run_analysis'] = args.run_analysis
    
    if args.timepoints:
        timepoints = []
        for tp in args.timepoints:
            hour, minute = map(int, tp.split(':'))
            label = f"{hour:02d}:{minute:02d}"
            timepoints.append((hour, minute, label))
    else:
        timepoints = config['timepoints']
    
    # ========================================================================
    # EXTRACTION
    # ========================================================================
    
    print("="*60)
    print("FREQUENCY EXTRACTION")
    print("="*60)
    print(f"Folders to process: {len(args.folders)}")
    print(f"Frequencies per timepoint: {config['n_fish']}")
    print(f"Window duration: {config['window_duration']}s ({config['window_duration']/60:.1f} min)")
    print(f"Ranking method: {config['method']}")
    print(f"Frequency tolerance: {config['freq_tolerance']} Hz")
    min_f = config.get('min_freq')
    max_f = config.get('max_freq')
    if min_f is not None or max_f is not None:
        min_str = f"{min_f}" if min_f is not None else "None"
        max_str = f"{max_f}" if max_f is not None else "None"
        print(f"Bandpass filter: {min_str} - {max_str} Hz")
    print(f"Timepoints: {', '.join([f'{h:02d}:{m:02d}' for h, m, _ in timepoints])}")
    print("="*60)
    
    all_results = []
    for folder in sorted(args.folders):
        day_results = process_day(
            folder,
            config['n_fish'],
            timepoints,
            config['window_duration'],
            config['method'],
            config['freq_tolerance'],
            config.get('min_freq'),
            config.get('max_freq')
        )
        all_results.extend(day_results)
    
    if not all_results:
        print("\n⚠️  No results generated!")
        return
    
    df = pd.DataFrame(all_results)
    df = df.sort_values(['date', 'timepoint_hour', 'rank'])
    
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print(f"Days processed: {df['date'].nunique()}")
    print(f"Timepoints per day: {len(timepoints)}")
    print(f"Frequency range: {df['median_freq_hz'].min():.1f} - {df['median_freq_hz'].max():.1f} Hz")
    
    # Save extracted data
    if args.output and args.output.suffix == '.csv':
        extracted_csv = args.output
    else:
        extracted_csv = parent_dir / "extracted_frequencies.csv"
    
    df.to_csv(extracted_csv, index=False)
    print(f"\nExtracted data saved to: {extracted_csv}")
    
    # ========================================================================
    # ANALYSIS (Optional)
    # ========================================================================
    
    if not config['run_analysis']:
        print("\n" + "="*60)
        print("Analysis disabled. Extraction complete.")
        print("="*60)
        return
    
    print("\n" + "="*60)
    print("TEMPORAL TRACE ANALYSIS")
    print("="*60)
    print(f"Trace tolerance: {config['trace_tolerance']} Hz")
    print("  (Tip: Increase --trace-tol if traces are fragmented)")
    
    df = connect_frequencies_temporal(df, config['trace_tolerance'])
    
    n_fish = df['fish_id'].nunique()
    trace_lengths = df.groupby('fish_id').size()
    mean_trace_length = trace_lengths.mean()
    max_trace_length = trace_lengths.max()
    short_traces = (trace_lengths < 5).sum()
    
    print(f"\n  Found {n_fish} distinct fish traces")
    print(f"  Mean trace length: {mean_trace_length:.1f} timepoints")
    print(f"  Longest trace: {max_trace_length} timepoints")
    if short_traces > 0:
        print(f"  ⚠️  {short_traces} short traces (<5 points) - consider increasing --trace-tol")
    
    # Set up output directory
    if args.output and args.output.is_dir():
        output_dir = args.output
    else:
        output_dir = parent_dir / "frequency_analysis"
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save traced data
    traced_csv = output_dir / 'temporal_traces.csv'
    df.to_csv(traced_csv, index=False)
    print(f"\nTraced data saved to: {traced_csv}")
    
    # Generate plot
    print("\nGenerating temporal trace plot...")
    plot_output = output_dir / 'temporal_traces.png'
    plot_temporal_traces(df, plot_output, config['trace_tolerance'])
    
    # Generate summary
    fish_stats = []
    for fish_id in sorted(df['fish_id'].unique()):
        fish_data = df[df['fish_id'] == fish_id]
        
        fish_stats.append({
            'fish_id': int(fish_id),
            'n_detections': len(fish_data),
            'first_seen': fish_data['datetime'].min(),
            'last_seen': fish_data['datetime'].max(),
            'duration_days': (fish_data['datetime'].max() - fish_data['datetime'].min()).days,
            'median_freq_hz': fish_data['median_freq_hz'].median(),
            'mean_freq_hz': fish_data['median_freq_hz'].mean(),
            'std_freq_hz': fish_data['median_freq_hz'].std(),
            'freq_range_hz': fish_data['median_freq_hz'].max() - fish_data['median_freq_hz'].min(),
            'min_freq_hz': fish_data['median_freq_hz'].min(),
            'max_freq_hz': fish_data['median_freq_hz'].max()
        })
    
    summary_df = pd.DataFrame(fish_stats)
    summary_df = summary_df.sort_values('n_detections', ascending=False)
    
    summary_csv = output_dir / 'fish_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    
    print("\n" + "="*60)
    print("FISH SUMMARY (Top 10 by detection count)")
    print("="*60)
    display_cols = ['fish_id', 'n_detections', 'duration_days', 'median_freq_hz', 
                    'std_freq_hz', 'freq_range_hz']
    print(summary_df[display_cols].head(10).to_string(index=False))
    
    print(f"\n" + "="*60)
    print("All outputs saved to:")
    print(f"  {output_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()
