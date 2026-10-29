#!/usr/bin/env python
"""Phase 3 calibration: Compare temporal metrics across baseline, freeze, and jitter videos."""

import subprocess
import json
import re
import os
import sys

def run_nqs(video_path, samples=10):
    """Run NQS and extract temporal metrics."""
    result = subprocess.run(
        [sys.executable, 'NQS.py', video_path, '--samples', str(samples), '--temporal'],
        capture_output=True,
        text=True,
        cwd=os.getcwd()
    )
    
    output = result.stdout + result.stderr
    
    # Parse temporal metrics from output
    metrics = {}
    
    # Extract temporal quality score
    match = re.search(r'Temporal Quality Score: ([\d.]+)', output)
    if match:
        metrics['temporal_quality'] = float(match.group(1))
    
    # Extract SSIM
    match = re.search(r'Avg SSIM: ([\d.]+)', output)
    if match:
        metrics['avg_ssim'] = float(match.group(1))
    
    # Extract freeze ratio
    match = re.search(r'Freeze Ratio: ([\d.]+)', output)
    if match:
        metrics['freeze_ratio'] = float(match.group(1))
    
    # Extract freeze events
    match = re.search(r'Freeze Events: (\d+)', output)
    if match:
        metrics['freeze_events'] = int(match.group(1))
    
    # Extract jitter index
    match = re.search(r'Avg Jitter Index: ([\d.]+)', output)
    if match:
        metrics['avg_jitter'] = float(match.group(1))
    
    # Extract overall quality score
    match = re.search(r'Average Quality Score: ([\d.]+)', output)
    if match:
        metrics['overall_quality'] = float(match.group(1))
    
    return metrics

def _fmt(value, spec):
    """Format a numeric value or return 'N/A' if it's missing/non-numeric."""
    if isinstance(value, (int, float)):
        return f"{value:{spec}}"
    return "N/A"


def main():
    videos = [
        ('testing_videos/video_small.mp4', 'Baseline (No Degradation)'),
        ('testing_videos/video_small_freeze.mp4', 'Freeze Artifacts'),
        ('testing_videos/video_small_jitter.mp4', 'Jitter Artifacts'),
    ]
    
    print("Phase 3 Calibration: Temporal Metrics Baseline")
    print("=" * 80)
    
    results = {}
    
    for video_path, label in videos:
        if not os.path.exists(video_path):
            print(f"Skip: {label} - {video_path} (not found)")
            continue
        
        print(f"\nAnalyzing: {label}")
        print(f"  File: {video_path}")
        metrics = run_nqs(video_path, samples=20)
        
        if metrics:
            results[label] = metrics
            print(f"  Temporal Quality: {_fmt(metrics.get('temporal_quality'), '.3f')}")
            print(f"  Avg SSIM: {_fmt(metrics.get('avg_ssim'), '.4f')}")
            print(f"  Freeze Ratio: {_fmt(metrics.get('freeze_ratio'), '.3f')}")
            print(f"  Freeze Events: {_fmt(metrics.get('freeze_events'), 'd')}")
            print(f"  Avg Jitter Index: {_fmt(metrics.get('avg_jitter'), '.3f')}")
            print(f"  Overall Quality: {_fmt(metrics.get('overall_quality'), '.3f')}")
        else:
            print(f"  ERROR: Could not extract metrics")
    
    print("\n" + "=" * 80)
    print("CALIBRATION SUMMARY")
    print("=" * 80)
    
    if len(results) >= 2:
        baseline = results.get('Baseline (No Degradation)', {})
        
        print(f"\nBaseline Temporal Quality: {_fmt(baseline.get('temporal_quality'), '.3f')}")
        
        for label, metrics in results.items():
            if label == 'Baseline (No Degradation)':
                continue
            
            base_q = baseline.get('temporal_quality') or 0
            delta = (metrics.get('temporal_quality') or 0) - base_q
            pct_change = (delta / base_q * 100) if base_q else 0
            
            print(f"\n{label}:")
            print(f"  Temporal Quality: {_fmt(metrics.get('temporal_quality'), '.3f')} (delta {delta:+.3f}, {pct_change:+.1f}%)")
            print(f"  SSIM: {_fmt(metrics.get('avg_ssim'), '.4f')}")
            print(f"  Jitter Index: {_fmt(metrics.get('avg_jitter'), '.3f')}")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("1. Freeze detection needs tuning if freeze video shows too-high temporal quality")
    print("2. Jitter detection sensitivity may need adjustment")
    print("3. Recommended final weight for temporal in composite: 10-15%")
    print("=" * 80)

if __name__ == '__main__':
    main()
