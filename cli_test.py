"""
Drishya terminal test runner — runs the classical (Phases 1-3) and deep MOS
(Phase 4) pipelines headlessly, without the Streamlit UI.

Usage:
    python cli_test.py testing_videos/video_small.mp4
    python cli_test.py testing_videos/video_small_blur.mp4 testing_videos/video_small.mp4 --temporal
    python cli_test.py testing_videos/video_small.mp4 --no-classical
    python cli_test.py testing_videos/*.mp4 --json report.json
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Drishya classical + deep MOS analysis in the terminal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('video_paths', nargs='+', help="Path(s) to video file(s)")
    p.add_argument('--samples', '-s', type=int, default=8,
                   help="Frames sampled per video (default: 8)")
    p.add_argument('--method', '-m', choices=['uniform', 'random'], default='uniform',
                   help="Classical frame sampling method (default: uniform)")
    p.add_argument('--temporal', action='store_true',
                   help="Enable temporal metrics (SSIM/flow/freeze)")
    p.add_argument('--temporal-stride', type=int, default=1,
                   help="Spacing between temporal-window frames (default: 1)")
    p.add_argument('--seed', type=int, default=None,
                   help="Seed for reproducible random sampling (default: None)")
    p.add_argument('--no-classical', action='store_true',
                   help="Skip the classical (Phases 1-3) analysis")
    p.add_argument('--no-deep', action='store_true',
                   help="Skip the deep MOS (Phase 4) prediction")
    p.add_argument('--ckpt-dir', default='checkpoints',
                   help="Folder holding svr_deep_*.pkl (default: checkpoints)")
    p.add_argument('--json', dest='json_out', default=None,
                   help="Write results to this JSON file")
    return p.parse_args()


def run_classical(path, args):
    from NQS import NoReferenceQualityScorer
    scorer = NoReferenceQualityScorer()
    return scorer.analyze_video(
        str(path),
        num_samples=args.samples,
        sample_method=args.method,
        temporal=args.temporal,
        temporal_stride=args.temporal_stride,
        seed=args.seed,
    )


def main():
    args = parse_args()

    for p in args.video_paths:
        if not Path(p).exists():
            print(f"Error: video not found: {p}")
            return 1

    # ── Load deep model once (shared across videos) ──────────
    deep = None
    if not args.no_deep:
        import deep_mos
        svr, backbone, transform, cfg, meta = deep_mos.load_deep_model(args.ckpt_dir)
        if svr is None:
            print(f"Warning: no svr_deep_*.pkl in '{args.ckpt_dir}'; deep MOS skipped.")
        else:
            deep = (deep_mos, svr, backbone, transform, cfg)
            print(f"Deep model loaded | {meta['backbone']} ({meta['weights_source']}) | "
                  f"device={cfg[0]} | frames={args.samples or cfg[1]}")

    reports = []
    for p in args.video_paths:
        path = Path(p)
        entry = {'file': path.name, 'path': str(path)}

        # ── Deep MOS ─────────────────────────────────────────
        if deep is not None:
            deep_mos, svr, backbone, transform, cfg = deep
            res = deep_mos.predict_mos(path, svr, backbone, transform, cfg,
                                       num_frames=args.samples)
            entry['mos_predicted'] = res['mos'] if res else None
            entry['mos_rating'] = res['rating'] if res else None
        else:
            entry['mos_predicted'] = None

        # ── Classical ────────────────────────────────────────
        if not args.no_classical:
            try:
                stats = run_classical(path, args)
            except Exception as e:
                stats = None
                print(f"  Classical analysis failed for {path.name}: {e}")
            if stats:
                entry['avg_quality'] = round(float(stats['avg_quality_score']), 4)
                entry['quality_rating'] = stats['quality_rating']
                entry['frames_analyzed'] = stats['total_frames_analyzed']
                if stats.get('temporal_metrics'):
                    tm = stats['temporal_metrics']
                    entry['temporal_score'] = tm.get('temporal_quality_score')
                    entry['avg_ssim'] = tm.get('avg_ssim')
                    entry['freeze_ratio'] = tm.get('freeze_ratio')
                    entry['jitter_index'] = tm.get('avg_jitter_index')

        reports.append(entry)

    # ── Summary table ────────────────────────────────────────
    print("\n" + "=" * 92)
    print("DRISHYA TERMINAL TEST SUMMARY")
    print("=" * 92)
    header = (f"{'File':<34} {'Deep MOS':>9} {'MOS Rating':>11} "
              f"{'AvgQual':>8} {'Rating':>10} {'Temporal':>9}")
    print(header)
    print("-" * 92)
    for r in reports:
        mos = r.get('mos_predicted')
        mos_s = f"{mos:.2f}" if mos is not None else "N/A"
        mos_rating = r.get('mos_rating', '-')
        avg = r.get('avg_quality')
        avg_s = f"{avg:.3f}" if avg is not None else "N/A"
        rating = r.get('quality_rating', 'N/A')
        temp = r.get('temporal_score')
        temp_s = f"{temp:.3f}" if temp is not None else "-"
        print(f"{r['file']:<34} {mos_s:>9} {mos_rating:>11} "
              f"{avg_s:>8} {rating:>10} {temp_s:>9}")
    print("=" * 92 + "\n")

    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(reports, f, indent=2)
        print(f"JSON report written to {args.json_out}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
