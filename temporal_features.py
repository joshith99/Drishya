import cv2
import numpy as np


class TemporalFeatures:
    """Temporal no-reference quality features from frame-to-frame behavior."""

    def __init__(self, freeze_ssim_threshold=0.85, freeze_flow_threshold=1.0, freeze_mad_threshold=8.0):
        self.freeze_ssim_threshold = freeze_ssim_threshold
        self.freeze_flow_threshold = freeze_flow_threshold
        self.freeze_mad_threshold = freeze_mad_threshold

    @staticmethod
    def _to_gray_float(frame):
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        return gray.astype(np.float32) / 255.0

    @staticmethod
    def ssim_score(frame_a, frame_b):
        """Compute SSIM between two frames (higher is better)."""
        a = TemporalFeatures._to_gray_float(frame_a)
        b = TemporalFeatures._to_gray_float(frame_b)

        c1 = (0.01 ** 2)
        c2 = (0.03 ** 2)

        mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
        mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)

        mu_a_sq = mu_a * mu_a
        mu_b_sq = mu_b * mu_b
        mu_ab = mu_a * mu_b

        sigma_a_sq = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a_sq
        sigma_b_sq = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b_sq
        sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

        numerator = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
        denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
        ssim_map = numerator / (denominator + 1e-12)

        return float(np.mean(ssim_map))

    @staticmethod
    def optical_flow_stats(prev_frame, curr_frame):
        """Compute optical flow magnitude statistics."""
        prev = (TemporalFeatures._to_gray_float(prev_frame) * 255.0).astype(np.uint8)
        curr = (TemporalFeatures._to_gray_float(curr_frame) * 255.0).astype(np.uint8)

        flow = cv2.calcOpticalFlowFarneback(
            prev,
            curr,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

        mean_mag = float(np.mean(mag))
        std_mag = float(np.std(mag))
        return mean_mag, std_mag

    def score_frame_pair(self, prev_frame, curr_frame):
        """Score one adjacent frame pair using temporal metrics."""
        prev_gray = self._to_gray_float(prev_frame)
        curr_gray = self._to_gray_float(curr_frame)
        ssim = self.ssim_score(prev_frame, curr_frame)
        flow_mean, flow_std = self.optical_flow_stats(prev_frame, curr_frame)
        mad = float(np.mean(np.abs(prev_gray - curr_gray)) * 255.0)

        is_freeze = (
            (ssim >= self.freeze_ssim_threshold)
            and (flow_mean <= self.freeze_flow_threshold)
            and (mad <= self.freeze_mad_threshold)
        )

        # Spatial flow roughness. Sequence-level jitter uses temporal variation below.
        jitter_index = flow_std / (flow_mean + 1e-6)

        return {
            'ssim': ssim,
            'flow_mean': flow_mean,
            'flow_std': flow_std,
            'mad': mad,
            'jitter_index': float(jitter_index),
            'is_freeze': bool(is_freeze),
        }

    def score_sequence(self, frames):
        """Aggregate temporal metrics over a frame sequence."""
        if len(frames) < 2:
            return {
                'avg_ssim': None,
                'avg_flow_mean': None,
                'avg_flow_std': None,
                'std_flow_mean': None,
                'avg_mad': None,
                'avg_jitter_index': None,
                'freeze_ratio': None,
                'freeze_events': None,
                'temporal_quality_score': None,
            }

        pair_scores = []
        for i in range(1, len(frames)):
            pair_scores.append(self.score_frame_pair(frames[i - 1], frames[i]))

        ssim_vals = [s['ssim'] for s in pair_scores]
        flow_mean_vals = [s['flow_mean'] for s in pair_scores]
        flow_std_vals = [s['flow_std'] for s in pair_scores]
        mad_vals = [s['mad'] for s in pair_scores]
        jitter_vals = [s['jitter_index'] for s in pair_scores]
        freeze_flags = [s['is_freeze'] for s in pair_scores]

        freeze_events = 0
        prev_freeze = False
        for flag in freeze_flags:
            if flag and not prev_freeze:
                freeze_events += 1
            prev_freeze = flag

        freeze_ratio = float(np.mean(freeze_flags)) if freeze_flags else 0.0
        avg_ssim = float(np.mean(ssim_vals))
        avg_flow_mean = float(np.mean(flow_mean_vals))
        avg_flow_std = float(np.mean(flow_std_vals))
        std_flow_mean = float(np.std(flow_mean_vals))
        avg_mad = float(np.mean(mad_vals))
        avg_jitter = float(np.mean(jitter_vals))

        jitter_quality = 1.0 / (1.0 + avg_jitter)
        motion_quality = 1.0 / (1.0 + avg_flow_mean)
        instability_quality = 1.0 / (1.0 + std_flow_mean)

        temporal_quality = (
            0.30 * avg_ssim
            + 0.30 * (1.0 - freeze_ratio)
            + 0.20 * motion_quality
            + 0.10 * instability_quality
            + 0.10 * jitter_quality
        )
        temporal_quality = float(np.clip(temporal_quality, 0.0, 1.0))

        return {
            'avg_ssim': avg_ssim,
            'avg_flow_mean': avg_flow_mean,
            'avg_flow_std': avg_flow_std,
            'std_flow_mean': std_flow_mean,
            'avg_mad': avg_mad,
            'avg_jitter_index': avg_jitter,
            'freeze_ratio': freeze_ratio,
            'freeze_events': int(freeze_events),
            'temporal_quality_score': temporal_quality,
        }