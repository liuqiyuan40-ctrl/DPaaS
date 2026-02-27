import cv2
import numpy as np
from typing import NamedTuple
from dataclasses import dataclass

class QualityCheckResult(NamedTuple):
    passed: bool
    reason: str
    rapid_cut_count: int
    rapid_cut_ratio: float
    max_cuts_in_window: int

class CutDensityAnalyzer:
    def __init__(
        self,
        rapid_cut_duration: float = 1.5,
        max_rapid_ratio: float = 0.3,
        window_size: float = 30.0,
        max_cuts_per_window: int = 10,
        rapid_cut_cpm_threshold: float = 15.0,
    ):
        self.rapid_cut_duration = rapid_cut_duration
        self.max_rapid_ratio = max_rapid_ratio
        self.window_size = window_size
        self.max_cuts_per_window = max_cuts_per_window
        self.rapid_cut_cpm_threshold = rapid_cut_cpm_threshold

    def analyze(
        self, clips: list[tuple[float, float]], total_duration: float
    ) -> QualityCheckResult:
        if not clips:
            return QualityCheckResult(True, "No clips detected", 0, 0.0, 0)

        rapid_cuts = [
            (start, end)
            for start, end in clips
            if (end - start) < self.rapid_cut_duration
        ]
        rapid_cut_count = len(rapid_cuts)
        total_clips = len(clips)
        rapid_ratio = rapid_cut_count / total_clips if total_clips > 0 else 0.0

        # Check CPM (Cuts Per Minute)
        cpm = (rapid_cut_count / total_duration) * 60 if total_duration > 0 else 0
        if cpm > self.rapid_cut_cpm_threshold:
            return QualityCheckResult(
                False,
                f"Rapid cut CPM too high: {cpm:.1f} (threshold: {self.rapid_cut_cpm_threshold})",
                rapid_cut_count,
                rapid_ratio,
                0,
            )

        if rapid_ratio > self.max_rapid_ratio:
            return QualityCheckResult(
                False,
                f"Rapid cut ratio too high: {rapid_ratio:.1%} (threshold: {self.max_rapid_ratio:.1%})",
                rapid_cut_count,
                rapid_ratio,
                0,
            )

        cut_points = [c[0] for c in clips]
        if len(cut_points) < self.max_cuts_per_window:
            return QualityCheckResult(
                True, "Passed", rapid_cut_count, rapid_ratio, len(cut_points)
            )

        cut_points.sort()
        max_density = 0
        n = len(cut_points)
        right = 0

        for left in range(n):
            while (
                right < n and cut_points[right] - cut_points[left] <= self.window_size
            ):
                right += 1
            current_density = right - left
            max_density = max(max_density, current_density)

        if max_density > self.max_cuts_per_window:
            return QualityCheckResult(
                False,
                f"Rapid cut accumulation: {max_density} cuts in {self.window_size}s (threshold: {self.max_cuts_per_window})",
                rapid_cut_count,
                rapid_ratio,
                max_density,
            )

        return QualityCheckResult(
            True, "Passed", rapid_cut_count, rapid_ratio, max_density
        )

class SceneDetector:
    def __init__(
        self,
        scene_threshold: float = 30.0,
        layout_threshold: float = 0.15,
        min_clip_duration: float = 1.0,
        density_config: dict | None = None,
    ):
        self.scene_threshold = scene_threshold
        self.layout_threshold = layout_threshold
        self.min_clip_duration = min_clip_duration
        config = density_config or {}
        self.density_analyzer = CutDensityAnalyzer(**config)

    def evaluate_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            fps = 25.0
        total_duration = total_frames / fps if fps > 0 else 0

        cut_frames = self._detect_cuts_legacy(cap, fps)
        cap.release()

        clips = self._frames_to_clips(cut_frames, total_frames, fps)
        quality_result = self.density_analyzer.analyze(clips, total_duration)
        
        return quality_result

    def _detect_cuts_legacy(self, cap: cv2.VideoCapture, fps: float) -> list[int]:
        cut_frames = [0]
        ret, prev_frame = cap.read()
        if not ret:
            return cut_frames

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_edges = cv2.Canny(prev_gray, 50, 150)
        frame_idx = 1
        sample_interval = max(1, int(fps / 10))

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                is_scene_cut = self._check_scene_change(prev_gray, curr_gray)
                
                curr_edges = cv2.Canny(curr_gray, 50, 150)
                is_layout_cut = self._check_layout_change(prev_edges, curr_edges)

                if is_scene_cut or is_layout_cut:
                    cut_frames.append(frame_idx)

                prev_gray = curr_gray
                prev_edges = curr_edges

            frame_idx += 1

        return cut_frames

    def _check_scene_change(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> bool:
        diff = cv2.absdiff(prev_gray, curr_gray)
        mean_diff = np.mean(diff)
        return mean_diff > self.scene_threshold

    def _check_layout_change(
        self, prev_edges: np.ndarray, curr_edges: np.ndarray
    ) -> bool:
        diff = cv2.absdiff(prev_edges, curr_edges)
        change_ratio = np.sum(diff > 0) / diff.size
        return change_ratio > self.layout_threshold

    def _frames_to_clips(
        self, cut_frames: list[int], total_frames: int, fps: float
    ) -> list[tuple[float, float]]:
        clips = []
        cut_frames = sorted(set(cut_frames))
        cut_frames.append(total_frames)

        for i in range(len(cut_frames) - 1):
            start_frame = cut_frames[i]
            end_frame = cut_frames[i + 1]
            start_time = start_frame / fps
            end_time = end_frame / fps
            duration = end_time - start_time

            if duration >= self.min_clip_duration:
                clips.append((start_time, end_time))

        return clips
