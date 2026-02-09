"""
Docstring for filter
    filter are applied to all files yet it is a file->[float, str] mapping, 
    the float is the confidence of the file being valid,
    the optional str is the reason for filtering out the file if the confidence is low,
    filter has an inherit threshold, if the confidence is below the threshold, the file will be filtered out,
    we always want batch process to enhance the throughput
"""

import random
import cv2
from dpaas.utils import randstr
from dpaas.modality import MODAL_UNCHANGED, MODAL_FILEPATH, MODAL_NUMPY
import numpy as np
from numpy import ndarray

FILTER_REGISTRY: dict[str, type["Filter"]] = {}

def dpaas_filter(cls):
    """Class decorator that registers a Filter subclass into the global registry."""
    FILTER_REGISTRY[cls.__name__] = cls
    return cls

def get_filter(name: str, config: dict, **kwargs) -> "Filter":
    """Instantiate a registered filter by name."""
    if name not in FILTER_REGISTRY:
        raise ValueError(
            f"Unknown filter '{name}'. Available: {list(FILTER_REGISTRY)}"
        )
    return FILTER_REGISTRY[name](config, **kwargs)

"""
Base class for filters, all filters should inherit from this class and implement the eval function
"""
class Filter:
    def __init__(self, config, thres=1.0, desc=""):
        self.config = config
        if desc:
            self.desc = desc
        else:
            self.desc = config.get("desc", \
                f"{self.__class__.__name__} with threshold {thres}")
        self.thres = thres
        self.id = self.__class__.__name__ + "_" + randstr(5)
        self.output_modality = MODAL_UNCHANGED

    def eval(self, fileobjs) -> tuple[list, list, list]:
        raise NotImplementedError("Shall not apply filter in base class")
    
    def filter(self, filenames, fileobjs) -> list:
        """
        :param filenames: list of file names
        :param fileobjs: a collection of file objects, len(fileobjs) == len(filenames)
            NOTE: fileobjs could be a batched tensor or a list of file objects, the filter should be able to handle both cases
        :rtype: retained_names, retained_fileobjs, eval_reports
        """

        retained_names = [] # short list
        retained_files = [] # short list
        eval_reports = {}   # full list, filename -> (conf, reason)
        processed, confs, reasons = self.eval(fileobjs)
        for i, (filename,
                conf,
                reason) in enumerate(zip(filenames, confs, reasons)):
            eval_reports[filename] = (conf, reason)
            if conf >= self.thres:
                retained_names.append(filename)
                retained_files.append(processed[i])
        if len(retained_files) == 0:
            print(f"Warning: all files filtered out, plz check the reports")
            return [], [], eval_reports
        return retained_names, self.output_modality.collate(retained_files), eval_reports

@dpaas_filter
class RandomFilter(Filter):
    def __init__(self, config, desc=""):
        super().__init__(config, thres=config.get("thres", 0.5), desc=desc)
        self.output_modality = MODAL_UNCHANGED

    def eval(self, fileobjs) -> tuple[list, list, list]:
        confs = [random.random() for _ in fileobjs]
        reasons = [
            "" if conf >= self.thres else "randomly filtered out"
            for conf in confs
        ]
        return fileobjs, confs, reasons

@dpaas_filter
class MP4MetaChecker(Filter):
    def __init__(self, config, desc=""):
        super().__init__(config, thres=config.get("thres", 1.0), desc=desc)
        self.min_fps = config.get("min_fps", 0)
        self.max_fps = config.get("max_fps", float("inf"))
        self.min_sec = config.get("min_sec", 0)
        self.max_sec = config.get("max_sec", float("inf"))
        self.min_pixel = config.get("min_pixel", 0)
        self.max_pixel = config.get("max_pixel", float("inf"))
        self.output_modality = MODAL_FILEPATH

    def eval(self, filepaths) -> tuple[list, list, list]:
        """
        Docstring for eval
        :param filepaths: file paths of mp4 to be checked
        :return: processed file paths, confidence scores, and reasons for each file
        :rtype: tuple[list, list, list]
        """
        processed = []
        confs = []
        reasons = []
        for path in filepaths:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                processed.append(path)
                confs.append(0.0)
                reasons.append(f"failed to open video {path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()

            duration = frame_count / fps if fps > 0 else 0
            pixel_count = width * height

            fails = []
            if fps < self.min_fps:
                fails.append(f"fps {fps:.1f} < min_fps {self.min_fps}")
            if fps > self.max_fps:
                fails.append(f"fps {fps:.1f} > max_fps {self.max_fps}")
            if duration < self.min_sec:
                fails.append(f"duration {duration:.1f}s < min_sec {self.min_sec}")
            if duration > self.max_sec:
                fails.append(f"duration {duration:.1f}s > max_sec {self.max_sec}")
            if pixel_count < self.min_pixel:
                fails.append(f"pixels {int(pixel_count)} < min_pixel {self.min_pixel}")
            if pixel_count > self.max_pixel:
                fails.append(f"pixels {int(pixel_count)} > max_pixel {self.max_pixel}")

            processed.append(path)
            if fails:
                confs.append(0.0)
                reasons.append("; ".join(fails))
            else:
                confs.append(1.0)
                reasons.append("")
        return processed, confs, reasons


@dpaas_filter
class MP4Sampler(Filter):
    def __init__(self, config, desc=""):
        super().__init__(config, thres=config.get("thres", 1.0), desc=desc)
        self.sample_fps = config.get("sample_fps", 1)
        self.num_frames = config.get("num_frames", 1)
        self.output_modality = MODAL_NUMPY

    def eval(self, filepaths) -> tuple[list, list, list]:
        """
        Docstring for eval
        :param filepaths: file paths of mp4 to be sampled
        :return: processed frames in ndarray, success or not, and error if there is any, for each file
        :rtype: tuple[list, list, list]
        """
        processed = []
        confs = []
        reasons = []
        for path in filepaths:
            cap = cv2.VideoCapture(path)
            src_fps = cap.get(cv2.CAP_PROP_FPS)
            if src_fps <= 0:
                cap.release()
                processed.append(None)
                confs.append(0.0)
                reasons.append(f"failed to get FPS from video {path}")
                continue

            interval = max(1, int(round(src_fps / self.sample_fps)))
            sampled = []
            frame_idx = 0
            while len(sampled) < self.num_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                sampled.append(frame)
                frame_idx += interval
            cap.release()

            if not sampled:
                processed.append(None)
                confs.append(0.0)
                reasons.append(f"no frames sampled from video {path}")
            elif len(sampled) < self.num_frames:
                processed.append(None)
                confs.append(0.0)
                reasons.append(f"only {len(sampled)}/{self.num_frames} frames sampled from video {path}")
            else:
                processed.append(np.stack(sampled))  # (num_frames, H, W, 3) BGR
                confs.append(1.0)
                reasons.append("")
        return processed, confs, reasons