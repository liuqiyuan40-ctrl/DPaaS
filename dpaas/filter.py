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
        self.target_max_side = config.get("target_max_side", None)
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
                
                if self.target_max_side is not None:
                    h, w = frame.shape[:2]
                    max_side = max(h, w)
                    if max_side > self.target_max_side:
                        scale = self.target_max_side / max_side
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
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

@dpaas_filter
class FastCutDetector(Filter):
    def __init__(self, config, desc=""):
        super().__init__(config, thres=config.get("thres", 1.0), desc=desc)
        self.scene_threshold = config.get("scene_threshold", 30.0)
        self.layout_threshold = config.get("layout_threshold", 0.15)
        self.min_clip_duration = config.get("min_clip_duration", 1.0)
        self.density_config = config.get("density_config", {
            "window_size": 30.0,
            "max_cuts_per_window": 15
        })
        self.output_modality = MODAL_FILEPATH
        
        from dpaas.scene_detector import SceneDetector
        self.detector = SceneDetector(
            scene_threshold=self.scene_threshold,
            layout_threshold=self.layout_threshold,
            min_clip_duration=self.min_clip_duration,
            density_config=self.density_config
        )

    def eval(self, filepaths) -> tuple[list, list, list]:
        processed = []
        confs = []
        reasons = []
        
        for path in filepaths:
            try:
                quality_result = self.detector.evaluate_video(path)
                processed.append(path)
                if quality_result.passed:
                    confs.append(1.0)
                    reasons.append("")
                else:
                    confs.append(0.0)
                    reasons.append(quality_result.reason)
            except Exception as e:
                processed.append(path)
                confs.append(0.0)
                reasons.append(f"Fast cut detection failed: {str(e)}")
                
        return processed, confs, reasons

@dpaas_filter
class VLMViewpointDetector(Filter):
    def __init__(self, config, desc=""):
        super().__init__(config, thres=config.get("thres", 1.0), desc=desc)
        self.api_key = config.get("api_key", "")
        self.api_base = config.get("api_base", "http://14.103.68.46/v1")
        self.model_name = config.get("model_name", "gemini-3-pro-preview")
        self.confidence_threshold = config.get("confidence_threshold", 0.85)
        self.allow_third_person = config.get("allow_third_person", False)
        self.output_modality = MODAL_NUMPY
        
    def _to_base64(self, frame: np.ndarray) -> str:
        import base64
        # Resize to save tokens
        frame = cv2.resize(frame, (512, 512))
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    def eval(self, fileobjs) -> tuple[list, list, list]:
        import httpx
        import json
        
        processed = []
        confs = []
        reasons = []
        
        client = httpx.Client(base_url=self.api_base, headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }, timeout=60.0)
        
        prompt = """
        You are an expert video analyst. Please analyze the provided sequence of image frames from a video.
        I need you to determine the following:
        1. Viewpoint: Is this a First-Person View (FPV) or a Third-Person View (TPV)?
           - FPV: The camera acts as the eyes of the person. You might see their hands, arms, or legs, but not their face or full body.
           - TPV: The camera is observing a subject from the outside. You can clearly see the subject's body, face, or back.
        2. Confidence: How confident are you in your viewpoint assessment (0.0 to 1.0)?
        
        Output your analysis strictly in JSON format with the following exact keys:
        {
            "viewpoint": "first_person" or "third_person" or "unknown",
            "viewpoint_confidence": 0.9
        }
        """
        
        for frames in fileobjs:
            if frames is None or len(frames) == 0:
                processed.append(frames)
                confs.append(0.0)
                reasons.append("No frames provided")
                continue
                
            try:
                content = [{"type": "text", "text": prompt}]
                
                # Sample up to 3 frames evenly
                num_frames = len(frames)
                step = max(1, num_frames // 3)
                for i in range(min(3, num_frames)):
                    idx = min(i * step, num_frames - 1)
                    b64_img = self._to_base64(frames[idx])
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                    })
                    
                data = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": content}],
                    "response_format": {"type": "json_object"}
                }
                
                resp = client.post("/chat/completions", json=data)
                resp.raise_for_status()
                
                resp_data = resp.json()
                content_text = resp_data["choices"][0]["message"]["content"].strip()
                
                if content_text.startswith("```json"):
                    content_text = content_text[7:]
                if content_text.endswith("```"):
                    content_text = content_text[:-3]
                    
                parsed_data = json.loads(content_text.strip())
                
                viewpoint = parsed_data.get("viewpoint", "unknown").lower()
                confidence = float(parsed_data.get("viewpoint_confidence", 0.0))
                
                processed.append(frames)
                
                if confidence < self.confidence_threshold:
                    confs.append(0.0)
                    reasons.append(f"Low confidence: {confidence} < {self.confidence_threshold}")
                elif viewpoint == "first_person":
                    confs.append(1.0)
                    reasons.append("")
                elif viewpoint == "third_person":
                    if self.allow_third_person:
                        confs.append(1.0)
                        reasons.append("")
                    else:
                        confs.append(0.0)
                        reasons.append("Third person view not allowed")
                else:
                    confs.append(0.0)
                    reasons.append(f"Unknown viewpoint: {viewpoint}")
                    
            except Exception as e:
                processed.append(frames)
                confs.append(0.0)
                reasons.append(f"VLM analysis failed: {str(e)}")
                
        return processed, confs, reasons