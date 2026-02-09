"""
pipeline is a list of filters, the order of the pipeline is the order of execution
"""

from dpaas.filter import Filter, get_filter
from dpaas.modality import MODAL_UNCHANGED, MODAL_FILEPATH, get_modality

class Pipeline:
    def __init__(self, name, stages, initial_modality):
        self.name = name
        if stages is None:
            print("Warning: pipeline stages is None, initializing with empty list")
            self.stages = []
        else:
            self.stages = []
            for s in stages:
                if isinstance(s, dict):
                    self.stages.append(get_filter(s["class"], s.get("config", {}), desc=s.get("desc", "")))
                elif isinstance(s, Filter):
                    self.stages.append(s)
                else:
                    raise ValueError(f"Unsupported stage type: {type(s)}")
        
        # propagate the modality of each stage to the next stage, if the stage is unchanged, it will inherit the modality from the previous stage
        # first stage with unchanged modality will be set to filepath modality by default
        if isinstance(initial_modality, str):
            initial_modality = get_modality(initial_modality)
        current_modality = initial_modality
        for stage in self.stages:
            if stage.output_modality != MODAL_UNCHANGED:
                current_modality = stage.output_modality
            else:
                stage.output_modality = current_modality
            
    def run(self, filenames, fileobjs, progress=False):
        final_report = {f:[] for f in filenames}
        divider = "-" * 40
        for stage in self.stages:
            if progress:
                print(divider)
                print(f"Running stage {stage.id}"
                      f"\n  Input modality: {stage.output_modality}"
                      f"\n  Input files: {len(filenames)}")
            filenames, fileobjs, stage_reports = stage.filter(filenames, fileobjs)
            for f, report in stage_reports.items():
                final_report[f].append(report)
            if len(filenames) == 0:
                print(f"All files filtered out at stage {stage.id}, early stopping pipeline execution")
                break
            if progress:
                print(f"  retained files: {len(filenames)}"
                      f"\n  filtered out files: {len(stage_reports) - len(filenames)}")
        return filenames, fileobjs, final_report
    
    def output_modality(self):
        return self.stages[-1].output_modality if self.stages else MODAL_FILEPATH
    
    @staticmethod
    def _wrap(text, width):
        """Wrap text into lines of at most `width` characters, breaking on words."""
        words = text.split()
        lines, cur = [], ""
        for word in words:
            if not cur:
                cur = word
            elif len(cur) + 1 + len(word) <= width:
                cur += " " + word
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines or [""]

    def __str__(self):
        W = 50  # total box width

        if not self.stages:
            return f"Pipeline: {self.name} (empty)"

        # collect stage info
        stages_info = []
        for stage in self.stages:
            label = stage.id
            modality = str(stage.output_modality)
            desc = stage.desc if stage.desc else ""
            stages_info.append((label, modality, desc))

        # title
        title = f"Pipeline: {self.name}"
        res = f"{'─' * W}\n"
        res += f"{title:^{W}}\n"
        res += f"{'─' * W}\n"

        inner = W - 2  # inside the box walls

        for i, (label, modality, desc) in enumerate(stages_info):
            # first row: label | modality split into two cells
            mid = inner // 2
            left_w = mid - 2       # content width for left cell
            right_w = inner - mid - 1  # content width for right cell

            # wrap label and modality, then pad rows to same height
            label_lines = self._wrap(label, left_w)
            modal_lines = self._wrap(modality, right_w)
            row1_h = max(len(label_lines), len(modal_lines))
            while len(label_lines) < row1_h:
                label_lines.append("")
            while len(modal_lines) < row1_h:
                modal_lines.append("")

            top = f"┌{'─' * (mid-1)}┬{'─' * (inner - mid)}┐"
            res += f"{top}\n"
            for li, mi in zip(label_lines, modal_lines):
                res += f"│ {li:<{left_w}}│ {mi:<{right_w}}│\n"

            # second row: desc spans full width, with word wrap
            divider = f"├{'─' * inner}┤"
            res += f"{divider}\n"
            desc_text = desc if desc else "-"
            desc_w = inner - 1  # 1 char left padding
            for line in self._wrap(desc_text, desc_w):
                res += f"│ {line:<{desc_w}}│\n"

            bottom = f"└{'─' * inner}┘"
            res += f"{bottom}\n"
            if i < len(stages_info) - 1:
                res += f"{'▼':^{W}}\n"

        return res.rstrip("\n")