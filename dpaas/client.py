import gzip
import requests

from dpaas import modality
from dpaas.pipeline import Pipeline
from dpaas.modality import MODAL_FILEPATH

class DPAASClient:

    def __init__(self, server="localhost:8080", task="DefaultChecker"):
        self.domain = server
        self.task = task
        self.local_pipeline = None

    def _url_get(self, api):
        url = f"{self.domain}{api}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error during GET request to {url}: {e}")
            return None

    def _url_post(self, api, data:dict = {}):
        url = f"{self.domain}{api}"
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error during POST request to {url}: {e}")
            return None

    def _url_file_post(self, api, names, objs, modality):
        url = f"{self.domain}{api}"

        files_bin = modality.serialize(names, objs)

        # encode the multipart body, then gzip-compress the entire payload
        req = requests.Request("POST", url, files=files_bin)
        try:
            response = requests.Session().send(req.prepare())
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error during file POST request to {url}: {e}")
            return None

    def handshake(self):
        ret = self._url_get(f"/handshake?task={self.task}")
        cfg = ret.get("pipeline", None)
        self.local_pipeline = Pipeline(
            name=cfg.get("name", "UnnamedPipeline"),
            stages=cfg["stages"],
            initial_modality=cfg.get("initial_modality", MODAL_FILEPATH)
        )
        self.remote_pipeline_print = ret.get("remote_pipeline_print", "No remote pipeline info received")

    def local_check(self, filenames, fileobjs, progress):
        """
        Run the local pipeline to check the files, return the filtered results and the evaluation report
        """
        return self.local_pipeline.run(filenames, fileobjs, progress=progress)

    def remote_check(self, filenames, fileobjs, progress):
        """
        Send the files to remote server for checking, return the evaluation report
        """
        local_output_modality = self.local_pipeline.output_modality()
        ret = self._url_file_post(f"/check?task={self.task}", filenames, fileobjs, modality=local_output_modality)
        if progress:
            divider = "-" * 40
            print(divider)
            print(f"Remote check completed"
                    f"\n  received reports: {len(ret.get('report', {}))}"
                    f"\n  valid files: {len(ret.get('valid_names', []))}")
        return ret.get("report", None)

    def check(self, filenames, fileobjs, progress=True):
        filenames, fileobjs, local_report = self.local_check(filenames, fileobjs, progress=progress)
        if len(filenames) == 0:
            print(f"All files filtered out by local pipeline, skipping remote check")
            return local_report
        if progress:
            divider = "-" * 40
            print(divider)
            print(f"Local pipeline [{self.local_pipeline.name}] finished, total retained files: {len(filenames)}")
        remote_report = self.remote_check(filenames, fileobjs, progress=progress)
        # fuse two report dict, local report as the base
        for name, report in remote_report.items():
            if name in local_report:
                local_report[name] += report
            else:
                raise ValueError(f"Remote report contains file {name} which is not in local report, this should not happen")

        return local_report