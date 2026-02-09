import io
import gzip
import json

from flask import Flask, request, jsonify

from dpaas.pipeline import Pipeline
from dpaas.modality import MODAL_FILEPATH, deserialize


class GzipMiddleware:
    """WSGI middleware that transparently decompresses gzip-encoded request bodies."""

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        if environ.get("HTTP_CONTENT_ENCODING") == "gzip":
            body = environ["wsgi.input"].read()
            decompressed = gzip.decompress(body)
            environ["wsgi.input"] = io.BytesIO(decompressed)
            environ["CONTENT_LENGTH"] = str(len(decompressed))
            del environ["HTTP_CONTENT_ENCODING"]
        return self.app(environ, start_response)


class DPAASServer:
    """
    Data Processing as a Service server.

    Parameters
    ----------
    host : str
        Bind address.
    port : int
        Bind port.
    tasks : dict
        Mapping of task_name -> {"local": path_to_local_config, "remote": path_to_remote_config}.
    """

    def __init__(self, host="0.0.0.0", port=8080, tasks=None):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.app.wsgi_app = GzipMiddleware(self.app.wsgi_app)
        self.tasks = {}

        if tasks:
            for task_name, paths in tasks.items():
                self.register_task(task_name, paths["local"], paths["remote"])

        self._register_routes()

    def register_task(self, task_name, local_config_path, remote_config_path):
        with open(local_config_path) as f:
            local_cfg = json.load(f)
        with open(remote_config_path) as f:
            remote_cfg = json.load(f)

        remote_pipeline = Pipeline(
            name=remote_cfg.get("name", "UnnamedPipeline"),
            stages=remote_cfg.get("stages", []),
            initial_modality=remote_cfg.get("initial_modality", MODAL_FILEPATH),
        )

        self.tasks[task_name] = {
            "local_config": local_cfg,
            "remote_pipeline": remote_pipeline,
        }

    def _register_routes(self):

        @self.app.route("/handshake", methods=["GET"])
        def handshake():
            task_name = request.args.get("task")
            if task_name not in self.tasks:
                return jsonify({"error": f"Unknown task: {task_name}"}), 404
            return jsonify(
                {
                    "pipeline": self.tasks[task_name]["local_config"],
                    "remote_pipeline_print": str(self.tasks[task_name]["remote_pipeline"])
                })

        @self.app.route("/check", methods=["POST"])
        def check():
            task_name = request.args.get("task")
            if task_name not in self.tasks:
                return jsonify({"error": f"Unknown task: {task_name}"}), 404

            task = self.tasks[task_name]

            # deserialize into (modality, filenames, fileobjs)
            _modality, filenames, fileobjs = deserialize(request.form, request.files)

            # run the remote pipeline
            assert _modality == task["remote_pipeline"].output_modality(), \
                f"Modality mismatch: expected {task['remote_pipeline'].output_modality()}, got {_modality}"
            _retained_names, _retained_objs, report = task["remote_pipeline"].run(filenames, fileobjs)

            return jsonify({
                    "report": self._serialize_report(report),
                    "valid_names": _retained_names
                })

    @staticmethod
    def _serialize_report(report):
        """Convert report values to JSON-safe types (tuples -> lists)."""
        return {
            name: [list(entry) if isinstance(entry, tuple) else entry for entry in entries]
            for name, entries in report.items()
        }

    def run(self):
        print(f"DPaaS server starting on {self.host}:{self.port}")
        for name in self.tasks:
            print(f"  task registered: {name}")
        self.app.run(host=self.host, port=self.port)
