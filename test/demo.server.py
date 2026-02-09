import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dpaas.server import DPAASServer

server = DPAASServer(
    host="0.0.0.0",
    port=8080,
    tasks={
        "test_task": {
            "local": "demo.local.json",
            "remote": "demo.remote.json",
        }
    },
)

server.run()
