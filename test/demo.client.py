import glob
import os
import sys
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dpaas.client import DPAASClient
from dpaas.utils import format_report

server = "http://0.0.0.0:8080"
task = "test_task"
save_dir = "demo_output"
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir, exist_ok=True)

client = DPAASClient(server, task)
client.handshake()

print(f"dump local pipeline for task {task} to {save_dir}/local_pipeline.txt")
with open(os.path.join(save_dir, "local_pipeline.txt"), "w") as f:
    f.write(str(client.local_pipeline))

print(f"dump remote pipeline info for task {task} to {save_dir}/remote_pipeline.txt")
with open(os.path.join(save_dir, "remote_pipeline.txt"), "w") as f:
    f.write(client.remote_pipeline_print)

filepaths = glob.glob("scannetpp/*.mp4")
filenames = [os.path.basename(fp) for fp in filepaths]

report = client.check(filenames, filepaths)
print(f"dump report to {save_dir}/report.txt")
with open(os.path.join(save_dir, "report.txt"), "w") as f:
    f.write(format_report(report))

