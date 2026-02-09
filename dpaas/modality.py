import os
import pickle
import tempfile
import numpy as np


class Modality:
    def __init__(self, name, collate=None, serialize=None, deserialize=None):
        self.name = name
        self.collate = collate
        self.serialize = serialize
        self.deserialize = deserialize

    def __eq__(self, other):
        if isinstance(other, Modality):
            return self.name == other.name
        return NotImplemented

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"Modality({self.name!r})"


# ── helpers ──────────────────────────────────────────────────────────

def _serialize_default(names, objs, modality_name):
    return [
        ('modality', (None, modality_name, 'text/plain')),
        ('names', ('names.pkl', pickle.dumps(names), 'application/octet-stream')),
        ('objs', ('objs.pkl', pickle.dumps(objs), 'application/octet-stream')),
    ]

def _serialize_filepath(names, objs):
    file_data = []
    for path in objs:
        with open(path, 'rb') as f:
            file_data.append(f.read())
    return _serialize_default(names, file_data, MODAL_FILEPATH.name)

def _deserialize_default(file_storage_map):
    names = pickle.loads(file_storage_map.get('names').read())
    objs = pickle.loads(file_storage_map.get('objs').read())
    return names, objs

def _deserialize_filepath(file_storage_map):
    names, objs_bytes = _deserialize_default(file_storage_map)
    tmp_dir = tempfile.mkdtemp(prefix="dpaas_")
    paths = []
    for name, data in zip(names, objs_bytes):
        tmp_path = os.path.join(tmp_dir, name)
        with open(tmp_path, 'wb') as f:
            f.write(data)
        paths.append(tmp_path)
    return names, paths


# ── singleton instances ──────────────────────────────────────────────

MODAL_NUMPY = Modality(
    name="ndarray",
    collate=lambda files: np.stack(files),
    serialize=lambda names, objs: _serialize_default(names, objs, "ndarray"),
    deserialize=_deserialize_default,
)

MODAL_BINARY = Modality(
    name="binary",
    collate=lambda files: files,
    serialize=lambda names, objs: _serialize_default(names, objs, "binary"),
    deserialize=_deserialize_default,
)

MODAL_FILEPATH = Modality(
    name="filepath",
    collate=lambda files: files,
    serialize=_serialize_filepath,
    deserialize=_deserialize_filepath,
)

MODAL_UNCHANGED = Modality(
    name="unchanged",
    collate=lambda files: files,
)

MODALITY_REGISTRY = {m.name: m for m in [MODAL_NUMPY, MODAL_BINARY, MODAL_FILEPATH, MODAL_UNCHANGED]}


def get_modality(name):
    """Look up a Modality singleton by its string name (e.g. from JSON config)."""
    if name not in MODALITY_REGISTRY:
        raise ValueError(f"Unknown modality: {name!r}. Available: {list(MODALITY_REGISTRY)}")
    return MODALITY_REGISTRY[name]


def deserialize(form, files):
    """
    Deserialize multipart upload data.
    Reads modality from the embedded form field, then dispatches
    to the appropriate Modality's deserialize method.

    Returns (modality, filenames, fileobjs).
    """
    modality_name = form.get('modality')
    modality = get_modality(modality_name)
    if modality.deserialize is None:
        raise ValueError(f"Modality {modality_name!r} does not support deserialization")
    names, objs = modality.deserialize(files)
    return modality, names, objs
