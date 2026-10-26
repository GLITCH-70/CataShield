"""
download_synapse_dataset.py
===========================
Fetches the Cataract-1K segmentation frames (Synapse syn53395479) that are missing from dataset/.

The Synapse folder holds Annotations/Images-and-Supervisely-Annotations/<case>/img/*.png (2,256 frames)
next to <case>/ann/*.png.json, plus 30 full surgery videos that this project does not need. This script
walks the img/ folders, downloads every frame whose file name is not already in dataset/ into
dataset_download/ (synapseclient verifies each file's MD5 on download), and then copies them into
dataset/. Re-running it skips frames that are already present.

Authentication is NOT handled here. Store a view+download personal access token once, yourself, in
~/.synapseConfig:

    [authentication]
    authtoken = <token>

(or set SYNAPSE_AUTH_TOKEN). syn.login() reads it from there. The Synapse download-list (cart) API is
deliberately not used: it needs the "modify" scope, which a download token should not have.
"""

import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from synapseclient import Synapse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
STAGING_DIR = os.path.join(BASE_DIR, "dataset_download")
SEGMENTATION_FOLDER = "syn53395479"
WORKERS = 4

_local = threading.local()


def client() -> Synapse:
    """One logged-in Synapse client per worker thread."""
    if not hasattr(_local, "syn"):
        syn = Synapse(silent=True)
        syn.login(silent=True)
        _local.syn = syn
    return _local.syn


def child(syn: Synapse, parent: str, name: str) -> str:
    for k in syn.getChildren(parent, includeTypes=["folder"]):
        if k["name"] == name:
            return k["id"]
    raise FileNotFoundError(f"folder '{name}' not found under {parent}")


def list_remote_images(syn: Synapse):
    """[(file name, synapse id)] for every frame in <case>/img/."""
    images_root = child(syn, child(syn, SEGMENTATION_FOLDER, "Annotations"), "Images-and-Supervisely-Annotations")
    images = []
    for case in syn.getChildren(images_root, includeTypes=["folder"]):
        for sub in syn.getChildren(case["id"], includeTypes=["folder"]):
            if sub["name"] == "img":
                images += [(k["name"], k["id"]) for k in syn.getChildren(sub["id"], includeTypes=["file"])]
    return images


def fetch(name: str, syn_id: str) -> str:
    target = os.path.join(STAGING_DIR, name)
    if not os.path.isfile(target):
        client().get(syn_id, downloadLocation=STAGING_DIR, ifcollision="overwrite.local")
    return name


def main():
    os.makedirs(STAGING_DIR, exist_ok=True)
    remote = list_remote_images(client())
    local = set(os.listdir(DATASET_DIR))
    missing = [(n, i) for n, i in remote if n not in local]
    print(f"remote frames: {len(remote)} | already in dataset/: {len(remote) - len(missing)} | "
          f"to download: {len(missing)}", flush=True)

    failed = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch, n, i): n for n, i in missing}
        for done, fut in enumerate(as_completed(futures), start=1):
            try:
                fut.result()
            except Exception as exc:  # keep going; report at the end
                failed.append((futures[fut], f"{type(exc).__name__}: {exc}"))
            if done % 50 == 0 or done == len(missing):
                print(f"  {done}/{len(missing)} processed, {len(failed)} failed", flush=True)

    copied = 0
    for name, _ in missing:
        src = os.path.join(STAGING_DIR, name)
        if os.path.isfile(src) and not os.path.isfile(os.path.join(DATASET_DIR, name)):
            shutil.copy2(src, os.path.join(DATASET_DIR, name))
            copied += 1
    print(f"copied {copied} new frames into dataset/", flush=True)
    if failed:
        print(f"FAILED ({len(failed)}); re-run the script to retry:", flush=True)
        for name, err in failed[:20]:
            print(f"  {name}: {err}", flush=True)
        return 1
    print("all frames present", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
