import os
import urllib.request
import zipfile
from dataclasses import dataclass


DRONE_AUDIO_ZIP_URL = "https://codeload.github.com/saraalemadi/DroneAudioDataset/zip/refs/heads/master"


@dataclass(frozen=True)
class DroneAudioDataMeta:
    root_dir: str
    extracted_dir: str
    zip_path: str
    binary_dir: str
    multiclass_dir: str


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_and_extract_drone_audio_dataset(root_dir: str = "data") -> DroneAudioDataMeta:
    _ensure_dir(root_dir)

    zip_path = os.path.join(root_dir, "DroneAudioDataset_master.zip")
    extracted_dir = os.path.join(root_dir, "DroneAudioDataset-master")

    binary_dir = os.path.join(extracted_dir, "Binary_Drone_Audio")
    multiclass_dir = os.path.join(extracted_dir, "Multiclass_Drone_Audio")


    if os.path.exists(binary_dir) and os.path.exists(multiclass_dir):
        return DroneAudioDataMeta(
            root_dir=root_dir,
            extracted_dir=extracted_dir,
            zip_path=zip_path,
            binary_dir=binary_dir,
            multiclass_dir=multiclass_dir
        )

    if not os.path.exists(zip_path):
        print(f"[download_drone_audio.py] Downloading dataset from: {DRONE_AUDIO_ZIP_URL}")
        urllib.request.urlretrieve(DRONE_AUDIO_ZIP_URL, zip_path)
        print(f"[download_drone_audio.py] Saved zip to: {zip_path}")


    print(f"[download_drone_audio.py] Extracting zip to: {root_dir}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(root_dir)


    if not os.path.exists(binary_dir):
        raise FileNotFoundError(f"Could not find Binary_Drone_Audio in: {binary_dir}")

    if not os.path.exists(multiclass_dir):
        raise FileNotFoundError(f"Could not find Multiclass_Drone_Audio in: {multiclass_dir}")

    return DroneAudioDataMeta(
        root_dir=root_dir,
        extracted_dir=extracted_dir,
        zip_path=zip_path,
        binary_dir=binary_dir,
        multiclass_dir=multiclass_dir
    )