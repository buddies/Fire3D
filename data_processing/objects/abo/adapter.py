"""Amazon Berkeley Objects adapter."""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_processing.objects.adapter_common import ObjectAdapter

OBJECT_ROOT = Path(os.environ.get("FIRE3D_OBJECT_ROOT", REPO_ROOT / "data/training_objects"))
CACHE_ROOT = Path(os.environ.get("FIRE3D_PROCESSING_CACHE", REPO_ROOT / "outputs/data_processing"))
ADAPTER = ObjectAdapter(
    name="ABO",
    data_dir=Path(os.environ.get("FIRE3D_ABO_ROOT", OBJECT_ROOT / "ABO/ABO/raw/3dmodels/original")),
    metadata_file=Path(os.environ.get("FIRE3D_ABO_METADATA", OBJECT_ROOT / "ABO/ABO/metadata.csv")),
    cache_file=CACHE_ROOT / "object_indexes/abo.json",
    metadata_cache_file=CACHE_ROOT / "object_metadata/abo.json",
    patterns=("*.glb",),
)
list_all_model_paths = ADAPTER.list_all_model_paths
build_metadata_mapping = ADAPTER.build_metadata_mapping
save_metadata_mapping = ADAPTER.save_metadata_mapping
load_model = ADAPTER.load_model

if __name__ == "__main__":
    ADAPTER.run_cli()
