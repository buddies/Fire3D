"""3D-FUTURE object adapter."""

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
    name="3D-FUTURE",
    data_dir=Path(os.environ.get("FIRE3D_3D_FUTURE_ROOT", OBJECT_ROOT / "3D-FUTURE/3D-FUTURE/raw/3D-FUTURE-model")),
    metadata_file=Path(os.environ.get("FIRE3D_3D_FUTURE_METADATA", OBJECT_ROOT / "3D-FUTURE/3D-FUTURE/metadata.csv")),
    cache_file=CACHE_ROOT / "object_indexes/3d_future.json",
    metadata_cache_file=CACHE_ROOT / "object_metadata/3d_future.json",
    patterns=("raw_model.obj",),
)
list_all_model_paths = ADAPTER.list_all_model_paths
build_metadata_mapping = ADAPTER.build_metadata_mapping
save_metadata_mapping = ADAPTER.save_metadata_mapping
load_model = ADAPTER.load_model

if __name__ == "__main__":
    ADAPTER.run_cli()
