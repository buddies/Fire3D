# Hugging Face Download Count

The `hongchi/Fire3D` model repository uses a root `config.json` as its standard
Hugging Face query file. The model card intentionally does not claim a library
integration because Fire3D uses its own runtime. The release downloader uses
`snapshot_download`, which requests this file as part of the model snapshot.

Hugging Face counts `GET` and `HEAD` requests to model query files on the
server. Fire3D does not add telemetry, cookies, or a separate counting request.
See the [Hugging Face model download-statistics
documentation](https://huggingface.co/docs/hub/models-download-stats) for the
server-side counting contract.
