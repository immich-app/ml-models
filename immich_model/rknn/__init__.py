"""RKNPU compilation and runtime tooling. Kept import-free so the package stays cheap to import
(compile pulls the heavy RKNN toolkit stack); import submodules directly, e.g.
from immich_model.rknn.compile import compile."""
