"""Machine-readable runtime identity for product discovery and pin validation."""

from __future__ import annotations

import json

from ltx_pipelines_mlx.utils.perf_profile import runtime_identity


def main() -> None:
    print(json.dumps(runtime_identity(), sort_keys=True))


if __name__ == "__main__":
    main()
