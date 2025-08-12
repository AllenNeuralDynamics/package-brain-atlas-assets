"""Base class for atlas assets with location and manifest management (moved from atlas_assets)."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AtlasAsset:
    """Base class for atlas assets.

    Attributes:
        name: Asset identifier
        version: Asset version string
        _asset_location: Storage directory name for this asset type
    """

    name: str
    version: str

    _asset_location = None

    @property
    def manifest(self) -> dict:
        """Generate manifest dictionary for this asset."""
        return {
            "name": self.name,
            "version": self.version,
            "location": str(self.location(Path("/"))),
        }

    def location(self, root) -> Path:
        """Get file system location for this asset."""
        return Path(root) / self._asset_location / f"{self.name}" / self.version

    def create_manifest(self, output_root: Path):
        """Create and write manifest.json file for this asset."""
        manifest_file = self.location(output_root) / "manifest.json"
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_file, "w") as f:
            json.dump(self.manifest, f, indent=3)
        logging.info(
            f"Created manifest for {self.__class__.__name__}: {self.name} at {manifest_file}"
        )
