"""Asset library for managing and retrieving atlas data assets (moved from atlas_assets)."""

from typing import Dict, List, Tuple

from atlas_builder.annotation_set import AnnotationSet
from atlas_builder.coordinate_space import CoordinateSpace
from atlas_builder.template import Template
from atlas_builder.atlas_asset import AtlasAsset
from atlas_builder.coordinate_transformation import CoordinateTransformation
from atlas_builder.atlas import Atlas
from atlas_builder.terminology import Terminology


class AssetLibrary:
    """Central registry for managing and retrieving atlas data assets."""

    def __init__(self):
        # Store assets by type and (name, version) key for efficient lookup
        self._assets: Dict[str, Dict[Tuple[str, str], AtlasAsset]] = {
            Template.__name__: {},
            AnnotationSet.__name__: {},
            CoordinateTransformation.__name__: {},
            Atlas.__name__: {},
            Terminology.__name__: {},
            CoordinateSpace.__name__: {},
        }

    def add(self, asset: AtlasAsset):
        """Add an asset to the library."""
        if not isinstance(asset, AtlasAsset):
            raise TypeError("Asset must be an instance of AtlasAsset")

        asset_type = asset.__class__.__name__
        if asset_type not in self._assets:
            raise ValueError(f"Unknown asset type: {asset_type}")

        key = (asset.name, asset.version)
        self._assets[asset_type][key] = asset

    def get_coordinate_space(self, name: str, version: str) -> AtlasAsset:
        """Get a coordinate space by name and version."""
        asset = self._assets[CoordinateSpace.__name__].get((name, version))
        if asset is None:
            raise KeyError(
                f"CoordinateSpace with name='{name}' and version='{version}' not found in library"
            )
        return asset

    def get_coordinate_transformation(self, name: str, version: str) -> AtlasAsset:
        """Get a coordinate transformation by name and version."""
        asset = self._assets[CoordinateTransformation.__name__].get((name, version))
        if asset is None:
            raise KeyError(f"CoordinateTransformation with name='{name}' and version='{version}' not found in library")
        return asset

    def get_template(self, name: str, version: str) -> AtlasAsset:
        """Get a template by name and version."""
        asset = self._assets[Template.__name__].get((name, version))
        if asset is None:
            raise KeyError(
                f"Template with name='{name}' and version='{version}' not found in library"
            )
        return asset

    def get_atlas(self, name: str, version: str) -> AtlasAsset:
        """Get an atlas by name and version."""
        asset = self._assets[Atlas.__name__].get((name, version))
        if asset is None:
            raise KeyError(
                f"Atlas with name='{name}' and version='{version}' not found in library"
            )
        return asset

    def get_terminology(self, name: str, version: str) -> AtlasAsset:
        """Get a terminology by name and version."""
        asset = self._assets[Terminology.__name__].get((name, version))
        if asset is None:
            raise KeyError(
                f"Terminology with name='{name}' and version='{version}' not found in library"
            )
        return asset

    def get_annotation_set(self, name: str, version: str) -> AtlasAsset:
        """Get an annotation set by name and version."""
        asset = self._assets[AnnotationSet.__name__].get((name, version))
        if asset is None:
            raise KeyError(
                f"AnnotationSet with name='{name}' and version='{version}' not found in library"
            )
        return asset

    @property
    def templates(self) -> List[AtlasAsset]:
        """List all templates."""
        return list(self._assets[Template.__name__].values())

    @property
    def annotation_sets(self) -> List[AtlasAsset]:
        """List all annotation sets."""
        return list(self._assets[AnnotationSet.__name__].values())

    @property
    def coordinate_transformations(self) -> List[AtlasAsset]:
        """List all coordinate transformations."""
        return list(self._assets[CoordinateTransformation.__name__].values())

    @property
    def atlases(self) -> List[AtlasAsset]:
        """List all atlases."""
        return list(self._assets[Atlas.__name__].values())

    @property
    def terminologies(self) -> List[AtlasAsset]:
        """List all terminologies."""
        return list(self._assets[Terminology.__name__].values())

    @property
    def coordinate_spaces(self) -> List[AtlasAsset]:
        """List all coordinate spaces."""
        return list(self._assets[CoordinateSpace.__name__].values())

    @property
    def all_assets(self) -> List[AtlasAsset]:
        """List all assets in the library."""
        all_assets = []
        for asset_dict in self._assets.values():
            all_assets.extend(asset_dict.values())
        return all_assets

    def remove_asset(self, name: str, version: str, asset_type: str) -> bool:
        """Remove an asset from the library. Returns True if removed, False if not found."""
        if asset_type not in self._assets:
            raise ValueError(f"Unknown asset type: {asset_type}")

        key = (name, version)
        if key in self._assets[asset_type]:
            del self._assets[asset_type][key]
            return True
        return False

    def asset_exists(self, name: str, version: str, asset_type: str) -> bool:
        """Check if an asset exists in the library."""
        if asset_type not in self._assets:
            return False
        return (name, version) in self._assets[asset_type]

    def get_asset_count(self) -> Dict[str, int]:
        """Get count of assets by type."""
        return {asset_type: len(assets) for asset_type, assets in self._assets.items()}

    def clear(self):
        """Remove all assets from the library."""
        for asset_dict in self._assets.values():
            asset_dict.clear()
