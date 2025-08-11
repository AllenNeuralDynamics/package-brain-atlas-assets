"""Asset library for managing and retrieving atlas data assets."""

from typing import Dict, List, Tuple

from allen_atlas_assets.anatomical_annotation_set import \
    AnatomicalAnnotationSet
from allen_atlas_assets.anatomical_space import AnatomicalSpace
from allen_atlas_assets.anatomical_template import AnatomicalTemplate
from allen_atlas_assets.atlas_asset import AtlasAsset
from allen_atlas_assets.coordinate_transformation import \
    CoordinateTransformation
from allen_atlas_assets.parcellation_atlas import ParcellationAtlas
from allen_atlas_assets.parcellation_terminology import ParcellationTerminology


class AssetLibrary:
    """Central registry for managing and retrieving atlas data assets."""

    def __init__(self):
        # Store assets by type and (name, version) key for efficient lookup
        self._assets: Dict[str, Dict[Tuple[str, str], AtlasAsset]] = {
            AnatomicalTemplate.__name__: {},
            AnatomicalAnnotationSet.__name__: {},
            CoordinateTransformation.__name__: {},
            ParcellationAtlas.__name__: {},
            ParcellationTerminology.__name__: {},
            AnatomicalSpace.__name__: {},
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

    def get_anatomical_space(self, name: str, version: str) -> AtlasAsset:
        """Get an anatomical space by name and version."""
        asset = self._assets[AnatomicalSpace.__name__].get((name, version))
        if asset is None:
            raise KeyError(
                f"AnatomicalSpace with name='{name}' and version='{version}' not found in library"
            )
        return asset

    def get_coordinate_transformation(self, name: str, version: str) -> AtlasAsset:
        """Get a coordinate transformation by name and version."""
        asset = self._assets[CoordinateTransformation.__name__].get((name, version))
        if asset is None:
            raise KeyError(
                f"CoordinateTransformation with name='{name}' and version='{version}' not found in library"
            )
        return asset

    def get_anatomical_template(self, name: str, version: str) -> AtlasAsset:
        """Get an anatomical template by name and version."""
        asset = self._assets[AnatomicalTemplate.__name__].get((name, version))
        if asset is None:
            raise KeyError(
                f"AnatomicalTemplate with name='{name}' and version='{version}' not found in library"
            )
        return asset

    def get_parcellation_atlas(self, name: str, version: str) -> AtlasAsset:
        """Get a parcellation atlas by name and version."""
        asset = self._assets[ParcellationAtlas.__name__].get((name, version))
        if asset is None:
            raise KeyError(
                f"ParcellationAtlas with name='{name}' and version='{version}' not found in library"
            )
        return asset

    def get_parcellation_terminology(self, name: str, version: str) -> AtlasAsset:
        """Get a parcellation terminology by name and version."""
        asset = self._assets[ParcellationTerminology.__name__].get((name, version))
        if asset is None:
            raise KeyError(
                f"ParcellationTerminology with name='{name}' and version='{version}' not found in library"
            )
        return asset

    def get_anatomical_annotation_set(self, name: str, version: str) -> AtlasAsset:
        """Get an anatomical annotation set by name and version."""
        asset = self._assets[AnatomicalAnnotationSet.__name__].get((name, version))
        if asset is None:
            raise KeyError(
                f"AnatomicalAnnotationSet with name='{name}' and version='{version}' not found in library"
            )
        return asset

    @property
    def anatomical_templates(self) -> List[AtlasAsset]:
        """List all anatomical templates."""
        return list(self._assets[AnatomicalTemplate.__name__].values())

    @property
    def anatomical_annotation_sets(self) -> List[AtlasAsset]:
        """List all anatomical annotation sets."""
        return list(self._assets[AnatomicalAnnotationSet.__name__].values())

    @property
    def coordinate_transformations(self) -> List[AtlasAsset]:
        """List all coordinate transformations."""
        return list(self._assets[CoordinateTransformation.__name__].values())

    @property
    def parcellation_atlases(self) -> List[AtlasAsset]:
        """List all parcellation atlases."""
        return list(self._assets[ParcellationAtlas.__name__].values())

    @property
    def parcellation_terminologies(self) -> List[AtlasAsset]:
        """List all parcellation terminologies."""
        return list(self._assets[ParcellationTerminology.__name__].values())

    @property
    def anatomical_spaces(self) -> List[AtlasAsset]:
        """List all anatomical spaces."""
        return list(self._assets[AnatomicalSpace.__name__].values())

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
