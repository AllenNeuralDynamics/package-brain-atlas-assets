"""Allen Atlas Builder package for brain atlas data management (renamed from atlas_assets)."""

from atlas_builder.annotation_set import AnnotationSet
from atlas_builder.coordinate_space import CoordinateSpace
from atlas_builder.template import Template
from atlas_builder.asset_library import AssetLibrary
from atlas_builder.atlas_asset import AtlasAsset
from atlas_builder.coordinate_transformation import CoordinateTransformation
from atlas_builder.mesh import Mesh
from atlas_builder.atlas import Atlas
from atlas_builder.terminology import Terminology

__all__ = [
    "AtlasAsset",
    "AnnotationSet",
    "CoordinateSpace",
    "Template",
    "AssetLibrary",
    "CoordinateTransformation",
    "Mesh",
    "Atlas",
    "Terminology",
]
