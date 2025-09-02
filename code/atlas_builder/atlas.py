"""Complete atlas combining coordinate space, annotations, and terminology."""

from dataclasses import dataclass

from atlas_builder.annotation_set import AnnotationSet
from atlas_builder.coordinate_space import CoordinateSpace
from atlas_builder.atlas_asset import AtlasAsset
from atlas_builder.terminology import Terminology


@dataclass
class Atlas(AtlasAsset):
    """Complete atlas with coordinate space, annotations, and terminology."""

    coordinate_space: CoordinateSpace
    annotation_set: AnnotationSet
    terminology: Terminology

    _asset_location = "atlases"

    @property
    def manifest(self):
        return super().manifest | {
            "coordinate_space": self.coordinate_space.manifest,
            "annotation_set": self.annotation_set.manifest,
            "terminology": self.terminology.manifest,
        }
