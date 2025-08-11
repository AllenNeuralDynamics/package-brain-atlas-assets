"""Complete parcellation atlas combining anatomical space, annotations, and terminology."""

from dataclasses import dataclass

from allen_atlas_assets.anatomical_annotation_set import \
    AnatomicalAnnotationSet
from allen_atlas_assets.anatomical_space import AnatomicalSpace
from allen_atlas_assets.atlas_asset import AtlasAsset
from allen_atlas_assets.parcellation_terminology import ParcellationTerminology


@dataclass
class ParcellationAtlas(AtlasAsset):
    """Complete parcellation atlas with anatomical space, annotations, and terminology."""

    anatomical_space: AnatomicalSpace
    anatomical_annotation_set: AnatomicalAnnotationSet
    parcellation_terminology: ParcellationTerminology

    _asset_location = "parcellation-atlases"

    @property
    def manifest(self):
        return super().manifest | {
            "anatomical_space": self.anatomical_space.manifest,
            "anatomical_annotation_set": self.anatomical_annotation_set.manifest,
            "parcellation_terminology": self.parcellation_terminology.manifest,
        }
