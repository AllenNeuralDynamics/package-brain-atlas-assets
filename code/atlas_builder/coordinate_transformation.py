"""Coordinate transformations between anatomical template spaces (moved)."""

from dataclasses import dataclass
from typing import ClassVar

from atlas_builder.template import Template
from atlas_builder.atlas_asset import AtlasAsset


@dataclass
class CoordinateTransformation(AtlasAsset):
    """Coordinate transformation between anatomical templates.

    Attributes:
        input_template: Source template space
        output_template: Target template space
    """

    input_template: Template
    output_template: Template

    _asset_location: ClassVar[str] = "coordinate-transformations"
    schema_version: ClassVar[str] = "0.1.0"

    @classmethod
    def init(
        cls,
        input_template: Template,
        output_template: Template,
        version: str,
    ):
        """Initialize coordinate transformation with auto-generated name.

        Returns:
            CoordinateTransformation: Initialized coordinate transformation instance
        """
        return cls(
            name=f"{input_template.name}-{input_template.version}_to_{output_template.name}-{output_template.version}",
            input_template=input_template,
            output_template=output_template,
            version=version,
        )

    @property
    def manifest(self) -> dict:
        """
        Generate manifest dictionary for this coordinate transformation.

        Returns:
            dict: Manifest containing transformation metadata including
                  input/output template information and location
        """
        return super().manifest | {
            "input_template": self.input_template.manifest,
            "output_template": self.output_template.manifest,
        }
