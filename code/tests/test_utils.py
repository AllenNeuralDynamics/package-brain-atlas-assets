import math
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import zarr

from atlas_builder.annotation_set import AnnotationSet
from atlas_builder.template import Template
from atlas_builder.terminology import Terminology
from utils import convert_mhd_to_nifti, correct_coordinate_transforms_rfc5, decompose_affine


class FakeSimpleITKImage:
    def __init__(self, spacing, origin, direction):
        self._spacing = tuple(spacing)
        self._origin = tuple(origin)
        self._direction = tuple(direction)

    def GetSpacing(self):
        return self._spacing

    def SetSpacing(self, spacing):
        self._spacing = tuple(spacing)

    def GetOrigin(self):
        return self._origin

    def SetOrigin(self, origin):
        self._origin = tuple(origin)

    def GetDirection(self):
        return self._direction

    def SetDirection(self, direction):
        self._direction = tuple(direction)


def rotation_x(theta: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(theta), -math.sin(theta)],
            [0.0, math.sin(theta), math.cos(theta)],
        ]
    )


def rotation_y(theta: float) -> np.ndarray:
    return np.array(
        [
            [math.cos(theta), 0.0, math.sin(theta)],
            [0.0, 1.0, 0.0],
            [-math.sin(theta), 0.0, math.cos(theta)],
        ]
    )


def rotation_z(theta: float) -> np.ndarray:
    return np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def compose_affine(
    scale: np.ndarray | None,
    rotation: np.ndarray | None,
    flip: np.ndarray | None,
    translation: np.ndarray | None,
) -> np.ndarray:
    linear = np.eye(3)
    if rotation is not None:
        linear = rotation @ linear
    if flip is not None:
        linear = linear @ flip
    if scale is not None:
        linear = linear @ np.diag(scale)

    affine = np.eye(4)
    affine[:3, :3] = linear
    if translation is not None:
        affine[:3, 3] = translation
    return affine


def build_transform_list(
    scale: list[float],
    rotation: np.ndarray | None = None,
    flip: np.ndarray | None = None,
    translation: list[float] | None = None,
) -> list[dict]:
    transforms = [{"type": "scale", "scale": scale}]
    if flip is not None:
        transforms.append({"type": "affine", "affine": flip.tolist()})
    if rotation is not None:
        transforms.append({"type": "rotation", "rotation": rotation.tolist()})
    if translation is not None:
        transforms.append({"type": "translation", "translation": translation})
    return transforms


class DecomposeAffineTests(unittest.TestCase):
    def assert_recomposes(self, affine: np.ndarray) -> None:
        scale, rotation, flip, translation = decompose_affine(affine)
        recomposed = compose_affine(scale, rotation, flip, translation)

        self.assertTrue(np.allclose(recomposed, affine, atol=1e-6))
        if rotation is not None:
            self.assertTrue(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6))
            self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=6)
        if flip is not None:
            self.assertTrue(np.allclose(flip, np.diag(np.diag(flip)), atol=1e-6))
            self.assertTrue(np.all(np.isin(np.diag(flip), [-1.0, 1.0])))

    def test_recomposes_identity_affine(self) -> None:
        self.assert_recomposes(np.eye(4))

    def test_recomposes_rotation_flip_scale_translation_affine(self) -> None:
        rotation = rotation_z(math.radians(20.0)) @ rotation_y(math.radians(-35.0)) @ rotation_x(math.radians(15.0))
        flip = np.diag([-1.0, 1.0, -1.0])
        scale = np.array([0.025, 0.040, 0.050])
        translation = np.array([1.2, -3.4, 5.6])

        affine = np.eye(4)
        affine[:3, :3] = rotation @ flip @ np.diag(scale)
        affine[:3, 3] = translation

        self.assert_recomposes(affine)

    def test_recomposes_flip_and_scale_only_affine(self) -> None:
        affine = np.eye(4)
        affine[:3, :3] = np.diag([-0.01, 0.02, -0.03])
        affine[:3, 3] = np.array([-2.0, 4.5, 0.0])

        self.assert_recomposes(affine)


class ConvertMhdToNiftiTests(unittest.TestCase):
    @patch("utils.sitk.WriteImage")
    @patch("utils.sitk.ReadImage")
    def test_converts_spacing_and_origin_to_millimeters(self, mock_read_image, mock_write_image) -> None:
        image = FakeSimpleITKImage(
            spacing=(25.0, 25.0, 50.0),
            origin=(100.0, 200.0, 300.0),
            direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        mock_read_image.return_value = image

        convert_mhd_to_nifti("input.mhd", "output.nii.gz")

        self.assertEqual(image.GetSpacing(), (0.025, 0.025, 0.05))
        self.assertEqual(image.GetOrigin(), (0.1, 0.2, 0.3))
        self.assertEqual(image.GetDirection(), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        mock_write_image.assert_called_once_with(image, "output.nii.gz")

    @patch("utils.sitk.WriteImage")
    @patch("utils.sitk.ReadImage")
    def test_overrides_direction_when_requested(self, mock_read_image, mock_write_image) -> None:
        image = FakeSimpleITKImage(
            spacing=(10.0, 10.0, 10.0),
            origin=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        mock_read_image.return_value = image

        convert_mhd_to_nifti(
            "input.mhd",
            "output.nii.gz",
            output_direction=[0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0],
        )

        self.assertEqual(image.GetDirection(), (0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0))
        mock_write_image.assert_called_once_with(image, "output.nii.gz")

    @patch("utils.sitk.WriteImage")
    @patch("utils.sitk.ReadImage")
    def test_overrides_origin_when_requested(self, mock_read_image, mock_write_image) -> None:
        image = FakeSimpleITKImage(
            spacing=(10.0, 10.0, 10.0),
            origin=(100.0, 200.0, 300.0),
            direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        mock_read_image.return_value = image

        convert_mhd_to_nifti(
            "input.mhd",
            "output.nii.gz",
            output_origin=[0.0, 0.0, 0.0],
        )

        self.assertEqual(image.GetOrigin(), (0.0, 0.0, 0.0))
        mock_write_image.assert_called_once_with(image, "output.nii.gz")


class AnnotationSetCreateFromMhdTests(unittest.TestCase):
    @patch.object(AnnotationSet, "create")
    @patch("atlas_builder.annotation_set.nib.load")
    @patch("atlas_builder.annotation_set.convert_mhd_to_nifti")
    def test_forwards_output_direction_and_origin_to_converter(
        self,
        mock_convert_mhd_to_nifti,
        mock_nib_load,
        mock_create,
    ) -> None:
        terminology_df = pd.DataFrame(
            {
                "identifier": ["DMBA:1"],
                "parent_identifier": [""],
                "name": ["root"],
                "abbreviation": ["RT"],
            }
        )
        annotation_set = AnnotationSet(
            name="test-annotation",
            version="2012",
            template=Template(name="test-template", version="2012", scales=(25,)),
            terminology=Terminology(name="test-terms", version="2012", df=terminology_df),
            scales=(25,),
        )

        class FakeNibImage:
            affine = np.eye(4)

            def get_fdata(self):
                return np.ones((2, 2, 2), dtype=np.int16)

        mock_nib_load.return_value = FakeNibImage()
        direction = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0]
        origin = [0.0, 0.0, 0.0]

        with tempfile.TemporaryDirectory() as tempdir:
            annotation_set.create_from_mhd(
                "annotation.mhd",
                tempdir,
                include_meshes=False,
                output_direction=direction,
                output_origin=origin,
            )

        mock_convert_mhd_to_nifti.assert_called_once()
        _, kwargs = mock_convert_mhd_to_nifti.call_args
        self.assertEqual(kwargs["output_direction"], direction)
        self.assertEqual(kwargs["output_origin"], origin)
        mock_create.assert_called_once()

    def test_rejects_multi_scale_annotation_sets(self) -> None:
        terminology_df = pd.DataFrame(
            {
                "identifier": ["DMBA:1"],
                "parent_identifier": [""],
                "name": ["root"],
                "abbreviation": ["RT"],
            }
        )
        annotation_set = AnnotationSet(
            name="test-annotation",
            version="2012",
            template=Template(name="test-template", version="2012", scales=(10, 25)),
            terminology=Terminology(name="test-terms", version="2012", df=terminology_df),
            scales=(10, 25),
        )

        with tempfile.TemporaryDirectory() as tempdir:
            with self.assertRaisesRegex(ValueError, "single-scale"):
                annotation_set.create_from_mhd("annotation.mhd", tempdir)


class CorrectCoordinateTransformsRfc5Tests(unittest.TestCase):
    def make_group(self, shape, axes, transforms_by_path):
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)

        group = zarr.open_group(tempdir.name, mode="w")
        datasets = []
        for path, transforms in transforms_by_path.items():
            group.create_array(path, data=np.zeros(shape, dtype=np.float32))
            datasets.append(
                {
                    "path": path,
                    "coordinateTransformations": transforms,
                }
            )

        group.attrs.put(
            {
                "ome": {
                    "multiscales": [
                        {
                            "name": "test",
                            "axes": axes,
                            "datasets": datasets,
                        }
                    ]
                }
            }
        )
        return group

    def test_splits_dataset_scale_from_shared_world_transform(self) -> None:
        intrinsic_axes = [
            {"name": "z", "type": "space", "unit": "millimeter"},
            {"name": "y", "type": "space", "unit": "millimeter"},
            {"name": "x", "type": "space", "unit": "millimeter"},
        ]
        world_axes = [
            {"name": "z", "type": "space", "unit": "millimeter", "orientation": {"type": "anatomical", "value": "inferior-to-superior"}},
            {"name": "y", "type": "space", "unit": "millimeter", "orientation": {"type": "anatomical", "value": "posterior-to-anterior"}},
            {"name": "x", "type": "space", "unit": "millimeter", "orientation": {"type": "anatomical", "value": "left-to-right"}},
        ]
        rotation = rotation_z(math.radians(30.0))
        translation = [1.2, -3.4, 5.6]
        transforms_by_path = {
            "0": build_transform_list([0.01, 0.01, 0.01], rotation=rotation, translation=translation),
            "1": build_transform_list([0.02, 0.02, 0.02], rotation=rotation, translation=translation),
        }

        group = self.make_group((2, 2, 2), intrinsic_axes, transforms_by_path)

        correct_coordinate_transforms_rfc5(group, world_axes)

        ome = dict(group.attrs)["ome"]
        self.assertEqual([cs["name"] for cs in ome["coordinateSystems"]], ["intrinsic", "mm RAS"])

        multiscales = ome["multiscales"][0]
        self.assertEqual(multiscales["coordinateTransformations"][0]["input"], "intrinsic")
        self.assertEqual(multiscales["coordinateTransformations"][0]["output"], "mm RAS")
        self.assertEqual(
            [t["type"] for t in multiscales["coordinateTransformations"][0]["transformations"]],
            ["rotation", "translation"],
        )

        dataset_transform = multiscales["datasets"][0]["coordinateTransformations"][0]
        self.assertEqual(dataset_transform["input"], "0")
        self.assertEqual(dataset_transform["output"], "intrinsic")
        self.assertEqual(dataset_transform["transformations"], [{"type": "scale", "scale": [0.01, 0.01, 0.01]}])

        dataset_ome = dict(group["0"].attrs)["ome"]
        self.assertEqual(dataset_ome["coordinateTransformations"], multiscales["datasets"][0]["coordinateTransformations"])

    def test_preserves_channel_identity_for_4d_intrinsic_scale(self) -> None:
        intrinsic_axes = [
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]
        world_axes = [
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer", "orientation": {"type": "anatomical", "value": "dorsal-to-ventral"}},
            {"name": "y", "type": "space", "unit": "micrometer", "orientation": {"type": "anatomical", "value": "anterior-to-posterior"}},
            {"name": "x", "type": "space", "unit": "micrometer", "orientation": {"type": "anatomical", "value": "left-to-right"}},
        ]
        rotation = np.eye(4)
        rotation[1:4, 1:4] = rotation_y(math.radians(20.0))
        translation = [0.0, 12.0, -8.0, 1.5]
        transforms_by_path = {
            "0": build_transform_list([1.0, 10.0, 10.0, 10.0], rotation=rotation, translation=translation),
            "1": build_transform_list([1.0, 25.0, 25.0, 25.0], rotation=rotation, translation=translation),
        }

        group = self.make_group((2, 2, 2, 2), intrinsic_axes, transforms_by_path)

        correct_coordinate_transforms_rfc5(group, world_axes)

        dataset_transform = dict(group.attrs)["ome"]["multiscales"][0]["datasets"][1]["coordinateTransformations"][0]
        self.assertEqual(
            dataset_transform["transformations"],
            [{"type": "scale", "scale": [1.0, 25.0, 25.0, 25.0]}],
        )
        shared_transform = dict(group.attrs)["ome"]["multiscales"][0]["coordinateTransformations"][0]
        self.assertEqual(shared_transform["transformations"][0]["type"], "rotation")
        self.assertEqual(shared_transform["transformations"][1]["type"], "translation")

    def test_keeps_reflection_in_shared_world_transform(self) -> None:
        intrinsic_axes = [
            {"name": "z", "type": "space", "unit": "millimeter"},
            {"name": "y", "type": "space", "unit": "millimeter"},
            {"name": "x", "type": "space", "unit": "millimeter"},
        ]
        world_axes = [
            {"name": "z", "type": "space", "unit": "millimeter", "orientation": {"type": "anatomical", "value": "inferior-to-superior"}},
            {"name": "y", "type": "space", "unit": "millimeter", "orientation": {"type": "anatomical", "value": "posterior-to-anterior"}},
            {"name": "x", "type": "space", "unit": "millimeter", "orientation": {"type": "anatomical", "value": "left-to-right"}},
        ]

        affine = np.eye(4)
        affine[:3, :3] = np.diag([-0.01, 0.02, 0.03])
        affine[:3, 3] = np.array([-2.0, 4.5, 0.0])
        scale, rotation, flip, translation = decompose_affine(affine)

        transforms_by_path = {
            "0": build_transform_list(
                scale.tolist(),
                rotation=rotation,
                flip=flip,
                translation=translation.tolist(),
            )
        }

        group = self.make_group((2, 2, 2), intrinsic_axes, transforms_by_path)

        correct_coordinate_transforms_rfc5(group, world_axes)

        shared_transform = dict(group.attrs)["ome"]["multiscales"][0]["coordinateTransformations"][0]
        self.assertIn("affine", [transform["type"] for transform in shared_transform["transformations"]])
        self.assertEqual(shared_transform["transformations"][-1]["type"], "translation")
        self.assertEqual(shared_transform["output"], "mm RAS")


if __name__ == "__main__":
    unittest.main()