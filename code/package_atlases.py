"""Main orchestration script for packaging brain atlas data assets."""

# Requirements: nibabel, ome-zarr, zarr

import logging
import shutil
from pathlib import Path

import CCFv3
import CCFv2020
import SmartSPIM
import devmouse
from allen_atlas_assets import AssetLibrary


def clear_directory(path):
    """Clear out the contents of the specified directory."""
    logging.info(f"Clearing contents of directory: {path}")
    if path.exists():
        # Remove the contents of the directory, but not the directory itself
        for item in path.iterdir():
            if item.is_file():
                item.unlink()
                logging.info(f"Removed file: {item}")
            elif item.is_dir():
                shutil.rmtree(item)
                logging.info(f"Removed directory: {item}")
    else:
        # Create the directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {path}")

    logging.info(f"Directory cleared: {path}")


def main():
    """
    Main function that orchestrates the atlas packaging pipeline.

    Processes multiple atlas datasets including CCF 2020, CCF 3, and SmartSPIM
    templates, creating standardized data packages with anatomical templates,
    annotation sets, and coordinate transformations.
    """
    scales = (10, 25, 50, 100)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    abc_dir = Path("/data/abc_atlas")
    ccf3_dir = Path("/data/allen_mouse_ccf")
    smartspim_dir = Path("/data/smartspim_lca_template_opendata")
    devmouse_dir = Path("/data/devmouse-atlas-assets")
    results_dir = Path("/results")

    # Initialize asset library
    library = AssetLibrary()

    # Clear the entire results directory
    clear_directory(results_dir)

    # Package DevMouse atlas assets first
    devmouse.package_devmouse(devmouse_dir, results_dir, library)

    # Package CCF 3 legacy annotations and templates
    CCFv3.package_ccf(ccf3_dir, results_dir, library, scales=scales)

    # Package CCF 2020 annotations 
    CCFv2020.package_ccf2020(abc_dir, results_dir, library, scales=(10,))

    # Package SmartSPIM template (uses library for assets)
    SmartSPIM.package_smartspim_template(smartspim_dir, results_dir, library, scales=scales)

    for a in library.anatomical_spaces:
        a.create_manifest(results_dir)


if __name__ == "__main__":
    main()
