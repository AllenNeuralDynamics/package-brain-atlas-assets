"""Main orchestration script for packaging brain atlas data assets."""

# Requirements: nibabel, ome-zarr, zarr

import argparse
import atexit
import logging
import shutil
import sys
from pathlib import Path

import CCFv3
import CCFv2020
import CCFv2026
import HMBA
import devmouse
import devmouse_spim
import SmartSPIM
import idisco
from atlas_builder import AssetLibrary


def clear_directory(path, keep_files=None):
    """Clear out the contents of the specified directory."""
    keep_files = set(keep_files or [])
    logging.info(f"Clearing contents of directory: {path}")
    if path.exists():
        # Remove the contents of the directory, but not the directory itself
        for item in path.iterdir():
            if item.is_file():
                if item.name in keep_files:
                    logging.info(f"Keeping file: {item}")
                    continue
                logging.info(f"Removing file: {item}")
                item.unlink()
                logging.info(f"Removed file: {item}")
            elif item.is_dir():
                logging.info(f"Removing directory (may take a while): {item}")
                shutil.rmtree(item)
                logging.info(f"Removed directory: {item}")
    else:
        # Create the directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {path}")

    logging.info(f"Directory cleared: {path}")


def configure_output_logging(results_dir: Path) -> Path:
    """Redirect stdout/stderr and logging output to results_dir/output.log."""
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "output.log"
    log_file = open(log_path, "w", buffering=1)
    atexit.register(log_file.close)

    class _TeeStream:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for stream in self.streams:
                stream.write(data)
                stream.flush()

        def flush(self):
            for stream in self.streams:
                stream.flush()

    tee_stdout = _TeeStream(sys.stdout, log_file)
    tee_stderr = _TeeStream(sys.stderr, log_file)

    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(tee_stdout)],
        force=True,
    )
    return log_path


def main(results_dir: str | Path):
    """
    Main function that orchestrates the atlas packaging pipeline.

    Processes multiple atlas datasets including CCF 2020, CCF 3, SmartSPIM,
    iDISCO, HMBA, and DevMouse templates/annotations, creating standardized
    data packages with anatomical templates, annotation sets, and coordinate
    transformations.

    Parameters
    ----------
    results_dir : str | Path
        Destination directory for packaged atlas assets.
    """
    scales = (10, 25, 50, 100)
    hmba_scales = {"hcp": (700,), "mac25": (160,), "riken25": (70,), "icbm": (500,)}

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    abc_dir = Path("/data/abc_atlas")
    ccf3_dir = Path("/data/allen_mouse_ccf")
    smartspim_dir = Path("/data/smartspim_lca_template_opendata")
    smartspim_remapped_dir = Path("/data/remapped_ccf_labels_on_spim_LCA_template")
    devmouse_dir = Path("/data/devmouse-atlas-assets")
    devmouse_lsfm_dir = Path("/data/devmouse-lsfm")
    hmba_dir = Path("/data/hmba")
    # Allow caller to override results directory
    results_dir = Path(results_dir)

    configure_output_logging(results_dir)

    # Initialize asset library
    library = AssetLibrary()

    # Clear the entire results directory
    clear_directory(results_dir, keep_files={"output.log"})

    ## Package DevMouse atlas assets first
    devmouse.package_devmouse(devmouse_dir, results_dir, library, include_meshes=True)
    devmouse_spim.package_devmouse(devmouse_lsfm_dir, results_dir, library)
    
    # Package CCF 3 legacy annotations and templates
    CCFv3.package_ccf(ccf3_dir, results_dir, library, scales=scales)

    # Package CCF 2020 annotations
    CCFv2020.package_ccf2020(abc_dir, results_dir, library, scales=(10, 25)) 

    # Package CCF 2026-03 terminology and annotations (hemisphere masks)
    CCFv2026.package_ccf2026(abc_dir, results_dir, library, scales=(10, 25))

    # Package SmartSPIM template (uses library for assets)
    SmartSPIM.package_smartspim_template(
        smartspim_dir, smartspim_remapped_dir, results_dir, library, scales=scales
    )

    # Package iDISCO template
    # no idisco for now
    # idisco.package_idisco_template(results_dir)       

    # Package HMBA atlases
    HMBA.package_ccf(hmba_dir, results_dir, library, scales=hmba_scales)

    # Re-write manifests now that all cross-asset associations are complete.
    # Templates associate their coordinate_space after Template.create() has
    # already written manifest.json, so re-render here to capture it.
    for a in library.coordinate_spaces:
        a.create_manifest(results_dir)

    for a in library.templates:
        a.create_manifest(results_dir)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Package brain atlas data assets.")
    parser.add_argument(
        "--results-dir",
        "-r",
        type=Path,
        default=Path("/results"),
        help="Directory to write packaged atlas assets (default: /results)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(results_dir=args.results_dir)
