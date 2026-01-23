"""Brain structure hierarchy and terminology management."""

import logging
from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from atlas_builder.atlas_asset import AtlasAsset


@dataclass
class Terminology(AtlasAsset):
    """Hierarchical brain structure terminology manager.

    Attributes:
        df: DataFrame with identifier, parent_identifier, name, abbreviation, descendant_identifiers columns
    """

    df: pd.DataFrame = None

    _asset_location: ClassVar[str] = "terminologies"
    schema_version: ClassVar[str] = "0.1.0"

    def __post_init__(self):
        """Initialize terminology and precompute descendant relationships."""
        # Make a copy of the DataFrame since we will be editing it
        self.df = self.df.copy()

        if (
            "identifier" not in self.df.columns
            or "parent_identifier" not in self.df.columns
            or "name" not in self.df.columns
            or "abbreviation" not in self.df.columns
        ):
            raise ValueError(
                "df must contain 'identifier', 'annotation_identifier', 'parent_identifier', 'name', and 'abbreviation' columns."
            )

        # Precompute all descendants
        logging.info("Pre-computing descendant lists for all terms in terminology...")

        # Add descendant_identifiers column to DataFrame (includes self)
        self.df["descendant_identifiers"] = self.df["identifier"].apply(self._compute_descendant_identifiers)

        logging.info(f"Pre-computed descendants for {len(self.df)} terms")

        # Precompute root-to-node paths (list of identifiers from root to current term)
        logging.info("Pre-computing root_identifier_path for all terms in terminology...")
        self.df["root_identifier_path"] = self.df["identifier"].apply(self._compute_root_identifier_path)
        logging.info("Pre-computed root_identifier_path for all terms")

    def _compute_descendant_identifiers(self, identifier):
        """Recursively compute all descendant identifiers for a given identifier (including self)."""
        descendants = []

        # Add the identifier itself if it exists
        if identifier in self.df["identifier"].values:
            descendants.append(identifier)

        # Find all direct children (terms whose parent_identifier matches this identifier)
        children_rows = self.df[self.df["parent_identifier"] == identifier]
        children = children_rows["identifier"].tolist()

        # Recursively find descendants of each child
        for child_identifier in children:
            child_descendants = self._compute_descendant_identifiers(child_identifier)
            descendants.extend(child_descendants)

        # Remove duplicates while preserving order
        seen = set()
        unique_descendants = []
        for desc_identifier in descendants:
            if desc_identifier not in seen:
                seen.add(desc_identifier)
                unique_descendants.append(desc_identifier)

        return unique_descendants

    def _compute_root_identifier_path(self, identifier):
        """Compute the path of identifiers from the root to the provided identifier.

        The path includes the identifier itself as the last element. Root is inferred
        by walking parent_identifier links until a parent is missing or null.
        """        

        # Build upward chain: current -> parent -> ... -> root
        chain = []
        current = identifier

        # Index for quick lookups
        id_to_parent = dict(zip(self.df["identifier"], self.df["parent_identifier"]))

        # Guard against cycles: limit to number of rows
        max_steps = len(self.df) + 1
        steps = 0
        while current is not None and steps < max_steps:
            chain.append(current)
            parent = id_to_parent.get(current, None)
            # Treat NaN as no parent
            if pd.isna(parent):
                parent = None
            # If parent not found in id_to_parent, consider current as root
            if parent is not None and parent not in id_to_parent:
                # Parent reference points outside known identifiers; stop here
                parent = None
            current = parent
            steps += 1

        # If we exceeded max_steps, we likely hit a cycle; fall back to just the identifier
        if steps >= max_steps:
            logging.warning(
                "Cycle detected or excessive parent chain for identifier %s; falling back to self-only path",
                identifier,
            )
            chain = [identifier]

        # Reverse to be from root -> ... -> identifier
        chain.reverse()
        return chain

    def get_descendants(self, identifier, include_self=True):
        """
        Get all descendant rows for a given identifier.

        This method provides efficient access to descendant relationships using
        the precomputed descendant mapping stored in the DataFrame.

        Args:
            identifier: The identifier to find descendants for
            include_self: Whether to include the identifier itself in the results (default: True)

        Returns:
            pd.DataFrame: DataFrame containing all descendant rows (optionally including the identifier itself)
        """
        row = self.df[self.df["identifier"] == identifier]
        if row.empty:
            return None

        descendant_identifiers = row.iloc[0]["descendant_identifiers"]

        if not include_self:
            # Remove the identifier itself from the list
            descendant_identifiers = [desc_id for desc_id in descendant_identifiers if desc_id != identifier]

        return self.df[self.df["identifier"].isin(descendant_identifiers)]

    def _set_column_values(self, column_name, values):
        """
        Private helper to add or update a column in the DataFrame.

        Args:
            column_name: Name of the column to set.
            values: Values to set for the column. Can be:
                   - A single value (applied to all rows)
                   - A list/array of values (must match DataFrame length)
                   - A dictionary mapping identifiers to values
                   - A callable that takes each row and returns values
        """
        if isinstance(values, dict):
            self.df[column_name] = self.df["identifier"].map(values)
        elif callable(values):
            self.df[column_name] = self.df.apply(values, axis=1)
        else:
            self.df[column_name] = values

    def set_descendant_annotation_values(self, values):
        """
        Add or update the descendant_annotation_values column in the DataFrame.
        """
        self._set_column_values("descendant_annotation_values", values)

    def set_term_set_name(self, values):
        """
        Add or update the term_set_name column in the DataFrame.
        """
        self._set_column_values("term_set_name", values)

    def write_terminology(self, output_root):
        output_dir = self.location(output_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_output_path = output_dir / "terminology.csv"
        parquet_output_path = output_dir / "terminology.parquet"
        self.df.to_csv(csv_output_path, index=False)
        self.df.to_parquet(parquet_output_path, index=False)
        logging.info(f"Terminology written to {csv_output_path}")
