import unittest

import pandas as pd

from atlas_builder.multires_mesh import build_hierarchy_remap_tables


def term(identifier, value, parent, descendants, root_path=None):
    row = dict(
        identifier=identifier,
        annotation_value=value,
        parent_identifier=parent,
        descendant_annotation_values=descendants,
    )
    if root_path is not None:
        row["root_identifier_path"] = root_path
    return row


# root -> {100, 200}; 100 -> {11, 12}; 200 -> {21}
THREE_LEVELS = [
    term("S:1000", 1000, "", [11, 12, 21]),
    term("S:100", 100, "S:1000", [11, 12]),
    term("S:200", 200, "S:1000", [21]),
    term("S:11", 11, "S:100", [11]),
    term("S:12", 12, "S:100", [12]),
    term("S:21", 21, "S:200", [21]),
]


class BuildHierarchyRemapTablesTests(unittest.TestCase):
    def test_groups_structures_by_depth(self) -> None:
        tables = build_hierarchy_remap_tables(pd.DataFrame(THREE_LEVELS))

        self.assertEqual(len(tables), 3)
        # Every leaf label maps to the one structure at that depth that contains it.
        self.assertEqual(tables[0], {11: 1000, 12: 1000, 21: 1000})
        self.assertEqual(tables[1], {11: 100, 12: 100, 21: 200})
        self.assertEqual(tables[2], {11: 11, 12: 12, 21: 21})

    def test_every_structure_appears_exactly_once(self) -> None:
        tables = build_hierarchy_remap_tables(pd.DataFrame(THREE_LEVELS))

        # Label sets must be disjoint across passes; that is what lets every pass write
        # into a single mesh directory without overwriting another pass's fragments.
        emitted = [value for table in tables for value in set(table.values())]
        self.assertEqual(sorted(emitted), [11, 12, 21, 100, 200, 1000])
        self.assertEqual(len(emitted), len(set(emitted)))

    def test_prefers_precomputed_root_identifier_path(self) -> None:
        # parent_identifier is deliberately wrong here; root_identifier_path must win.
        rows = [
            term("S:1000", 1000, "S:BOGUS", [11], root_path=["S:1000"]),
            term("S:11", 11, "S:BOGUS", [11], root_path=["S:1000", "S:11"]),
        ]

        tables = build_hierarchy_remap_tables(pd.DataFrame(rows))

        self.assertEqual(tables, [{11: 1000}, {11: 11}])

    def test_falls_back_to_walking_parents(self) -> None:
        rows = [r.copy() for r in THREE_LEVELS]
        tables = build_hierarchy_remap_tables(pd.DataFrame(rows))

        self.assertNotIn("root_identifier_path", pd.DataFrame(rows).columns)
        self.assertEqual([sorted(set(t.values())) for t in tables], [[1000], [100, 200], [11, 12, 21]])

    def test_skips_structures_without_an_annotation_value(self) -> None:
        rows = THREE_LEVELS + [term("S:none", None, "S:1000", [])]

        tables = build_hierarchy_remap_tables(pd.DataFrame(rows))

        self.assertEqual(sorted(set(tables[1].values())), [100, 200])

    def test_rejects_overlapping_descendants_within_a_depth(self) -> None:
        # Two structures at the same depth claiming the same leaf would make the remap
        # ambiguous, so it must fail rather than silently pick one.
        rows = [
            term("S:a", 1, "", [99]),
            term("S:b", 2, "", [99]),
        ]

        with self.assertRaisesRegex(ValueError, "disjoint descendants"):
            build_hierarchy_remap_tables(pd.DataFrame(rows))

    def test_requires_descendant_annotation_values(self) -> None:
        rows = pd.DataFrame([{"identifier": "S:1", "annotation_value": 1, "parent_identifier": ""}])

        with self.assertRaisesRegex(ValueError, "descendant_annotation_values"):
            build_hierarchy_remap_tables(rows)


if __name__ == "__main__":
    unittest.main()
