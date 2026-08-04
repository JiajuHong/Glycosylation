"""Regression tests for the O4_EXTERNAL role-channel ablation."""

from __future__ import annotations

import unittest

import torch

from layer1.train_hard_feasibility import (
    ACCEPTOR_ROLE_NAMES,
    CACHED_ACCEPTOR_ROLE_NAMES,
    HardFeasibilityDataset,
)


class O4ExternalRoleAblationTests(unittest.TestCase):
    def test_ablation_schema_removes_only_external_role(self) -> None:
        self.assertEqual(
            set(CACHED_ACCEPTOR_ROLE_NAMES) - set(ACCEPTOR_ROLE_NAMES),
            {"O4_EXTERNAL"},
        )

    def test_cached_role_projection_preserves_atoms_and_other_roles(self) -> None:
        dataset = HardFeasibilityDataset.__new__(HardFeasibilityDataset)
        dataset.stress = False
        dataset.acceptor_role_names = list(ACCEPTOR_ROLE_NAMES)
        full = torch.arange(
            3 * len(CACHED_ACCEPTOR_ROLE_NAMES), dtype=torch.float32
        ).reshape(3, len(CACHED_ACCEPTOR_ROLE_NAMES))
        dataset.cached_samples = {
            "sample": {
                "acceptor_site_role_matrix": full,
            }
        }

        metadata = dataset._metadata({"Sample_ID": "sample"})
        projected = metadata["acceptor_site_role_matrix"]

        self.assertEqual(
            tuple(projected.shape),
            (3, len(ACCEPTOR_ROLE_NAMES)),
        )
        self.assertTrue(
            torch.equal(
                projected,
                full[:, : len(ACCEPTOR_ROLE_NAMES)],
            )
        )


if __name__ == "__main__":
    unittest.main()
