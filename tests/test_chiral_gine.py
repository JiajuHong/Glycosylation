from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.data import Batch

from chiral_graph import mol_to_chiral_pyg_graph
from glyco_dataset import (
    GlycoDataset,
    atom_features,
    bond_features,
    glyco_collate_fn,
    mol_from_smiles,
)
from models.chiral_gine_encoder import ChiralGINEEncoder
from models.gine_encoder import GINEEncoder
from models.glyco_gine_models import (
    GlycoGINECrossAttn,
    GlycoGINECrossAttnTri,
    build_glyco_gine_model,
)
from models.local_modules import RFUOHCrossAttentionTri
from models.tetra_permutation import EVEN_TETRAHEDRAL_PERMUTATIONS, PermCatAggregator
from train_gine import (
    apply_encoder_training_defaults,
    get_feature_dims,
    make_optimizer,
    make_scheduler,
)


ROOT = Path(__file__).resolve().parents[1]


def make_chiral_graph(smiles: str):
    return mol_to_chiral_pyg_graph(
        mol_from_smiles(smiles),
        smiles,
        atom_feature_fn=atom_features,
        bond_feature_fn=bond_features,
    )


class ChiralGraphTests(unittest.TestCase):
    def test_explicit_hydrogen_preserves_heavy_atom_indices(self):
        graph = make_chiral_graph("F[C@H](Cl)Br")

        self.assertEqual(graph.num_heavy_atoms, 4)
        self.assertEqual(graph.num_nodes, 5)
        self.assertEqual(graph.heavy_atom_mask.tolist(), [True, True, True, True, False])
        self.assertEqual(graph.atom_idx.tolist(), list(range(5)))
        self.assertEqual(graph.tetra_center_index.tolist(), [1])
        self.assertEqual(graph.tetra_neighbor_index.shape, (1, 4))
        self.assertIn(4, graph.tetra_neighbor_index[0].tolist())
        self.assertIn(int(graph.parity_atoms[1]), (-1, 1))

    def test_batch_offsets_node_and_edge_indices(self):
        first = make_chiral_graph("F[C@H](Cl)Br")
        second = make_chiral_graph("F[C@@H](Cl)Br")
        batch = Batch.from_data_list([first, second])

        self.assertEqual(batch.tetra_neighbor_index.shape, (2, 4))
        self.assertEqual(batch.tetra_edge_index.shape, (2, 4))
        self.assertEqual(batch.tetra_center_index[1].item(), first.num_nodes + 1)
        self.assertTrue((batch.tetra_neighbor_index[1] >= first.num_nodes).all())
        self.assertTrue((batch.tetra_edge_index[1] >= first.num_edges).all())

    def test_nonchiral_graph_has_empty_tetrahedral_metadata(self):
        graph = make_chiral_graph("CCO")
        self.assertEqual(graph.tetra_center_index.shape, (0,))
        self.assertEqual(graph.tetra_neighbor_index.shape, (0, 4))
        self.assertEqual(graph.tetra_edge_index.shape, (0, 4))


class PermCatTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.aggregator = PermCatAggregator(hidden_dim=8, dropout=0.0).eval()
        self.messages = torch.randn(3, 4, 8)

    def test_is_invariant_to_all_even_tetrahedral_permutations(self):
        expected = self.aggregator(self.messages)
        for permutation in EVEN_TETRAHEDRAL_PERMUTATIONS:
            actual = self.aggregator(self.messages[:, permutation, :])
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_is_sensitive_to_an_odd_permutation(self):
        expected = self.aggregator(self.messages)
        mirrored = self.aggregator(self.messages[:, [1, 0, 2, 3], :])
        self.assertFalse(torch.allclose(expected, mirrored))

    def test_mean_normalization_is_scaled_reference(self):
        mean_aggregator = PermCatAggregator(
            hidden_dim=8, dropout=0.0, normalization="mean"
        ).eval()
        mean_aggregator.load_state_dict(self.aggregator.state_dict())
        self.aggregator.output_mlp = torch.nn.Identity()
        mean_aggregator.output_mlp = torch.nn.Identity()
        reference = self.aggregator(self.messages)
        mean = mean_aggregator(self.messages)
        torch.testing.assert_close(mean * 4.0, reference)


class TrainingStabilityTests(unittest.TestCase):
    class Args:
        lr = 3e-4
        weight_decay = 1e-4
        perm_cat_lr_scale = 0.5
        lr_scheduler = "cosine"
        warmup_epochs = 2
        epochs = 10

    @staticmethod
    def make_model():
        model = torch.nn.Module()
        model.base = torch.nn.Linear(8, 8)
        model.perm_cat = PermCatAggregator(hidden_dim=8)
        return model

    def test_perm_cat_uses_its_own_learning_rate(self):
        model = self.make_model()
        optimizer = make_optimizer(model, self.Args())
        rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
        self.assertEqual(rates, {"base": 3e-4, "perm_cat": 1.5e-4})

    def test_warmup_and_cosine_scheduler_preserve_group_ratio(self):
        model = self.make_model()
        optimizer = make_optimizer(model, self.Args())
        scheduler = make_scheduler(optimizer, self.Args())
        initial = {group["name"]: group["lr"] for group in optimizer.param_groups}
        self.assertAlmostEqual(initial["base"], 1.5e-4)
        self.assertAlmostEqual(initial["perm_cat"], 0.75e-4)
        optimizer.step()
        scheduler.step()
        warmed = {group["name"]: group["lr"] for group in optimizer.param_groups}
        self.assertAlmostEqual(warmed["base"], 3e-4)
        self.assertAlmostEqual(warmed["perm_cat"], 1.5e-4)

    @staticmethod
    def make_default_args(encoder_type):
        return SimpleNamespace(
            encoder_type=encoder_type,
            deterministic=None,
            perm_cat_dropout=None,
            perm_cat_normalization=None,
            perm_cat_lr_scale=None,
            gradient_clip=None,
            warmup_epochs=None,
            lr_scheduler=None,
            patience=None,
            epochs=200,
        )

    def test_chiral_gine_automatically_uses_validated_stable_defaults(self):
        args = self.make_default_args("chiral_gine")
        apply_encoder_training_defaults(args)
        self.assertEqual(args.training_defaults, "chiral_stable_v1")
        self.assertTrue(args.deterministic)
        self.assertEqual(args.perm_cat_dropout, 0.1)
        self.assertEqual(args.perm_cat_normalization, "reference")
        self.assertEqual(args.perm_cat_lr_scale, 0.5)
        self.assertEqual(args.gradient_clip, 2.0)
        self.assertEqual(args.warmup_epochs, 10)
        self.assertEqual(args.lr_scheduler, "cosine")
        self.assertEqual(args.patience, 40)

    def test_ordinary_gine_keeps_original_training_defaults(self):
        args = self.make_default_args("gine")
        apply_encoder_training_defaults(args)
        self.assertEqual(args.training_defaults, "gine_original")
        self.assertFalse(args.deterministic)
        self.assertIsNone(args.perm_cat_dropout)
        self.assertEqual(args.perm_cat_lr_scale, 1.0)
        self.assertEqual(args.gradient_clip, 5.0)
        self.assertEqual(args.warmup_epochs, 0)
        self.assertEqual(args.lr_scheduler, "none")
        self.assertEqual(args.patience, 20)

    def test_explicit_chiral_override_is_preserved(self):
        args = SimpleNamespace(
            encoder_type="chiral_gine",
            deterministic=False,
            perm_cat_dropout=0.25,
            perm_cat_normalization="mean",
            perm_cat_lr_scale=0.75,
            gradient_clip=3.0,
            warmup_epochs=4,
            lr_scheduler="none",
            patience=25,
            epochs=200,
        )
        apply_encoder_training_defaults(args)
        self.assertFalse(args.deterministic)
        self.assertEqual(args.perm_cat_dropout, 0.25)
        self.assertEqual(args.perm_cat_normalization, "mean")
        self.assertEqual(args.perm_cat_lr_scale, 0.75)
        self.assertEqual(args.gradient_clip, 3.0)
        self.assertEqual(args.warmup_epochs, 4)
        self.assertEqual(args.lr_scheduler, "none")
        self.assertEqual(args.patience, 25)


class LocalInteractionTests(unittest.TestCase):
    def test_three_output_block_shapes_masks_and_gradients(self):
        torch.manual_seed(11)
        block = RFUOHCrossAttentionTri(hidden_dim=16, num_heads=4, dropout=0.0)
        h_d = torch.randn(2, 4, 16, requires_grad=True)
        h_a = torch.randn(2, 3, 16, requires_grad=True)
        d_mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
        a_mask = torch.tensor([[True, True, True], [True, True, False]])
        output = block(
            h_d,
            h_a,
            d_mask,
            a_mask,
            donor_c1_local_pos=torch.tensor([0, 1]),
            acceptor_o4_local_pos=torch.tensor([1, 0]),
        )

        self.assertEqual(output.z_d_local.shape, (2, 16))
        self.assertEqual(output.z_a_local.shape, (2, 16))
        self.assertEqual(output.z_int.shape, (2, 16))
        self.assertTrue((output.h_d_updated[1, 3] == 0).all())
        self.assertTrue((output.h_a_updated[1, 2] == 0).all())
        self.assertIsNot(block.d_from_a_attention, block.a_from_d_attention)

        loss = output.z_d_local.sum() + output.z_a_local.sum() + output.z_int.sum()
        loss.backward()
        self.assertIsNotNone(h_d.grad)
        self.assertIsNotNone(h_a.grad)
        self.assertTrue(
            all(parameter.grad is not None for parameter in block.parameters())
        )


class ModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = GlycoDataset(
            ROOT / "data/processed/glyco_model_local.csv",
            split="train",
            encoder_type="chiral_gine",
            chiral_graph_cache_path=None,
            validate=False,
        )

    def test_encoder_selection_keeps_ordinary_gine_available(self):
        atom_dim, bond_dim, num_solvent, num_catalyst = get_feature_dims(self.dataset)
        ordinary = build_glyco_gine_model(
            "global",
            atom_dim,
            bond_dim,
            num_solvent,
            num_catalyst,
            hidden_dim=16,
            num_layers=1,
            encoder_type="gine",
        )
        chiral = build_glyco_gine_model(
            "global",
            atom_dim,
            bond_dim,
            num_solvent,
            num_catalyst,
            hidden_dim=16,
            num_layers=1,
            encoder_type="chiral_gine",
        )
        self.assertIsInstance(ordinary.donor_encoder, GINEEncoder)
        self.assertIsInstance(chiral.donor_encoder, ChiralGINEEncoder)

    def test_all_existing_model_variants_support_forward_and_backward(self):
        batch = glyco_collate_fn([self.dataset[0], self.dataset[1]])
        atom_dim, bond_dim, num_solvent, num_catalyst = get_feature_dims(self.dataset)

        for model_type in ("global", "local", "crossattn", "crossattn_tri"):
            with self.subTest(model_type=model_type):
                model = build_glyco_gine_model(
                    model_type,
                    atom_dim,
                    bond_dim,
                    num_solvent,
                    num_catalyst,
                    hidden_dim=16,
                    num_layers=1,
                    dropout=0.0,
                    encoder_type="chiral_gine",
                )
                logits = model(batch)
                self.assertEqual(logits.shape, (2, 2))
                torch.nn.functional.cross_entropy(logits, batch["label"]).backward()
                self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
                perm_cat_parameters = [
                    parameter
                    for name, parameter in model.named_parameters()
                    if "perm_cat" in name
                ]
                self.assertTrue(perm_cat_parameters)
                self.assertTrue(all(parameter.grad is not None for parameter in perm_cat_parameters))

    def test_old_crossattn_is_preserved_and_tri_modes_have_expected_inputs(self):
        atom_dim, bond_dim, num_solvent, num_catalyst = get_feature_dims(self.dataset)
        old_model = build_glyco_gine_model(
            "crossattn",
            atom_dim,
            bond_dim,
            num_solvent,
            num_catalyst,
            hidden_dim=16,
            num_layers=1,
            encoder_type="chiral_gine",
        )
        self.assertIsInstance(old_model, GlycoGINECrossAttn)

        expected_classifier_inputs = {"l1": 16 * 8, "l2": 16 * 7, "l3": 16 * 9}
        for mode, expected_input in expected_classifier_inputs.items():
            with self.subTest(mode=mode):
                model = build_glyco_gine_model(
                    "crossattn_tri",
                    atom_dim,
                    bond_dim,
                    num_solvent,
                    num_catalyst,
                    hidden_dim=16,
                    num_layers=1,
                    encoder_type="chiral_gine",
                    local_output_mode=mode,
                )
                self.assertIsInstance(model, GlycoGINECrossAttnTri)
                self.assertEqual(model.local_output_mode, mode)
                self.assertEqual(model.classifier[0].in_features, expected_input)


if __name__ == "__main__":
    unittest.main()
