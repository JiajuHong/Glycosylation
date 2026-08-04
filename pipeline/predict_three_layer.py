#!/usr/bin/env python
"""冻结版三层统一推理入口。

第一层判断目标 O4 的硬结构可行性；只有通过第一层的候选才进入第二层
文献支持度评分和第三层 α/β 立体选择性预测。第二层分数和第三层 softmax
输出均不是经实验失败数据校准的反应成功概率。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from layer1.predict_hard_feasibility import (
    InferenceDataset,
    annotate_row,
    checkpoint_acceptor_role_names,
    extract_acceptor_target_site,
    load_model as load_layer1_model,
    run_probabilities as run_layer1_probabilities,
)
from layer1.train_hard_feasibility import collate as layer1_collate
from layer2.score_soft_compatibility import (
    INTERPRETATION,
    missing,
    prepare_row,
    score_prepared_queries,
)
from layer3.extract_donor_rfu import extract_donor_rfu
from layer3.glyco_dataset import GlycoDataset, glyco_collate_fn
from layer3.train_gine import get_feature_dims, move_batch_to_device
from models.glyco_gine_models import build_glyco_gine_model
from pipeline.status import (
    FINAL_INTERNAL_ERROR,
    FINAL_INVALID_INPUT,
    FINAL_PREDICTED,
    FINAL_STRUCTURALLY_INFEASIBLE,
    SOFT_NOT_EVALUATED,
    STEREO_NOT_EVALUATED,
)


OUTPUT_DEFAULTS: dict[str, Any] = {
    "Final_Status": FINAL_INVALID_INPUT,
    "Final_Error": "",
    "Final_Interpretation": "",
    "Input_Status": "parse_error",
    "Donor_RFU_Status": "not_evaluated",
    "Acceptor_O4_Status": "not_evaluated",
    "Warnings": "",
    "Canonical_Donor_SMILES": "",
    "Canonical_Acceptor_SMILES": "",
    "Resolved_Donor_Type": "",
    "Target_O4_Source": "",
    "Target_O4_Index_Canonical": np.nan,
    "Target_O4_State": "",
    "Target_O4_Candidate_Count": np.nan,
    "Target_O4_Free_Candidate_Count": np.nan,
    "Target_O4_Blocked_Candidate_Count": np.nan,
    "Layer1_Mean_Feasibility_Score": np.nan,
    "Layer1_Seed_Disagreement": False,
    "Layer1_Seed_Votes": "",
    "Layer1_Decision": "not_evaluated",
    "Soft_Compatibility_Score": np.nan,
    "Soft_TopK_Mean_Score": np.nan,
    "Soft_Support_Percentile": np.nan,
    "Soft_Domain_Q05": np.nan,
    "Soft_Domain_Q10": np.nan,
    "Soft_Domain_Status": SOFT_NOT_EVALUATED,
    "Nearest_Success_Reaction_ID": "",
    "Nearest_Success_Similarity": np.nan,
    "Soft_Score_Interpretation": INTERPRETATION,
    "Stereoselectivity_Label": STEREO_NOT_EVALUATED,
    "Stereoselectivity_Mean_Beta_Score": np.nan,
    "Stereoselectivity_Seed_Disagreement": False,
    "Stereoselectivity_Vote_Count": "",
    "Stereoselectivity_Skip_Reason": "",
    "Pipeline_Version": "",
    "Inference_UTC": "",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact(root: Path, entry: dict[str, Any]) -> Path:
    path = Path(entry["path"])
    path = path if path.is_absolute() else root / path
    if not path.exists():
        raise FileNotFoundError(f"模型清单中的文件不存在: {path}")
    expected = entry.get("sha256")
    if expected and sha256_file(path) != expected:
        raise ValueError(f"文件 SHA256 与模型清单不一致: {path}")
    return path


def load_manifest(path: Path) -> tuple[dict[str, Any], Path]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"pipeline_version", "layer1", "layer2", "layer3", "label_mapping"}
    absent = required - set(manifest)
    if absent:
        raise ValueError(f"模型清单缺少字段: {sorted(absent)}")
    if manifest["label_mapping"] != {"0": "Alpha", "1": "Beta"}:
        raise ValueError("第三层标签映射必须是 0=Alpha, 1=Beta")
    root_value = manifest.get("project_root", ".")
    root = Path(root_value)
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    return manifest, root


def initialize_output(source: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    output = source.copy()
    for column, value in OUTPUT_DEFAULTS.items():
        output[column] = value
    output["Pipeline_Version"] = str(manifest["pipeline_version"])
    output["Inference_UTC"] = datetime.now(timezone.utc).isoformat()
    for entry in manifest["layer1"]["checkpoints"]:
        seed = int(entry["seed"])
        output[f"Layer1_Seed{seed}_Feasibility_Score"] = np.nan
        output[f"Layer1_Seed{seed}_Vote"] = "not_evaluated"
    for entry in manifest["layer3"]["checkpoints"]:
        seed = int(entry["seed"])
        output[f"Layer3_Seed{seed}_Beta_Score"] = np.nan
        output[f"Layer3_Seed{seed}_Threshold"] = float(entry["threshold"])
        output[f"Layer3_Seed{seed}_Vote"] = "not_evaluated"
    return output


def annotate_inputs(
    source: pd.DataFrame,
    output: pd.DataFrame,
    strict: bool,
    acceptor_role_names: list[str] | None = None,
) -> tuple[list[int], list[dict[str, Any]]]:
    valid_positions: list[int] = []
    items: list[dict[str, Any]] = []
    errors: list[str] = []
    for position, (_, row) in enumerate(source.iterrows()):
        try:
            item, audit = annotate_row(
                row,
                position,
                acceptor_role_names=acceptor_role_names,
            )
        except Exception as exc:  # 单行输入错误不能使非严格批处理整体中断
            item, audit = None, {"Inference_Error": f"{type(exc).__name__}: {exc}"}
        if item is None:
            if audit.get("Inference_Status") == "structurally_infeasible":
                output.iat[position, output.columns.get_loc("Final_Status")] = (
                    FINAL_STRUCTURALLY_INFEASIBLE
                )
                output.iat[position, output.columns.get_loc("Input_Status")] = "parsed"
                output.iat[position, output.columns.get_loc("Donor_RFU_Status")] = "ok"
                output.iat[position, output.columns.get_loc("Acceptor_O4_Status")] = (
                    "no_free_target_site"
                )
                output.iat[position, output.columns.get_loc("Layer1_Decision")] = "infeasible"
                output.iat[
                    position, output.columns.get_loc("Stereoselectivity_Skip_Reason")
                ] = "no_free_target_site"
                for source_key, output_key in {
                    "Canonical_Donor_SMILES": "Canonical_Donor_SMILES",
                    "Canonical_Acceptor_SMILES": "Canonical_Acceptor_SMILES",
                    "Resolved_Donor_Type": "Resolved_Donor_Type",
                    "Target_O4_Source": "Target_O4_Source",
                    "Target_O4_State": "Target_O4_State",
                    "Target_O4_Candidate_Count": "Target_O4_Candidate_Count",
                    "Target_O4_Free_Candidate_Count": "Target_O4_Free_Candidate_Count",
                    "Target_O4_Blocked_Candidate_Count": "Target_O4_Blocked_Candidate_Count",
                }.items():
                    output.iat[position, output.columns.get_loc(output_key)] = audit.get(
                        source_key, ""
                    )
                continue
            message = str(audit.get("Inference_Error", "input annotation failed"))
            output.iat[position, output.columns.get_loc("Final_Status")] = FINAL_INVALID_INPUT
            output.iat[position, output.columns.get_loc("Final_Error")] = message
            errors.append(f"row {position}: {message}")
            continue
        valid_positions.append(position)
        items.append(item)
        output.iat[position, output.columns.get_loc("Input_Status")] = "parsed"
        output.iat[position, output.columns.get_loc("Donor_RFU_Status")] = "ok"
        output.iat[position, output.columns.get_loc("Acceptor_O4_Status")] = "ok"
        mapping = {
            "Canonical_Donor_SMILES": "Canonical_Donor_SMILES",
            "Canonical_Acceptor_SMILES": "Canonical_Acceptor_SMILES",
            "Resolved_Donor_Type": "Resolved_Donor_Type",
            "Target_O4_Source": "Target_O4_Source",
            "Target_O4_Index": "Target_O4_Index_Canonical",
            "Target_O4_State": "Target_O4_State",
            "Target_O4_Candidate_Count": "Target_O4_Candidate_Count",
            "Target_O4_Free_Candidate_Count": "Target_O4_Free_Candidate_Count",
            "Target_O4_Blocked_Candidate_Count": "Target_O4_Blocked_Candidate_Count",
        }
        for source_key, output_key in mapping.items():
            output.iat[position, output.columns.get_loc(output_key)] = audit.get(source_key, "")
    if strict and errors:
        raise ValueError("严格模式发现无效输入:\n" + "\n".join(errors[:20]))
    return valid_positions, items


def run_layer1(
    output: pd.DataFrame,
    positions: list[int],
    items: list[dict[str, Any]],
    manifest: dict[str, Any],
    root: Path,
    device: torch.device,
    batch_size: int,
) -> list[int]:
    if not items:
        return []
    loader = DataLoader(
        InferenceDataset(items), batch_size=batch_size, shuffle=False, collate_fn=layer1_collate
    )
    threshold = float(manifest["layer1"]["ensemble"]["threshold"])
    all_scores: list[np.ndarray] = []
    all_votes: list[np.ndarray] = []
    for entry in manifest["layer1"]["checkpoints"]:
        checkpoint = resolve_artifact(root, entry)
        model, checkpoint_seed = load_layer1_model(checkpoint, device)
        seed = int(entry["seed"])
        if checkpoint_seed != seed:
            raise ValueError(f"第一层 checkpoint seed 不一致: {checkpoint}")
        scores = np.asarray(run_layer1_probabilities(model, loader, device), dtype=float)
        votes = scores >= threshold
        all_scores.append(scores)
        all_votes.append(votes)
        output.loc[output.index[positions], f"Layer1_Seed{seed}_Feasibility_Score"] = scores
        output.loc[output.index[positions], f"Layer1_Seed{seed}_Vote"] = np.where(
            votes, "feasible", "infeasible"
        )
    score_matrix = np.vstack(all_scores)
    vote_matrix = np.vstack(all_votes)
    mean_scores = score_matrix.mean(axis=0)
    decisions = mean_scores >= threshold
    output.loc[output.index[positions], "Layer1_Mean_Feasibility_Score"] = mean_scores
    output.loc[output.index[positions], "Layer1_Seed_Disagreement"] = (
        vote_matrix.min(axis=0) != vote_matrix.max(axis=0)
    )
    output.loc[output.index[positions], "Layer1_Decision"] = np.where(
        decisions, "feasible", "infeasible"
    )
    output.loc[output.index[positions], "Layer1_Seed_Votes"] = [
        json.dumps([int(value) for value in vote_matrix[:, column]], separators=(",", ":"))
        for column in range(vote_matrix.shape[1])
    ]
    pass_positions: list[int] = []
    for local_pos, output_pos in enumerate(positions):
        if decisions[local_pos]:
            pass_positions.append(output_pos)
        else:
            output.iat[output_pos, output.columns.get_loc("Final_Status")] = FINAL_STRUCTURALLY_INFEASIBLE
            output.iat[output_pos, output.columns.get_loc("Stereoselectivity_Skip_Reason")] = "rejected_by_layer1"
    return pass_positions


def run_layer2(
    source: pd.DataFrame,
    output: pd.DataFrame,
    positions: list[int],
    manifest: dict[str, Any],
    root: Path,
    top_k: int,
) -> tuple[list[int], dict[int, dict[str, Any]], list[dict[str, Any]]]:
    prepared: list[dict[str, Any]] = []
    scored_positions: list[int] = []
    prepared_by_position: dict[int, dict[str, Any]] = {}
    for position in positions:
        row = source.iloc[position].copy()
        # 统一使用第一层已经确认过的规范化结构和目标位点。
        row["Donor_Canonical_SMILES"] = output.iloc[position]["Canonical_Donor_SMILES"]
        row["Acceptor_Canonical_SMILES"] = output.iloc[position]["Canonical_Acceptor_SMILES"]
        row["Donor_Type"] = output.iloc[position]["Resolved_Donor_Type"]
        row["Acceptor_O4_Index"] = output.iloc[position]["Target_O4_Index_Canonical"]
        row["Target_O4_Index"] = np.nan
        try:
            item, audit = prepare_row(row)
        except Exception as exc:
            item, audit = None, {"Soft_Score_Error": f"{type(exc).__name__}: {exc}"}
        if item is None:
            output.iat[position, output.columns.get_loc("Final_Status")] = FINAL_INTERNAL_ERROR
            output.iat[position, output.columns.get_loc("Final_Error")] = str(
                audit.get("Soft_Score_Error", "layer2 preparation failed")
            )
            continue
        prepared.append(item)
        scored_positions.append(position)
        prepared_by_position[position] = item

    if not prepared:
        return [], prepared_by_position, []
    reference_entry = manifest["layer2"]["positive_reference"]
    positives = pd.read_csv(resolve_artifact(root, reference_entry), encoding="utf-8-sig")
    scored, neighbor_lists, _ = score_prepared_queries(
        pd.DataFrame(prepared),
        positives,
        reference_split=str(manifest["layer2"].get("reference_split", "all")),
        top_k_neighbors=top_k,
    )
    for local_pos, output_pos in enumerate(scored_positions):
        for column in scored.columns:
            output.iat[output_pos, output.columns.get_loc(column)] = scored.iloc[local_pos][column]
        if neighbor_lists[local_pos]:
            nearest = neighbor_lists[local_pos][0]
            output.iat[
                output_pos, output.columns.get_loc("Nearest_Success_Reaction_ID")
            ] = nearest["reaction_id"]
            output.iat[
                output_pos, output.columns.get_loc("Nearest_Success_Similarity")
            ] = nearest["total_similarity"]
    long_neighbors: list[dict[str, Any]] = []
    for local_pos, output_pos in enumerate(scored_positions):
        sample_id = source.iloc[output_pos].get("Sample_ID", source.iloc[output_pos].get("ID", output_pos))
        for neighbor in neighbor_lists[local_pos]:
            long_neighbors.append({"input_row": output_pos, "sample_id": sample_id, **neighbor})
    return scored_positions, prepared_by_position, long_neighbors


def make_layer3_item(
    source_row: pd.Series,
    output_row: pd.Series,
    prepared: dict[str, Any],
    template: GlycoDataset,
    position: int,
) -> dict[str, Any]:
    donor_smiles = str(output_row["Canonical_Donor_SMILES"])
    acceptor_smiles = str(output_row["Canonical_Acceptor_SMILES"])
    donor = extract_donor_rfu(donor_smiles, str(output_row["Resolved_Donor_Type"]))
    target = extract_acceptor_target_site(
        acceptor_smiles, int(output_row["Target_O4_Index_Canonical"])
    )
    if donor["Donor_RFU_Status"] != "ok" or target.get("status") != "ok":
        raise ValueError("第三层局部反应中心提取失败")
    has_solvent = int(
        source_row.get(
            "has_solvent",
            0
            if missing(source_row.get("Solvent_Component_IDs")) and missing(source_row.get("Solvent"))
            else 1,
        )
    )
    has_catalyst = int(
        source_row.get(
            "has_catalyst",
            0
            if missing(source_row.get("Catalyst_Component_IDs")) and missing(source_row.get("Catalyst"))
            else 1,
        )
    )
    row = pd.Series(
        {
            "ID": position,
            "Reaction_ID": source_row.get("Reaction_ID", source_row.get("Sample_ID", position)),
            "Donor_Canonical_SMILES": donor_smiles,
            "Acceptor_Canonical_SMILES": acceptor_smiles,
            "Donor_RFU_Atom_Indices": donor["Donor_RFU_Atom_Indices"],
            "Donor_RFU_Role_Indices": donor["Donor_RFU_Role_Indices"],
            "Donor_C1_Index": donor["Donor_C1_Index"],
            "Acceptor_OH_Local_Atom_Indices": ";".join(map(str, target["local_indices"])),
            "Acceptor_OH_Role_Indices": json.dumps(target["roles"], separators=(",", ":")),
            "Acceptor_O4_Index": target["o4_index"],
            "Solvent_Component_IDs": prepared["Solvent_Component_IDs"],
            "Catalyst_Component_IDs": prepared["Catalyst_Component_IDs"],
            "has_solvent": has_solvent,
            "has_catalyst": has_catalyst,
            "Temp_C": prepared["Temp_C"],
            "has_temp": prepared["has_temp"],
            "Time_min": prepared["Time_min"],
            "has_time": prepared["has_time"],
            "Label": 0,
            "split_pair_group": "inference",
            "split_random_stratified": "inference",
            "split_year": "inference",
        }
    )
    return template.build_item(row)


def load_layer3_template_and_dims(
    checkpoint: dict[str, Any], root: Path
) -> tuple[GlycoDataset, tuple[int, int, int, int]]:
    args = checkpoint["args"]
    csv_path = Path(args["csv"])
    cache_path = Path(args.get("chiral_graph_cache", "data/processed/rdkit_chiral_graph_cache.pt"))
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    if not cache_path.is_absolute():
        cache_path = root / cache_path
    dataset = GlycoDataset(
        csv_path=csv_path,
        split_column=args["split_column"],
        split="train",
        encoder_type=args.get("encoder_type", "gine"),
        solvent_vocab_path=root / "data/processed/solvent_vocab.json",
        catalyst_vocab_path=root / "data/processed/catalyst_vocab.json",
        chiral_graph_cache_path=cache_path,
        validate=True,
    )
    return dataset, get_feature_dims(dataset)


def build_layer3_model(
    checkpoint: dict[str, Any], dims: tuple[int, int, int, int], device: torch.device
):
    args = checkpoint["args"]
    atom_dim, bond_dim, num_solvent, num_catalyst = dims
    model = build_glyco_gine_model(
        model_type=args["model_type"],
        atom_feat_dim=atom_dim,
        bond_feat_dim=bond_dim,
        num_solvent_tokens=num_solvent,
        num_catalyst_tokens=num_catalyst,
        hidden_dim=args["hidden_dim"],
        num_layers=args["num_layers"],
        dropout=args["dropout"],
        encoder_type=args.get("encoder_type", "gine"),
        perm_cat_dropout=args.get("perm_cat_dropout"),
        perm_cat_normalization=args.get("perm_cat_normalization", "reference"),
        local_output_mode=args.get("local_output_mode", "l3"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def run_layer3(
    source: pd.DataFrame,
    output: pd.DataFrame,
    positions: list[int],
    prepared_by_position: dict[int, dict[str, Any]],
    manifest: dict[str, Any],
    root: Path,
    device: torch.device,
    batch_size: int,
) -> None:
    if not positions:
        return
    entries = manifest["layer3"]["checkpoints"]
    first_path = resolve_artifact(root, entries[0])
    first_checkpoint = torch.load(first_path, map_location="cpu", weights_only=False)
    template, dims = load_layer3_template_and_dims(first_checkpoint, root)
    items: list[dict[str, Any]] = []
    valid_positions: list[int] = []
    for position in positions:
        try:
            items.append(
                make_layer3_item(
                    source.iloc[position], output.iloc[position], prepared_by_position[position], template, position
                )
            )
            valid_positions.append(position)
        except Exception as exc:
            output.iat[position, output.columns.get_loc("Final_Status")] = FINAL_INTERNAL_ERROR
            output.iat[position, output.columns.get_loc("Final_Error")] = (
                f"layer3 preparation failed: {type(exc).__name__}: {exc}"
            )
    if not items:
        return
    loader = DataLoader(items, batch_size=batch_size, shuffle=False, collate_fn=glyco_collate_fn)
    all_scores: list[np.ndarray] = []
    all_votes: list[np.ndarray] = []
    for entry in entries:
        path = resolve_artifact(root, entry)
        checkpoint = first_checkpoint if path == first_path else torch.load(
            path, map_location="cpu", weights_only=False
        )
        seed = int(entry["seed"])
        if int(checkpoint["args"]["seed"]) != seed:
            raise ValueError(f"第三层 checkpoint seed 不一致: {path}")
        model = build_layer3_model(checkpoint, dims, device)
        probabilities: list[float] = []
        with torch.no_grad():
            for batch in loader:
                moved = move_batch_to_device(batch, device)
                probabilities.extend(torch.softmax(model(moved), dim=-1)[:, 1].cpu().tolist())
        scores = np.asarray(probabilities, dtype=float)
        threshold = float(entry["threshold"])
        votes = scores >= threshold
        all_scores.append(scores)
        all_votes.append(votes)
        output.loc[output.index[valid_positions], f"Layer3_Seed{seed}_Beta_Score"] = scores
        output.loc[output.index[valid_positions], f"Layer3_Seed{seed}_Vote"] = np.where(
            votes, "Beta", "Alpha"
        )
    score_matrix = np.vstack(all_scores)
    vote_matrix = np.vstack(all_votes)
    beta_votes = vote_matrix.sum(axis=0)
    ensemble_beta = beta_votes > (len(entries) / 2.0)
    output.loc[output.index[valid_positions], "Stereoselectivity_Label"] = np.where(
        ensemble_beta, "Beta", "Alpha"
    )
    output.loc[output.index[valid_positions], "Stereoselectivity_Mean_Beta_Score"] = score_matrix.mean(axis=0)
    output.loc[output.index[valid_positions], "Stereoselectivity_Seed_Disagreement"] = (
        vote_matrix.min(axis=0) != vote_matrix.max(axis=0)
    )
    winning_votes = np.maximum(beta_votes, len(entries) - beta_votes)
    output.loc[output.index[valid_positions], "Stereoselectivity_Vote_Count"] = [
        f"{int(count)}/{len(entries)}" for count in winning_votes
    ]
    output.loc[output.index[valid_positions], "Final_Status"] = FINAL_PREDICTED


def finalize_output(output: pd.DataFrame) -> None:
    """根据各层正交状态补充面向使用者的解释和警告。"""
    for position in range(len(output)):
        row = output.iloc[position]
        warnings: list[str] = []
        if bool(row["Layer1_Seed_Disagreement"]):
            warnings.append("layer1_seed_disagreement")
        if bool(row["Stereoselectivity_Seed_Disagreement"]):
            warnings.append("layer3_seed_disagreement")
        if row["Soft_Domain_Status"] in {"borderline", "out_of_domain"}:
            warnings.append(f"soft_domain_{row['Soft_Domain_Status']}")
        output.iat[position, output.columns.get_loc("Warnings")] = ";".join(warnings)

        status = row["Final_Status"]
        if status == FINAL_INVALID_INPUT:
            interpretation = "输入结构或目标位点无法可靠解析；未进入模型。"
            output.iat[position, output.columns.get_loc("Stereoselectivity_Skip_Reason")] = "invalid_input"
        elif status == FINAL_STRUCTURALLY_INFEASIBLE:
            interpretation = "目标 O4 不满足硬结构前提；第二、第三层均已跳过。"
        elif status == FINAL_PREDICTED:
            interpretation = (
                "已通过硬结构过滤；第二层为文献支持度，第三层为多数投票立体选择性。"
            )
        else:
            interpretation = "流水线内部处理失败；该行结果不可使用。"
            output.iat[position, output.columns.get_loc("Stereoselectivity_Skip_Reason")] = "internal_error"
        output.iat[position, output.columns.get_loc("Final_Interpretation")] = interpretation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/model_manifest_v2.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k-neighbors", type=int, default=5)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--neighbors-output",
        type=Path,
        help="可选：另存第二层近邻明细；主输出不嵌入大段 JSON。",
    )
    parser.add_argument(
        "--save-neighbors",
        action="store_true",
        help="将近邻明细保存为主输出同目录下的 *.neighbors.csv。",
    )
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    if cli.batch_size < 1 or cli.top_k_neighbors < 1:
        raise ValueError("batch size and top-k must be positive")
    manifest, root = load_manifest(cli.manifest.resolve())
    source = pd.read_csv(cli.input, encoding="utf-8-sig")
    required = {"Donor_Canonical_SMILES", "Acceptor_Canonical_SMILES"}
    missing_columns = required - set(source)
    if missing_columns:
        raise ValueError(f"输入缺少字段: {sorted(missing_columns)}")
    output = initialize_output(source, manifest)
    device = torch.device(cli.device)

    layer1_checkpoint_paths = [
        resolve_artifact(root, entry)
        for entry in manifest["layer1"]["checkpoints"]
    ]
    acceptor_role_names = checkpoint_acceptor_role_names(layer1_checkpoint_paths)
    valid_positions, layer1_items = annotate_inputs(
        source,
        output,
        cli.strict,
        acceptor_role_names=acceptor_role_names,
    )
    passed_positions = run_layer1(
        output, valid_positions, layer1_items, manifest, root, device, cli.batch_size
    )
    layer2_positions, prepared_by_position, neighbors = run_layer2(
        source,
        output,
        passed_positions,
        manifest,
        root,
        cli.top_k_neighbors,
    )
    run_layer3(
        source,
        output,
        layer2_positions,
        prepared_by_position,
        manifest,
        root,
        device,
        cli.batch_size,
    )
    finalize_output(output)

    cli.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(cli.output, index=False, encoding="utf-8-sig")
    neighbors_output = cli.neighbors_output
    if cli.save_neighbors and neighbors_output is None:
        neighbors_output = cli.output.with_name(f"{cli.output.stem}.neighbors.csv")
    if neighbors_output:
        neighbors_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(neighbors).to_csv(neighbors_output, index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "rows": len(output),
                "status_counts": output["Final_Status"].value_counts().to_dict(),
                "layer1_seed_disagreements": int(output["Layer1_Seed_Disagreement"].sum()),
                "layer3_seed_disagreements": int(output["Stereoselectivity_Seed_Disagreement"].sum()),
                "output": str(cli.output),
                "warning": "Beta score and soft compatibility score are not calibrated reaction probabilities.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
