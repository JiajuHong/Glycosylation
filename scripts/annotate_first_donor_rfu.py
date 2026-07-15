#!/usr/bin/env python3
"""Draw donor RFU and acceptor OH-local annotations for selected CSV rows.

Examples (row numbers are 1-based data rows, excluding the header):
  python scripts/annotate_first_donor_rfu.py --rows 1
  python scripts/annotate_first_donor_rfu.py --rows 1,3,8-12
  python scripts/annotate_first_donor_rfu.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, rdBase
from rdkit.Chem.Draw import rdMolDraw2D


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/processed/glyco_model_local.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/donor_acceptor_annotations"

ROLE_COLORS = {
    "C1": (0.90, 0.25, 0.20),
    "O5": (0.20, 0.48, 0.85),
    "C2": (0.96, 0.58, 0.16),
    "LG_ENTRY": (0.56, 0.32, 0.76),
    "LG_CORE": (0.18, 0.67, 0.43),
    "RING_CONTEXT": (0.20, 0.72, 0.76),
    "C2_SUB_ENTRY": (0.82, 0.38, 0.62),
    "C3": (0.55, 0.36, 0.78),
    "C4": (0.16, 0.66, 0.55),
    "O4": (0.93, 0.42, 0.18),
    "C5": (0.48, 0.68, 0.20),
    "N2_SUB_ENTRY": (0.75, 0.30, 0.68),
    "N2_SUBGRAPH": (0.30, 0.60, 0.82),
    "PG_ENTRY": (0.58, 0.48, 0.32),
    "LOCAL_CONTEXT": (0.48, 0.53, 0.60),
}
FALLBACK_COLOR = (0.48, 0.53, 0.60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate donor RFU and acceptor OH-local indices and roles."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--rows",
        default="1",
        help="1-based data rows, e.g. '1', '1,3,5-8' (default: 1)",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="draw every data row",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"input CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def parse_row_spec(spec: str, total: int) -> list[int]:
    """Convert '1,3,5-8' to sorted zero-based positions."""
    selected: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            if not all(part.strip().isdigit() for part in parts):
                raise ValueError(f"Invalid row range: {token!r}")
            start, end = (int(part.strip()) for part in parts)
            if start > end:
                raise ValueError(f"Range start exceeds end: {token!r}")
            selected.update(range(start, end + 1))
        elif token.isdigit():
            selected.add(int(token))
        else:
            raise ValueError(f"Invalid row selector: {token!r}")
    if not selected:
        raise ValueError("No rows were selected")
    invalid = sorted(number for number in selected if number < 1 or number > total)
    if invalid:
        raise IndexError(f"Row number(s) outside 1-{total}: {invalid}")
    return [number - 1 for number in sorted(selected)]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def pil_color(rgb: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(round(255 * value) for value in rgb)


def fragment_match_status(
    molecule: Chem.Mol, fragment_smiles: str, expected_indices: list[int]
) -> tuple[str, tuple[int, ...] | None]:
    """Check whether a fragment maps to exactly the stored local atom set."""
    # Primary validation: re-extract the induced fragment from the canonical
    # molecule using the stored indices. This correctly handles boundary atoms
    # whose valence changes when their bonds to the rest of the molecule are cut.
    extracted = Chem.MolFragmentToSmiles(
        molecule,
        atomsToUse=expected_indices,
        isomericSmiles=True,
        canonical=True,
    )
    # Some extracted RFUs contain a lowercase aromatic boundary atom that is
    # valid as a query fragment but not as a sanitized standalone molecule.
    with rdBase.BlockLogs():
        fragment = Chem.MolFromSmiles(fragment_smiles)
    parser_kind = "SMILES"
    if fragment is not None:
        normalized_fragment = Chem.MolToSmiles(
            fragment, isomericSmiles=True, canonical=True
        )
        if normalized_fragment == extracted:
            return (
                "exact induced-fragment match (specified chirality checked; SMILES)",
                tuple(expected_indices),
            )
    elif fragment_smiles == extracted:
        return (
            "exact induced-fragment text match (aromatic boundary fragment)",
            tuple(expected_indices),
        )
    if fragment is None:
        with rdBase.BlockLogs():
            fragment = Chem.MolFromSmarts(fragment_smiles)
        parser_kind = "SMARTS fallback"
    if fragment is None:
        return "fragment could not be parsed as SMILES or query", None
    expected = set(expected_indices)
    matches = molecule.GetSubstructMatches(fragment, useChirality=True, uniquify=True)
    for match in matches:
        if set(match) == expected:
            return f"exact atom-set match (specified chirality checked; {parser_kind})", match
    if matches:
        return "substructure found, but atom set differs", matches[0]
    return "no chirality-aware substructure match", None


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._") or "unknown"


def draw_panel(
    row: dict[str, str],
    row_number: int,
    *,
    title: str,
    canonical_field: str,
    indices_field: str,
    roles_field: str,
    fragment_field: str,
    unassigned_role: str,
) -> Image.Image:
    """Create one structure-left/table-right annotation panel."""
    smiles = row[canonical_field]
    local_indices = [
        int(value) for value in row[indices_field].split(";") if value.strip()
    ]
    if not local_indices:
        raise ValueError(f"{indices_field} is empty")
    role_indices = json.loads(row[roles_field])
    roles_by_atom: dict[int, list[str]] = {idx: [] for idx in local_indices}
    for role, indices in role_indices.items():
        for idx in indices:
            roles_by_atom.setdefault(int(idx), []).append(role)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse {canonical_field}")
    if min(local_indices) < 0 or max(local_indices) >= mol.GetNumAtoms():
        raise IndexError(f"An index in {indices_field} is outside the canonical molecule")

    match_text, _ = fragment_match_status(
        mol, row.get(fragment_field, ""), local_indices
    )

    # RDKit atom indices are zero-based, matching the values stored in the CSV.
    for idx in local_indices:
        mol.GetAtomWithIdx(idx).SetProp("atomNote", str(idx))

    secondary_roles = {"RING_CONTEXT", "N2_SUBGRAPH"}
    primary_role = {
        idx: next(
            (role for role in roles_by_atom.get(idx, []) if role not in secondary_roles),
            next(iter(roles_by_atom.get(idx, [])), unassigned_role),
        )
        for idx in local_indices
    }
    highlight_colors = {
        idx: ROLE_COLORS.get(primary_role[idx], FALLBACK_COLOR) for idx in local_indices
    }
    highlight_radii = {idx: 0.40 for idx in local_indices}

    drawer = rdMolDraw2D.MolDraw2DCairo(1300, 920)
    options = drawer.drawOptions()
    options.padding = 0.06
    options.bondLineWidth = 2.2
    options.annotationFontScale = 0.85
    options.highlightRadius = 0.40
    options.fillHighlights = True
    options.continuousHighlight = False
    drawer.DrawMolecule(
        mol,
        highlightAtoms=local_indices,
        highlightAtomColors=highlight_colors,
        highlightAtomRadii=highlight_radii,
    )
    drawer.FinishDrawing()
    molecule = Image.open(BytesIO(drawer.GetDrawingText())).convert("RGB")

    canvas_height = max(1200, 330 + 70 * len(local_indices))
    canvas = Image.new("RGB", (2200, canvas_height), "white")
    molecule_y = max(155, (canvas_height - molecule.height) // 2)
    canvas.paste(molecule, (20, molecule_y))
    draw = ImageDraw.Draw(canvas)

    record_id = row.get("ID", "")
    draw.text(
        (55, 35),
        f"{title} — CSV row {row_number} (ID {record_id or 'NA'})",
        fill=(20, 25, 35),
        font=font(38, True),
    )
    draw.text(
        (55, 92),
        f"Colored circles mark {indices_field}; numbers are zero-based canonical atom indices.",
        fill=(70, 78, 92),
        font=font(24),
    )
    match_color = (25, 125, 70) if match_text.startswith("exact") else (180, 75, 35)
    draw.text(
        (55, 128),
        f"{fragment_field} validation: {match_text}",
        fill=match_color,
        font=font(22, True),
    )

    table_x, table_y = 1370, 175
    table_bottom = table_y + 125 + 70 * len(local_indices)
    draw.rounded_rectangle(
        (table_x - 30, table_y - 25, 2160, table_bottom),
        radius=22,
        fill=(247, 249, 252),
        outline=(215, 220, 228),
        width=2,
    )
    draw.text((table_x, table_y), "Index", fill=(30, 35, 45), font=font(27, True))
    draw.text((table_x + 115, table_y), "Atom", fill=(30, 35, 45), font=font(27, True))
    draw.text((table_x + 225, table_y), "Role(s)", fill=(30, 35, 45), font=font(27, True))
    draw.line((table_x, table_y + 42, 2120, table_y + 42), fill=(185, 192, 202), width=2)

    y = table_y + 65
    for idx in local_indices:
        atom_symbol = mol.GetAtomWithIdx(idx).GetSymbol()
        role_text = ", ".join(roles_by_atom.get(idx, [])) or unassigned_role
        color = pil_color(ROLE_COLORS.get(primary_role[idx], FALLBACK_COLOR))
        draw.ellipse((table_x, y + 1, table_x + 28, y + 29), fill=color, outline=(80, 80, 80), width=1)
        draw.text((table_x + 42, y - 2), str(idx), fill=(25, 30, 40), font=font(24, True))
        draw.text((table_x + 125, y - 2), atom_symbol, fill=(25, 30, 40), font=font(24))
        draw.text((table_x + 225, y - 2), role_text, fill=(25, 30, 40), font=font(22))
        y += 70

    return canvas


def draw_row(row: dict[str, str], row_number: int, output_dir: Path) -> Path:
    donor_panel = draw_panel(
        row,
        row_number,
        title="Donor RFU annotation",
        canonical_field="Donor_Canonical_SMILES",
        indices_field="Donor_RFU_Atom_Indices",
        roles_field="Donor_RFU_Role_Indices",
        fragment_field="Donor_RFU_SMILES",
        unassigned_role="RFU_CONTEXT",
    )
    acceptor_panel = draw_panel(
        row,
        row_number,
        title="Acceptor OH-local annotation",
        canonical_field="Acceptor_Canonical_SMILES",
        indices_field="Acceptor_OH_Local_Atom_Indices",
        roles_field="Acceptor_OH_Role_Indices",
        fragment_field="Acceptor_OH_Local_SMILES",
        unassigned_role="LOCAL_CONTEXT",
    )

    divider_height = 24
    canvas = Image.new(
        "RGB",
        (2200, donor_panel.height + divider_height + acceptor_panel.height),
        "white",
    )
    canvas.paste(donor_panel, (0, 0))
    divider_y = donor_panel.height
    divider = ImageDraw.Draw(canvas)
    divider.rectangle((40, divider_y + 8, 2160, divider_y + 15), fill=(205, 211, 220))
    canvas.paste(acceptor_panel, (0, donor_panel.height + divider_height))

    record_id = row.get("ID", "")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        f"row_{row_number:04d}_ID_{safe_name(record_id)}_donor_acceptor.png"
    )
    canvas.save(output, dpi=(200, 200))
    return output


def main() -> None:
    args = parse_args()
    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    positions = list(range(len(rows))) if args.all else parse_row_spec(args.rows, len(rows))
    failures: list[tuple[int, str]] = []
    for position in positions:
        row_number = position + 1
        try:
            output = draw_row(rows[position], row_number, args.output_dir)
            print(f"OK row {row_number}: {output}")
        except Exception as exc:  # Continue so --all is useful on mixed-quality data.
            failures.append((row_number, str(exc)))
            print(f"ERROR row {row_number}: {exc}")

    print(f"Finished: {len(positions) - len(failures)} succeeded, {len(failures)} failed")
    if failures:
        print("Failed rows: " + ", ".join(str(number) for number, _ in failures))


if __name__ == "__main__":
    main()
