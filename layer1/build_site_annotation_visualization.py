#!/usr/bin/env python3
"""第一层人工审计工具：生成代表性位点标注的交互式可视化。"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/hard_feasibility_site_annotated_2632.csv"
GROUPS = ROOT / "data/processed/hard_feasibility_structural_groups_1316.csv"
OUTPUT = Path(
    "/Users/lesliehung/.codex/visualizations/2026/07/15/"
    "019f660e-9cd9-7892-b3c1-053ad77571ce/site-annotation-check.html"
)


def molecule_svg(smiles: str, roles_json: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    roles = json.loads(roles_json)
    role_colors = {
        "C1": (0.22, 0.52, 0.86, 0.34),
        "O5": (0.22, 0.52, 0.86, 0.34),
        "C2": (0.22, 0.52, 0.86, 0.34),
        "C3": (0.22, 0.52, 0.86, 0.34),
        "C4": (0.22, 0.52, 0.86, 0.34),
        "C5": (0.22, 0.52, 0.86, 0.34),
        "O4": (0.95, 0.58, 0.12, 0.50),
        "O4_EXTERNAL": (0.82, 0.24, 0.24, 0.50),
    }
    highlight_atoms: list[int] = []
    highlight_colors: dict[int, tuple[float, float, float, float]] = {}
    for role, values in roles.items():
        if not values:
            continue
        atom_index = int(values[0])
        molecule.GetAtomWithIdx(atom_index).SetProp("atomNote", role)
        highlight_atoms.append(atom_index)
        highlight_colors[atom_index] = role_colors[role]

    drawer = rdMolDraw2D.MolDraw2DSVG(500, 340)
    options = drawer.drawOptions()
    options.useBWAtomPalette()
    options.annotationFontScale = 0.72
    options.padding = 0.08
    options.setBackgroundColour((1, 1, 1, 0))
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        molecule,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace("#000000", "var(--foreground)")
    svg = svg.replace("#FFFFFF00", "transparent")
    svg = svg.replace("#FFFFFF", "transparent")
    # RDKit converts the three highlight RGBA colors to these hex RGB values.
    svg = svg.replace("#3884DB", "var(--viz-series-1)")
    svg = svg.replace("#F2931E", "var(--viz-series-2)")
    svg = svg.replace("#D13D3D", "var(--viz-series-3)")
    return svg


def encode_svg(svg: str) -> str:
    return base64.b64encode(svg.encode("utf-8")).decode("ascii")


def main() -> None:
    annotated = pd.read_csv(INPUT, encoding="utf-8-sig")
    groups = pd.read_csv(GROUPS, encoding="utf-8-sig")
    positive = annotated.loc[annotated["hard_feasibility"] == 1].copy()
    positive["Atom_Count"] = positive["Acceptor_Canonical_SMILES"].map(
        lambda smiles: Chem.MolFromSmiles(smiles).GetNumAtoms()
    )
    if "Duplicate_Count" not in positive.columns:
        positive = positive.merge(
            groups[["Structural_Group_ID", "Duplicate_Count"]],
            on="Structural_Group_ID",
            how="left",
            validate="one_to_one",
        )

    ordered = positive.sort_values(["Atom_Count", "Structural_Group_ID"]).reset_index(drop=True)
    candidate_indices = {
        "较小受体": int(len(ordered) * 0.12),
        "中等受体": int(len(ordered) * 0.50),
        "较复杂受体": int(len(ordered) * 0.88),
    }
    selected: list[tuple[str, str]] = [
        (label, ordered.iloc[index]["Structural_Group_ID"])
        for label, index in candidate_indices.items()
    ]
    repeated = positive.sort_values(["Duplicate_Count", "Atom_Count"], ascending=[False, True]).iloc[0]
    selected.append(("重复记录最多的结构组", repeated["Structural_Group_ID"]))

    entries: list[dict[str, object]] = []
    seen: set[str] = set()
    for label, group_id in selected:
        if group_id in seen:
            continue
        seen.add(group_id)
        pair = annotated.loc[annotated["Structural_Group_ID"] == group_id]
        pos = pair.loc[pair["hard_feasibility"] == 1].iloc[0]
        neg = pair.loc[pair["hard_feasibility"] == 0].iloc[0]
        group = groups.loc[groups["Structural_Group_ID"] == group_id].iloc[0]
        entries.append(
            {
                "label": label,
                "group": group_id,
                "duplicateCount": int(group["Duplicate_Count"]),
                "positiveSample": pos["Sample_ID"],
                "negativeSample": neg["Sample_ID"],
                "positiveCore": int(pos["Acceptor_Target_Core_Num_Atoms"]),
                "negativeCore": int(neg["Acceptor_Target_Core_Num_Atoms"]),
                "positiveSvg": encode_svg(molecule_svg(pos["Acceptor_Canonical_SMILES"], pos["Acceptor_Target_Role_Indices"])),
                "negativeSvg": encode_svg(molecule_svg(neg["Acceptor_Canonical_SMILES"], neg["Acceptor_Target_Role_Indices"])),
            }
        )

    entries_json = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    fragment = f"""<div id="site-annotation-viz">
  <div class="viz-grid" aria-label="标注检查汇总">
    <div class="card viz-stat"><span class="text-muted">通过配对检查</span><span class="viz-stat-value">1316 / 1316</span></div>
    <div class="card viz-stat"><span class="text-muted">唯一模型输入</span><span class="viz-stat-value">2632 / 2632</span></div>
    <div class="card viz-stat"><span class="text-muted">标注失败</span><span class="viz-stat-value">0</span></div>
  </div>
  <div class="viz-controls">
    <label class="form-label" for="site-example-select">代表性结构
      <select class="form-select" id="site-example-select"></select>
    </label>
  </div>
  <div class="site-pair-grid">
    <section class="card site-molecule" aria-labelledby="site-positive-label">
      <div class="viz-row"><strong id="site-positive-label">正样本 · O4-H</strong><span class="text-muted" id="site-positive-id"></span></div>
      <div class="site-svg" id="site-positive-svg"></div>
      <div class="text-small text-muted">7 个显式核心角色；O4_EXTERNAL 为隐式氢</div>
    </section>
    <section class="card site-molecule" aria-labelledby="site-negative-label">
      <div class="viz-row"><strong id="site-negative-label">负样本 · O4-C(=O)CH3</strong><span class="text-muted" id="site-negative-id"></span></div>
      <div class="site-svg" id="site-negative-svg"></div>
      <div class="text-small text-muted">8 个显式核心角色；O4_EXTERNAL 为羰基碳</div>
    </section>
  </div>
  <div class="site-legend" aria-label="颜色图例">
    <span><i class="site-swatch site-ring"></i>糖环 C1/O5/C2/C3/C4/C5</span>
    <span><i class="site-swatch site-o4"></i>目标 O4</span>
    <span><i class="site-swatch site-external"></i>负样本 O4_EXTERNAL</span>
  </div>
  <div class="text-small text-muted" id="site-selected-detail"></div>
</div>
<style>
  #site-annotation-viz {{ display: grid; gap: 1rem; color: var(--foreground); }}
  #site-annotation-viz .site-pair-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }}
  #site-annotation-viz .site-molecule {{ min-width: 0; display: grid; gap: .5rem; }}
  #site-annotation-viz .site-svg {{ width: 100%; min-height: 260px; display: grid; place-items: center; color: var(--foreground); }}
  #site-annotation-viz .site-svg svg {{ width: 100%; height: auto; max-height: 360px; }}
  #site-annotation-viz .site-legend {{ display: flex; flex-wrap: wrap; gap: .75rem 1.25rem; align-items: center; }}
  #site-annotation-viz .site-legend span {{ display: inline-flex; gap: .4rem; align-items: center; }}
  #site-annotation-viz .site-swatch {{ width: .8rem; height: .8rem; border-radius: 50%; display: inline-block; }}
  #site-annotation-viz .site-ring {{ background: var(--viz-series-1); }}
  #site-annotation-viz .site-o4 {{ background: var(--viz-series-2); }}
  #site-annotation-viz .site-external {{ background: var(--viz-series-3); }}
  @media (max-width: 620px) {{
    #site-annotation-viz .site-pair-grid {{ grid-template-columns: 1fr; }}
    #site-annotation-viz .site-svg {{ min-height: 220px; }}
  }}
</style>
<script>
(() => {{
  const root = document.getElementById('site-annotation-viz');
  const entries = {entries_json};
  const select = root.querySelector('#site-example-select');
  const decodeSvg = value => new TextDecoder().decode(Uint8Array.from(atob(value), char => char.charCodeAt(0)));
  entries.forEach((entry, index) => {{
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = entry.label;
    select.appendChild(option);
  }});
  const update = () => {{
    const entry = entries[Number(select.value || 0)];
    root.querySelector('#site-positive-svg').innerHTML = decodeSvg(entry.positiveSvg);
    root.querySelector('#site-negative-svg').innerHTML = decodeSvg(entry.negativeSvg);
    root.querySelector('#site-positive-id').textContent = entry.positiveSample;
    root.querySelector('#site-negative-id').textContent = entry.negativeSample;
    root.querySelector('#site-selected-detail').textContent = `${{entry.group}} · 原始反应记录 ${{entry.duplicateCount}} 条 · 核心原子 ${{entry.positiveCore}} → ${{entry.negativeCore}}`;
  }};
  select.addEventListener('change', update);
  update();
}})();
</script>
"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(fragment, encoding="utf-8")
    print({"output": str(OUTPUT), "examples": len(entries), "bytes": OUTPUT.stat().st_size})


if __name__ == "__main__":
    main()
