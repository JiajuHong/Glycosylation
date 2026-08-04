// 第一层审计工具：读取阴性数据工作簿并生成结构化清洗结果与可视化文件。
import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const ROOT = process.cwd();
const inputNegative = path.join(ROOT, "data/raw/阴性数据.xlsx");
const inputAudit = path.join(ROOT, "results/negative_data_feasibility_audit.xlsx");
const outputDir = path.join(ROOT, "outputs/019f660e-9cd9-7892-b3c1-053ad77571ce");
const outputXlsx = path.join(outputDir, "negative_data_cleaning_861.xlsx");

async function workbookSummary(filePath) {
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(filePath));
  const summary = await workbook.inspect({
    kind: "workbook,sheet,table",
    maxChars: 12000,
    tableMaxRows: 8,
    tableMaxCols: 30,
    tableMaxCellChars: 160,
  });
  return { workbook, summary: summary.ndjson };
}

if (process.argv.includes("--inspect")) {
  for (const filePath of [inputNegative, inputAudit]) {
    try {
      const { summary } = await workbookSummary(filePath);
      console.log(JSON.stringify({ filePath, summary }));
    } catch (error) {
      console.log(JSON.stringify({ filePath, error: String(error) }));
    }
  }
  process.exit(0);
}

if (process.argv.includes("--extract-source")) {
  const outputDir = path.join(ROOT, ".codex-tmp-negative-cleaning");
  await fs.mkdir(outputDir, { recursive: true });
  const negativeWorkbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputNegative));
  const negativeSheet = negativeWorkbook.worksheets.getItem("乙酰基封闭4号位");
  const negativeValues = negativeSheet.getRange("A1:AE877").values;
  const auditWorkbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputAudit));
  const auditSheet = auditWorkbook.worksheets.getItem("封闭样本明细");
  const auditValues = auditSheet.getRange("A1:Q877").values;
  await fs.writeFile(
    path.join(outputDir, "negative_source_rows.json"),
    JSON.stringify({ negativeValues, auditValues }),
  );
  console.log(JSON.stringify({ negativeRows: negativeValues.length - 1, auditRows: auditValues.length - 1 }));
  process.exit(0);
}

async function csvValues(filePath, sheetName) {
  const csvText = await fs.readFile(filePath, "utf8");
  const imported = await Workbook.fromCSV(csvText, { sheetName });
  return imported.worksheets.getItem(sheetName).getUsedRange(true).values;
}

function columnName(index) {
  let value = index + 1;
  let name = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    name = String.fromCharCode(65 + remainder) + name;
    value = Math.floor((value - 1) / 26);
  }
  return name;
}

function addDataSheet(workbook, name, values, options = {}) {
  const sheet = workbook.worksheets.add(name);
  const rows = values.length;
  const cols = values[0].length;
  const lastColumn = columnName(cols - 1);
  const used = sheet.getRange(`A1:${lastColumn}${rows}`);
  used.values = values;
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  sheet.getRange(`A1:${lastColumn}1`).format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "outside", style: "thin", color: "#163A5C" },
  };
  sheet.getRange(`A1:${lastColumn}1`).format.rowHeight = 34;
  sheet.getRange(`A2:${lastColumn}${rows}`).format = {
    verticalAlignment: "top",
    borders: { insideHorizontal: { style: "thin", color: "#E6EEF5" } },
  };
  used.format.autofitColumns();
  used.format.autofitRows();
  used.format.rowHeight = 18;
  sheet.getRange(`A1:${lastColumn}1`).format.rowHeight = 34;
  for (let col = 0; col < cols; col += 1) {
    const header = String(values[0][col] ?? "");
    const column = sheet.getRange(`${columnName(col)}1:${columnName(col)}${rows}`);
    if (/SMILES|Canonical|Reason|Criterion|Verification/.test(header)) {
      column.format.columnWidth = options.longWidth ?? 46;
    } else if (/ID|Index|Row|Count|Cost|Size|Value|feasibility|has_/.test(header)) {
      column.format.columnWidth = 14;
    } else {
      column.format.columnWidth = 20;
    }
  }
  const table = sheet.tables.add(`A1:${lastColumn}${rows}`, true, `${name.replace(/[^A-Za-z0-9]/g, "")}Table`);
  table.style = "TableStyleMedium2";
  table.showBandedRows = true;
  return sheet;
}

const workbook = Workbook.create();
const readme = workbook.worksheets.add("README");
readme.showGridLines = false;
readme.getRange("A1:F1").merge();
readme.getRange("A1").values = [["目标 4-O 糖基化负样本清洗交付（861 条）"]];
readme.getRange("A1:F1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", fontSize: 16 },
  verticalAlignment: "center",
};
readme.getRange("A1:F1").format.rowHeight = 32;
readme.getRange("A3:B12").values = [
  ["项目", "说明"],
  ["任务定义", "hard_feasibility=0：目标 O4 已乙酰化，预定义 C1—O4 糖苷键在结构上不可形成。"],
  ["原始负样本", "data/raw/阴性数据.xlsx；保持原文件不修改。"],
  ["SMILES 修复", "只执行确定性替换 ]] → ]，并逐行记录修改位置及修复前后 RDKit 解析状态。"],
  ["结构复现", "从 1561 条正样本的原始受体与 Acceptor_O4_Index 出发，程序添加 O-C(=O)CH3；验证价态、O4 封闭、C4—O4 键和原有手性标签。"],
  ["保留标准", "负样本的供体与修复后受体必须精确匹配某一程序生成的 O4 乙酰化结构组；结构组内正负数量必须相等。"],
  ["正反应绑定", "每条保留负样本一对一使用一个 Parent_Reaction_ID；条件字段全部取自权威正样本，仅替换受体为 O4 乙酰化版本。"],
  ["泄漏防护", "清洗表不含产物 SMILES、产物 α/β、产率、Label 或 Acceptor_OH_Status。"],
  ["排除规则", "15 条无法与程序生成结构精确匹配的记录进入 Manual_Review，不用于训练。"],
  ["软件", "RDKit 2026.03.2；确定性处理，无随机抽样。"],
];
readme.getRange("A3:B3").format = { fill: "#D9EAF7", font: { bold: true, color: "#17365D" } };
readme.getRange("A3:B12").format = { wrapText: true, verticalAlignment: "top" };
readme.getRange("A3:B12").format.borders = { preset: "outside", style: "thin", color: "#A6B8C8" };
readme.getRange("A3:A12").format.columnWidth = 20;
readme.getRange("B3:B12").format.columnWidth = 88;
readme.getRange("A3:B12").format.autofitRows();

const summaryValues = await csvValues(path.join(ROOT, "results/negative_cleaning_summary.csv"), "QA_Summary");
const cleanedValues = await csvValues(path.join(ROOT, "data/processed/negative_cleaned_861.csv"), "Cleaned_Negatives");
const repairValues = await csvValues(path.join(ROOT, "results/negative_smiles_repair_report.csv"), "SMILES_Repair_Report");
const manualValues = await csvValues(path.join(ROOT, "results/negative_unmatched_manual_review.csv"), "Manual_Review");

const summarySheet = addDataSheet(workbook, "QA_Summary", summaryValues, { longWidth: 54 });
summarySheet.getRange(`B2:B${summaryValues.length}`).format.numberFormat = "#,##0";
const cleanedSheet = addDataSheet(workbook, "Cleaned_Negatives", cleanedValues, { longWidth: 52 });
cleanedSheet.freezePanes.freezeColumns(6);
cleanedSheet.getRange(`F2:F${cleanedValues.length}`).format.numberFormat = "0";
const repairSheet = addDataSheet(workbook, "SMILES_Repair_Report", repairValues, { longWidth: 55 });
repairSheet.freezePanes.freezeColumns(2);
const manualSheet = addDataSheet(workbook, "Manual_Review", manualValues, { longWidth: 55 });
manualSheet.freezePanes.freezeColumns(2);

await fs.mkdir(outputDir, { recursive: true });
const previewSpecs = [
  ["README", "A1:F12"],
  ["QA_Summary", `A1:C${summaryValues.length}`],
  ["Cleaned_Negatives", "A1:J14"],
  ["SMILES_Repair_Report", "A1:J14"],
  ["Manual_Review", `A1:M${manualValues.length}`],
];
for (const [sheetName, range] of previewSpecs) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(
    path.join(outputDir, `preview_${sheetName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const inspection = await workbook.inspect({
  kind: "table",
  range: "QA_Summary!A1:C12",
  include: "values,formulas",
  tableMaxRows: 15,
  tableMaxCols: 5,
  maxChars: 5000,
});
console.log(inspection.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputXlsx);
console.log(JSON.stringify({ outputXlsx, cleanedRows: cleanedValues.length - 1, manualRows: manualValues.length - 1 }));
