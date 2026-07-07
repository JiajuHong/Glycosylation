from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("raw.csv")
OUTPUT_PATH = Path("condition_tokenized.csv")


SOLVENT_TOKEN_MAP = {
    "Dichloromethane": "SOLV_DCM",
    "Diethyl ether": "SOLV_DIETHYL_ETHER",
    "Toluene": "SOLV_TOLUENE",
    "Acetonitrile": "SOLV_ACN",
    "Methanol": "SOLV_MEOH",
    "Hexane": "SOLV_HEXANE",
    "Dichloroethane": "SOLV_DCE",
    "Triethylamine": "SOLV_TRIETHYLAMINE",
    "Chloroform": "SOLV_CHLOROFORM",
    "Tetrahydrofuran": "SOLV_THF",
    "Ethyl acetate": "SOLV_ETOAC",
    "Benzene": "SOLV_BENZENE",
    "Heptane": "SOLV_HEPTANE",
    "Cyclohexane": "SOLV_CYCLOHEXANE",
    "Ethanol": "SOLV_ETHANOL",
    "Isopropanol": "SOLV_IPA",
    "Trichloroacetonitrile": "SOLV_TRICHLOROACETONITRILE",
    "Propionitrile": "SOLV_PROPIONITRILE",
    "Nitromethane": "SOLV_NITROMETHANE",
    "Acetic acid": "SOLV_ACOH",
}


SOLVENT_VOCAB = {
    "SOLV_PAD": 0,
    "SOLV_MISSING": 1,
    "SOLV_UNKNOWN": 2,
    "SOLV_DCM": 3,
    "SOLV_DIETHYL_ETHER": 4,
    "SOLV_TOLUENE": 5,
    "SOLV_ACN": 6,
    "SOLV_MEOH": 7,
    "SOLV_HEXANE": 8,
    "SOLV_DCE": 9,
    "SOLV_TRIETHYLAMINE": 10,
    "SOLV_CHLOROFORM": 11,
    "SOLV_THF": 12,
    "SOLV_ETOAC": 13,
    "SOLV_BENZENE": 14,
    "SOLV_HEPTANE": 15,
    "SOLV_CYCLOHEXANE": 16,
    "SOLV_ETHANOL": 17,
    "SOLV_IPA": 18,
    "SOLV_TRICHLOROACETONITRILE": 19,
    "SOLV_PROPIONITRILE": 20,
    "SOLV_NITROMETHANE": 21,
    "SOLV_ACOH": 22,
}


CATALYST_TOKEN_MAP = {
    "Trimethylsilyl triflate": "CAT_TMSOTF",
    "Triethylamine": "CAT_TRIETHYLAMINE",
    "N-Iodosuccinimide": "CAT_NIS",
    "Trifluoromethanesulfonic acid": "CAT_TFOH",
    "Boron trifluoride etherate": "CAT_BF3_ET2O",
    "Sodium bicarbonate": "CAT_NAHCO3",
    "Trifluoromethanesulfonic anhydride": "CAT_TF2O",
    "Silver triflate": "CAT_AGOTF",
    "Methyl triflate": "CAT_MEOTF",
    "Pyridine": "CAT_PYRIDINE",
    "Tosyl chloride": "CAT_TSCL",
    "1-(Phenylsulfonyl)piperidine": "CAT_PHENYLSULFONYL_PIPERIDINE",
    "Diphenyl sulfoxide": "CAT_DIPHENYL_SULFOXIDE",
    "Sodium thiosulfate": "CAT_SODIUM_THIOSULFATE",
    "Trifluoroacetic acid": "CAT_TFA",
    "Bromine": "CAT_BR2",
    "Tetrabutylammonium bromide": "CAT_TBAB",
    "Copper bromide": "CAT_CUBR",
    "Methyleugenol": "CAT_METHYLEUGENOL",
    "Iodonium": "CAT_IODONIUM",
    "Triethylsilyl 1,1,1-trifluoromethanesulfonate": "CAT_TESOTF",
    "Sulfobromophthalein": "CAT_SULFOBROMOPHTHALEIN",
    "2,6-Di-tert-butyl-4-methylpyridine": "CAT_DTBMP",
    "Nitrosonium tetrafluoroborate": "CAT_NOBF4",
    "Potassium carbonate": "CAT_K2CO3",
    "Triethyl phosphite": "CAT_TRIETHYL_PHOSPHITE",
    "Sodium chloride": "CAT_NACL",
    "Diisopropylamine": "CAT_DIPA",
    "Diisopropylethylamine": "CAT_DIPEA",
    "(1,1-Dimethylethyl)dimethylsilyl 1,1,1-trifluoromethanesulfonate": "CAT_TBSOTF",
    "Silylium": "CAT_SILYLIUM",
    "Copper(II) triflate": "CAT_CUOTF2",
    "Perchloric acid": "CAT_HCLO4",
    "Zinc iodide": "CAT_ZNI2",
    "Boron trifluoride": "CAT_BF3",
    "Tetrabutylammonium iodide": "CAT_TBAI",
    "Tetrabutylammonium fluoride": "CAT_TBAF",
    "Ytterbium triflate": "CAT_YBOTF3",
    "Methyl formate": "CAT_METHYL_FORMATE",
    "Benzenesulfenyl chloride": "CAT_PHENYLSULFENYL_CHLORIDE",
    "Methyl trifluoroacetate": "CAT_METHYL_TRIFLUOROACETATE",
}


CATALYST_VOCAB = {
    "CAT_PAD": 0,
    "CAT_MISSING": 1,
    "CAT_UNKNOWN": 2,
    "CAT_TMSOTF": 3,
    "CAT_TRIETHYLAMINE": 4,
    "CAT_NIS": 5,
    "CAT_TFOH": 6,
    "CAT_BF3_ET2O": 7,
    "CAT_NAHCO3": 8,
    "CAT_TF2O": 9,
    "CAT_AGOTF": 10,
    "CAT_MEOTF": 11,
    "CAT_PYRIDINE": 12,
    "CAT_TSCL": 13,
    "CAT_PHENYLSULFONYL_PIPERIDINE": 14,
    "CAT_DIPHENYL_SULFOXIDE": 15,
    "CAT_SODIUM_THIOSULFATE": 16,
    "CAT_TFA": 17,
    "CAT_BR2": 18,
    "CAT_TBAB": 19,
    "CAT_CUBR": 20,
    "CAT_METHYLEUGENOL": 21,
    "CAT_IODONIUM": 22,
    "CAT_TESOTF": 23,
    "CAT_SULFOBROMOPHTHALEIN": 24,
    "CAT_DTBMP": 25,
    "CAT_NOBF4": 26,
    "CAT_K2CO3": 27,
    "CAT_TRIETHYL_PHOSPHITE": 28,
    "CAT_NACL": 29,
    "CAT_DIPA": 30,
    "CAT_DIPEA": 31,
    "CAT_TBSOTF": 32,
    "CAT_SILYLIUM": 33,
    "CAT_CUOTF2": 34,
    "CAT_HCLO4": 35,
    "CAT_ZNI2": 36,
    "CAT_BF3": 37,
    "CAT_TBAI": 38,
    "CAT_TBAF": 39,
    "CAT_YBOTF3": 40,
    "CAT_METHYL_FORMATE": 41,
    "CAT_PHENYLSULFENYL_CHLORIDE": 42,
    "CAT_METHYL_TRIFLUOROACETATE": 43,
}


def is_missing(value: object) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def split_components(value: object) -> list[str]:
    if is_missing(value):
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def map_components(
    raw_value: object,
    token_map: dict[str, str],
    vocab: dict[str, int],
    missing_token: str,
    unknown_token: str,
) -> tuple[str, str, int]:
    names = split_components(raw_value)

    if not names:
        tokens = [missing_token]
        ids = [vocab[missing_token]]
        num_components = 0
    else:
        tokens = [token_map.get(name, unknown_token) for name in names]
        ids = [vocab[token] for token in tokens]
        num_components = len(tokens)

    return ";".join(tokens), ";".join(map(str, ids)), num_components


def main() -> None:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

    solvent_results = df["Solvent"].apply(
        lambda value: map_components(
            value,
            SOLVENT_TOKEN_MAP,
            SOLVENT_VOCAB,
            "SOLV_MISSING",
            "SOLV_UNKNOWN",
        )
    )
    df["Solvent_Components"] = solvent_results.apply(lambda value: value[0])
    df["Solvent_Component_IDs"] = solvent_results.apply(lambda value: value[1])
    df["Num_Solvent_Components"] = solvent_results.apply(lambda value: value[2])

    catalyst_results = df["Catalyst"].apply(
        lambda value: map_components(
            value,
            CATALYST_TOKEN_MAP,
            CATALYST_VOCAB,
            "CAT_MISSING",
            "CAT_UNKNOWN",
        )
    )
    df["Catalyst_Components"] = catalyst_results.apply(lambda value: value[0])
    df["Catalyst_Component_IDs"] = catalyst_results.apply(lambda value: value[1])
    df["Num_Catalyst_Components"] = catalyst_results.apply(lambda value: value[2])

    unknown_solvent_rows = df[df["Solvent_Components"].str.contains("SOLV_UNKNOWN", regex=False)]
    unknown_catalyst_rows = df[df["Catalyst_Components"].str.contains("CAT_UNKNOWN", regex=False)]

    print("Unknown solvent rows:", len(unknown_solvent_rows))
    print("Unknown catalyst rows:", len(unknown_catalyst_rows))
    print("SOLV_MISSING rows:", int((df["Solvent_Components"] == "SOLV_MISSING").sum()))
    print("CAT_MISSING rows:", int((df["Catalyst_Components"] == "CAT_MISSING").sum()))

    if len(unknown_solvent_rows):
        print("Unknown solvent names:", sorted(set(sum(unknown_solvent_rows["Solvent"].apply(split_components), []))))
    if len(unknown_catalyst_rows):
        print("Unknown catalyst names:", sorted(set(sum(unknown_catalyst_rows["Catalyst"].apply(split_components), []))))

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    Path("solvent_vocab.json").write_text(
        json.dumps(SOLVENT_VOCAB, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path("catalyst_vocab.json").write_text(
        json.dumps(CATALYST_VOCAB, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
