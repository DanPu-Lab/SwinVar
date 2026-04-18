import pandas as pd
import pysam
import os
import re


def common_prefix(strs):
    """返回多个字符串的最短公共前缀"""
    if not strs:
        return ""
    shortest = min(strs, key=len)
    for i, ch in enumerate(shortest):
        if any(s[i] != ch for s in strs):
            return shortest[:i]
    return shortest


def make_vcf_record(row, ref):
    chrom, pos, gt = row["CHROM"], row["POS"], row["Prediction_Genotype"]
    p1, p2 = row["Prediction_1"], row["Prediction_2"]
    in1, in2 = row["INDEL_1"], row["INDEL_2"]
    ref_base = ref.fetch(chrom, pos - 1, pos)  # 获取参考碱基

    ref_seq = ref_base
    genotype = "./."

    # 处理单等位
    if gt == 1 or gt == 2:
        genotype = "0/1" if gt == 1 else "1/1"
        p = p1
        seq = re.sub(r"^[\+\-]*\d*", "", in1)
        if p == "Insert":
            alts = ref_base + seq
        elif p == "Deletion":
            ref_seq = ref_base + seq
            alts = ref_base
        else:
            alts = p

    # 处理双等位
    elif gt == 3:
        genotype = "1/2"
        ref_seq, alts = build_ref_alt(ref_base, p1, in1, p2, in2)

    return {
        "CHROM": chrom,
        "POS": pos,
        "ID": ".",
        "REF": ref_seq,
        "ALT": alts,
        "QUAL": ".",
        "FILTER": "PASS",
        "INFO": ".",
        "FORMAT": "GT",
        "SAMPLE": genotype,
    }


def write_vcf_with_header(df, output_path, reference):

    contigs = []
    with open(f"{reference}.fai", "r") as f:
        for line in f:
            chrom, length = line.strip().split("\t")[:2]
            contigs.append(f"##contig=<ID={chrom},length={length}>")

    header_lines = [
        "##fileformat=VCFv4.2",
        '##FILTER=<ID=PASS,Description="All filters passed">',
        '##FILTER=<ID=LowQual,Description="Low quality variant">',
        '##FILTER=<ID=RefCall,Description="Reference call">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        *contigs,
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(header_lines) + "\n")
        df.to_csv(f, sep="\t", header=False, index=False)


def get_vcf(df, output_vcf, reference):

    df = df.drop_duplicates(subset=["CHROM", "POS"]).reset_index(drop=True)
    df = df[df["Prediction_Genotype"] != 0]
    df = df[(df["Prediction_1"] != "REF_BASE") | (df["Prediction_2"] != "REF_BASE")]

    df["POS"] = df["POS"].astype(int)
    df[["INDEL_1", "INDEL_2"]] = (
        df["INDEL"].str.split("|", expand=True).iloc[:, :2].fillna("")
    )
    # df["INDEL_1"] = df["INDEL_1"].str.replace(r"^[\+\-]*\d*", "", regex=True)
    # df["INDEL_2"] = df["INDEL_2"].str.replace(r"^[\+\-]*\d*", "", regex=True)

    ref = pysam.FastaFile(reference)

    records = [make_vcf_record(r, ref) for _, r in df.iterrows()]
    vcf_df = pd.DataFrame(records)

    write_vcf_with_header(vcf_df, output_vcf, reference)

    # 运行shell命令
    os.system(f"bcftools sort {output_vcf} -o {output_vcf}.sorted")
    os.system(
        f"bcftools norm -f {reference} {output_vcf}.sorted -cs -Ov -o {output_vcf}"
    )
    os.system(f"rm -f {output_vcf}.sorted")


def build_ref_alt(ref_base: str, p1: str, indel1: str, p2: str, indel2: str):
    import re

    V = set("ACGTN")
    rb = ref_base.strip().upper()
    if len(rb) != 1 or rb not in V:
        raise ValueError(f"bad ref_base: {ref_base}")

    def parse_by_p(p: str, s: str):

        p = p.strip().upper()
        s = (s or "").strip().upper()

        m = re.fullmatch(r"([+-])?(\d+)?([ACGTN]+)", s) if s else None
        if p in V - {"N"}:
            return ("sub", p)
        if p in {"Insert", "INSERT", "INS"}:
            if not m:
                return ("sub", "N")
            sign, n, seq = m.groups()
            if n and int(n) != len(seq):
                return ("sub", "N")
            return ("ins", seq)
        if p in {"Deletion", "DELETION", "DEL"}:
            if not m:
                return ("sub", "N")
            sign, n, seq = m.groups()
            if n and int(n) != len(seq):
                return ("sub", "N")
            return ("del", seq)
        return ("sub", "N")

    (k1, s1), (k2, s2) = parse_by_p(p1, indel1), parse_by_p(p2, indel2)

    dels = [s for k, s in ((k1, s1), (k2, s2)) if k == "del"]
    L = max(dels, key=len) if dels else ""

    if dels and any(not L.startswith(d) for d in dels):
        raise ValueError(f"conflicting deletions {dels}; need reference context")
    REF = rb + L

    def alt_of(k, s):
        if k == "ins":
            return rb + s + L
        if k == "del":
            return rb + L[len(s) :]
        if k == "sub":
            return s + L
        raise ValueError

    a1, a2 = alt_of(k1, s1), alt_of(k2, s2)

    while len(REF) > 1 and all(len(x) > 1 and x[-1] == REF[-1] for x in (a1, a2)):
        REF, a1, a2 = REF[:-1], a1[:-1], a2[:-1]

    return REF, f"{a1},{a2}"
