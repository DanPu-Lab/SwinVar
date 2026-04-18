import numpy as np
import os

from cyvcf2 import VCF

from swinvar.preprocess.parameters import VARIANT_LABELS, GENOTYPE_LABELS
from swinvar.preprocess.utils import execute_cmd, check_directory


def get_genotype(gt1, gt2):
    gt = gt1 + gt2
    if gt == 0:
        return GENOTYPE_LABELS.get("hom_ref")
    if gt == 1:
        return GENOTYPE_LABELS.get("het")
    if gt == 2:
        return GENOTYPE_LABELS.get("hom_alt")
    if gt == 3:
        return GENOTYPE_LABELS.get("het_alt")


def get_variant_type(reference, alt):
    if len(reference) == 1 and len(alt) == 1:
        return alt
    if len(reference) > len(alt):
        return "Deletion"
    if len(reference) < len(alt):
        return "Insert"
    if len(reference) == len(alt):
        return alt[0]


def get_variant_label(reference, alt_list, genotype):

    if len(alt_list) == 1:
        alt_list = alt_list * 2

    variant_type_list = [get_variant_type(reference, alt) for alt in alt_list]
    variant_label = sorted(variant_type_list, key=lambda x: VARIANT_LABELS.get(x))

    return VARIANT_LABELS[variant_label[0]], VARIANT_LABELS[variant_label[1]]


def get_vcf_label(vcf_file, output_path):

    check_directory(output_path)
    illumina_vcf_file = os.path.join(output_path, "Illumina.vcf.gz")

    illumina_gz_vcf = execute_cmd(
        f"bcftools view -i 'INFO/platformnames=\"Illumina\"' -Oz -o {illumina_vcf_file} {vcf_file}"
    )
    illumina_gz_vcf.wait()

    gz_index = execute_cmd(f"bcftools index -f -t {illumina_vcf_file}")
    gz_index.wait()

    vcf = VCF(illumina_vcf_file)

    vcf_dict = {}

    for info in vcf:
        reference = info.REF
        alt_list = info.ALT
        gt1, gt2, phased = info.genotypes[0]

        key = f"{info.CHROM}:{info.POS}"

        if "*" in info.ALT:
            gt1 = 0
            gt2 = 1
            alt_list.remove("*")

        genotype = get_genotype(gt1, gt2)
        label_1, label_2 = get_variant_label(reference, alt_list, genotype)

        # 提供变异类型和基因型
        vcf_dict[key] = [label_1, label_2, genotype]
    return vcf_dict


def get_reference_label(reference):

    genotype = 0

    return VARIANT_LABELS[reference], VARIANT_LABELS[reference], genotype
