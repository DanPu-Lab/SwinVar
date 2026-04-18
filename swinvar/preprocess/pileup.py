import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np
import subprocess
import tables
import time
import os
import re
from rich.console import Console


from swinvar.preprocess.bed_splitter import BedSplitter
from swinvar.preprocess.parallelizer import Parallelizer
from swinvar.preprocess.parameters import CHANNEL_SIZE, BASE2INDEX, flank_size, windows_size
from swinvar.preprocess.label import get_vcf_label, get_reference_label
from swinvar.preprocess.utils import execute_cmd, check_directory, setup_logger


@dataclass
class PileupConfig:
    """Pileup处理配置类

    Attributes:
        min_mapping_quality: 最小映射质量
        min_base_quality: 最小碱基质量
        min_freq: 最小频率阈值
        samtools_path: samtools工具路径
        windows_size: 窗口大小
        flank_size: 侧翼区域大小
        compression_level: 压缩级别
    """

    min_mapping_quality: int = 5
    min_base_quality: int = 5
    min_freq: float = 0.005
    samtools_path: str = "samtools"
    flank_size: int = flank_size
    windows_size: int = windows_size
    compression_level: int = 5


class PileupProcessor:
    """Pileup数据处理器类

    该类负责从BAM文件生成pileup数据并转换为特征矩阵。

    Attributes:
        bam_file: BAM文件路径
        bed_file: BED文件路径
        vcf_file: VCF文件路径
        reference_file: 参考基因组文件路径
        output_path: 输出目录路径
        config: Pileup处理配置
        console: Rich控制台对象
        vcf_dict: VCF标签字典
    """

    def __init__(
        self,
        bam_file: str,
        bed_file: str,
        vcf_file: str,
        reference_file: str,
        output_path: str,
        config: Optional[PileupConfig] = None,
    ):
        """初始化Pileup处理器

        Args:
            bam_file: BAM文件路径
            bed_file: BED文件路径
            vcf_file: VCF文件路径
            reference_file: 参考基因组文件路径
            output_path: 输出目录路径
            config: Pileup处理配置，如果为None则使用默认配置
        """
        self.bam_file = bam_file
        self.bed_file = bed_file
        self.vcf_file = vcf_file
        self.reference_file = reference_file
        self.output_path = output_path
        self.config = config or PileupConfig()
        self.bed_splitter = BedSplitter(bed_file, output_path)
        self.parallelizer = Parallelizer()

        self.bed_name = os.path.basename(bed_file).split(".")[0]
        self.vcf_dict = get_vcf_label(vcf_file, output_path)

    @staticmethod
    def _quality_to_scores(qual: str) -> int:
        """将质量字符串转换为分数

        Args:
            qual: 质量字符串

        Returns:
            质量分数
        """
        return ord(qual) - 33

    def process_parallel_tasks(self, max_workers: Optional[int] = None) -> int:

        # 分割BED文件
        bed_list = self.bed_splitter.split_by_chromosome()

        # 构建参数列表
        args_list = [(bed_chrom_file, self.output_path, self.config, self.bam_file, self.reference_file, self.vcf_dict) for bed_chrom_file in bed_list.values()]

        # 执行并行任务
        self.parallelizer.execute(
            create_pileup_data,
            args_list,
            pool_fn_index=0,
            max_workers=max_workers or len(args_list),
            show_progress=True,
            use_threads=False,
        )

        return len(bed_list)



def create_pileup_data(bed_file: List[str], output_path: str, config: PileupConfig, bam_file: str, reference_file: str, vcf_dict: Dict[str, List]) -> int:
    """创建pileup数据

    Args:
        bed_file: 可选的BED文件路径，如果为None则使用self.bed_file

    Returns:
        处理的数据点数量
    """

    start_time = time.time()

    bed_name = Path(bed_file).stem

    log_path = os.path.join(output_path, "log", "pileup")
    check_directory(log_path)
    log_file = os.path.join(log_path, f"{bed_name}.log")
    pileup_logger = setup_logger(filename=log_file, mode="w", level=logging.INFO, format="%(asctime)s - %(message)s")

    pileup_logger.info(f"[START] Generating mpileup data...")

    output_path = os.path.join(output_path, "pileup")
    check_directory(output_path)

    output_file = os.path.join(output_path, f"{bed_name}.h5")

    # 保存结果文件
    # tables.set_blosc_max_threads(64)
    filters = tables.Filters(complevel=5, complib="blosc")
    features_atom = tables.Float32Atom()
    variant_labels_atom = tables.Int32Atom()
    ChromPosRef_atom = tables.StringAtom(itemsize=5000)
    np_output_file = tables.open_file(output_file, mode="w", filters=filters)
    features_np = np_output_file.create_earray(
        where=np_output_file.root,
        name="Features",
        atom=features_atom,
        shape=(0, windows_size, 3, CHANNEL_SIZE),
        filters=filters,
    )
    variant_labels_np = np_output_file.create_earray(
        where=np_output_file.root,
        name="Variant_labels",
        atom=variant_labels_atom,
        shape=(0, 3),
        filters=filters,
    )
    ChromPosRef_np = np_output_file.create_earray(
        where=np_output_file.root,
        name="ChromPosRef",
        atom=ChromPosRef_atom,
        shape=(0, 1),
        filters=filters,
    )

    samtools_mpileup = execute_cmd(
        f"{config.samtools_path} mpileup {bam_file} -f {reference_file} -l {bed_file} -B --reverse-del --min-MQ {config.min_mapping_quality} --min-BQ {config.min_base_quality}",
        stdout=subprocess.PIPE
    )

    windows_np = np.zeros((windows_size, 3, CHANNEL_SIZE), dtype=np.float32)
    pre_pos = float("-inf")
    pre_chrom = 0
    np_index = 0
    candidate = []

    for row in samtools_mpileup.stdout:
        info = row.rstrip().split("\t")
        chrom = int(info[0][3:]) if info[0].startswith("chr") else int(info[0])
        pos = int(info[1])
        ref = info[2].upper()
        depth = int(info[3])
        alt = info[4]
        qual = info[5]

        if depth < 2:
            continue

        # 计算碱基频率
        alt = re.sub(r"\^.", "", alt)
        alt = alt.replace("$", "")
        alt = alt.replace(".", ref)
        alt = alt.replace("N", ref)
        alt = alt.replace(",", ref.lower())
        alt = alt.replace("n", ref.lower())

        # 大写: . *
        # 小写: , #
        # indel不参与depth计算
        bases_list = list(alt)
        alt_list = list(alt)
        M_I = 0
        m_i = 0
        M_D = 0
        m_d = 0

        # 处理indel
        if "+" in bases_list or "-" in bases_list:
            for base_index in range(len(bases_list)):
                base = bases_list[base_index]
                if base == "+" or base == "-":
                    # 获取indel的长度
                    for i in range(base_index + 1, len(bases_list)):
                        if not bases_list[i].isdigit():
                            indel_num = int("".join(bases_list[base_index + 1 : i]))
                            break
                    indel = "".join(bases_list[base_index: i + indel_num])
                    alt_list[base_index: i + indel_num] = [None] * (i + indel_num - base_index)
                    alt_list.append(indel)


        alt_list = list(filter(None, alt_list))
        alt_counter = Counter(alt_list)

        # 计算质量
        base_list = np.array([item for item in alt_list if "+" not in item and "-" not in item])
        qual_list = np.array([ord(q) - 33 for q in qual])

        unique_bases = np.unique(base_list)
        qual_np = np.zeros((2, CHANNEL_SIZE), dtype=np.float32)
        for base in unique_bases:
            qual_np[0, BASE2INDEX[base]] = np.mean(qual_list[base_list == base])
            qual_np[1, BASE2INDEX[base]] = np.var(qual_list[base_list == base])

        qual_np[0, BASE2INDEX[ref]] *= -1
        qual_np[0, BASE2INDEX[ref.lower()]] *= -1
        qual_np[1, BASE2INDEX[ref]] *= -1
        qual_np[1, BASE2INDEX[ref.lower()]] *= -1

        # 计算碱基数量
        pileup_dict = defaultdict(int)
        alt_dict = defaultdict(int)
        pileup_np = np.zeros(CHANNEL_SIZE, dtype=np.float32)
        for base, count in alt_counter.items():
            alt_dict[base.upper()] += count
            if base.startswith("+"):
                pileup_dict["I"] += count
                if base[2] in "ACGT":
                    pileup_np[BASE2INDEX["I"]] += count
                    M_I = max(M_I, count)
                else:
                    pileup_np[BASE2INDEX["i"]] += count
                    m_i = max(m_i, count)
            elif base.startswith("-"):
                pileup_dict["D"] += count
                if base[2] in "ACGT":
                    pileup_np[BASE2INDEX["D"]] += count
                    M_D = max(M_D, count)
                else:
                    pileup_np[BASE2INDEX["d"]] += count
                    m_d = max(m_d, count)
            else:
                pileup_np[BASE2INDEX[base]] += count
                if base.upper() in "ACGT":
                    pileup_dict[base.upper()] += count
            
        pileup_np[BASE2INDEX["M_I"]] = M_I
        pileup_np[BASE2INDEX["m_i"]] = m_i
        pileup_np[BASE2INDEX["M_D"]] = M_D
        pileup_np[BASE2INDEX["m_d"]] = m_d

        pileup_np[BASE2INDEX[ref]] *= -1
        pileup_np[BASE2INDEX[ref.lower()]] *= -1

        # 统计indel信息
        alt_dict.pop(ref, None)
        for key in list(alt_dict.keys()):
            if "+" not in key and "-" not in key:
                alt_dict.pop(key, None)
        indel_list = "|".join([k for k, v in sorted(alt_dict.items(), key=lambda x: x[1], reverse=True)])

        # 计算碱基频率
        pileup_dict.pop(ref, None)
        af_dict = {k: v/depth for k, v in pileup_dict.items()}
        af_list = sorted(af_dict.items(), key=lambda x: x[1], reverse=True)

        # 候选位点
        alt_af = af_list[0][1] if len(af_list) > 0 else 0
        
        if len(candidate) != 0 and (pre_chrom != chrom or pos > candidate[0][1] + flank_size):
            if pre_chrom == candidate[0][0] and pre_pos >= candidate[0][1] and pre_pos <= candidate[0][1] + flank_size:

                candidate_chrom, candidate_pos, candidate_ref, candidate_index, indel_info = candidate.pop(0)
                key = f"{candidate_chrom}:{candidate_pos}"
                if key in vcf_dict:
                    variant_label_1, variant_label_2, genotype_label = vcf_dict.get(key)
                    variant_labels_np.append(np.array([variant_label_1, variant_label_2, genotype_label]).reshape(1, -1))
                    
                else:
                    variant_label_1, variant_label_2, genotype_label = get_reference_label(candidate_ref)
                    variant_labels_np.append(np.array([variant_label_1, variant_label_2, genotype_label]).reshape(1, -1))

                shift_index = candidate_index - flank_size
                zeros_index = (pre_pos - candidate_pos) - flank_size
                candidate_np = np.concatenate((windows_np[shift_index: ], windows_np[ :shift_index]), axis=0, dtype=np.float32)
                candidate_np[zeros_index: ] = np.zeros_like(candidate_np[zeros_index: ], dtype=np.float32)

                features_np.append(candidate_np.reshape(1, windows_size, 3, CHANNEL_SIZE))

                info = ":".join([key, candidate_ref+candidate_ref, indel_info])
                ChromPosRef_np.append(np.array([info]).reshape(-1, 1))

                for i in range(len(candidate)):
                    candidate_chrom, candidate_pos, candidate_ref, candidate_index, indel_info = candidate[i]
                    key = f"{candidate_chrom}:{candidate_pos}"
                    if key in vcf_dict:
                        variant_label_1, variant_label_2, genotype_label = vcf_dict.get(key)
                        variant_labels_np.append(np.array([variant_label_1, variant_label_2, genotype_label]).reshape(1, -1))
                        
                    else:
                        variant_label_1, variant_label_2, genotype_label = get_reference_label(candidate_ref)
                        variant_labels_np.append(np.array([variant_label_1, variant_label_2, genotype_label]).reshape(1, -1))
                        
                    shift_index = candidate_index - flank_size
                    zeros_index = (pre_pos - candidate_pos) - flank_size
                    candidate_np = np.concatenate((windows_np[shift_index: ], windows_np[ :shift_index]), axis=0, dtype=np.float32)
                    candidate_np[zeros_index: ] = np.zeros_like(candidate_np[zeros_index: ], dtype=np.float32)

                    features_np.append(candidate_np.reshape(1, windows_size, 3, CHANNEL_SIZE))

                    info = ":".join([key, candidate_ref+candidate_ref, indel_info])
                    ChromPosRef_np.append(np.array([info]).reshape(-1, 1))

                candidate = []

            np_index = 0
            windows_np = np.zeros((windows_size, 3, CHANNEL_SIZE), dtype=np.float32)
        
        if alt_af >= config.min_freq:
            candidate.append((chrom, pos, ref, np_index, indel_list))

        pre_chrom = chrom
        pre_pos = pos

        windows_np[np_index, 0] = pileup_np / depth
        windows_np[np_index, 1:] = qual_np
        np_index = (np_index + 1) % windows_size


        if len(candidate) != 0 and chrom == candidate[0][0] and pos == candidate[0][1]:

            candidate_chrom, candidate_pos, candidate_ref, candidate_index, indel_info = candidate[0]

            shift_index = candidate_index - flank_size
            windows_np = np.concatenate((windows_np[shift_index: ], windows_np[ :shift_index]), axis=0, dtype=np.float32)
            windows_np[flank_size + 1: ] = np.zeros_like(windows_np[flank_size + 1: ], dtype=np.float32)

            raw_index = -shift_index
            windows_np = np.concatenate((windows_np[-shift_index: ], windows_np[: -shift_index]), axis=0, dtype=np.float32)

        if len(candidate) != 0 and chrom == candidate[0][0] and pos - candidate[0][1] == flank_size:
            candidate_chrom, candidate_pos, candidate_ref, candidate_index, indel_info = candidate.pop(0)
            key = f"{candidate_chrom}:{candidate_pos}"
            if key in vcf_dict:
                variant_label_1, variant_label_2, genotype_label = vcf_dict.get(key)
                variant_labels_np.append(np.array([variant_label_1, variant_label_2, genotype_label]).reshape(1, -1))
                
            else:
                variant_label_1, variant_label_2, genotype_label = get_reference_label(candidate_ref)
                variant_labels_np.append(np.array([variant_label_1, variant_label_2, genotype_label]).reshape(1, -1))
                
            shift_index = candidate_index - flank_size
            candidate_np = np.concatenate((windows_np[shift_index: ], windows_np[ :shift_index]), axis=0, dtype=np.float32)
            features_np.append(candidate_np.reshape(1, windows_size, 3, CHANNEL_SIZE))

            info = ":".join([key, candidate_ref+candidate_ref, indel_info])
            ChromPosRef_np.append(np.array([info]).reshape(-1, 1))

    if len(candidate) != 0:
        for i in range(len(candidate)):
            candidate_chrom, candidate_pos, candidate_ref, candidate_index, indel_info = candidate[i]
            key = f"{candidate_chrom}:{candidate_pos}"

            if key in vcf_dict:
                variant_label_1, variant_label_2, genotype_label = vcf_dict.get(key)
                variant_labels_np.append(np.array([variant_label_1, variant_label_2, genotype_label]).reshape(1, -1))
                
            else:
                variant_label_1, variant_label_2, genotype_label = get_reference_label(candidate_ref)
                variant_labels_np.append(np.array([variant_label_1, variant_label_2, genotype_label]).reshape(1, -1))
                
            shift_index = candidate_index - flank_size
            zeros_index = (pre_pos - candidate_pos) - flank_size
            candidate_np = np.concatenate((windows_np[shift_index: ], windows_np[ :shift_index]), axis=0, dtype=np.float32)
            candidate_np[zeros_index: ] = np.zeros_like(candidate_np[zeros_index: ], dtype=np.float32)

            features_np.append(candidate_np.reshape(1, windows_size, 3, CHANNEL_SIZE))

            info = ":".join([key, candidate_ref+candidate_ref, indel_info])
            ChromPosRef_np.append(np.array([info]).reshape(-1, 1))

    samtools_mpileup.stdout.close()
    samtools_mpileup.wait()

    num_data = len(variant_labels_np)

    np_output_file.close()

    end_time = time.time()
    consum_time = end_time - start_time
    hours, remainder = divmod(consum_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    pileup_logger.info(f"[SUMMARY] Total number of data: {num_data}")
    pileup_logger.info(f"[COMPLETED] Samtools mpileup data generation finished successfully !")
    pileup_logger.info(f"[TIME TAKEN] Total execution time: {hours:.0f}h{minutes:.0f}m{seconds:.0f}s")






if __name__ == "__main__":
    # 示例用法
    # 配置参数
    config = PileupConfig(min_mapping_quality=0, min_base_quality=0, min_freq=0.005)

    bam_file = "/data2/lijie/data/hs37d5/HG002/151002_7001448_0359_AC7F6GANXX_Sample_HG002-EEogPU_v02-KIT-Av5_AGATGTAC_L008.posiSrt.markDup.bam"
    bed_file = "/data2/lijie/data/hs37d5/HG002/HG002.bed"
    vcf_file = (
        "/data2/lijie/data/hs37d5/HG002/HG002_GRCh37_1_22_v4.2.1_benchmark.vcf.gz"
    )
    reference_file = "/data2/lijie/reference/hs37d5/hs37d5.fa"
    output_path = "/data2/lijie/result/Transformer_pileup_3_channel/HG002_WES"

    try:
        # 创建处理器并执行并行任务
        processor = PileupProcessor(
            bam_file, bed_file, vcf_file, reference_file, output_path, config
        )
        processor.process_parallel_tasks()

    except Exception as e:
        print(f"处理失败: {e}")
