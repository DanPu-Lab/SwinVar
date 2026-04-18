import pandas as pd
import pysam
import os

from collections import Counter
from typing import Tuple, Optional, Dict



class DeepVariantPostProcessor:
    """
    基于深度学习预测结果和BAM文件生成VCF的后处理类。
    实现了Indel的BAM回溯重组装和多等位基因的对齐逻辑。
    """

    def __init__(self, bam_path: str, ref_fasta_path: str, class_map: Dict[int, str]={0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'INS', 5: 'DEL'}):
        """
        初始化处理器。
        
        Args:
            bam_path: BAM文件路径 (必须有 .bai 索引)
            ref_fasta_path: 参考基因组 FASTA 路径 (必须有 .fai 索引)
            class_map: 类别映射字典 {0: 'A', 4: 'INS', ...}
        """
        self.bam_path = bam_path
        self.ref_fasta_path = ref_fasta_path
        self.class_map = class_map
        self.min_mq = 20  # 最小比对质量过滤

    def _get_ref_sequence(self, fasta: pysam.FastaFile, chrom: str, start: int, end: int) -> str:
        """获取参考基因组序列 (Safe fetch, 0-based [start, end))"""
        try:
            return fasta.fetch(chrom, start, end).upper()
        except Exception as e:
            return "N" * (end - start)

    def _resolve_indel_from_bam(self, 
                                samfile: pysam.AlignmentFile, 
                                fasta: pysam.FastaFile, 
                                chrom: str, 
                                pos: int, 
                                indel_type: str, 
                                rank: int = 1) -> Tuple[Optional[str], Optional[str]]:
        """
        核心方法：回溯 BAM 获取 Indel 序列。
        
        Args:
            pos: 1-based VCF position
            indel_type: 'INS' or 'DEL'
            rank: 1 for most common, 2 for second most common
            
        Returns:
            (REF, ALT) strings compliant with VCF standard, or (None, None)
        """
        # pysam fetch 使用 0-based 坐标
        # 我们搜索 pos-1 (anchor) 到 pos 这个极小窗口
        try:
            iter_reads = samfile.fetch(chrom, pos - 1, pos)
        except ValueError:
            return None, None

        candidates = []
        
        for read in iter_reads:
            if read.mapping_quality < self.min_mq:
                continue
            
            # 快速跳过不包含 Indel 的 reads
            if indel_type == 'INS' and 1 not in [op for op, _ in read.cigar]: continue
            if indel_type == 'DEL' and 2 not in [op for op, _ in read.cigar]: continue

            read_ref_pos = read.reference_start
            read_seq_pos = 0
            found = False
            
            # CIGAR codes: 0=M, 1=I, 2=D, 3=N, 4=S
            for cigar_op, cigar_len in read.cigar:
                # 优化：如果当前操作位置已超过目标，提前退出
                if read_ref_pos > pos: 
                    break
                
                # 判定 Indel 发生的位置 (通常紧跟在 POS 之后)
                # 这里的逻辑判定需非常严谨，通常 Indel 发生在 Anchor (pos) 之后
                if read_ref_pos == pos or read_ref_pos == (pos - 1):
                    if indel_type == 'INS' and cigar_op == 1:
                        # 提取插入序列
                        ins_seq = read.query_sequence[read_seq_pos : read_seq_pos + cigar_len]
                        candidates.append(ins_seq)
                        found = True
                    elif indel_type == 'DEL' and cigar_op == 2:
                        # 记录缺失长度
                        candidates.append(cigar_len)
                        found = True
                
                if found: break
                
                # Update coordinates
                if cigar_op in [0, 2, 3, 7, 8]: # Consumes Ref
                    read_ref_pos += cigar_len
                if cigar_op in [0, 1, 4, 7, 8]: # Consumes Read
                    read_seq_pos += cigar_len

        if not candidates:
            return None, None

        # 投票
        counts = Counter(candidates).most_common()
        if len(counts) >= rank:
            best_candidate = counts[rank-1][0]
        else:
            best_candidate = counts[0][0] # Fallback to rank 1

        # 构建 VCF 格式 (Anchor + Event)
        # 获取 POS 位置的碱基作为 Anchor (1-based POS -> 0-based index POS-1)
        ref_anchor = self._get_ref_sequence(fasta, chrom, pos - 1, pos)
        
        if indel_type == 'INS':
            # VCF Spec: REF=Anchor, ALT=Anchor+Insert
            return ref_anchor, ref_anchor + best_candidate
        
        elif indel_type == 'DEL':
            # VCF Spec: REF=Anchor+Deleted, ALT=Anchor
            del_len = best_candidate
            deleted_seq = self._get_ref_sequence(fasta, chrom, pos, pos + del_len)
            return ref_anchor + deleted_seq, ref_anchor
            
        return None, None

    def _harmonize_alleles(self, 
                           ref_v1: str, alt_v1: str, 
                           ref_v2: str, alt_v2: str) -> Tuple[str, str, str]:
        """
        对齐两个等位基因 (SNP+Indel 或 Indel+Indel) 到统一的 REF。
        实现右侧填充 (Right Padding)。
        """
        len_ref1 = len(ref_v1)
        len_ref2 = len(ref_v2)
        
        # 确定最长的 REF 作为基准
        if len_ref1 >= len_ref2:
            final_ref = ref_v1
            final_alt1 = alt_v1
            # 补齐 V2: 将 final_ref 多出来的后缀加到 alt_v2 后面
            padding = final_ref[len_ref2:]
            final_alt2 = alt_v2 + padding
        else:
            final_ref = ref_v2
            final_alt2 = alt_v2
            # 补齐 V1
            padding = final_ref[len_ref1:]
            final_alt1 = alt_v1 + padding
            
        return final_ref, final_alt1, final_alt2

    def process_to_vcf(self, df: pd.DataFrame, output_path: str):
        """
        主入口：将 DataFrame 转换为 VCF 文件。
        """

        contigs = []
        with open(f"{self.ref_fasta_path}.fai", "r") as f:
            for line in f:
                chrom, length = line.strip().split("\t")[:2]
                contigs.append(f"##contig=<ID={chrom},length={length}>")

        headers = [
            "##fileformat=VCFv4.2",
            '##FILTER=<ID=PASS,Description="All filters passed">',
            '##FILTER=<ID=LowQual,Description="Low quality variant">',
            '##FILTER=<ID=RefCall,Description="Reference call">',
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            *contigs,
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE",
        ]

        # 使用上下文管理器安全打开文件
        with pysam.AlignmentFile(self.bam_path, "rb") as samfile, \
             pysam.FastaFile(self.ref_fasta_path) as fasta, \
             open(output_path, 'w') as vcf_out:
            
            # Write Header
            vcf_out.write("\n".join(headers) + "\n")
            
            count_processed = 0
            count_failed = 0

            for _, row in df.iterrows():
                try:
                    chrom = str(row['CHROM'])
                    pos = int(row['POS'])
                    ref_base_df = row['REF']
                    pred_gt = row['Prediction_Genotype']
                    
                    if pred_gt == '0/0': 
                        continue

                    pred_v1 = self.class_map[row['Prediction_1']]
                    pred_v2 = self.class_map[row['Prediction_2']]

                    # -------------------------------------------------
                    # 内部闭包：解析单个 Allele (SNP or Indel)
                    # -------------------------------------------------
                    def resolve_allele(pred_class, rank=1):
                        if pred_class in ['A', 'C', 'G', 'T']:
                            return ref_base_df, pred_class
                        elif pred_class in ['INS', 'DEL']:
                            return self._resolve_indel_from_bam(
                                samfile, fasta, chrom, pos, pred_class, rank
                            )
                        return None, None

                    # 1. 解析 Allele 1
                    raw_ref1, raw_alt1 = resolve_allele(pred_v1, rank=1)
                    if not raw_ref1:
                        count_failed += 1
                        continue

                    final_ref = raw_ref1
                    final_alt_str = raw_alt1
                    gt_str = "0/1"

                    # 2. 根据基因型处理 Allele 2
                    if pred_gt == '1/1':
                        gt_str = "1/1"
                        
                    elif pred_gt == '1/2':
                        # 复合杂合处理
                        rank_v2 = 2 if pred_v1 == pred_v2 else 1
                        raw_ref2, raw_alt2 = resolve_allele(pred_v2, rank=rank_v2)
                        
                        if raw_ref2:
                            # 成功解析出两个变异，进行对齐
                            final_ref, fixed_alt1, fixed_alt2 = self._harmonize_alleles(
                                raw_ref1, raw_alt1, raw_ref2, raw_alt2
                            )
                            final_alt_str = f"{fixed_alt1},{fixed_alt2}"
                            gt_str = "1/2"
                        else:
                            # V2 解析失败 (比如 reads 支持不够)，降级为 0/1
                            # 这是一个保底策略，防止因为 V2 失败丢掉整个变异
                            gt_str = "0/1"

                    # 3. 写入 VCF
                    # 简单的质量分数占位符，如果你的模型有概率值，可以在这里填入
                    qual = "." 
                    record = f"{chrom}\t{pos}\t.\t{final_ref}\t{final_alt_str}\t{qual}\tPASS\t.\tGT\t{gt_str}\n"
                    vcf_out.write(record)
                    count_processed += 1

                except Exception as e:
                    count_failed += 1
                    continue


def get_vcf(df, bam, output_vcf, reference):

    processor = DeepVariantPostProcessor(bam, reference)
    processor.process_to_vcf(df, output_vcf)

    # 运行shell命令
    os.system(f"bcftools sort {output_vcf} -o {output_vcf}.sorted")
    os.system(
        f"bcftools norm -m - -f {reference} {output_vcf}.sorted -cs -Ov -o {output_vcf}"
    )
    os.system(f"rm -f {output_vcf}.sorted")
