import argparse

from swinvar.preprocess.parameters import (
    flank_size,
    windows_size,
    CHANNEL_SIZE,
    VARIANT_SIZE,
)
from swinvar.preprocess.pileup import PileupConfig, PileupProcessor
from swinvar.preprocess.balance import DataBalancer
from swinvar.training.train import train_model
from swinvar.inference.call_variant import call_model


def get_parser():
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Variant Calling",
        allow_abbrev=False,
    )

    parser.add_argument("--bam", type=str, nargs='+', required=True, help="bam文件路径")
    parser.add_argument("--bed", type=str, nargs='+', required=True, help="bed文件路径")
    parser.add_argument("--vcf", type=str, nargs='+', required=True, help="vcf文件路径")
    parser.add_argument("--output_dir", type=str, nargs='+', required=True, help="输出目录")
    parser.add_argument("--reference", type=str, required=True, help="参考基因组文件路径")
    parser.add_argument("--min_mapping_quality", type=int, default=10, help="最小比对质量")
    parser.add_argument("--min_base_quality", type=int, default=10, help="最小碱基质量")
    parser.add_argument("--min_freq", type=float, default=0.12, help="最小突变频率")
    parser.add_argument("--samtools", type=str, default="samtools", help="samtools路径")
    parser.add_argument("--pileup", action='store_true', help="是否启用pileup module")

    parser.add_argument("--ref_var_ratio", type=int, default=2, help="参考与突变数据的比例")
    parser.add_argument("--chunk_size", type=int, default=500000, help="分块大小")
    parser.add_argument("--balance", action='store_true', help="是否启用balance module")

    parser.add_argument("--train_input", type=str, nargs='+', help="训练数据目录(默认用chr20用于验证)")
    parser.add_argument("--train_output", type=str, help="训练输出目录")
    parser.add_argument("--call_input", type=str, help="突变识别数据目录")
    parser.add_argument("--call_bam", type=str, help="突变识别bam文件")
    parser.add_argument("--output_vcf", type=str, default=None, help="VCF输出路径")
    parser.add_argument("--patience", type=int, default=50, help="早停的等待轮数")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader启用的子进程数")
    parser.add_argument("--train", action='store_true', help="是否启用训练")
    parser.add_argument("--checkpoint", action='store_true', help="是否启用checkpoint恢复")
    parser.add_argument("--call", action='store_true', help="是否启用突变识别")
    parser.add_argument("--fune_turning", action='store_true', help="是否启用微调")

    return parser


def main():

    parser = get_parser()
    args = parser.parse_args()

    if args.pileup or args.balance:

        bam_file_list = args.bam
        bed_file_list = args.bed
        vcf_file_list = args.vcf
        output_path_list = args.output_dir
        call_bam = args.call_bam

        assert (
            len(bam_file_list) == len(bed_file_list)
            and len(vcf_file_list) == len(output_path_list)
            and len(bam_file_list) == len(vcf_file_list)
        ), "文件提供错误"

        samtools_path = args.samtools
        reference_file = args.reference
        min_mapping_quality = args.min_mapping_quality
        min_base_quality = args.min_base_quality
        min_freq = args.min_freq

        ref_variant_ratio = args.ref_var_ratio
        chunk_size = args.chunk_size

        for i in range(len(bam_file_list)):
            bam_file = bam_file_list[i]
            bed_file = bed_file_list[i]
            vcf_file = vcf_file_list[i]
            output_path = output_path_list[i]

            if args.pileup:
                pileup_config = PileupConfig(
                    min_mapping_quality,
                    min_base_quality,
                    min_freq,
                    samtools_path,
                    flank_size,
                    windows_size,
                )
                pileup_processor = PileupProcessor(
                    bam_file,
                    bed_file,
                    vcf_file,
                    reference_file,
                    output_path,
                    pileup_config,
                )
                pileup_processor.process_parallel_tasks()

            if args.balance:
                balancer = DataBalancer(output_path, ref_variant_ratio, chunk_size)
                balancer.balance_data()

    # Train
    args_model = {
        "input_path": args.train_input,
        "output_path": args.train_output,
        "reference": args.reference,
        "ref_var_ratio": args.ref_var_ratio,
        "file": f"balance_{args.ref_var_ratio}",
        "epochs": 300,
        "batch_size": 600,
        "feature_size": (windows_size, CHANNEL_SIZE),
        "num_classes": VARIANT_SIZE,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "embed_dim": 192,
        "patch_size": 2,
        "window_size": 3,
        "n_routed_experts": 8,
        "n_activated_experts": 2,
        "n_expert_groups": 1,
        "n_limited_groups": 1,
        "score_func": "sigmoid",
        "route_scale": 1,
        "moe_inter_dim": 64,
        "n_shared_experts": 1,
        "drop_rate": 0.1,
        "drop_path_rate": 0.1,
        "attn_drop_rate": 0.1,
        "lr": 0.001,
        "weight_decay": 0.01,
        "num_workers": args.num_workers,
        "patience": args.patience,
        "model_save_path": "best_model.pth",
        "log_file_train": "train_log.txt",
        "log_file_call": "call_log.txt",
        "matplot_save_path": "train_process.png",
        "hyperparams_log": "hyperparams.xlsx",
        "checkpoint": args.checkpoint,
        "call_bam":call_bam,
        "call_batch_size": 5000,
        "call_input_path": args.call_input,
        "call_file": "pileup",
        "ft": args.fune_turning,
        "ft_file": "pileup",
        "output_vcf": args.output_vcf,
        "ft_batch_size": 1200,
        "ft_epochs": 30,
        "ft_patience": 6,
        "pct_start": 0.3,
        "factor": [1, 1],
        "ft_strategy": "last_k_blocks_lora", # heads_only / last_stage_core / last_k_blocks_lora
        "ft_use_llrd": True,
        "ft_last_k_blocks": 4,
        "ft_lora_r": 16,
        "ft_lora_alpha": 32,
        "ft_lora_dropout": 0.1,
        "ft_base_lr": 0.001,
        "ft_head_lr": 0.001,
        "ft_layer_decay": 0.75,
        "genotype": True,
    }

    if args.train:
        train_model(args_model)

    if args_model["genotype"]:
        from swinvar.trainers.call_f1_opt import optimize_thresholds
        optimize_thresholds(args_model)

    if args.call:
        from swinvar.inference.f1_variant_caller import call_model
        call_model(args_model)


if __name__ == "__main__":

    main()
