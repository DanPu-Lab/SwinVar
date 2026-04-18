import tables
import torch
import time
import os

from torch.utils.data import DataLoader

from swinvar.postprocess.metrics_calculator import variant_df, calculate_metrics
from swinvar.models.swin_var import SwinVar
from swinvar.models.dataset import CallingDataset
from swinvar.preprocess.utils import setup_logger, check_directory


def test_model(args):

    start_time = time.time()

    if args["LoRA"]:
        output_path = os.path.join(args["output_path"], "train_moe", f"LoRA_{args["LoRA_file"]}_{args["ref_var_ratio"]}")
        model_path = os.path.join(output_path, f"LoRA_{args["model_save_path"]}")
    else:
        output_path = os.path.join(args["output_path"], "train_moe", args["file"])
        model_path = os.path.join(output_path, f"{args["model_save_path"]}")

    check_directory(output_path)

    log_file = os.path.join(output_path, args["log_file_test"])

    input_path = os.path.join(args["test_input_path"], args["test_file"])
    inputs_files = [os.path.join(input_path, file) for file in os.listdir(input_path)]

    tables_data_list = [tables.open_file(file, "r") for file in inputs_files]
    dataset = CallingDataset(tables_data_list)

    test_dataloader = DataLoader(
        dataset,
        batch_size=args["test_batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=args["num_workers"],
    )

    device = torch.device(f"cuda")
    model = SwinVar(
        feature_size=args["feature_size"],
        num_classes=args["num_classes"],
        embed_dim=args["embed_dim"],
        patch_size=args["patch_size"],
        window_size=args["window_size"],
        n_routed_experts=args["n_routed_experts"],
        n_activated_experts=args["n_activated_experts"],
        n_expert_groups=args["n_expert_groups"],
        n_limited_groups=args["n_limited_groups"],
        score_func=args["score_func"],
        route_scale=args["route_scale"],
        moe_inter_dim=args["moe_inter_dim"],
        n_shared_experts=args["n_shared_experts"],
        depths=args["depths"],
        num_heads=args["num_heads"],
        drop_rate=args["drop_rate"],
        drop_path_rate=args["drop_path_rate"],
        attn_drop_rate=args["attn_drop_rate"],
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    test_num = 0
    test_variant_1_acc, test_variant_2_acc, test_genotype_acc = .0, .0, .0
    test_variant_1_predictions, test_variant_2_predictions, test_genotype_predictions = [], [], []
    test_variant_1_labels, test_variant_2_labels, test_genotype_labels = [], [], []
    test_chrom = []
    test_pos = []
    test_ref = []
    test_indel_info = []

    # ------------画图------------
    y_true_genotype, y_score_genotype = [], []
    y_true_variant_1, y_score_variant_1 = [], []
    y_true_variant_2, y_score_variant_2 = [], []
    # 保存结果文件
    filters = tables.Filters(complevel=5, complib="blosc")
    scores_atom = tables.Float32Atom()
    labels_atom = tables.Int32Atom()
    np_output_file = tables.open_file(os.path.join(output_path, "pr_data.h5"), mode="w", filters=filters)
    y_score_genotype_np = np_output_file.create_earray(
        where=np_output_file.root,
        name="Score_Genotype",
        atom=scores_atom,
        shape=(0, 4),
        filters=filters,
    )
    y_true_genotype_np = np_output_file.create_earray(
        where=np_output_file.root,
        name="Labels_Genotype",
        atom=labels_atom,
        shape=(0, 1),
        filters=filters,
    )
    y_score_variant_1_np = np_output_file.create_earray(
        where=np_output_file.root,
        name="Score_Variant_1",
        atom=scores_atom,
        shape=(0, 6),
        filters=filters,
    )
    y_true_variant_1_np = np_output_file.create_earray(
        where=np_output_file.root,
        name="Labels_Variant_1",
        atom=labels_atom,
        shape=(0, 1),
        filters=filters,
    )
    y_score_variant_2_np = np_output_file.create_earray(
        where=np_output_file.root,
        name="Score_Variant_2",
        atom=scores_atom,
        shape=(0, 6),
        filters=filters,
    )
    y_true_variant_2_np = np_output_file.create_earray(
        where=np_output_file.root,
        name="Labels_Variant_2",
        atom=labels_atom,
        shape=(0, 1),
        filters=filters,
    )

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            features, variant_1_labels, variant_2_labels, genotype_labels, chrom, pos, ref, indel_info = batch

            features, variant_1_labels, variant_2_labels, genotype_labels = (
                features.to(device),
                variant_1_labels.reshape(-1).to(device),
                variant_2_labels.reshape(-1).to(device),
                genotype_labels.reshape(-1).to(device),
            )

            with torch.amp.autocast("cuda"):
                variant_1_classification, variant_2_classification, genotype_classification = model(features)


            # ------------画图------------
            # Genotype
            prob_genotype = torch.softmax(genotype_classification, dim=1)

            # Variant 1
            mask_1 = (genotype_labels == 1) | (genotype_labels == 2) | (genotype_labels == 3)
            prob_variant_1 = torch.softmax(variant_1_classification[mask_1], dim=1)

            # Variant 2
            mask_2 = (genotype_labels == 3)
            prob_variant_2 = torch.softmax(variant_2_classification[mask_2], dim=1)

            if prob_genotype.cpu().numpy().size !=0:
                y_score_genotype_np.append(prob_genotype.cpu().numpy())
                y_true_genotype_np.append(genotype_labels.cpu().numpy().reshape(-1, 1))
            if prob_variant_1.cpu().numpy().size !=0:
                y_score_variant_1_np.append(prob_variant_1.cpu().numpy())
                y_true_variant_1_np.append(variant_1_labels[mask_1].cpu().numpy().reshape(-1, 1))
            if prob_variant_2.cpu().numpy().size !=0:
                y_score_variant_2_np.append(prob_variant_2.cpu().numpy())
                y_true_variant_2_np.append(variant_2_labels[mask_2].cpu().numpy().reshape(-1, 1))


            # ------------统计------------

            test_variant_1_pred = torch.argmax(variant_1_classification, dim=-1)
            test_variant_2_pred = torch.argmax(variant_2_classification, dim=-1)
            test_genotype_pred = torch.argmax(genotype_classification, dim=-1)

            test_variant_1_predictions.extend(test_variant_1_pred.cpu().numpy())
            test_variant_2_predictions.extend(test_variant_2_pred.cpu().numpy())
            test_genotype_predictions.extend(test_genotype_pred.cpu().numpy())
            test_variant_1_labels.extend(variant_1_labels.cpu().numpy())
            test_variant_2_labels.extend(variant_2_labels.cpu().numpy())
            test_genotype_labels.extend(genotype_labels.cpu().numpy())
            test_chrom.extend(chrom)
            test_pos.extend(pos)
            test_ref.extend(ref)
            test_indel_info.extend(indel_info)

            test_variant_1_acc += (test_variant_1_pred.detach() == variant_1_labels.detach()).sum().item()
            test_variant_2_acc += (test_variant_2_pred.detach() == variant_2_labels.detach()).sum().item()
            test_genotype_acc += (test_genotype_pred == genotype_labels).sum().item()
            test_num += features.size(0)

            logger = setup_logger(log_file)
            logger.info(f"Batch {batch_idx + 1}/{len(test_dataloader)} processed.")

    logger = setup_logger(log_file, mode="a")
    end_time = time.time()
    consum_time = end_time - start_time
    hours, remainder = divmod(consum_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"[TIME TAKEN] All batches processed successfully! Total time: {hours:.0f}h{minutes:.0f}m{seconds:.0f}s")


    test_1_acc = test_variant_1_acc / test_num
    test_2_acc = test_variant_2_acc / test_num
    test_genotype_acc = test_genotype_acc / test_num

    test_df = variant_df(test_chrom, test_pos, test_ref, test_indel_info, test_variant_1_predictions, test_variant_1_labels, test_variant_2_predictions, test_variant_2_labels, test_genotype_predictions, test_genotype_labels, args["call_bam"], args["output_vcf"], args["reference"])
    # test_df.to_csv(os.path.join(output_path, "test_df.csv"), index=False)

    test_df_snp = test_df[~test_df["Label"].str.contains("Insert|Deletion", regex=True)]
    test_df_indel = test_df[test_df["Label"].str.contains("Insert|Deletion", regex=True)]

    test_count = len(test_df)
    test_variant_count = (test_df["Label_Genotype"] != 0).sum()

    test_variant, test_sensitivity, test_precision, test_f1_score = calculate_metrics(test_df)
    test_variant_snp, test_sensitivity_snp, test_precision_snp, test_f1_score_snp = calculate_metrics(test_df_snp)
    test_variant_indel, test_sensitivity_indel, test_precision_indel, test_f1_score_indel = calculate_metrics(test_df_indel)


    for tables_data in tables_data_list:
        tables_data.close()


    logger.info(f"测试集ACC: Genotype: {test_genotype_acc:.6f}\tVariant 1: {test_1_acc:.6f}\tVariant 2: {test_2_acc:.6f}")
    logger.info(f"Total: {test_count}")
    logger.info(f"Total number of variants: {test_variant_count}")
    logger.info(f"True negative: {test_variant["True negative"]}\tTrue positive: {test_variant['True positive']}\tFalse positive: {test_variant['False positive']}\tFalse negative: {test_variant['False negative']}")
    logger.info(f"Sensitivity: {test_sensitivity:.6f}\tPrecision: {test_precision:.6f}\tF1 score: {test_f1_score:.6f}")
    logger.info(f"SNP:")
    logger.info(f"True negative: {test_variant_snp["True negative"]}\tTrue positive: {test_variant_snp['True positive']}\tFalse positive: {test_variant_snp['False positive']}\tFalse negative: {test_variant_snp['False negative']}")
    logger.info(f"Sensitivity: {test_sensitivity_snp:.6f}\tPrecision: {test_precision_snp:.6f}\tF1 score: {test_f1_score_snp:.6f}")
    logger.info(f"INDEL:")
    logger.info(f"True negative: {test_variant_indel["True negative"]}\tTrue positive: {test_variant_indel['True positive']}\tFalse positive: {test_variant_indel['False positive']}\tFalse negative: {test_variant_indel['False negative']}")
    logger.info(f"Sensitivity: {test_sensitivity_indel:.6f}\tPrecision: {test_precision_indel:.6f}\tF1 score: {test_f1_score_indel:.6f}")

    consum_time = time.time() - start_time
    hours, remainder = divmod(consum_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"[TIME TAKEN] Total time: {hours:.0f}h{minutes:.0f}m{seconds:.0f}s")

    np_output_file.close()
